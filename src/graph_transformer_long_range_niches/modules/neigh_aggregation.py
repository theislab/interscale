import torch
import torch.nn as nn
from torch_scatter import scatter_mean  # For neighborhood aggregation
import pytorch_lightning as L

from graph_transformer_long_range_niches.modules import BaseModule
from typing import List

class LitNeighAggregation(BaseModule):
    def __init__(self, 
                 cfg, 
                 class_weights: List = None, 
                 **model_kwargs
        ):
        super().__init__(cfg, class_weights, **model_kwargs)
        
        self._cfg = cfg
        
        # Define output dimension based on the task
        self.output_dim = cfg.model.hidden_dim if hasattr(cfg.model, 'hidden_dim') else 64
        
        self.model_type = 'NeighborhoodAggregation'
        
        # MLP for transforming neighborhood features
        self.neigh_transform = nn.Sequential(
            nn.Linear(cfg.dataset.num_features, self.output_dim * 2),
            nn.ReLU(),
            nn.Linear(self.output_dim * 2, self.output_dim)
        )
        
        self.norm_input = nn.LayerNorm(self.output_dim)
        
        # Final prediction layer
        if 'classification' in self.prediction_task:
            self.pred_linear = nn.Linear(self.output_dim, cfg.dataset.num_classes)
        elif 'regression' in self.prediction_task:
            self.pred_linear = nn.Linear(self.output_dim, 1)
        else:
            raise ValueError("Invalid prediction task. Must be 'classification' or 'regression'")
        
    def aggregate_neighbors(self, x, edge_index):
        """
        Perform simple mean aggregation over neighbors.
        
        Parameters:
        - x: Tensor of shape [N, F] (node features)
        - edge_index: Tensor of shape [2, E] (edges)

        Returns:
        - Aggregated node features of shape [N, F]
        """
        row, col = edge_index  # row: sources, col: targets
        aggregated_x = scatter_mean(x[row], col, dim=0, out=torch.zeros_like(x))
        return aggregated_x

    def forward(self, batched_data):
        """
        Input: 
            batched_data: Pytorch geometric object 
                batched_data.x = [N, F]
                batched_data.edge_index = [2, E]
        """
        # Aggregate neighbor features
        h_neighbor = self.aggregate_neighbors(batched_data.x, batched_data.edge_index)
        
        # Transform neighborhood features using MLP
        h_neighbor = self.neigh_transform(h_neighbor)
        
        # Normalize features
        h_node = self.norm_input(h_neighbor)

        # Make predictions
        if 'graph' in self.prediction_task:
            # For graph-level prediction, use the last node's features
            out = self.pred_linear(h_node[-1].unsqueeze(0))
        else:  # node-level prediction
            out = self.pred_linear(h_node)
            
        return h_node, out

    def configure_optimizers(self):
        return self.common_configure_optimizers()

    def training_step(self, batch):
        return self.common_training_step(batch)

    def validation_step(self, batch):
        return self.common_validation_step(batch)

    def test_step(self, batch):
        return self.common_test_step(batch)
    
    def _common_step(self, batch):
        """Shared step between train, val and test.
        """
        # Run forward pass
        h_node, out = self.forward(batch)
        
        # Prepare ground truth
        if 'classification' in self.prediction_task:
            if 'node' in self.prediction_task:
                y_true = batch.y
            elif 'graph' in self.prediction_task:
                y_true = batch.y[-1].unsqueeze(0)
        elif 'regression' in self.prediction_task:
            if 'node' in self.prediction_task:
                y_true = batch.x
            elif 'graph' in self.prediction_task:
                y_true = batch.x[-1].unsqueeze(0)
        else:
            raise ValueError("Invalid prediction task. Must be 'classification' or 'regression'")

        if 'classification' in self.prediction_task:
            return self._common_step_classification_metrics(out, y_true, None)
        elif 'regression' in self.prediction_task:
            return self._common_step_regression_metrics(out, y_true, None) 