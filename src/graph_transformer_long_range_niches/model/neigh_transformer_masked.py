import torch
import torch.nn as nn
from torch_scatter import scatter_mean  # For neighborhood aggregation
import pytorch_lightning as L

from graph_transformer_long_range_niches.modules import TransformerNodeEncoder, BaseModule, TransformerNodeEncoderHook
from graph_transformer_long_range_niches.tl.utils import pad_batch, create_transformer_attention_mask_from_edges
from graph_transformer_long_range_niches.tl.masking import apply_mask
from typing import List

class LitNeighTransformerMasked(BaseModule):
    def __init__(self, 
                 cfg, 
                 class_weights: List = None, 
                 **model_kwargs
        ):
        super().__init__(cfg, class_weights, **model_kwargs)
        
        self._cfg = cfg
        
        assert cfg.transformer.d_model, "Transformer embedding dimension must be defined."
        
        self.model_type = 'NeighborhoodAggregation_Transformer'
        self.output_dim = cfg.transformer.d_model
        
        # MLP for transforming neighborhood features
        self.neigh_transform = nn.Sequential(
            nn.Linear(cfg.dataset.num_features, self.output_dim * 2),
            nn.ReLU(),
            nn.Linear(self.output_dim * 2, self.output_dim)
        )
        
        self.norm_input = nn.LayerNorm(self.output_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.output_dim], requires_grad=True))

        # Transformer encoder initialization
        self.transformer_encoder = TransformerNodeEncoderHook(cfg)
        
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
        # Aggregate neighbor features instead of GCN
        h_neighbor = self.aggregate_neighbors(batched_data.x, batched_data.edge_index)
        
        # Transform neighborhood features using MLP
        h_neighbor = self.neigh_transform(h_neighbor)
        
        h_node = self.norm_input(h_neighbor)

        if self.masked_nodes:
            keep_indices = batched_data.mask
        else:
            keep_indices = None

        # Padding for transformer input
        padded_h_node, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, 
            batched_data.batch, 
            self.transformer_encoder.max_seq_len, 
            get_mask=self.masked_nodes,
            keep_indices=keep_indices
        )

        if self._cfg.transformer.long_range_attention:
            attention_mask = create_transformer_attention_mask_from_edges(
                batched_data.edge_index, 
                len(batched_data.obs_names), 
                batched_data.batch, 
                index_nodes, 
                self.transformer_encoder.n_heads
            )
            # Convert attention_mask to same dtype as src_padding_mask
            attention_mask = attention_mask.to(dtype=src_padding_mask.dtype)
        else:
            attention_mask = None
            
        transformer_out = padded_h_node
        transformer_out, src_padding_mask = self.transformer_encoder(
            transformer_out, 
            src_padding_mask, 
            mask=attention_mask
        )

        ## Graph-level prediction
        if 'graph' in self.prediction_task:
            cls = transformer_out[-1, :, :]  # [B, E]
            out = self.graph_pred_linear(cls)
            return h_node, out, index_nodes

        ## Node-level prediction
        elif 'node' in self.prediction_task:
            h_graph = transformer_out[:-1]  # [S, B, E]
            h_graph = torch.permute(h_graph, (1, 0, 2))  # [B, S, E]
            src_padding_mask = src_padding_mask[:, :-1]  # [B, S]
            masked_output = h_graph[~src_padding_mask]  # [N, E]
            out = self.graph_pred_linear(masked_output)
            return h_node, out, index_nodes

        else:
            raise Exception('Choose a valid prediction task (graph or node).')
        
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
        # Mask nodes 
        input_data_masked, mask_idx = apply_mask(batch)
        # Run forward pass on masked data
        out_gnn, out_transformer, pad_index_nodes = self.forward(input_data_masked)
        
        y_true = []
        adjusted_mask_idx = []  # Track new positions of masked nodes
        current_offset = 0
        
        start = 0
        
        for i in range(batch.batch[-1] + 1):
            mask = batch.batch.eq(i)
            pad_indices = torch.tensor(pad_index_nodes[i], device=batch.x.device)
            end = start + len(pad_indices)

            for idx in mask_idx[start:end]:
                # Convert to tensor if not already
                if idx in pad_indices:
                    new_idx = torch.where(pad_indices == idx)[0].item()
                    adjusted_mask_idx.append(new_idx + current_offset)
            
            current_offset += len(pad_indices)
            start = end
            
            if 'classification' in self.prediction_task:
                if 'node' in self.prediction_task:
                    y_true += batch.y[mask][pad_index_nodes[i]].clone().detach()
                elif 'graph' in self.prediction_task:
                    y_true.append(batch.y[mask][-1].clone().detach())
            elif 'regression' in self.prediction_task:
                y_true += batch.x[mask][pad_index_nodes[i]].clone().detach()
            else:
                raise Exception('Choose a valid prediction tasks (graph or node).')
                    
        y_true = torch.stack(y_true)
        adjusted_mask_idx = torch.tensor(adjusted_mask_idx, device=y_true.device)

        if 'classification' in self.prediction_task:
            if 'node' in self.prediction_task:
                return self._common_step_classification_metrics(out_transformer, y_true, adjusted_mask_idx)
            elif 'graph' in self.prediction_task:
                return self._common_step_classification_metrics(out_transformer, y_true, None)

        if 'regression' in self.prediction_task:
            return self._common_step_regression_metrics(out_transformer, y_true, adjusted_mask_idx)