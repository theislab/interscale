import torch
import torch.nn as nn
from typing import Optional, Type
from .base_module import BaseModule

class LocalBaseModule(BaseModule):
    """Base class for local models that process individual nodes or neighborhoods"""
    
    def __init__(self, cfg, class_weights=None, **model_kwargs):
        super().__init__(cfg, class_weights, **model_kwargs)
        self.model_type = 'LocalBase'
        
    def forward(self, x, edge_index):
        """Process local node features and return node embeddings
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
        Returns:
            h_node: Node embeddings [N, E]
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _common_step(self, batch):
        """Process local features and compute metrics"""
        h_node = self.forward(batch.x, batch.edge_index)
        
        # Prepare ground truth
        if 'classification' in self.prediction_task:
            y_true = batch.y
        elif 'regression' in self.prediction_task:
            y_true = batch.x
        else:
            raise ValueError("Invalid prediction task. Must be 'classification' or 'regression'")
            
        # Make predictions
        out = self.graph_pred_linear(h_node)
        
        # Compute metrics
        if 'classification' in self.prediction_task:
            return self._common_step_classification_metrics(out, y_true, None)
        elif 'regression' in self.prediction_task:
            return self._common_step_regression_metrics(out, y_true, None)

class GlobalBaseModule(BaseModule):
    """Base class for global models that process entire graphs"""
    
    def __init__(self, cfg, class_weights=None, **model_kwargs):
        super().__init__(cfg, class_weights, **model_kwargs)
        self.model_type = 'GlobalBase'
        
    def forward(self, x, edge_index):
        """Process global graph features and return graph embeddings
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
        Returns:
            h_graph: Graph embeddings [1, E]
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _common_step(self, batch):
        """Process global features and compute metrics"""
        h_graph = self.forward(batch.x, batch.edge_index)
        
        # Prepare ground truth
        if 'classification' in self.prediction_task:
            y_true = batch.y[-1].unsqueeze(0)  # Use last node's label for graph-level prediction
        elif 'regression' in self.prediction_task:
            y_true = batch.x[-1].unsqueeze(0)  # Use last node's features for graph-level prediction
        else:
            raise ValueError("Invalid prediction task. Must be 'classification' or 'regression'")
            
        # Make predictions
        out = self.graph_pred_linear(h_graph)
        
        # Compute metrics
        if 'classification' in self.prediction_task:
            return self._common_step_classification_metrics(out, y_true, None)
        elif 'regression' in self.prediction_task:
            return self._common_step_regression_metrics(out, y_true, None)

class CombinedModule(BaseModule):
    """Class that can combine local and global components"""
    
    def __init__(self, 
                 cfg, 
                 class_weights=None,
                 local_module: Optional[Type[LocalBaseModule]] = None,
                 global_module: Optional[Type[GlobalBaseModule]] = None,
                 **model_kwargs):
        super().__init__(cfg, class_weights, **model_kwargs)
        self.model_type = 'Combined'
        
        # Initialize components if provided
        self.local_module = local_module(cfg, class_weights, **model_kwargs) if local_module else None
        self.global_module = global_module(cfg, class_weights, **model_kwargs) if global_module else None
        
        if self.local_module is None and self.global_module is None:
            raise ValueError("At least one of local_module or global_module must be provided")
    
    def forward(self, x, edge_index):
        """Process input through local and/or global components
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
        Returns:
            Tuple of (local_output, global_output) where either can be None
        """
        local_out = None
        global_out = None
        
        if self.local_module:
            local_out = self.local_module.forward(x, edge_index)
            
        if self.global_module:
            global_out = self.global_module.forward(x, edge_index)
            
        return local_out, global_out
    
    def _common_step(self, batch):
        """Process batch through local and/or global components and compute metrics"""
        local_out, global_out = self.forward(batch.x, batch.edge_index)
        
        # Prepare ground truth
        if 'classification' in self.prediction_task:
            y_true_local = batch.y
            y_true_global = batch.y[-1].unsqueeze(0)
        elif 'regression' in self.prediction_task:
            y_true_local = batch.x
            y_true_global = batch.x[-1].unsqueeze(0)
        else:
            raise ValueError("Invalid prediction task. Must be 'classification' or 'regression'")
        
        # Process local predictions if available
        if local_out is not None:
            local_pred = self.graph_pred_linear(local_out)
            if 'classification' in self.prediction_task:
                local_metrics = self._common_step_classification_metrics(local_pred, y_true_local, None)
            else:
                local_metrics = self._common_step_regression_metrics(local_pred, y_true_local, None)
        else:
            local_metrics = None
            
        # Process global predictions if available
        if global_out is not None:
            global_pred = self.graph_pred_linear(global_out)
            if 'classification' in self.prediction_task:
                global_metrics = self._common_step_classification_metrics(global_pred, y_true_global, None)
            else:
                global_metrics = self._common_step_regression_metrics(global_pred, y_true_global, None)
        else:
            global_metrics = None
            
        # Combine metrics if both components are present
        if local_metrics is not None and global_metrics is not None:
            # Average the losses and combine other metrics
            loss = (local_metrics[0] + global_metrics[0]) / 2
            if 'classification' in self.prediction_task:
                metrics = [
                    (local_metrics[1][0] + global_metrics[1][0]) / 2,  # accuracy
                    (local_metrics[1][1] + global_metrics[1][1]) / 2,  # f1_micro
                    (local_metrics[1][2] + global_metrics[1][2]) / 2,  # f1_macro
                    [(l + g) / 2 for l, g in zip(local_metrics[1][3], global_metrics[1][3])]  # f1_per_class
                ]
            else:
                metrics = [
                    (local_metrics[1][0] + global_metrics[1][0]) / 2,  # mse
                    (local_metrics[1][1] + global_metrics[1][1]) / 2,  # r2
                    (local_metrics[1][2] + global_metrics[1][2]) / 2   # pearson_corr
                ]
        else:
            # Use whichever metrics are available
            loss, metrics = local_metrics if local_metrics is not None else global_metrics
            
        return loss, metrics 