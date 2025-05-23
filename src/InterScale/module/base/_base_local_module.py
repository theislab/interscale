from InterScale.module.base._base_module import BaseModuleClass
from abc import abstractmethod
from typing import Literal
from InterScale.tl import apply_mask

import torch

class LocalModuleClass(BaseModuleClass):
    def __init__(self,
                 **base_module_kwargs):
        
        super().__init__(**base_module_kwargs)
        
        self.registered_local_component = True
        self.registered_global_component = False
        
    @abstractmethod
    def forward(self):
        """Forward pass."""
        
    def _common_step(self,
                     batch,
                     prediction_task: str,
                     prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.
        
        Returns
        -------
        local_embedding: torch.Tensor 
            Size: [N, E]
        global_embedding: torch.Tensor 
            Size: [N, E]
        y_pred: torch.Tensor 
            Size: [N, C] (classification) or [N, F] (regression)
        y_true: torch.Tensor 
            Size: [N, ] (classification) or [N, F] (regression)
        """
        # Mask nodes 
        if self.pct_mask_nodes > 0:
            batch_masked, mask_idx = apply_mask(batch)
        else:
            # pretend as if all nodes are masked
            mask_idx = torch.arange(batch.x.shape[0])
            batch_masked = batch
        
        local_embedding = self.forward(batch_masked.x, batch_masked.edge_index)
        y_pred = self.decoder.forward(local_embedding)
        
        assert y_pred.shape[0] == len(batch.obs_names), f"Mismatch: y_pred.shape: {y_pred.shape[0]}, batch.obs_names: {len(batch.obs_names)}"
        assert y_pred.shape[1] == self.n_output, f"Mismatch: y_pred.shape: {y_pred.shape[1]}, self.n_output: {self.n_output}"
        
        y_pred = y_pred[mask_idx]
        
        if 'classification' in prediction_task:
            y_true = batch.y[mask_idx] # batch without mask because constant otherwise
            assert y_true.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true
            
        if 'regression' in prediction_task:
            y_true = batch.x[mask_idx] # batch without mask because constant otherwise
            assert y_true.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true
            
        assert False, "Prediction task not supported"
    
    def get_local_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)

    # acts as a factory method to create a module from a config
    @staticmethod
    def from_config(cfg, **kwargs):
        module_name = cfg.model.local_component.name
        params = cfg.model.local_component.parameters.copy()  # Make a copy to avoid modifying the original
            
        if module_name == 'GCN':
            from InterScale.module.local_modules import GCN
            return GCN(n_layers = params['num_layers'],
                       hidden_dim = params['hidden_dim'],
                       dropout_local = params['dropout_local'],
                       **kwargs)
        # Add more elifs for other modules
        else:
            raise ValueError(f"Unknown local module name: {module_name}")
