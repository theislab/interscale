from InterScale.module.base._base_module import BaseModuleClass
from abc import abstractmethod
from typing import Literal
from InterScale.tl import apply_mask
from scvi.nn import Encoder
import torch.nn as nn

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
            
    def predict(self,
                local_embedding,
                prediction_level: Literal["node", "graph"] | None = None):
        """Predict with the decoder.
        
        Parameters
        ----------
        local_embedding: torch.Tensor
            Size: [N, E]
        prediction_level: Literal["node", "graph"]
        """ 
        return self.decoder.forward(local_embedding)
        
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
            Size: [B, C] (classification) or [B, F] (regression)
        y_true: torch.Tensor 
            Size: [B, ] (classification) or [B, F] (regression)
        """
        # Mask nodes 
        batch_masked, mask_idx = self._common_step_masking(batch)
        
        local_embedding = self.forward(batch_masked.x, batch_masked.edge_index)
        y_pred = self.decoder.forward(local_embedding)
        
        assert y_pred.shape[0] == len(batch.obs_names), f"Mismatch: y_pred.shape: {y_pred.shape[0]}, batch.obs_names: {len(batch.obs_names)}"
        assert y_pred.shape[1] == self.n_output, f"Mismatch: y_pred.shape: {y_pred.shape[1]}, self.n_output: {self.n_output}"
        assert y_pred.isnan().sum() == 0, "y_pred contains NaN values"
        
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
        elif module_name == 'GIN':
            from InterScale.module.local_modules import GIN
            return GIN(n_layers = params['num_layers'],
                       hidden_dim = params['hidden_dim'],
                       dropout_local = params['dropout_local'],
                       **kwargs)
        elif module_name == 'SCVI':
            print("Creating SCVI Local Module")
            from InterScale.module.local_modules import SCVILocalModule
            n_input = kwargs.pop('n_input')
            n_embed = kwargs.pop('n_embed')
            return SCVILocalModule(
                n_input=n_input,
                n_latent=n_embed,
                n_layers=params.get('num_layers', 2),
                n_hidden=params.get('hidden_dim', 128),
                dropout_rate=params.get('dropout_local', 0.1),
                **kwargs
            )
        # elif module_name == 'Precomputed':
        #     print(f"Creating Precomputed Embedding Module from {cfg.dataset.precomputed}")
        #     from InterScale.module.local_modules import PrecomputedEmbeddingModule
        #     return PrecomputedEmbeddingModule(
        #         **kwargs
        #     )
        # Add more elifs for other modules
        else:
            raise ValueError(f"Unknown local module name: {module_name}")
        # # Add more elifs for other modules
        # else:
        #     raise ValueError(f"Unknown local module name: {module_name}")

    
