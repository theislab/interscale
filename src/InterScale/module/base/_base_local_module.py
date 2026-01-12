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
                prediction_level: Literal["node", "graph"]):
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
        # # Add more elifs for other modules
        # else:
        #     raise ValueError(f"Unknown local module name: {module_name}")

class SCVILocalModule(LocalModuleClass):
    def __init__(self, 
                 n_input: int,
                 n_latent: int,
                 n_layers: int = 2,
                 n_hidden: int = 128,
                 dropout_rate: float = 0.1,
                 **base_module_kwargs):
        """
        Wrapper for scVI Encoder to act as a LocalModule.
        """

        base_module_kwargs['n_input'] = n_input
        base_module_kwargs['n_embed'] = n_latent
        super().__init__(**base_module_kwargs)
        
        # scVI Encoder: maps input counts to latent space
        # Reference: https://docs.scvi-tools.org/en/stable/api/reference/scvi.nn.Encoder.html
        self.encoder = Encoder(
            n_input=n_input,
            n_output=n_latent, # n_latent corresponds to your n_embed
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution="normal" # Standard VAE approach
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None):
        """
        Forward pass. edge_index is accepted for compatibility but ignored.
        
        Parameters
        ----------
        x: torch.Tensor
            Input features (e.g. raw counts or normalized data).
        edge_index: torch.Tensor
            Graph connectivity (ignored by scVI encoder).
        """
        # scVI encoder returns (mean, variance, latent_sample)
        # We take the mean (q_m) or the sample (z) as the embedding.
        # Usually, for downstream tasks, the mean is more stable.
        q_m, q_v, z = self.encoder(x)
        
        return {
            'embedding': z, 
            'q_m': q_m,
            'q_v': q_v
        }
    
    def loss_kl(self, outputs_dict: dict) -> torch.Tensor:
        """
        Calculates the analytical KL Divergence for the normal distribution.
        """
        q_m = outputs_dict['q_m']
        q_v = outputs_dict['q_v'] # This is the variance from scvi.nn.Encoder
        
        # Formula for KL divergence between N(mu, sigma^2) and N(0, 1):
        # KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        # Note: scvi.nn.Encoder returns variance (q_v), not log-variance.
        kl_element = 0.5 * (q_v + q_m.pow(2) - 1 - torch.log(q_v + 1e-8))
        
        return torch.mean(torch.sum(kl_element, dim=1))    

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict method to act as a placeholder for the local decoder.
        If your local module has a specific decoder (e.g. for ZINB), 
        implement the reconstruction logic here.
        """
        if hasattr(self, 'decoder'):
            return self.decoder(z)
        # If no internal decoder, we assume it's handled by the CombinedModule's loss_fn
        return z

    def get_model_summary(self) -> str:
        try:
            latent_dim = self.encoder.mean_encoder.out_features
        except AttributeError:
            latent_dim = "Unknown"

        return (
            f"scVI Encoder Wrapper:\n"
            f"  - Latent Dim (n_embed): {latent_dim}\n"
            f"  - Full Architecture:\n{str(self.encoder)}" 
        )


    # def get_model_summary(self) -> str:
    #     """
    #     Overrides the base method to provide scVI-specific details.
    #     """
    #     # Accessing internal scVI encoder attributes for the summary
    #     # Note: scVI stores dimensions in specific attributes, usually accessible via the module
    #     try:
    #         latent_dim = self.encoder.mean_encoder.out_features
    #     except AttributeError:
    #         latent_dim = "Unknown"

    #     return (
    #         f"scVI Encoder Wrapper:\n"
    #         f"  - Latent Dim (n_embed): {latent_dim}\n"
    #         f"  - Full Architecture:\n{str(self.encoder)}" 
    #     )

    @staticmethod
    def from_config(cfg, **kwargs):
        """
        Factory method to instantiate the SCVI-based local module.
        """
        params = cfg.model.local_component.parameters

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
    
class PrecomputedEmbeddingModule(LocalModuleClass):
    def __init__(self, n_embed: int, **kwargs):
        """
        Module for using frozen, pre-computed embeddings.
        """
        super().__init__(n_embed=n_embed, **kwargs)
        # Dummy parameter to avoid optimizer errors if no other params exist
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, **kwargs):
        """
        In this case, x is expected to be already the embedding.
        """
        return x

    def get_model_summary(self) -> str:
        return "Local Module: Precomputed/Frozen Embeddings (Pass-through)"

    @staticmethod
    def from_config(cfg, **kwargs):
        return PrecomputedEmbeddingModule(
            **kwargs
        )