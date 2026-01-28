from scvi.nn import Encoder
from InterScale.module.base import LocalModuleClass
import torch



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