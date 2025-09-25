from typing import Literal, List
import torch
import torch.nn as nn 

class LinearLSEDecoder(nn.Module):
    """
    Adapted from: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/linear.py#L50
    Date: 25.08.2025
    
    Applies a log-sum-exp transformation to the incoming data:
    y_g = log(sum_k exp(z_k * W_{k,g})) + b_g

    Parameters
    ----------
    n_input: int
        Number of input (latent) features.
    n_output: int
        Number of output features.
    """

    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        # Weight and bias similar to Linear layer, but we handle them manually
        self.weight = nn.Parameter(torch.Tensor(n_output, n_input))
        self.bias = nn.Parameter(torch.Tensor(n_output))
        self.n_input = n_input
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / fan_in**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, n_input]
        Returns: [batch_size, n_output]
        """
        # contrib shape: [batch_size, n_input, n_output]
        contrib = x.unsqueeze(2) * self.weight.t().unsqueeze(0)
        
        # Apply log-sum-exp across latent dimension (dim=1)
        y = torch.logsumexp(contrib, dim=1) - math.log(self.n_input) # [batch_size, n_output]

        # Add bias
        return y + self.bias
    

class LinearDecoder(nn.Module):
    """Applies an affine linear transformation to the incoming data: y = xA^T + b
    
    Parameters
    ----------
    n_input: int
        Number of input features.
    n_output: int
        Number of output features.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
    ):
        super().__init__()
        self.decoder = nn.Linear(n_input, n_output)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
class NonLinearDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_dims: List[int] = [128, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers_dim = [n_input] + hidden_dims + [n_output]
        
        self.decoder = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.LayerNorm(n_out),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ) for n_in, n_out in zip(layers_dim[:-1], layers_dim[1:-1])],
            nn.Linear(layers_dim[-2], layers_dim[-1])  # Final layer without activation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)