from typing import Literal, List

import torch
from torch import nn

class LinearDecoder(nn.Module):
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