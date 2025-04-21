# PyTorch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, MessagePassing
import torch.nn.functional as F

import typing as List

# PyTorch Lightning
import pytorch_lightning as L

from InterScale.module.base import LocalComponent

class GCN(LocalComponent):
    def __init__(self,
        n_layers: int = 2,
        hidden_dim: int = 16,
        ):
        super().__init__()      
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        layers = []
        for l_idx in range(n_layers - 1):
            layers += [
                GCNConv(in_channels=self.n_input, out_channels=self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout)
            ]
            in_dim = self.n_input
        
        layers += [GCNConv(in_channels=self.n_input, out_channels=self.embed_dim)]
        self.layers = nn.ModuleList(layers)
            
    def forward(self, x, edge_index):
        """
        Input:
            x: Adjacency matrix (n x obs)
            edge_index: gene expressiong (var x obs)
        """
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        h = F.relu(x)
        return h
    