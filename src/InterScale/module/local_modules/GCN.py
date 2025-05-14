# PyTorch
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing
import torch.nn.functional as F

from typing import Literal

# PyTorch Lightning
import pytorch_lightning as L

from InterScale.module.base import LocalModuleClass

class GCN(LocalModuleClass):
    def __init__(self,
                 n_layers: int = 2,
                 hidden_dim: int = 16,
                 dropout_local: float = 0.1,
                 **base_module_kwargs):
        
        super().__init__(**base_module_kwargs)      
        
        self.module_name = 'GCN'
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout_local = dropout_local
        
        layers = []
        in_dim = self.n_input
        hidden_dim = self.hidden_dim
        for l_idx in range(n_layers - 1):
            layers += [
                GCNConv(in_channels=in_dim, out_channels=hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_local)
            ]
            in_dim = hidden_dim
        
        layers += [GCNConv(in_channels=in_dim, out_channels=self.n_embed)]
        self.layers = nn.ModuleList(layers)
            
    def forward(self, x, edge_index):
        """
        Parameters:
        -----------
            x: gene expression (var x obs)
            edge_index: Adjacency matrix (n x obs)
            
        Returns:
        --------
            h: Embeddings (n x embed_dim)
        """
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        h = F.relu(x)
        return h
    