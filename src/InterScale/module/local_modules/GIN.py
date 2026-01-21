# PyTorch
from torch import nn
from torch_geometric.nn import GINConv, MessagePassing
import torch.nn.functional as F

from typing import Literal

# PyTorch Lightning
import pytorch_lightning as L

from InterScale.module.base import LocalModuleClass

class GIN(LocalModuleClass):
    def __init__(self,
                 n_layers: int = 2,
                 hidden_dim: int = 16,
                 dropout_local: float = 0.1,
                 **base_module_kwargs):
        
        super().__init__(**base_module_kwargs)      
        
        self.module_name = 'GIN'
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout_local = dropout_local
        
        layers = []
        in_dim = self.n_input
        hidden_dim = self.hidden_dim
        for l_idx in range(n_layers - 1):
            # Create MLP for GIN layer
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            layers += [
                GINConv(nn=mlp, train_eps=True),
                nn.Dropout(self.dropout_local)
            ]
            in_dim = hidden_dim
        
        # Final layer
        final_mlp = nn.Sequential(
            nn.Linear(in_dim, self.n_embed)
        )
        layers += [GINConv(nn=final_mlp, train_eps=True)]
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
    
    def get_model_summary(self) -> str:
        """Returns a string containing the model's parameters summary.

        Returns:
            str: Summary string with model parameters
        """
        summary = (
            f"GIN Local Component: \n"
            f"n_layers: {self.n_layers}, \n"
            f"n_hidden: {self.hidden_dim}, \n"
            f"n_embed: {self.n_embed}, \n"
            f"dropout_local: {self.dropout_local}"
        )
        return summary

