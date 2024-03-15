import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, MessagePassing
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, 
                 cfg,
                 num_classes,
                 dp_rate = 0.1
        ):
        super().__init__()      
        #dp_rate = cfg['dp_rate'] if cfg['dp_rate'] is not None else dp_rate
        self.num_classes = num_classes

        in_dim, hidden_dim = cfg['num_features'], cfg['hidden_dim']
        layers = []
        for l_idx in range(cfg['num_layers'] - 1):
            layers += [
                GCNConv(in_channels=in_dim, out_channels=hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_dim = hidden_dim
        
        layers += [GCNConv(in_channels=in_dim, out_channels=cfg['embed_dim'])]
        self.layers = nn.ModuleList(layers)
        self.out = Linear(cfg['embed_dim'], num_classes)

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
        z = self.out(h)
        return x, z