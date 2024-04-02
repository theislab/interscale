# PyTorch
import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, MessagePassing
import torch.nn.functional as F

# PyTorch Lightning
import pytorch_lightning as L

class GCN(torch.nn.Module):
    def __init__(self, 
                 cfg,
                 dp_rate = 0.1
        ):
        super().__init__()      
        #dp_rate = cfg['dp_rate'] if cfg['dp_rate'] is not None else dp_rate
        self.num_classes = cfg.get('dataset/num_classes')
        in_dim, hidden_dim, embed_dim = cfg.get('gnn/num_features'), cfg.get('gnn/hidden_dim'), cfg.get('gnn/embed_dim')

        layers = []
        for l_idx in range(cfg.get('gnn/num_layers') - 1):
            layers += [
                GCNConv(in_channels=in_dim, out_channels=hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_dim = hidden_dim
        
        layers += [GCNConv(in_channels=in_dim, out_channels=embed_dim)]
        self.layers = nn.ModuleList(layers)
        self.out = Linear(embed_dim, self.num_classes)

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
    
class LitGCN(L.LightningModule):
    def __init__(self, 
                 cfg,
                 dp_rate = 0.1
        ):
        super().__init__()      
        #dp_rate = cfg['dp_rate'] if cfg['dp_rate'] is not None else dp_rate
        self._cfg = cfg
        self.num_classes = cfg.get('dataset/num_classes')
        in_dim, hidden_dim, embed_dim = cfg.get('gnn/num_features'), cfg.get('gnn/hidden_dim'), cfg.get('gnn/embed_dim')
        self.loss_criterion = torch.nn.CrossEntropyLoss()

        layers = []
        for l_idx in range(cfg.get('gnn/num_layers') - 1):
            layers += [
                GCNConv(in_channels=in_dim, out_channels=hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_dim = hidden_dim
        
        layers += [GCNConv(in_channels=in_dim, out_channels=embed_dim)]
        self.layers = nn.ModuleList(layers)
        self.out = Linear(embed_dim, self.num_classes)

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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self._cfg.get('model/lr')), weight_decay=float(self._cfg.get('model/wd')))
        return optimizer

    def training_step(self, batch):
        print('train')
        loss = self._common_step(batch)
        self.log('train_loss', loss, batch_size=int(self._cfg.get('dataset/batch_size')))
        return loss

    def validation_step(self, batch):
        print('val')
        loss = self._common_step(batch)
        self.log('val_loss', loss, batch_size=int(self._cfg.get('dataset/batch_size')))
        return loss

    def test_step(self, batch):
        loss = self._common_step(batch)
        self.log('test_loss', loss, batch_size=int(self._cfg.get('dataset/batch_size')))
        return loss

    def _common_step(self, batch):
        """Shared step between train, val and test.
        """
        # Forward pass
        gnn_x, gnn_z = self.forward(batch.x, batch.edge_index) # [B, C] with C being the number of tasks to predict, e.i.        
        # Calculate loss function
        loss = self.loss_criterion(gnn_z, batch.y)

        return loss