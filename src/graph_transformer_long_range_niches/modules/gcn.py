# PyTorch
import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, MessagePassing
import torch.nn.functional as F

import torchmetrics

# PyTorch Lightning
import pytorch_lightning as L

from graph_transformer_long_range_niches.tl.evaluation import accuracy 

from graph_transformer_long_range_niches.tl.scheduler import CosineWarmupScheduler
    
class LitGCN(L.LightningModule):
    def __init__(self, 
                 cfg
        ):
        super().__init__()      
        #dp_rate = cfg['dp_rate'] if cfg['dp_rate'] is not None else dp_rate
        self._cfg = cfg
        self.num_classes = cfg.get('dataset/num_classes')
        in_dim, hidden_dim, embed_dim = cfg.get('gnn/num_features'), cfg.get('gnn/hidden_dim'), cfg.get('gnn/embed_dim')
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.lr = float(self._cfg.get('optim/lr'))
        self.wd = float(self._cfg.get('optim/wd'))
        # Define metrics
        self.accurary = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes) 

        layers = []
        for l_idx in range(cfg.get('gnn/num_layers') - 1):
            layers += [
                GCNConv(in_channels=in_dim, out_channels=hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.get('gnn/dropout'))
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=int(self._cfg.get('optim/warm_up')),
                                             max_epochs=int(self._cfg.get('model/n_epochs')))
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch):
        loss, acc, f1_score = self._common_step(batch)
        self.log_dict({'train_loss': loss, 'train_acc': acc, 'train_f1': f1_score}, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        loss, acc, f1_score = self._common_step(batch)
        self.log_dict({'val_loss': loss, 'val_acc': acc, 'val_f1': f1_score}, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch):
        loss, acc, f1_score = self._common_step(batch)
        self.log_dict({'test_loss': loss, 'test_acc': acc, 'test_f1': f1_score}, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
        return loss

    def _common_step(self, batch):
        """Shared step between train, val and test.
        """
        # Forward pass
        gnn_x, gnn_z = self.forward(batch.x, batch.edge_index) # [B, C] with C being the number of tasks to predict, e.i.        
        # Calculate loss function
        loss = self.loss_criterion(gnn_z, batch.y)
        print('predicted and true: ', gnn_z.argmax(dim=1)[:10], batch.y[:10])
        acc = self.accurary(gnn_z, batch.y)
        f1_score = self.f1_score(gnn_z, batch.y)

        return loss, acc, f1_score