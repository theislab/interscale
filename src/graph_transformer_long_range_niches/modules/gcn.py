# PyTorch
import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, MessagePassing
import torch.nn.functional as F

import typing as List

import torchmetrics

# PyTorch Lightning
import pytorch_lightning as L
from graph_transformer_long_range_niches.tl.scheduler import CosineWarmupScheduler
    
class LitGCN(L.LightningModule):
    def __init__(self,
                 cfg,
                 class_weights: List = None,
        ):
        super().__init__()      
        #dp_rate = cfg['dp_rate'] if cfg['dp_rate'] is not None else dp_rate
        self._cfg = cfg
        self.save_hyperparameters()
        self.num_classes = cfg.dataset.num_classes
        in_dim, hidden_dim, embed_dim = cfg.dataset.num_features, cfg.gnn.hidden_dim, cfg.gnn.embed_dim
        self.lr = float(self._cfg.optim.lr)
        self.wd = float(self._cfg.optim.wd)
        self.prediction_task = cfg.dataset.prediction_task
        self.num_classes = cfg.dataset.num_classes
        self.num_features = cfg.dataset.num_features

        if 'classification' in self.prediction_task:
            if cfg.optim.loss == 'CrossEntropy':
                self.loss = torch.nn.CrossEntropyLoss()
            elif cfg.optim.loss == 'WeightedCE':
                self.loss = torch.nn.CrossEntropyLoss(torch.from_numpy(class_weights))
            else:
                raise Exception("Classification must be run with CrossEntropy or WeightedCE loss.")
        elif 'regression' in self.prediction_task:
            if cfg.optim.loss == 'MSELoss':
                self.loss = torch.nn.MSELoss()
            elif cfg.optim.loss == 'GaussianNLL':
                self.loss = torch.nn.GaussianNLLLoss()
            elif cfg.optim.loss == 'SmoothL1':
                self.loss = torch.nn.SmoothL1Loss()
            else:
                raise Exception("Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss.")
        else:
            raise Exception("Prediction task must define 'classification' or 'regression'.")

         # Define metrics
        if 'classification' in self.prediction_task:
            self.accurary = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.f1_score_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="micro")
            self.f1_score_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
            self.f1_score_per_class = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average=None)
        elif 'regression' in self.prediction_task:
            self.mse = torchmetrics.MeanSquaredError()
            self.r2 = torchmetrics.R2Score(num_outputs=self.num_features, multioutput = 'uniform_average')
            self.pearson_corr = torchmetrics.PearsonCorrCoef(num_outputs=self.num_features)

        layers = []
        for l_idx in range(cfg.gnn.num_layers - 1):
            layers += [
                GCNConv(in_channels=in_dim, out_channels=hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.gnn.dropout)
            ]
            in_dim = hidden_dim
        
        layers += [GCNConv(in_channels=in_dim, out_channels=embed_dim)]
        self.layers = nn.ModuleList(layers)
        if 'classification' in self.prediction_task:
            self.out = Linear(embed_dim, self.num_classes)
        elif 'regression' in self.prediction_task:
            self.out = Linear(embed_dim, self.num_features)

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
                                             warmup=int(self._cfg.optim.warm_up),
                                             #max_epochs=int(self._cfg.get('model/n_epochs')))
                                             max_epochs=1000000)
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        #loss, acc, f1_score_micro, f1_score_macro, f1_score_per_class = self._common_step(batch)
        loss, metric_list = self._common_step(batch)
        if 'classification' in self.prediction_task:
            acc, f1_score_micro, f1_score_macro, f1_score_per_class = metric_list
            log_dict = {
                'train_loss': loss,
                'train_acc': acc,
                'train_f1_micro/avg': f1_score_micro,
                'train_f1_macro/avg': f1_score_macro,
            }
            for class_idx in range(self.num_classes):
                log_dict[f'train_f1/class_{class_idx}'] = f1_score_per_class[class_idx]
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        elif 'regression' in self.prediction_task:
            mse, r2, pearson_corr = metric_list
            log_dict = {
                'train_mse': mse,
                'train_r2': r2,
                'val_pearson_corr': pearson_corr
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric_list = self._common_step(batch)
        if 'classification' in self.prediction_task:
            acc, f1_score_micro, f1_score_macro, f1_score_per_class = metric_list
            log_dict = {
                'val_loss': loss,
                'val_acc': acc,
                'val_f1_micro/avg': f1_score_micro,
                'val_f1_macro/avg': f1_score_macro,
            }
            for class_idx in range(self.num_classes):
                log_dict[f'val_f1/class_{class_idx}'] = f1_score_per_class[class_idx]
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        elif 'regression' in self.prediction_task:
            mse, r2, pearson_corr = metric_list
            log_dict = {
                'val_mse': mse,
                'val_r2': r2,
                'val_pearson_corr': pearson_corr
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch):
        loss, metric_list = self._common_step(batch)
        if 'classification' in self.prediction_task:
            acc, f1_score_micro, f1_score_macro, f1_score_per_class = metric_list
            log_dict = {
                'test_loss': loss,
                'test_acc': acc,
                'test_f1_micro/avg': f1_score_micro,
                'test_f1_macro/avg': f1_score_macro,
            }
            for class_idx in range(self.num_classes):
                log_dict[f'test_f1/class_{class_idx}'] = f1_score_per_class[class_idx]
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        elif 'regression' in self.prediction_task:
            mse, r2, pearson_corr = metric_list
            log_dict = {
                'test_mse': mse,
                'test_r2': r2,
                'test_pearson_corr': pearson_corr
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss


    def _common_step(self, batch):
        """Shared step between train, val and test.
        """
        # Forward pass
        gnn_x, gnn_z = self.forward(batch.x, batch.edge_index) # [B, C] with C being the number of tasks to predict, e.i.        
        
        if 'classification' in self.prediction_task:
            loss = self.loss(gnn_z, batch.y)
            acc = self.accurary(gnn_z.argmax(dim=1), batch.y.argmax(dim=1))
            f1_score_micro = self.f1_score_micro(gnn_z.argmax(dim=1), batch.y.argmax(dim=1))
            f1_score_macro = self.f1_score_macro(gnn_z.argmax(dim=1), batch.y.argmax(dim=1))
            f1_score_per_class = self.f1_score_per_class(gnn_z.argmax(dim=1), batch.y.argmax(dim=1))
            return loss, [acc, f1_score_micro, f1_score_macro, f1_score_per_class]

        if 'regression' in self.prediction_task:
            y_var = torch.var(batch.x, dim=1, keepdim=True)  # You can adjust the estimation method
            # Ensure variance is non-zero and positive
            y_var = y_var.clamp(min=1e-6)
            loss = self.loss(gnn_z, batch.x, y_var)
            mse = self.mse(gnn_z, batch.x)
            r2 = self.r2(gnn_z, batch.x)
            pearson_corr = torch.mean(self.pearson_corr(gnn_z, batch.x))
            return loss, [mse, r2, pearson_corr]

        loss = self.loss(gnn_z, batch.y)
        #print('predicted and true: ', gnn_z.argmax(dim=1)[:10], batch.y.argmax(dim=1)[:10])
        acc = self.accurary(gnn_z.argmax(dim=1), batch.y.argmax(dim=1))
        f1_score_micro = self.f1_score_micro(gnn_z.argmax(dim=1), batch.y.argmax(dim=1))
        f1_score_macro = self.f1_score_macro(gnn_z.argmax(dim=1), batch.y.argmax(dim=1))
        f1_score_per_class = self.f1_score_per_class(gnn_z.argmax(dim=1), batch.y.argmax(dim=1))
        # #print(f'acc: {acc}, f1_score: {f1_score}, loss: {loss}')

        # return loss, acc, f1_score_micro, f1_score_macro, f1_score_per_class

    def evaluation(self, model, batched_data):
         return model.forward(batched_data.x, batched_data.edge_index)