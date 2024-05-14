import torch
import torch.nn as nn
import pytorch_lightning as pl

import numpy as np
import torchmetrics


# PCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

class BaselinePCA():

    def __init__(self):
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Scale the data
            ('pca', PCA(n_components=50)),  # You can adjust the number of components
        ])
        self.linear_regression = LinearRegression()

    def run(self, X_train, y_train, X_test, y_test):

        # Train
        X_train_scaled_pca = self.pipeline.fit_transform(X_train)
        self.linear_regression.fit(X_train_scaled_pca, y_train)
        y_pred_train = self.linear_regression.predict(X_train_scaled_pca)

        # Test
        X_test_scaled_pca = self.pipeline.transform(X_test)
        y_pred_test = self.linear_regression.predict(X_test_scaled_pca)

        return y_pred_train, y_pred_test

class BaselineFCNN(pl.LightningModule):
    def __init__(self, 
                 cfg
        ):
        super().__init__()      
        #dp_rate = cfg['dp_rate'] if cfg['dp_rate'] is not None else dp_rate
        self._cfg = cfg
        self.num_classes = cfg.get('dataset/num_classes')
        in_dim, hidden_dim, embed_dim = cfg.get('dataset/num_features'), cfg.get('fcnn/hidden_dim'), cfg.get('fcnn/embed_dim')
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.lr = float(self._cfg.get('optim/lr'))
        self.wd = float(self._cfg.get('optim/wd'))
        # Define metrics
        self.accurary = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes) 

        print(in_dim)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(embed_dim, self.num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        z = self.out(x)
        return x, z

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
        x, z = self.forward(batch.x) 
        # Calculate loss function
        loss = self.loss_criterion(z, batch.y)
        #print('predicted and true: ', gnn_z.argmax(dim=1)[:10], batch.y.argmax(dim=1)[:10])
        acc = self.accurary(z.argmax(dim=1), batch.y.argmax(dim=1))
        f1_score = self.f1_score(z.argmax(dim=1), batch.y.argmax(dim=1))
        #print(f'acc: {acc}, f1_score: {f1_score}, loss: {loss}')

        return loss, acc, f1_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return [optimizer]
    
