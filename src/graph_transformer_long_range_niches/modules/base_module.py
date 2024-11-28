# Standard libraries

# PyTorch
# PyTorch Lightning
import pytorch_lightning as L
import torch
import torchmetrics
from torch import nn

from typing import List

from graph_transformer_long_range_niches.tl.scheduler import CosineWarmupScheduler

class BaseModule(L.LightningModule):
    """Base class for all models (Local, Global, Local+Global)"""
    
    def __init__(self, cfg, class_weights: List = None, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self._cfg = cfg
        self.lr = float(self._cfg.optim.lr)
        self.wd = float(self._cfg.optim.wd)
        self.class_weights = class_weights

        self.model_type = 'BaseClass' # Will be overwritten by subclasses
        self.prediction_task = cfg.dataset.prediction_task
        self.num_classes = cfg.dataset.num_classes
        self.num_features = cfg.dataset.num_features
        self.max_seq_len = cfg.transformer.max_seq_len

        # Loss function
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

    def common_configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=int(self._cfg.optim.warm_up),
                                             max_epochs=100000)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def common_training_step(self, batch, batch_idx):
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
                'train_pearson_corr': pearson_corr,
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss

    def common_validation_step(self, batch, batch_idx):
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
                'val_pearson_corr': pearson_corr,
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss

    def common_test_step(self, batch):
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
                'test_pearson_corr': pearson_corr,
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss
    
    def _common_step_classification_metrics(self, y_pred, y_true):
        """Calculate classification metrics
        
        Input:
            y_pred: List[torch.Tensor] #ToDo: size???
                Predicted values
            y_true: List[torch.Tensor] #ToDo: size???
                True values
        """
        loss = self.loss(y_pred, y_true.argmax(dim=1))
        acc = self.accurary(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        f1_score_micro = self.f1_score_micro(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        f1_score_macro = self.f1_score_macro(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        f1_score_per_class = self.f1_score_per_class(y_pred.argmax(dim=1), y_true.argmax(dim=1))

        return loss, [acc, f1_score_micro, f1_score_macro, f1_score_per_class]
    
    def _common_step_regression_metrics(self, y_pred, y_true):
        """Calculate regression metrics
        """
        # Estimate variance based on the true values (e.g., using batch variance)
        y_var = torch.var(y_true, dim=1, keepdim=True)  # You can adjust the estimation method
        # Ensure variance is non-zero and positive
        y_var = y_var.clamp(min=1e-6)
        loss = self.loss(y_pred, y_true, y_var)
        mse = self.mse(y_pred, y_true)
        r2 = self.r2(y_pred, y_true)
        pearson_corr = torch.mean(self.pearson_corr(y_pred, y_true))
        return loss, [mse, r2, pearson_corr]

