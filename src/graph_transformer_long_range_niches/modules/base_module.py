# Standard libraries

# PyTorch
# PyTorch Lightning
import pytorch_lightning as L
import torch
import torchmetrics
from torch import nn

from typing import List
from scipy.stats import pearsonr

import numpy as np

from graph_transformer_long_range_niches.tl.scheduler import CosineWarmupScheduler
from graph_transformer_long_range_niches.tl.masking import apply_mask
from graph_transformer_long_range_niches.tl.utils import define_loss, define_classification_metrics, define_regression_metrics, compute_dynamic_variance, create_transformer_attention_mask_from_edges, pad_batch

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
        
        if cfg.dataset.pct_mask_nodes > 0:
            self.masked_nodes = True
        else:
            self.masked_nodes = False

        #define loss
        self.loss = define_loss(cfg, class_weights)
        
        #define metrics
        if 'classification' in self.prediction_task:
            self.accurary, self.f1_score_micro, self.f1_score_macro, self.f1_score_per_class = define_classification_metrics(cfg)
        elif 'regression' in self.prediction_task:
            if cfg.optim.cross_corr == 'gene':
                print('cross-gene per cellcorrelation metrics')
                self.AXIS = 1 # selecting rows / cells
            elif cfg.optim.cross_corr == 'cell':
                print('cross-cell per gene correlation metrics')
                self.mse, self.r2_raw, self.r2, self.r2_single = define_regression_metrics(cfg.dataset.num_features)
                # define in common_step because nr cells is variable
                self.AXIS = 0 # selecting columns / genes
            else:
                raise Exception("Cross-correlation must be run with 'gene' or 'cell'.")
            
        ## Prediction units
        self.graph_pred_linear_list = torch.nn.ModuleList()
        n_in = self._cfg.transformer.d_model
        if 'classification' in self.prediction_task:
            n_out = self.num_classes
            layers_dim = [n_in] + self._cfg.model.decoder.hidden_dims + [self.num_classes]
        elif 'regression' in self.prediction_task:
            n_out = self.num_features
            layers_dim = [n_in] + self._cfg.model.decoder.hidden_dims + [self.num_features]
                
        if self._cfg.model.decoder.type == 'linear':
            self.graph_pred_linear = torch.nn.Linear(n_in, n_out)
        elif self._cfg.model.decoder.type == 'nonlinear':
            self.graph_pred_linear = nn.Sequential(
                    *[nn.Sequential(
                        nn.Linear(n_in, n_out),
                        nn.LayerNorm(n_out),
                        nn.ReLU(),
                        nn.Dropout(p=self._cfg.model.decoder.dropout)
                    ) for n_in, n_out in zip(layers_dim[:-1], layers_dim[1:-1])],
                    nn.Linear(layers_dim[-2], layers_dim[-1])  # Final layer without activation
            )
        else:
            raise ValueError(f"Invalid decoder type: {self._cfg.model.decoder}. Must be either 'linear' or 'nonlinear'")

    def common_configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=int(self._cfg.optim.warm_up),
                                             max_epochs=100000)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]
    
    def _common_step(self, batch):
        raise NotImplementedError("Subclasses must implement the _common_step method.")
    
    def common_training_step(self, batch):
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

    def common_validation_step(self, batch):
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
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True, sync_dist=True)
        elif 'regression' in self.prediction_task:
            mse, r2, pearson_corr = metric_list
            log_dict = {
                'test_mse': mse,
                'test_r2': r2,
                'test_pearson_corr': pearson_corr,
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True, sync_dist=True)
            print(f"Test step (regression) - Loss: {loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
        return loss
    
    def _common_step_classification_metrics(self, y_pred, y_true, mask_idx=None):
        """Calculate classification metrics fir=
        
        Input:
            y_pred: List[torch.Tensor] [N, C]
                Predicted values
            y_true: List[torch.Tensor] [N, C]
                True values
            mask_idx: List[torch.Tensor]
                Indices of masked nodes to calculate metrics on. Default is None, i.e. graph-level prediction.
        Output:
            loss: torch.Tensor[int]
            
        """
        if mask_idx is None: # graph-level prediction
            mask_idx = torch.arange(len(y_pred))
        mask_idx = mask_idx.numpy()
        loss = self.loss(y_pred[mask_idx, :], y_true[mask_idx, :])
        acc = self.accurary(y_pred.argmax(dim=1)[mask_idx], y_true.argmax(dim=1)[mask_idx])
        f1_score_micro = self.f1_score_micro(y_pred.argmax(dim=1)[mask_idx], y_true.argmax(dim=1)[mask_idx])
        f1_score_macro = self.f1_score_macro(y_pred.argmax(dim=1)[mask_idx], y_true.argmax(dim=1)[mask_idx])
        f1_score_per_class = self.f1_score_per_class(y_pred.argmax(dim=1)[mask_idx], y_true.argmax(dim=1)[mask_idx])

        return loss, [acc, f1_score_micro, f1_score_macro, f1_score_per_class]
    
    def _common_step_regression_metrics(self, y_pred, y_true, mask_idx=None):
        """Calculate regression metrics
        Input:
            y_pred: List[torch.Tensor] #ToDo: size???
                Predicted values
            y_true: List[torch.Tensor] #ToDo: size???
                True values
            mask_idx: List[torch.Tensor]
                Indices of masked nodes to calculate metrics on
        """
        if mask_idx is None: # graph-level prediction
            mask_idx = torch.arange(len(y_pred))
        y_pred = y_pred[mask_idx]
        y_true = y_true[mask_idx]
        assert y_true.shape == y_pred.shape
        
        # Estimate variance based on the true values (e.g., using batch variance)
        y_var = compute_dynamic_variance(y_true, y_pred, axis=self.AXIS)
        nr_cells = y_true.shape[0]
            
        if self._cfg.optim.cross_corr == 'gene':
            # score per cell, cell numbers dependent on sliding windows / spatial slide
            self.mse, self.r2_raw, self.r2, self.r2_single = define_regression_metrics(nr_cells) 
            # for GPU usage
            self.mse = self.mse.to(y_pred.device)
            self.r2_raw = self.r2_raw.to(y_pred.device)
            self.r2 = self.r2.to(y_pred.device)
            self.r2_single = self.r2_single.to(y_pred.device)
            loss = self.loss(y_pred, y_true, y_var)
            y_pred = y_pred.T.contiguous()
            y_true = y_true.T.contiguous()
            assert y_pred.shape[0] == self.num_features
        elif self._cfg.optim.cross_corr == 'cell':
            loss = self.loss(y_pred.T.contiguous(), y_true.T.contiguous(), y_var) # loss calculated over [N,:]
            assert y_pred.shape[1] == self.num_features

        if nr_cells > 1:
            mse = self.mse(y_pred, y_true)
            assert y_pred.shape[1] == y_true.shape[1] == self.r2_raw.num_outputs # multioutput (N, M)
            r2_raw = self.r2_raw(y_pred, y_true)
            r2 = self.r2(y_pred, y_true)
            pearson_corr_raw = pearsonr(y_pred.detach().cpu().numpy(), 
                                        y_true.detach().cpu().numpy())
                                        #axis=self.AXIS)
            pearson_corr = torch.tensor(np.nanmean(pearson_corr_raw[0]), 
                                        dtype=torch.float32, 
                                        device=y_pred.device)
            if self._cfg.optim.cross_corr == 'cell':
                # metric value for each gene
                assert r2_raw.shape[0] == pearson_corr_raw[0].shape[0] == self.num_features
            if self._cfg.optim.cross_corr == 'gene':
                # metric value for each cell
                assert r2_raw.shape[0] == pearson_corr_raw[0].shape[0] == nr_cells
            return loss, [mse, r2, pearson_corr]
        else: # single data obect in the batch
            mse = self.mse(y_pred, y_true)
            r2 = self.r2_single(y_pred, y_true)
            # For single element, correlation is undefined, return NaN or 1.0 if values are identical
            if torch.allclose(y_pred, y_true):
                pearson_corr = torch.tensor(1.0, dtype=torch.float32, device=y_pred.device)
            else:
                pearson_corr = torch.tensor(float('nan'), dtype=torch.float32, device=y_pred.device)
            return loss, [mse, torch.mean(r2), pearson_corr]

