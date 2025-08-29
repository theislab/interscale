import warnings
from typing import Any, Literal, List
import lightning.pytorch as pl
import torch
import torch.nn as nn
from typing import List, Optional, Literal, Dict, Any
import numpy as np
from InterScale.tl import CosineWarmupScheduler, compute_dynamic_variance
from InterScale.model.base._base_model import BaseModelClass
from InterScale.module.base._base_module import BaseModuleClass
from .losses import BalancedPearsonCorrelationLoss

import torchmetrics
from torchmetrics import MetricCollection
# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/train/_trainingplans.py
# accessed on 28 April 2025
class TrainingPlan(pl.LightningModule):
    """Lightning module task to train scvi-tools modules.

    The training plan is a PyTorch Lightning Module that is initialized
    with a scvi-tools module object. It configures the optimizers, defines
    the training step and validation step, and computes metrics to be recorded
    during training. The training step and validation step are functions that
    take data, run it through the model and return the loss, which will then
    be used to optimize the model parameters in the Trainer. Overall, custom
    training plans can be used to develop complex inference schemes on top of
    modules.

    The following developer tutorial will familiarize you more with training plans
    and how to use them: :doc:`/tutorials/notebooks/dev/model_user_guide`.

    Parameters
    ----------
    
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        prediction_task: str,
        prediction_level: Literal["node", "graph"],
        loss: Literal["CrossEntropy", "WeightedCE", "MSELoss", "GaussianNLL", "SmoothL1"],
        cross_corr: Literal["gene", "cell"],
        batch_size: int,
        class_weights: np.ndarray | None = None,
        class_labels: List[str] | None = None,
        *,
        use_lr_scheduler: bool = True,
        weight_decay: float = 1e-6,
        lr: float = 1e-3,
        lr_warmup: int = 0,
        lr_max_epochs: int = 100000,    
        **kwargs,
    ):
        super().__init__()
        self.module = module
        self.prediction_task = prediction_task
        self.prediction_level = prediction_level
        self.loss_type = loss
        self.cross_corr = cross_corr
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.class_labels = class_labels
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_warmup = lr_warmup
        self.lr_max_epochs = lr_max_epochs
        self.lr = lr
        if self.prediction_task == 'regression':
            if self.cross_corr == 'gene':
                print('cross-gene per cell correlation metrics')
                self.AXIS = 1 # selecting rows / cells
                self.rel_weight_gene = 1.0
                self.rel_weight_cell = 0.0
            elif self.cross_corr == 'cell':
                print('cross-cell per gene correlation metrics')
                self.AXIS = 0 # selecting columns / genes
                self.rel_weight_gene = 0.0
                self.rel_weight_cell = 1.0
            self.pearson_corr = BalancedPearsonCorrelationLoss(
                rel_weight_gene=self.rel_weight_gene,
                rel_weight_cell=self.rel_weight_cell,
            )
        
        # setup metrics and loss
        if 'classification' in self.prediction_task:
            metrics = self._setup_classification_metrics(self.module.n_output)
            self.loss = self._setup_classification_loss(self.loss_type, self.class_weights)
        elif 'regression' in self.prediction_task:
            metrics = self._setup_regression_metrics()
            self.loss = self._setup_regression_loss(self.loss_type)
        else:
            raise ValueError("Prediction task must define 'classification' or 'regression'.")
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
    
    @staticmethod
    def _setup_classification_loss(loss: Literal["CrossEntropy", "WeightedCE"], class_weights: torch.Tensor | None = None):
        """Setup loss function based on prediction task and configuration."""
        assert loss == 'CrossEntropy' or loss == 'WeightedCE', "Classification must be run with CrossEntropy or WeightedCE loss."
        if loss == 'CrossEntropy':
            return nn.CrossEntropyLoss()
        elif loss == 'WeightedCE':
            assert class_weights is not None, "Class weights must be provided for WeightedCE loss."
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch tensor"
            return nn.CrossEntropyLoss(class_weights)
            
    
    def _setup_regression_loss(self, loss: Literal["MSELoss", "GaussianNLL", "SmoothL1", "BalancedPearsonCorrelationLoss"]):
        """Setup loss function based on prediction task and configuration."""
        assert loss == 'MSELoss' or loss == 'GaussianNLL' or loss == 'SmoothL1' or loss == "BalancedPearsonCorrelationLoss", "Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss."
        if loss == 'MSELoss':
            return nn.MSELoss()
        elif loss == 'GaussianNLL':
            return nn.GaussianNLLLoss()
        elif loss == 'SmoothL1':
            return nn.SmoothL1Loss()
        elif loss == "BalancedPearsonCorrelationLoss":
            return self.pearson_corr
        
    @staticmethod
    def _setup_classification_metrics(num_outputs: int):
        return MetricCollection({
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs),
            "f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="micro"),
            "f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="macro"),
            "f1_per_class": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average=None)
        })
    
    @staticmethod
    def _setup_regression_metrics():
        return MetricCollection({
            "mse": torchmetrics.MeanSquaredError(),
            "r2": torchmetrics.R2Score(multioutput='uniform_average'),
            # "r2_single": torchmetrics.R2Score()
        })
        
    def _classification_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        mask_idx: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate classification metrics."""
        ## TODO: Currently mask_idx is applied in module._common_step. Maybe move to here?
        # if mask_idx is not None:
        #     y_pred = y_pred[mask_idx]
        #     y_true = y_true[mask_idx]
            
        loss = self.loss(y_pred, y_true)
        metrics = metrics(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        metrics[f'{mode}_loss'] = loss
        
        return metrics
        
    def _regression_metrics(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            mode: str,
            metrics: MetricCollection,
            mask_idx: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate regression metrics.
            y_true, y_pred: torch.Tensor
                True and predicted values of shape [N, G], where N is the number of cells and G is the number of genes
            """
        if self.loss_type == 'GaussianNLL':
            y_var = compute_dynamic_variance(y_true, y_pred, axis=self.AXIS)
       
        if self.loss_type == 'GaussianNLL':
            loss = self.loss(y_pred, y_true, y_var)
        else:
            loss = self.loss(y_pred, y_true)
            
        metrics = metrics(y_pred, y_true)
                                
        if torch.std(y_pred) == 0 or torch.std(y_true) == 0:
            print('constant array')
            print(y_pred[:5], y_true[:5])
            pearson_corr = torch.tensor(1.0 if np.allclose(y_pred, y_true) else float('nan'), 
                                      dtype=torch.float32, 
                                      device=y_pred.device)
        else:
            pearson_corr = self.pearson_corr(y_pred, y_true)
        
        metrics[f'{mode}_loss'] = loss
        metrics[f'{mode}_pearson_corr'] = pearson_corr
        return metrics

    def forward(self, *args, **kwargs):
        """Passthrough to the module's forward method."""
        return self.module(
            *args,
            **kwargs,
        )
        
    @torch.inference_mode()
    def _compute_and_log_metrics(self, 
                     y_pred: torch.Tensor,
                     y_true: torch.Tensor,
                     mode: str, 
                     metrics: MetricCollection):
        """Helper method to log metrics for training, validation, or test steps.
        
        Parameters
        ----------
        y_true, y_pred: torch.Tensor
            True and predicted values of shape [N, G], where N is the number of cells and G is the number of genes
        mode
            One of 'train', 'val', or 'test'
        metrics: MetricCollection
            Metrics to log
        """
        print(y_pred.shape, y_true.shape)
        
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        #TODO: where is the batch size?
        assert y_pred.shape[0] == self.module.n_input, "y_pred must have same number of inputs as module.n_input"
        assert y_pred.shape[1] == self.module.n_output, "y_pred must have same number of outputs as module.n_output"
        
        if 'classification' in self.prediction_task:
            metrics = self._classification_metrics(y_pred, y_true, mode, metrics)
            for class_idx, class_score in enumerate(metrics[f'{mode}_f1_per_class']):
                metrics[f'{mode}_f1_{self.class_labels[class_idx]}'] = class_score
            metrics.pop(f'{mode}_f1_per_class')
            
        elif 'regression' in self.prediction_task:
            metrics = self._regression_metrics(y_pred, y_true, mode, metrics)
            
        # Set sync_dist=True only for test mode
        sync_dist = (mode == 'test')
        self.log_dict(metrics, 
                     batch_size=int(self.batch_size), 
                     on_step=False, 
                     on_epoch=True,
                     sync_dist=sync_dist)
        
        return metrics[f'{mode}_loss']

    def training_step(self, batch):
        """Training step for the model."""
        local_embedding, global_embedding, y_pred, y_true = self.module._common_step(batch, self.prediction_task, self.prediction_level)
        return self._compute_and_log_metrics(y_pred, y_true, 'train', self.train_metrics)

    def validation_step(self, batch):
        """Validation step for the model."""
        local_embedding, global_embedding, y_pred, y_true = self.module._common_step(batch, self.prediction_task, self.prediction_level)
        return self._compute_and_log_metrics(y_pred, y_true, 'val', self.valid_metrics)
    
    def test_step(self, batch):
        """Test step for the model."""
        local_embedding, global_embedding, y_pred, y_true = self.module._common_step(batch, self.prediction_task, self.prediction_level)
        return self._compute_and_log_metrics(y_pred, y_true, 'test', self.test_metrics)

    def configure_optimizers(self):
        params = []
        params.extend(filter(lambda p: p.requires_grad, self.module.parameters()))
        # if self.model.local_component is not None:
        #     params.extend(filter(lambda p: p.requires_grad, self.module.local_component.parameters()))
        # if self.model.global_component is not None:
        #     params.extend(filter(lambda p: p.requires_grad, self.model.global_component.parameters()))
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.use_lr_scheduler:
            lr_scheduler = CosineWarmupScheduler(optimizer,
                                                warmup=self.lr_warmup,
                                                max_epochs=self.lr_max_epochs)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]