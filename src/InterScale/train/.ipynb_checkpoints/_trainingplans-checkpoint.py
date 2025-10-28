import warnings
from typing import Any, Literal, List
import lightning.pytorch as pl
import torch
import torch.nn as nn
from typing import List, Optional, Literal, Dict, Any

from InterScale.tl import CosineWarmupScheduler, compute_dynamic_variance
from InterScale.model.base._base_model import BaseModelClass
from InterScale.module.base._base_module import BaseModuleClass


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
        loss: Literal["CrossEntropy", "WeightedCE", "MSELoss", "GaussianNLL", "SmoothL1"],
        cross_corr: Literal["gene", "cell"],
        batch_size: int,
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
        self.loss_type = loss
        self.cross_corr = cross_corr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_warmup = lr_warmup
        self.lr_max_epochs = lr_max_epochs
        self.lr = lr

        self.metrics = self._setup_metrics(self.module.n_input)
        self._setup_loss(self.loss_type)

    def _setup_loss(self, 
                    loss: Literal["CrossEntropy", "WeightedCE", "MSELoss", "GaussianNLL", "SmoothL1"]):
        """Setup loss function based on prediction task and configuration."""
                
        if 'classification' in self.prediction_task:
            assert loss == 'CrossEntropy' or loss == 'WeightedCE', "Classification must be run with CrossEntropy or WeightedCE loss."
            if loss == 'CrossEntropy':
                self.loss = nn.CrossEntropyLoss()
            elif loss == 'WeightedCE':
                assert self.class_weights is not None, "Class weights must be provided for WeightedCE loss."
                self.loss = nn.CrossEntropyLoss(torch.from_numpy(self.class_weights))
        elif 'regression' in self.prediction_task:
            print('loss:', loss)
            assert loss == 'MSELoss' or loss == 'GaussianNLL' or loss == 'SmoothL1', "Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss."
            if loss == 'MSELoss':
                self.loss = nn.MSELoss()
            elif loss == 'GaussianNLL':
                self.loss = nn.GaussianNLLLoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
            else:
                raise ValueError("Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss.")
        else:
            raise ValueError("Prediction task must define 'classification' or 'regression'.")
        
    def _setup_metrics(self, num_outputs: int):
        if 'classification' in self.prediction_task:
            return MetricCollection({
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs),
                "f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="micro"),
                "f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="macro"),
                "f1_per_class": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average=None)
            })
        elif 'regression' in self.prediction_task:
            if self.cross_corr == 'gene':
                print('cross-gene per cellcorrelation metrics')
                self.AXIS = 1 # selecting rows / cells
                return None
            elif self.cross_corr == 'cell':
                self.AXIS = 0 # selecting columns / genes
                print('cross-cell per gene correlation metrics')
                return MetricCollection({
                    "mse": torchmetrics.MeanSquaredError(),
                    "r2_raw": torchmetrics.R2Score(num_outputs=num_outputs, multioutput='raw_values'),
                    "r2": torchmetrics.R2Score(num_outputs=num_outputs, multioutput='uniform_average'),
                    "r2_single": torchmetrics.R2Score()
                })
            else:
                raise Exception("Cross-correlation must be run with 'gene' or 'cell'.")
            
        else:
            raise ValueError("Prediction task must define 'classification' or 'regression'.")
        
    def _classification_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask_idx: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate classification metrics."""
        if mask_idx is not None:
            y_pred = y_pred[mask_idx]
            y_true = y_true[mask_idx]
            
        loss = self.loss(y_pred, y_true)
        metrics = self.metrics(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        metrics['loss'] = loss
        
        return loss, metrics
        
    def _regression_metrics(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            mask_idx: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        """Calculate regression metrics."""
        if mask_idx is not None:
            y_pred = y_pred[mask_idx]
            y_true = y_true[mask_idx]
            
            
        if self.loss_type == 'GaussianNLL':
            y_var = compute_dynamic_variance(y_true, y_pred, axis=self.AXIS)
            loss = self.loss(y_pred, y_true, y_var)
        else:
            loss = self.loss(y_pred, y_true)
            
        nr_cells = y_true.shape[0]
        
        if self.cross_corr == 'gene':
            # score per cell, cell numbers dependent on sliding windows / spatial slide
            self.metrics = self._setup_metrics(nr_cells) 
            y_pred = y_pred.T.contiguous()
            y_true = y_true.T.contiguous()
            assert y_pred.shape[0] == self.module.n_input
        elif self.cross_corr == 'cell':
            loss = self.loss(y_pred.T.contiguous(), y_true.T.contiguous(), y_var) # loss calculated over [N,:]
            assert y_pred.shape[1] == self.module.n_input
       
        metrics = self.metrics(y_pred, y_true)
        metrics['loss'] = loss
        
        return loss, metrics

    def forward(self, *args, **kwargs):
        """Passthrough to the module's forward method."""
        return self.module(
            *args,
            **kwargs,
            get_inference_input_kwargs={"full_forward_pass": not self.update_only_decoder},
        )

    def _compute_and_log_metrics(self, 
                     y_pred: torch.Tensor,
                     y_true: torch.Tensor,
                     mode: str):
        """Helper method to log metrics for training, validation, or test steps.
        
        Parameters
        ----------
        loss
            Loss value
        metric_list
            List of metrics to log
        mode
            One of 'train', 'val', or 'test'
        """
        if 'classification' in self.prediction_task:
            loss, metrics = self._classification_metrics(y_pred, y_true)
            log_dict = {
                f'{mode}_loss': loss,
                f'{mode}_acc': metrics['accuracy'],
                f'{mode}_f1_micro/avg': metrics['f1_micro'],
                f'{mode}_f1_macro/avg': metrics['f1_macro'],
            }
            for class_idx in range(self.module.n_output):
                log_dict[f'{mode}_f1/class_{class_idx}'] = metrics['f1_per_class'][class_idx]
        elif 'regression' in self.prediction_task:
            loss, metrics = self._regression_metrics(y_pred, y_true)
            log_dict = {
                f'{mode}_loss': loss,
                f'{mode}_mse': metrics['mse'],
                f'{mode}_r2': metrics['r2'],
                f'{mode}_r2_raw': metrics['r2_raw'],
                f'{mode}_r2_single': metrics['r2_single'],
            }
        
        # Set sync_dist=True only for test mode
        sync_dist = (mode == 'test')
        self.log_dict(log_dict, 
                     batch_size=int(self.batch_size), 
                     on_step=False, 
                     on_epoch=True,
                     sync_dist=sync_dist)
        return loss

    def training_step(self, batch):
        """Training step for the model."""
        local_embedding, global_embedding, y_pred, y_true = self.module._common_step(batch, self.prediction_task)
        return self._compute_and_log_metrics(y_pred, y_true, 'train')

    def validation_step(self, batch):
        """Validation step for the model."""
        local_embedding, global_embedding, y_pred, y_true = self.module._common_step(batch, self.prediction_task)
        return self._compute_and_log_metrics(y_pred, y_true, 'val')
    
    def test_step(self, batch):
        """Test step for the model."""
        local_embedding, global_embedding, y_pred, y_true = self.module._common_step(batch, self.prediction_task)
        return self._compute_and_log_metrics(y_pred, y_true, 'test')

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