from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Dict, Any
import torch
from torch import nn
import pytorch_lightning as L
from torchmetrics import MetricCollection
from InterScale.tl import CosineWarmupScheduler

class BaseModule(L.LightningModule, ABC):
    """Abstract base class for all models defining the common training interface.
    
    Parameters
    ----------
    n_input: int
        Number of input features.
    prediction_task: Literal["classification", "regression"]
        Prediction task
    prediction_level: Literal["node", "graph"]
        If classification, choose from "node" or "graph".
    loss: Literal["CrossEntropy", "WeightedCE", "MSELoss", "GaussianNLL", "SmoothL1"]
        If prediction_task is "classification", choose from "CrossEntropy", "WeightedCE".
        If prediction_task is "regression", choose from "MSELoss", "GaussianNLL", "SmoothL1".
    n_classes: int
        If classification, number of output features / classes.
        For example, number of cell types.
    n_embed: int
        Number of embedding features.
    warmup: int
        Number of warmup steps.
    max_epochs: int
        Maximum number of epochs.   
    lr: float
        Learning rate.
    wd: float
        Weight decay.
    dropout: float
        Dropout rate.   
    class_weights: List
        Class weights.
    """
    
    def __init__(
        self,
        n_input: int,
        prediction_task: Literal["classification", "regression"] = "classification",
        prediction_level: Literal["node", "graph"] = None,
        loss: Literal["CrossEntropy", "WeightedCE", "MSELoss", "GaussianNLL", "SmoothL1"] = None,
        n_classes: int = None,
        n_embed: int = 16,
        warmup: int = 10,
        max_epochs: int = 1000,
        lr: float = 1e-3,
        wd: float = 1e-4,
        dropout: float = 0.2,
        class_weights: Optional[List] = None,
    ):
        super().__init__()
        
        self.n_input = n_input
        self.n_embed = n_embed
        self.warmup = warmup
        self.max_epochs = max_epochs
        self.lr = lr
        self.wd = wd
        self.dropout = dropout
        self.save_hyperparameters()
        
        self.class_weights = class_weights
        self.prediction_task = prediction_task
        self.prediction_level = prediction_level
        self.loss = loss
        if self.prediction_task == "classification":
            assert self.prediction_level is not None, "Prediction level must be provided for classification task."
            self.n_classes = n_classes
            if self.loss is None:
                print("No loss function provided for classification task. Using CrossEntropy loss.")
                self.loss = "CrossEntropy"
        elif self.prediction_task == "regression":
            if self.loss is None:
                print("No loss function provided for regression task. Using MSELoss loss.")
                self.loss = "MSELoss"
        
        # Define components 
        self.local_component = None
        self.global_component = None
        
        # Initialize metrics and loss
        self._setup_metrics()
        self._setup_loss()
        
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]
            mask: Node mask [N]
            
        Returns:
            z: Embeddings [N, E]
            out: Model predictions
            index_nodes: Node indices [N]
        """
        pass
        
    def _setup_metrics(self):
        """Setup metrics based on prediction task."""
        if 'classification' in self.prediction_task:
            self.metrics = MetricCollection({
                'accuracy': self.accuracy,
                'f1_micro': self.f1_score_micro,
                'f1_macro': self.f1_score_macro,
                **{f'f1_class_{i}': self.f1_score_per_class[i] for i in range(self.num_classes)}
            })
        elif 'regression' in self.prediction_task:
            self.metrics = MetricCollection({
                'mse': self.mse,
                'r2': self.r2
            })
    
    def _setup_loss(self):
        """Setup loss function based on prediction task and configuration."""
        if 'classification' in self.prediction_task:
            assert self.loss == 'CrossEntropy' or self.loss == 'WeightedCE', "Classification must be run with CrossEntropy or WeightedCE loss."
            if self.loss == 'CrossEntropy':
                self.loss = nn.CrossEntropyLoss()
            elif self.loss == 'WeightedCE':
                assert self.class_weights is not None, "Class weights must be provided for WeightedCE loss."
                self.loss = nn.CrossEntropyLoss(torch.from_numpy(self.class_weights))
        elif 'regression' in self.prediction_task:
            assert self.loss == 'MSELoss' or self.loss == 'GaussianNLL' or self.loss == 'SmoothL1', "Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss."
            if self.loss == 'MSELoss':
                self.loss = nn.MSELoss()
            elif self.loss == 'GaussianNLL':
                self.loss = nn.GaussianNLLLoss()
            elif self.loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
            else:
                raise ValueError("Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss.")
        else:
            raise ValueError("Prediction task must define 'classification' or 'regression'.")
            
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._common_step(batch, "train")
        
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._common_step(batch, "val")
        
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self._common_step(batch, "test")
        
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
            
        loss = self.loss(y_pred, y_true)
        metrics = self.metrics(y_pred, y_true)
        metrics['loss'] = loss
        
        return loss, metrics
        
    def configure_optimizers(self) -> tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.wd)
        )
        
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=int(self.warmup),
            max_epochs=int(self.max_epochs)
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}] 