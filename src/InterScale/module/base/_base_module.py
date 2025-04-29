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
    """
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_embed: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.module_name = None
        
        self.n_input = n_input
        self.n_embed = n_embed
        self.dropout = dropout
        self.save_hyperparameters()
        
        # Define components 
        self.local_component = None
        self.global_component = None
        
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
        
