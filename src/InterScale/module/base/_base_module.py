from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Dict, Any
import torch
from torch import nn
import pytorch_lightning as L
from InterScale.nn import LinearDecoder, NonLinearDecoder

class BaseModuleClass(L.LightningModule, ABC):
    """Abstract base class for all models defining the common training interface.
    
    
    """
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_embed: int = 16,
        decoder_type: None |Literal["linear", "nonlinear"] = "linear",
        dropout_decoder: float = 0.2,
        decoder_hidden_dims: List[int] = [128, 128],
        pct_mask_nodes: float = 0.0,
    ):
        """
        Parameters
        ----------
        n_input: int
            Number of input features.
        n_classes: int
            If classification, number of output features / classes.
            For example, number of cell types.
        n_embed: int
            Number of embedding dimensions.
        decoder_type: Literal["linear", "nonlinear"]
            Type of decoder to use. For combined module the submodules will potentially not have their own decoder (set to None).
        dropout_decoder: float
            Dropout rate for the decoder only if decoder_type is "nonlinear".
        decoder_hidden_dims: List[int]
            Hidden dimensions for the decoder only if decoder_type is "nonlinear".
        mask_nodes: bool
            Whether to mask nodes.
        """
        super().__init__()
        
        self.module_name = None
        
        self.n_input = n_input
        self.n_embed = n_embed
        self.n_output = n_output
        self.dropout_decoder = dropout_decoder
        self.decoder_type = decoder_type
        self.decoder_hidden_dims = decoder_hidden_dims
        self.pct_mask_nodes = pct_mask_nodes
        # Define components 
        self.local_component = None
        self.global_component = None
        
        if self.decoder_type == 'linear':
            self.decoder = LinearDecoder(n_input = self.n_embed,
                                        n_output = self.n_output)
        elif self.decoder_type == 'nonlinear':
            self.decoder = NonLinearDecoder(n_input = self.n_embed,
                                           n_output = self.n_output,
                                           hidden_dims = self.decoder_hidden_dims,
                                           dropout = self.dropout_decoder)
        elif self.decoder_type == None: # If Local + Global model sequential and no decoder needed
            self.decoder = None
        else:
            raise ValueError(f"Decoder {self.decoder_type} not found.")
        
    @abstractmethod
    def _common_step(self,
                    batch):
        """Shared step between train, val and test.
        """
        
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
        
