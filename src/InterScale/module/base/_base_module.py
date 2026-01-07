from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Dict, Any
import torch
from torch import nn
import pytorch_lightning as L
from InterScale.nn import LinearDecoder, NonLinearDecoder, LinearLSEDecoder
from InterScale.tl.masking import apply_mask

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
        dual_decoder: bool = False,
        pct_mask_nodes: float = 0.0,
        type_gex_embedding: Literal["PCA", "NMF","scvi"] | None = None,
    ):
        """
        Parameters
        ----------
        n_input: int
            Number of input features.
        n_output: int
            If classification, number of output features / classes.
            For example, number of cell types.
        n_embed: int
            Number of embedding dimensions.
        decoder_type: Literal["linear", "nonlinear", linear-lse"]
            Type of decoder to use. For combined module the submodules will potentially not have their own decoder (set to None).
        dropout_decoder: float
            Dropout rate for the decoder only if decoder_type is "nonlinear".
        decoder_hidden_dims: List[int]
            Hidden dimensions for the decoder only if decoder_type is "nonlinear".
        dual_decoder: bool
            If True, use dual decoder for combined module. Both local and global decoders are used.
        pct_mask_nodes: float
            percentage of nodes to mask.
        type_gex_embedding: Literal["PCA", "NMF","scvi"] | None
            Type of GEX embedding to use.
        """
        super().__init__()
        
        self.module_name = None
        
        self.n_input = n_input
        self.n_embed = n_embed
        self.n_output = n_output
        self.dropout_decoder = dropout_decoder
        self.decoder_type = decoder_type
        self.decoder_hidden_dims = decoder_hidden_dims
        self.dual_decoder = dual_decoder
        self.pct_mask_nodes = pct_mask_nodes
        self.type_gex_embedding = type_gex_embedding
        if self.pct_mask_nodes > 0:
            self.masked_nodes = True
        else:
            self.masked_nodes = False
        
        # Define components 
        self.local_component = None
        self.global_component = None
                
        if self.decoder_type == 'linear-lse':
            self.decoder = LinearLSEDecoder(n_input = self.n_embed,
                                           n_output = self.n_output)
        elif self.decoder_type == 'linear':
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
        
    def _common_step_masking(self, batch):
        """Mask nodes in the batch.
        
        Parameters
        ----------
        batch: Batch
            Batch of data.
        Returns
        -------
        batch_masked: Batch
            Batch of data with masked nodes having value MASK_VALUE.
        mask_idx: torch.Tensor
            Indices of masked nodes. Size: [N_masked_nodes, ]
        """
        if self.pct_mask_nodes > 0:
            batch_masked, mask_idx = apply_mask(batch)
        else:
            mask_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
            batch_masked = batch
        return batch_masked, mask_idx
        
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
        mask: Optional[torch.Tensor] = None,
        compute_loss: bool = True
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
    
    # @abstractmethod
    # def loss(self, *args, **kwargs):
    #     """Compute the loss for a minibatch of data.

    #     This function uses the outputs of the inference and generative functions to compute
    #     a loss. This many optionally include other penalty terms, which should be computed here.
    #     """
        
        
        
