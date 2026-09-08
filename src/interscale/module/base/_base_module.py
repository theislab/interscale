from abc import ABC, abstractmethod
from typing import Literal

import pytorch_lightning as L
import torch

from interscale.nn import LinearDecoder, LinearLSEDecoder, NonLinearDecoder
from interscale.tl.masking import MASK_STRATEGIES, apply_mask


class BaseModule(L.LightningModule, ABC):
    """Abstract base class for all models defining the common training interface."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_embed: int = 16,
        decoder_type: None | Literal["linear", "nonlinear"] = "linear",
        dropout_decoder: float = 0.2,
        decoder_hidden_dims: list[int] = [128, 128],
        dual_decoder: bool = False,
        mask_percentage: float = 0.0,
        mask_strategy: Literal["node", "gene"] = "node",
        type_gex_embedding: Literal["PCA", "NMF", "scvi"] | None = None,
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
        mask_percentage: float
            Bernoulli masking probability -- per cell under ``mask_strategy="node"``, per
            (cell, gene) entry under ``"gene"``.
        mask_strategy: Literal["node", "gene"]
            Granularity of the reconstruction corruption; see ``interscale.tl.masking``.
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
        self.mask_percentage = mask_percentage
        if mask_strategy not in MASK_STRATEGIES:
            raise ValueError(f"mask_strategy must be one of {MASK_STRATEGIES}, got {mask_strategy!r}.")
        self.mask_strategy = mask_strategy
        self.type_gex_embedding = type_gex_embedding
        # `masked_nodes` means "the reconstruction input is corrupted at all", not "whole cells
        # are blanked". Both strategies set it: under gene masking every cell is a supervision
        # target, so `pad_batch`'s keep_indices is all-True and it degenerates to the plain
        # subsampling branch, which is the correct behaviour.
        self.masked_nodes = self.mask_percentage > 0

        # Define components
        self.local_component = None
        self.global_component = None

        if self.decoder_type == "linear-lse":
            self.decoder = LinearLSEDecoder(n_input=self.n_embed, n_output=self.n_output)
        elif self.decoder_type == "linear":
            self.decoder = LinearDecoder(n_input=self.n_embed, n_output=self.n_output)
        elif self.decoder_type == "nonlinear":
            self.decoder = NonLinearDecoder(
                n_input=self.n_embed,
                n_output=self.n_output,
                hidden_dims=self.decoder_hidden_dims,
                dropout=self.dropout_decoder,
            )
        elif self.decoder_type == None:  # If Local + Global model sequential and no decoder needed
            self.decoder = None
        else:
            raise ValueError(f"Decoder {self.decoder_type} not found.")

    def _common_step_masking(self, batch):
        """Corrupt the reconstruction input of the batch.

        Parameters
        ----------
        batch: Batch
            Batch of data.

        Returns
        -------
        batch_masked: Batch
            Batch of data with the masked entries set to MASK_VALUE.
        mask_idx: torch.Tensor
            Indices of the cells that are supervision targets. Size: [N_masked_nodes, ]
        entry_mask: torch.Tensor | None
            ``[N, G]`` boolean over the full node ordering when ``mask_strategy == "gene"``,
            marking the entries the loss must be restricted to; ``None`` under cell masking,
            where the whole row of every target cell is scored. Callers must subset it with the
            same row indices they apply to ``y_true``.
        """
        if self.masked_nodes:
            batch_masked, mask_idx, entry_mask = apply_mask(batch, self.mask_strategy)
        else:
            mask_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
            batch_masked = batch
            entry_mask = None
        return batch_masked, mask_idx, entry_mask

    @abstractmethod
    def _common_step(self, batch):
        """Shared step between train, val and test."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor | None = None,
        compute_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]
            mask: Node mask [N]

        Returns
        -------
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
