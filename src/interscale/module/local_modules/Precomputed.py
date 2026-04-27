import torch

from interscale.module.base import LocalModule


class PrecomputedEmbeddingModule(LocalModule):
    """Module for using frozen, pre-computed embeddings. Essentially a pass-through module."""

    def __init__(self, n_embed: int, **kwargs):
        """
        Module for using frozen, pre-computed embeddings.
        """
        super().__init__(n_embed=n_embed, **kwargs)
        # Dummy parameter to avoid optimizer errors if no other params exist
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, **kwargs):
        """
        In this case, x is expected to be already the embedding.
        """
        return x

    def get_model_summary(self) -> str:
        return "Local Module: Precomputed/Frozen Embeddings (Pass-through)"

    # @staticmethod
    # def from_config(cfg, **kwargs):
    #     return PrecomputedEmbeddingModule(
    #         **kwargs
    #     )
