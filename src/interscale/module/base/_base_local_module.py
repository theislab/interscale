from abc import abstractmethod
from typing import Literal

from ._base_module import BaseModule


class LocalModule(BaseModule):
    def __init__(self, **base_module_kwargs):

        super().__init__(**base_module_kwargs)

        self.registered_local_component = True
        self.registered_global_component = False

    @abstractmethod
    def forward(self):
        """Forward pass."""

    def predict(self, local_embedding, prediction_level: Literal["node", "graph"] | None = None):
        """Predict with the decoder.

        Parameters
        ----------
        local_embedding: torch.Tensor
            Size: [N, E]
        prediction_level: Literal["node", "graph"]
        """
        return self.decoder.forward(local_embedding)

    def _common_step(self, batch, prediction_task: str, prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.

        Returns
        -------
        local_embedding: torch.Tensor
            Size: [N, E]
        global_embedding: torch.Tensor
            Size: [N, E]
        y_pred: torch.Tensor
            Size: [B, C] (classification, masked cells) or [N, F] (regression, ALL cells)
        y_true: torch.Tensor
            Size: [B, ] (classification, masked cells) or [N, F] (regression, ALL cells)
        attn: None
            This module has no attention; returned for a uniform `_common_step` contract.
        entry_mask: torch.Tensor | None
            Size: [N, F] for regression, marking the entries the LOSS is scored on; the metrics
            use every cell. None for classification.
        """
        # Mask nodes
        batch_masked, mask_idx, entry_mask = self._common_step_masking(batch)

        local_embedding = self.forward(batch_masked.x, batch_masked.edge_index)
        y_pred = self.decoder.forward(local_embedding)

        assert y_pred.shape[0] == len(batch.obs_names), (
            f"Mismatch: y_pred.shape: {y_pred.shape[0]}, batch.obs_names: {len(batch.obs_names)}"
        )
        assert y_pred.shape[1] == self.n_output, (
            f"Mismatch: y_pred.shape: {y_pred.shape[1]}, self.n_output: {self.n_output}"
        )
        assert y_pred.isnan().sum() == 0, "y_pred contains NaN values"

        if "classification" in prediction_task:
            # The masked cells ARE the supervision targets here, so both loss and metrics stay
            # restricted to them. Only reconstruction moved to scoring every cell.
            y_pred = y_pred[mask_idx]
            y_true = batch.y[mask_idx]  # batch without mask because constant otherwise
            assert y_true.shape == y_pred.shape
            # Class labels are not gene entries, so there is nothing for an entry mask to select.
            return local_embedding, None, y_pred, y_true, None, None

        if "regression" in prediction_task:
            # Every cell is scored. `entry_mask` says which entries the LOSS uses: the gene mask
            # under gene masking, the per-cell mask broadcast over all genes under cell masking
            # (masked_loss recognises the whole-row form and subsets rows, so the loss is
            # unchanged from when this returned masked cells only).
            y_true = batch.x  # batch without mask because constant otherwise
            assert y_true.shape == y_pred.shape
            if entry_mask is None:
                entry_mask = batch.mask.bool().unsqueeze(1).expand_as(y_true)
            assert entry_mask.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true, None, entry_mask

        assert False, "Prediction task not supported"

    def get_local_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)

    # acts as a factory method to create a module from a config
    @staticmethod
    def from_config(cfg, **kwargs):
        module_name = cfg.model.local_component.name
        params = cfg.model.local_component.parameters.copy()  # Make a copy to avoid modifying the original

        if module_name == "GCN":
            from interscale.module.local_modules import GCN

            return GCN(
                n_layers=params["num_layers"],
                hidden_dim=params["hidden_dim"],
                dropout_local=params["dropout_local"],
                **kwargs,
            )
        elif module_name == "GIN":
            from interscale.module.local_modules import GIN

            return GIN(
                n_layers=params["num_layers"],
                hidden_dim=params["hidden_dim"],
                dropout_local=params["dropout_local"],
                **kwargs,
            )
        elif module_name == "SCVI":
            print("Creating SCVI Local Module")
            from interscale.module.local_modules import SCVILocalModule

            n_input = kwargs.pop("n_input")
            n_embed = kwargs.pop("n_embed")
            return SCVILocalModule(
                n_input=n_input,
                n_latent=n_embed,
                n_layers=params.get("num_layers", 2),
                n_hidden=params.get("hidden_dim", 128),
                dropout_rate=params.get("dropout_local", 0.1),
                **kwargs,
            )
        # elif module_name == 'Precomputed':
        #     print(f"Creating Precomputed Embedding Module from {cfg.dataset.precomputed}")
        #     from interscale.module.local_modules import PrecomputedEmbeddingModule
        #     return PrecomputedEmbeddingModule(
        #         **kwargs
        #     )
        # Add more elifs for other modules
        else:
            raise ValueError(f"Unknown local module name: {module_name}")
        # # Add more elifs for other modules
        # else:
        #     raise ValueError(f"Unknown local module name: {module_name}")
