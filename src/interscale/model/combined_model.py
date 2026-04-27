import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from yacs.config import CfgNode as CN

from interscale.model.base._base_model import BaseModel
from interscale.module import CombinedModule, DualDecoderCombinedModule
from interscale.tl import prepare_a2d_dataset
from interscale.train import NodeMaskingTrainingPlan


class CombinedModel(NodeMaskingTrainingPlan, BaseModel):
    """Combined model with both local and global components."""

    _module_cls = CombinedModule

    def __init__(
        self,
        adata: AnnData,
        cfg: CN,
    ):
        super().__init__(adata, cfg)

        self._module_kwargs = self._cfg.model

        self.local_component = True
        self.global_component = True
        # Initialize the combined module with both local and global components
        if self._cfg.model.decoder.dual_decoder:
            self.module = DualDecoderCombinedModule(
                cfg=self._cfg,
                n_input=self.n_input,
                n_output=self.n_output,
                n_embed=self.n_embed,
                decoder_type=None,  # Container doesn't need its own decoder, only submodules do
                dropout_decoder=self._cfg.model.decoder.dropout_decoder,
                decoder_hidden_dims=self._cfg.model.decoder.hidden_dims,
                pct_mask_nodes=self._cfg.dataset.pct_mask_nodes,
            )
        else:
            self.module = CombinedModule(
                cfg=self._cfg,
                n_input=self.n_input,
                n_output=self.n_output,
                n_embed=self.n_embed,
                decoder_type=None,  # Container doesn't need its own decoder, only global module does
                dropout_decoder=self._cfg.model.decoder.dropout_decoder,
                decoder_hidden_dims=self._cfg.model.decoder.hidden_dims,
                pct_mask_nodes=self._cfg.dataset.pct_mask_nodes,
            )

        self._model_summary_string = self._model_summary_string + self.module.get_model_summary()

    def get_model_output(self, adata: AnnData | None = None, prefix: str = ""):
        """Save the embeddings, predictions and attentionsin the adata object.

        Parameters
        ----------
        adata
            AnnData object to run the model on. If `None`, the model's AnnData object is used.
        prefix
            Prefix for the output columns.
        """
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)

        # Check for duplicate observation names
        adata.obs_names = [
            str(i) for i in range(1, len(adata.obs_names) + 1)
        ]  # ensure that no duplicate observation names are present
        assert len(adata.obs_names) == len(adata.obs_names.unique()), (
            f"Duplicate observation names found. Expected {len(adata.obs_names)} unique names but found {len(adata.obs_names.unique())}"
        )

        assert "spatial" in adata.obsm, "Missing spatial coordinates"
        assert adata.obsm["spatial"].shape[0] > 0, "No spatial coordinates found"

        a2d = prepare_a2d_dataset(self._cfg)
        pyg, _ = list(a2d(adata))

        obs_names_str = adata.obs_names.astype(int).astype(str)

        # Check for duplicate observation names
        assert len(obs_names_str) == len(obs_names_str.unique()), (
            f"Duplicate observation names found. Expected {len(obs_names_str)} unique names but found {len(obs_names_str.unique())}"
        )

        # Create empty DataFrame with correct shape
        local_embeddings_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_embed), dtype=np.float32)

        global_embeddings_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_embed), dtype=np.float32)
        attention_matrix_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self._cfg.model.global_component.parameters.max_seq_len),
            dtype=np.float32,
        )

        if self._cfg.model.decoder.dual_decoder:
            y_pred_local_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_output), dtype=np.float32)
        else:
            y_pred_local_df = None

        y_pred_global_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_output), dtype=np.float32)

        cls_token_horizontal = np.full(len(adata.obs_names), np.nan)
        cls_token_vertical = np.full(len(adata.obs_names), np.nan)

        for batch in pyg:
            ## Local model output
            local_embedding = self.module.local_module.forward(batch.x, batch.edge_index)
            if self._cfg.model.decoder.dual_decoder:
                mask_idx = torch.arange(local_embedding.size(0), device=local_embedding.device)
                y_pred_local = self.module.predict_local(local_embedding, mask_idx)

            sample_mask_local = local_embeddings_df.index.isin(batch.obs_names.numpy().astype(int).astype(str))
            local_embeddings_df.loc[sample_mask_local] = local_embedding.detach().cpu().numpy()
            if self._cfg.model.decoder.dual_decoder:
                y_pred_local_df.loc[sample_mask_local] = y_pred_local.detach().cpu().numpy()

            ## Global model output
            transformer_in, global_embedding, src_padding_mask, pad_index_nodes, I = self.module.global_module.evaluate(
                batch, local_embedding
            )
            # no masking during evaluation
            y_pred_global = self.module.predict_global(global_embedding, src_padding_mask, self.prediction_level)
            ## Save model output

            batch_obs_names_str = batch.obs_names.numpy().astype(int).astype(str)[pad_index_nodes[0]]
            sample_mask = global_embeddings_df.index.isin(batch_obs_names_str)
            global_embeddings_df.loc[sample_mask] = global_embedding[:-1].squeeze(1).detach().cpu().numpy()
            cls_token_horizontal[sample_mask] = I[-1, :-1].squeeze().cpu().detach().numpy()
            cls_token_vertical[sample_mask] = I[:-1, -1].squeeze().cpu().detach().numpy()
            attn_matrix = I[:-1, :-1].cpu().detach().numpy()
            # Pad attention matrix to match max_seq_len with NaN
            padded_attn = np.full(
                (attn_matrix.shape[0], self._cfg.model.global_component.parameters.max_seq_len), np.nan
            )
            padded_attn[:, : attn_matrix.shape[1]] = attn_matrix
            attention_matrix_df.loc[sample_mask] = padded_attn

            y_pred_global_df.loc[sample_mask] = y_pred_global.detach().cpu().numpy()

        adata = self.save_evaluation_results(
            adata,
            prefix,
            y_pred_local_df=y_pred_local_df,
            y_pred_global_df=y_pred_global_df,
            local_embeddings_df=local_embeddings_df,
            global_embeddings_df=global_embeddings_df,
            attention_matrix_df=attention_matrix_df,
            cls_token_horizontal=cls_token_horizontal,
            cls_token_vertical=cls_token_vertical,
        )

        return adata
