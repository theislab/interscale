import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch.nn import functional as F
from yacs.config import CfgNode as CN

from interscale.model.base._base_model import BaseModel
from interscale.module.base import GlobalModule
from interscale.tl import SelfAttentionRelevance, prepare_a2d_dataset
from interscale.train._training import NodeMaskingTrainingPlan


class GlobalModel(NodeMaskingTrainingPlan, BaseModel):
    """Global model with only global component."""

    _module_cls = GlobalModule

    def __init__(
        self,
        adata: AnnData,
        cfg: CN,
    ):
        super().__init__(adata, cfg)

        self._module_kwargs = self._cfg.model.global_component.parameters

        self.local_component = False
        self.global_component = True

        # self.module = self._register_global_component()

        self.module = GlobalModule.from_config(
            cfg,
            n_input=self.n_input,
            n_output=self.n_output,
            n_embed=self.n_embed,
            decoder_type=self._cfg.model.decoder.type,
            dropout_decoder=self._cfg.model.decoder.dropout_decoder,
            decoder_hidden_dims=self._cfg.model.decoder.hidden_dims,
            pct_mask_nodes=self._cfg.dataset.pct_mask_nodes,
            type_gex_embedding=self._cfg.model.global_component.parameters.type_gex_embedding,
        )

    # @torch.inference_mode() Not possible because of pytorch hook for self attention relevance
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

        a2d = prepare_a2d_dataset(self._cfg)
        pyg, _ = list(a2d(adata))

        obs_names_str = adata.obs_names.astype(int).astype(str)

        # Create empty DataFrame with correct shape
        global_embeddings_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_embed), dtype=np.float32)
        attention_matrix_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self._cfg.model.global_component.parameters.max_seq_len),
            dtype=np.float32,
        )

        y_pred_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_output), dtype=np.float32)

        cls_token_horizontal = np.full(len(adata.obs_names), np.nan)
        cls_token_vertical = np.full(len(adata.obs_names), np.nan)
        self_attention_relevance = SelfAttentionRelevance(self.module)

        for batch in pyg:
            if hasattr(batch, "embeddings"):
                embedding = batch.embeddings
            else:
                embedding = self.module.create_gex_embedding(
                    batch.x, type=self._cfg.model.global_component.parameters.type_gex_embedding
                )
            embedding = torch.tensor(embedding, dtype=torch.float32, device=batch.x.device)
            transformer_in, global_embedding, src_padding_mask, pad_index_nodes, I = self.module.evaluate(
                batch, embedding
            )
            # no masking during evaluation
            y_pred = self.module.predict(global_embedding, src_padding_mask, self.prediction_level)

            cosine_sim = F.cosine_similarity(batch.x, y_pred, dim=1)

            # I = self_attention_relevance.generate_relevance(transformer_in, src_padding_mask)
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
            y_pred_df.loc[sample_mask] = y_pred.detach().cpu().numpy()

        # Save embeddings in adata.obsm
        adata = self.save_evaluation_results(
            adata,
            prefix,
            # decoder_weight_df = decoder_weight_df,
            y_pred_local_df=None,
            y_pred_global_df=y_pred_df,
            global_embeddings_df=global_embeddings_df,
            attention_matrix_df=attention_matrix_df,
            cls_token_horizontal=cls_token_horizontal,
            cls_token_vertical=cls_token_vertical,
        )

        return adata
