from anndata import AnnData
from yacs.config import CfgNode as CN

import numpy as np
import pandas as pd
import torch

from InterScale.tl import prepare_a2d_dataset, SelfAttentionRelevance
from InterScale.model.base._base_model import BaseModelClass
from InterScale.train._training import NodeMaskingTrainingPlan
from InterScale.module import CombinedModuleClass

class CombinedModel(NodeMaskingTrainingPlan,
                 BaseModelClass):
    
    _module_cls = CombinedModuleClass
    
    def __init__(self, 
                 adata: AnnData,
                 cfg: CN,):
        super().__init__(adata, cfg)
        
        self._module_kwargs = self._cfg.model
        
        self.local_component = True
        self.global_component = True

        # Initialize the combined module with both local and global components
        self.module = CombinedModuleClass(
            cfg=self._cfg,
            n_input=self.n_input,
            n_output=self.n_output,
            n_embed=self.n_embed,
            decoder_type=self._cfg.model.decoder.type,
            dropout_decoder=self._cfg.model.decoder.dropout_decoder,
            decoder_hidden_dims=self._cfg.model.decoder.hidden_dims,
            pct_mask_nodes=self._cfg.dataset.pct_mask_nodes
        )
        
        self._model_summary_string = self._model_summary_string + self.module.get_model_summary()
        
    def get_model_output(self,
                         adata: AnnData | None = None,
                         prefix: str = ""):
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
        
        a2d = prepare_a2d_dataset(self._cfg)
        pyg, _ = list(a2d(adata))
        
        obs_names_str = adata.obs_names.astype(int).astype(str)
        
        # Check for duplicate observation names
        assert len(obs_names_str) == len(obs_names_str.unique()), f"Duplicate observation names found. Expected {len(obs_names_str)} unique names but found {len(obs_names_str.unique())}"
        
        # Create empty DataFrame with correct shape
        local_embeddings_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self.n_embed)
        )

        global_embeddings_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self.n_embed)
        )
        attention_matrix_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self._cfg.model.global_component.parameters.max_seq_len)
        )
        
        decoder_weight_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self.n_output)
        )
        
        y_pred_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self.n_output)
        )
        
        
        cls = np.full(len(adata.obs_names), np.nan)
        
        for batch in pyg:
            ## Get model output
            local_embedding = self.module.local_module.forward(batch.x, batch.edge_index)
            transformer_in, global_embedding, src_padding_mask, index_nodes, I = self.module.global_module.evaluate(batch, local_embedding)
            y_pred = self.module.predict(global_embedding, src_padding_mask, self.prediction_level)
            
            ## Save model output
            # Get indices for this sample
            sample_mask = local_embeddings_df.index.isin(batch.obs_names.numpy().astype(int).astype(str))
            # Fill embeddings directly into the DataFrame
            local_embeddings_df.loc[sample_mask] = local_embedding.detach().cpu().numpy()
            batch_obs_names_str = batch.obs_names.numpy().astype(int).astype(str)[index_nodes[0]]
            sample_mask = global_embeddings_df.index.isin(batch_obs_names_str)
            global_embeddings_df.loc[sample_mask] = global_embedding[:-1].squeeze(1).detach().cpu().numpy()
            cls[sample_mask] = I[:1, 1:].squeeze().cpu().detach().numpy() 
            attn_matrix = I[1:, 1:].cpu().detach().numpy()
            # Pad attention matrix to match max_seq_len with NaN
            padded_attn = np.full((attn_matrix.shape[0], self._cfg.model.global_component.parameters.max_seq_len), np.nan)
            padded_attn[:, :attn_matrix.shape[1]] = attn_matrix
            attention_matrix_df.loc[sample_mask] = padded_attn
            y_pred_df.loc[sample_mask] = y_pred.detach().cpu().numpy()
            
            if self.module.decoder_type == 'linear':
                W = self.module.decoder.decoder.weight
                contribution = torch.matmul(global_embedding[:-1].squeeze(1), torch.transpose(W, 0, 1))
                decoder_weight_df.loc[sample_mask] = contribution.detach().numpy()
                
        adata.obsm[f'{prefix}_local_emb'] = local_embeddings_df.values
        adata.obsm[f'{prefix}_global_emb'] = global_embeddings_df.values
        adata.obsm[f'{prefix}_attn_matrix'] = attention_matrix_df.values
        adata.obsm[f'{prefix}_decoder_weight'] = decoder_weight_df.values
        adata.obs[f'{prefix}_cls'] = cls
        adata.layers[f'{prefix}_y_pred'] = y_pred_df.values
        
        return adata