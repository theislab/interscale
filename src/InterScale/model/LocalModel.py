from InterScale.model.base._base_model import BaseModelClass
from InterScale.train._training import NodeMaskingTrainingPlan
from InterScale.module.base import LocalModuleClass
from InterScale.module.local_modules import GCN
from anndata import AnnData
from yacs.config import CfgNode as CN
from InterScale.tl.geome_utils import prepare_a2d_dataset
import numpy as np
import pandas as pd
import torch

class LocalModel(NodeMaskingTrainingPlan,
                 BaseModelClass):
    
    _module_cls = LocalModuleClass
    
    def __init__(self, 
                 adata: AnnData,
                 cfg: CN,):
        super().__init__(adata, cfg)
        
        self.local_component = True
        self.global_component = False

        self.module = LocalModuleClass.from_config(
            cfg,
            n_input=self.n_input,
            n_output=self.n_output,
            n_embed=self.n_embed,
            decoder_type=self._cfg.model.decoder.type,
            dropout_decoder=self._cfg.model.decoder.dropout_decoder,
            decoder_hidden_dims=self._cfg.model.decoder.hidden_dims,
            pct_mask_nodes=self._cfg.dataset.pct_mask_nodes
        )
            
    @torch.inference_mode()
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
        
        # Check for duplicate observation names
        adata.obs_names = [str(i) for i in range(1, len(adata.obs_names) + 1)] # ensure that no duplicate observation names are present
        assert len(adata.obs_names) == len(adata.obs_names.unique()), f"Duplicate observation names found. Expected {len(adata.obs_names)} unique names but found {len(adata.obs_names.unique())}"
        
        a2d = prepare_a2d_dataset(self._cfg)
        pyg, _ = list(a2d(adata))
        
        obs_names_str = adata.obs_names.astype(int).astype(str)
        
        # Create empty DataFrame with correct shape
        local_embeddings_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self.n_embed),
            dtype=np.float32
        )
        # decoder_weight_df = pd.DataFrame(
        #     index=obs_names_str,
        #     columns=range(self.n_output)
        # )
        y_pred_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self.n_output),
            dtype=np.float32
        )
        
        for batch in pyg:
            local_embedding = self.module.forward(batch.x, batch.edge_index)
            # Get indices for this sample
            sample_mask = local_embeddings_df.index.isin(batch.obs_names.numpy().astype(int).astype(str))
            # Fill embeddings directly into the DataFrame
            local_embeddings_df.loc[sample_mask] = local_embedding.detach().cpu().numpy()
            
            # if self.module.decoder_type == 'linear':
            #     W = self.module.decoder.decoder.weight
            #     #contribution = torch.matmul(local_embedding, torch.transpose(W, 0, 1))
            #     #decoder_weight_df.loc[sample_mask] = contribution.detach().cpu().numpy()
            #     decoder_weight_df.loc[sample_mask] = W.detach().cpu().numpy()
            
            y_pred = self.module.predict(local_embedding, self.prediction_level)
            y_pred_df.loc[sample_mask] = y_pred.detach().cpu().numpy()

        # Save embeddings in adata.obsm
        adata = self.save_evaluation_results(adata, 
                                             prefix, 
                                             local_embeddings_df = local_embeddings_df, 
                                             #decoder_weight_df = decoder_weight_df, 
                                             y_pred_local_df = y_pred_df)

        
        return adata
    
        
        
        
       