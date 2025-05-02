from InterScale.model.base._base_model import BaseModelClass
from InterScale.train._training import NodeMaskingTrainingPlan
from InterScale.module.base import LocalModuleClass
from anndata import AnnData
from yacs.config import CfgNode as CN
from InterScale.tl.geome_utils import prepare_a2d_dataset
import numpy as np
import pandas as pd
import torch

class LocalModel(NodeMaskingTrainingPlan,
                 BaseModelClass):
    
    def __init__(self, 
                 adata: AnnData,
                 prediction_task: str,
                 cfg: CN,):
        super().__init__(adata, prediction_task, cfg)
        
        self.local_component = self._register_local_component()
        
    def _common_step(self,
                     batch):
        """Shared step between train, val and test.
        
        Returns:
            local_embedding: torch.Tensor
            global_embedding: torch.Tensor
            y_pred: torch.Tensor
        """
        mask_idx = None
        
        local_embedding = self.local_component.forward(batch.x, batch.edge_index)
        y_pred = self.decoder.forward(local_embedding)
        
        if 'classification' in self.prediction_task:
            y_true = batch.y
            assert y_true.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true
            
        if 'regression' in self.prediction_task:
            y_true = batch.x
            assert y_true.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true
            
        assert False, "Prediction task not supported"
            
    @torch.inference_mode()
    def get_model_output(self,
                         adata: AnnData | None = None):
        """Save the embeddings, predictions and attentionsin the adata object.

        Parameters
        ----------
        adata
            AnnData object to run the model on. If `None`, the model's AnnData object is used.
        """
        
        if not self.is_trained:
            raise RuntimeError("Please train the model first.")
        
        adata = self._validate_anndata(adata)
        
        a2d = prepare_a2d_dataset(self._cfg)
        pyg, _ = list(a2d(adata))
        
        # Create empty DataFrame with correct shape
        local_embeddings_df = pd.DataFrame(
            index=adata.obs_names,
            columns=range(self.n_embed)
        )
        decoder_weight_df = pd.DataFrame(
            index=adata.obs_names,
            columns=range(self.n_output)
        )
        
        for batch in pyg:
            local_embedding = self.local_component.forward(batch.x, batch.edge_index)
            # Get indices for this sample
            sample_mask = local_embeddings_df.index.isin(batch.obs_names.numpy().astype(int).astype(str))
            # Fill embeddings directly into the DataFrame
            local_embeddings_df.loc[sample_mask] = local_embedding.detach().cpu().numpy()
            
            if self.decoder_type == 'linear':
                W = self.decoder.decoder.weight
                contribution = torch.matmul(local_embedding, torch.transpose(W, 0, 1))
                decoder_weight_df.loc[sample_mask] = contribution.detach().cpu().numpy()
                    
        # Save embeddings in adata.obsm
        adata.obsm['local_emb'] = local_embeddings_df.values
        adata.obsm['decoder_weight'] = decoder_weight_df.values
        
        return adata
    
        
        
        
       