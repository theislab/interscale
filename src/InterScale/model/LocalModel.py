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
    
    _module_cls = LocalModuleClass
    
    def __init__(self, 
                 adata: AnnData,
                 cfg: CN,):
        super().__init__(adata, cfg)
        
        self._module_kwargs = self._cfg.model.local_component.parameters
        
        self.local_component = True
        self.global_component = False

        self.module = self._register_local_component()
            
    @torch.inference_mode()
    def get_model_output(self,
                         adata: AnnData | None = None):
        """Save the embeddings, predictions and attentionsin the adata object.

        Parameters
        ----------
        adata
            AnnData object to run the model on. If `None`, the model's AnnData object is used.
        """
        
        if not self.is_trained_:
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
            local_embedding = self.module.forward(batch.x, batch.edge_index)
            # Get indices for this sample
            sample_mask = local_embeddings_df.index.isin(batch.obs_names.numpy().astype(int).astype(str))
            # Fill embeddings directly into the DataFrame
            local_embeddings_df.loc[sample_mask] = local_embedding.detach().cpu().numpy()
            
            if self.module.decoder_type == 'linear':
                W = self.module.decoder.decoder.weight
                contribution = torch.matmul(local_embedding, torch.transpose(W, 0, 1))
                decoder_weight_df.loc[sample_mask] = contribution.detach().cpu().numpy()
                    
        # Save embeddings in adata.obsm
        adata.obsm['local_emb'] = local_embeddings_df.values
        adata.obsm['decoder_weight'] = decoder_weight_df.values
        
        return adata
    
        
        
        
       