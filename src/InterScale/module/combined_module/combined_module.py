from abc import abstractmethod
import torch
from typing import Optional

from InterScale.module.base import BaseModuleClass, LocalModuleClass, GlobalModuleClass
from yacs.config import CfgNode as CN

from InterScale.tl import apply_mask
from typing import Literal


# MODULE_REGISTRY = {
#     "GIN": LocalModuleClass,
#     "GCN": LocalModuleClass,
#     'scVI': SCVILocalModule,
#     "Precomputed": PrecomputedEmbeddingModule
# }


class CombinedModuleClass(BaseModuleClass):
    
    def __init__(self,
                 cfg: CN,
                 **base_module_kwargs):
        super().__init__(**base_module_kwargs)
        
        self.local_module_args = cfg.model.local_component
        self.global_module_args = cfg.model.global_component
        
        # module_name = cfg.model.local_component.name
        # local_class = MODULE_REGISTRY.get(module_name)

        # if local_class is None:
        #     raise ValueError(f"Module {module_name} not found in MODULE_REGISTRY")
        
        # print(local_class)
        
        self.registered_local_component = True
        self.registered_global_component = True
        
        self.local_module = LocalModuleClass.from_config(cfg,
                                                         n_input=self.n_input,
                                                         n_output=self.n_output,
                                                         n_embed=self.n_embed,
                                                         decoder_type=None, # don't need decoder for local module
                                                         dropout_decoder=0,
                                                         decoder_hidden_dims=[],
                                                         pct_mask_nodes=self.pct_mask_nodes)
        self.global_module = GlobalModuleClass.from_config(cfg,
                                                           n_input=self.n_input,
                                                           n_output=self.n_output,
                                                           n_embed=self.n_embed,
                                                           decoder_type=cfg.model.decoder.type,
                                                           pct_mask_nodes=self.pct_mask_nodes)
        
    def predict_local(self,
                      local_embedding):
        """Predict with the decoder."""
        raise Exception("For CombinedModel the local module does not have a decoder")
        
    def predict_global(self,
                global_embedding,
                src_padding_mask,
                prediction_level):
        """Predict with the decoder."""
        return self.global_module.predict(global_embedding, src_padding_mask, prediction_level)
    
    def forward(
        self,
        batch_masked):
        """Forward pass through the model"""
        local_input = getattr(batch_masked, 'embeddings', batch_masked.x)
        
        local_embedding = self.local_module.forward(local_input, batch_masked.edge_index)
        
        padded_emb, src_padding_mask, pad_index_nodes, attention_mask = self.global_module.common_step_local_to_global(batch_masked, local_embedding)
        assert not torch.any(torch.isnan(padded_emb)), "padded_emb contains NaN values"
        global_embedding, src_padding_mask, attn_matrix = self.global_module.forward(padded_emb, src_padding_mask, attention_mask)
        assert not torch.any(torch.isnan(global_embedding)), "global_embedding contains NaN values"
        
        return local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn_matrix
        
    def _common_step(self,
                    batch, 
                    prediction_task, 
                    prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.
        """
        batch_masked, mask_idx = self._common_step_masking(batch)
            
        local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn_matrix = self.forward(batch_masked)
        y_pred = self.predict_global(global_embedding, src_padding_mask, prediction_level)
        
        if prediction_task == 'classification' and prediction_level == 'graph':
            y_true = batch.y[batch.ptr[:-1]]
        else:
            y_true, adjusted_mask_idx = self.global_module._process_batch_for_metrics(batch, prediction_task, prediction_level, pad_index_nodes, mask_idx)
            y_pred = y_pred[adjusted_mask_idx]
            y_true = y_true[adjusted_mask_idx]
            
        assert len(y_pred) == len(y_true), "y_pred and y_true are not consistent"
        assert not torch.any(torch.isnan(y_pred)), "y_pred contains NaN values"
        assert not torch.any(torch.isnan(y_true)), "y_true contains NaN values"

        return local_embedding, global_embedding, y_pred, y_true, attn_matrix
    
    def get_model_summary(self) -> str:
        """Returns a string containing the model's parameters summary.

        Returns:
            str: Summary string with model parameters
        """
        summary = (
            f"Combined Module: \n"
            f"Local Module: {self.local_module.get_model_summary()}\n"
            f"Global Module: {self.global_module.get_model_summary()}\n"
        )
        return summary
        