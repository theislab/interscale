from abc import abstractmethod
import torch
from typing import Optional

from InterScale.module.base import BaseModuleClass, LocalModuleClass, GlobalModuleClass
from yacs.config import CfgNode as CN

from InterScale.tl import apply_mask
from typing import Literal


class CombinedModuleClass(BaseModuleClass):
    
    def __init__(self,
                 cfg: CN,
                 **base_module_kwargs):
        super().__init__(**base_module_kwargs)
        
        self.local_module_args = cfg.model.local_component
        self.global_module_args = cfg.model.global_component
        
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
                                                           decoder_type="linear",
                                                           pct_mask_nodes=self.pct_mask_nodes)
        
    def predict(self,
                global_embedding,
                src_padding_mask,
                prediction_level,
                prediction_task,
                pad_index_nodes,
                mask_idx):
        """Predict with the decoder."""
        return self.global_module.predict(global_embedding, src_padding_mask, prediction_level)
        
    def _common_step(self,
                    batch, 
                    prediction_task, 
                    prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.
        """
        batch_masked, mask_idx = self._common_step_masking(batch)
            
        local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask = self.forward(batch_masked)
        y_pred, y_true = self.predict(global_embedding, src_padding_mask, prediction_level, prediction_task, pad_index_nodes, mask_idx)

        return local_embedding, global_embedding, y_pred, y_true
    
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
        