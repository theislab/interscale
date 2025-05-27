from abc import abstractmethod
import torch
from typing import Optional

from InterScale.module.base import BaseModuleClass, LocalModuleClass, GlobalModuleClass
from yacs.config import CfgNode as CN

from InterScale.tl import apply_mask


class CombinedModuleClass(BaseModuleClass):
    
    def __init__(self,
                 cfg: CN,
                 **base_module_kwargs,):
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
                                                           decoder_type=None,
                                                           dropout_decoder=0,
                                                           decoder_hidden_dims=[],
                                                           pct_mask_nodes=self.pct_mask_nodes)
        
    def _common_step(self,
                    batch, 
                    prediction_task, 
                    prediction_level):
        """Shared step between train, val and test.
        """
        # Mask nodes 
        if self.pct_mask_nodes > 0:
            batch_masked, mask_idx = apply_mask(batch)
        else:
            # pretend as if all nodes are masked
            mask_idx = torch.arange(batch.x.shape[0])
            batch_masked = batch
            
        local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask = self.forward(batch_masked)
        
        y_true, adjusted_mask_idx = self.global_module._process_batch_for_metrics(batch, 
                                                                    prediction_task, 
                                                                    prediction_level, 
                                                                    pad_index_nodes, 
                                                                    mask_idx)
        
        ## Graph-level prediction: get cls
        if 'graph' in prediction_level:
            cls = global_embedding[-1,:, :] # [B, E]
            y_pred = self.decoder(cls)
        ## Node-level prediction: remove cls
        elif 'node' in prediction_level: 
            h_graph = global_embedding[:-1] # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node
            masked_output = h_graph[~ src_padding_mask] # [N, E]
            y_pred = self.decoder(masked_output)
        else:
            raise Exception('Choose a valid prediction tasks (graph or node).')
    
        y_pred_masked = y_pred[adjusted_mask_idx]
        y_true_masked = y_true[adjusted_mask_idx]
        
        assert len(y_pred_masked) == len(y_true_masked), "y_pred and y_true are not consistent" 
        return local_embedding, global_embedding, y_pred_masked, y_true_masked
        
    def forward(
        self,
        batch_masked):
        """Forward pass through the model"""
        
        local_embedding = self.local_module.forward(batch_masked.x, batch_masked.edge_index)
        
        padded_emb, src_padding_mask, pad_index_nodes, attention_mask = self.global_module.common_step_local_to_global(batch_masked, local_embedding)
        global_embedding, src_padding_mask = self.global_module.forward(padded_emb, src_padding_mask, attention_mask)
        
        return local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask
    
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
        