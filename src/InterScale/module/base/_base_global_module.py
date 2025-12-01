from abc import abstractmethod

import torch

from InterScale.module.base import BaseModuleClass
from InterScale.tl import pad_batch, apply_mask
from typing import Literal
from sklearn.decomposition import PCA
import numpy as np

class GlobalModuleClass(BaseModuleClass):
    def __init__(self,
                 **base_module_kwargs):
        
        super().__init__(**base_module_kwargs)
        
        self.registered_local_component = False
        self.registered_global_component = True
        
        self.pca = PCA(n_components=self.n_embed)
    
    @abstractmethod
    def forward(self, embeddings: torch.Tensor):
        """
        Input: 
            embeddings: torch.Tensor
                Size: [N, E], either output of local component or user-provided embeddings.
        """
        
    def create_gex_embedding(self, 
                             embeddings: torch.Tensor,
                             type: Literal["PCA", "scvi"]):
        """Generate embeddings for GEX if no local component is used.
        
        Parameters
        ----------
        batch.x: torch.Tensor
            Size: [N, F]
        type: Literal["PCA", "scvi"]
            Type of embedding to generate.
        
        Returns
        -------
        gex_embedding: torch.Tensor
            Size: [N, E]
        """
        if type == "PCA":
            # Fit PCA only once (on first batch), then use transform for subsequent batches
            # This avoids expensive refitting on every batch during training
            if not hasattr(self.pca, 'components_'):
                return self.pca.fit_transform(embeddings)
            else:
                return self.pca.transform(embeddings)
        # elif type == "scvi":
        #     return scvi.model.SCVI(embeddings)
        else:
            raise ValueError(f"Invalid embedding type: {type}")
        
    def _process_batch_for_metrics(self, batch, prediction_task, prediction_level, pad_index_nodes, mask_idx_tensor):
        """Process batch to extract y_true and adjusted_mask_idx for metrics calculation.
        
        mask_idx = torch.tensor([0, 2, 3, 7, 8])
        pad_index_nodes = [[0, 1, 2, 3], [0, 1], [0, 1, 2, 3]]
        
        Parameters
        ----------
        batch
            Input batch
        prediction_task: str
            Type of prediction task ('classification' or 'regression')
        prediction_level: str
            Level of prediction ('node' or 'graph')
        pad_index_nodes: List[List[int]]
            List of padded node indices: [B, S] or [B,N] if number of nodes in graph are smaller than max_seq_len (S)
        mask_idx_tensor: torch.Tensor
            Indices of masked nodes of shape [N_masked_nodes] with range [0, N_nodes-1]
            
        Returns
        -------
        y_true: torch.Tensor
            Ground truth values
        adjusted_mask_idx: torch.Tensor
            Adjusted indices for masked nodes
        """
        assert prediction_level == "node", "Node specific retrieval only necessary for node-level prediction."
                
        y_true = []
        adjusted_mask_idx = []  # Track new positions of masked nodes
        current_offset = 0
        start = 0
        mask_j = 0
        
        for i in range(batch.batch[-1] + 1):
            mask = batch.batch.eq(i)
            pad_indices = torch.tensor(pad_index_nodes[i], device=batch.x.device) + start
            end = start + torch.sum(mask)

            # can not assume that pad_indices is a subset of mask_idx
            #TODO: use stack and pop instead 
            for j, mask_idx in enumerate(mask_idx_tensor[mask_j:]):
                if mask_idx > end:
                    break
                if mask_idx in pad_indices:
                    new_idx = torch.where(pad_indices == mask_idx)[0].item()
                    adjusted_mask_idx.append(new_idx + current_offset)
            
            current_offset += len(pad_indices)
            start = end
            mask_j = j
            
            if 'classification' in prediction_task:
                y_true += batch.y[mask][pad_index_nodes[i]].clone().detach()
            elif 'regression' in prediction_task:
                y_true += batch.x[mask][pad_index_nodes[i]].clone().detach()
            else:
                raise Exception('Choose a valid prediction tasks (graph or node).')
            assert len(mask) >= len(pad_indices) >= len(adjusted_mask_idx), "mask, pad_indices, adjusted_mask_idx are not consistent"
            
        y_true = torch.stack(y_true)
        adjusted_mask_idx = torch.tensor(adjusted_mask_idx, device=y_true.device)
        
        return y_true, adjusted_mask_idx
    
    def predict(self,
                global_embedding,
                src_padding_mask,
                prediction_level):
        """Predict with the decoder.
        
        Parameters
        ----------
        global_embedding: torch.Tensor
            Size: [N, E]
        prediction_level: Literal["node", "graph"]
            Level of prediction
        """
        ## Graph-level prediction: get cls_token from last position
        if 'graph' in prediction_level:
            cls_token = global_embedding[-1,:, :] # [B, E]
            return self.decoder(cls_token)
        ## Node-level prediction: remove cls_token from last position
        elif 'node' in prediction_level: 
            h_graph = global_embedding[:-1] # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node
            masked_output = h_graph[~ src_padding_mask] # [N, E]
            return self.decoder(masked_output)
        else:
            raise Exception('Choose a valid prediction tasks (graph or node).')
        
    def _common_step(self,
                     batch,
                     prediction_task: str, 
                     prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.
        
        Returns
        -------
        local_embedding: torch.Tensor 
            Size: [N, E]
        global_embedding: torch.Tensor 
            Size: [N, E] with SEQ_LEN_MASK for padding nodes.
        y_pred: torch.Tensor 
            Size: [N, C] (classification) or [N, F] (regression) with SEQ_LEN_MASK for padding nodes.
        y_true: torch.Tensor 
            Size: [N, C] (classification) or [N, F] (regression) with SEQ_LEN_MASK for padding nodes.
        """
        # Mask nodes  - before GEX embedding because otherwise embedding contains information about masked nodes
        batch_masked, mask_idx = self._common_step_masking(batch)
        
        embedding = self.create_gex_embedding(batch_masked.x.cpu().numpy(), type="PCA")
        embedding = torch.tensor(embedding, dtype=torch.float32, device=batch_masked.x.device)
        
        assert embedding.shape == (batch_masked.x.shape[0], self.n_embed), f"Mismatch: embedding.shape: {embedding.shape}, batch_masked.x.shape: {batch_masked.x.shape}"
        assert not torch.any(torch.isnan(embedding)), "embedding contains NaN values"
        
        padded_emb, src_padding_mask, pad_index_nodes, attention_mask = self.common_step_local_to_global(batch_masked, embedding)
        assert not torch.any(torch.isnan(padded_emb)), "padded_emb contains NaN values"
        global_embedding, src_padding_mask = self.forward(padded_emb, src_padding_mask, attention_mask)
        assert not torch.any(torch.isnan(global_embedding)), "global_embedding contains NaN values"
        
        y_pred = self.predict(global_embedding, src_padding_mask, prediction_level)
        
        if prediction_task == 'classification' and prediction_level == 'graph':
            y_true = batch.y[batch.ptr[:-1]]
        else:
            y_true, adjusted_mask_idx = self._process_batch_for_metrics(batch, prediction_task, prediction_level, pad_index_nodes, mask_idx)
            y_pred = y_pred[adjusted_mask_idx]
            y_true = y_true[adjusted_mask_idx]
            
        assert len(y_pred) == len(y_true), "y_pred and y_true are not consistent"
        assert not torch.any(torch.isnan(y_pred)), "y_pred contains NaN values"
        assert not torch.any(torch.isnan(y_true)), "y_true contains NaN values"
        
        return None, global_embedding, y_pred, y_true

    def get_global_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
    
    # acts as a factory method to create a module from a config
    @staticmethod
    def from_config(cfg, **kwargs):
        module_name = cfg.model.global_component.name
        params = cfg.model.global_component.parameters.copy()  # Make a copy to avoid modifying the original
            
        if module_name == 'self-attn-transformer':
            from InterScale.module.global_modules import TransformerNodeEncoderHook
            return TransformerNodeEncoderHook(max_seq_len = params['max_seq_len'],
                                        n_heads = params['n_heads'],
                                        dropout_global = params['dropout_global'],
                                        act_func = params['activation_func'],
                                        num_layers = params['num_layers'],
                                        dim_feedforward = params['dim_feedforward'],
                                        long_range_attention = params['long_range_attention'],
                                        **kwargs)
        # Add more elifs for other modules
        else:
            raise ValueError(f"Unknown local module name: {module_name}")
        