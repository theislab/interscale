from abc import abstractmethod

import torch

from InterScale.module.base import BaseModuleClass
from InterScale.tl import pad_batch, apply_mask
from typing import Literal
from sklearn.decomposition import PCA

SEQ_LEN_MASK = None

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
        """Generate embeddings for GEX if no local component is used."""
        if type == "PCA":
            return self.pca.fit_transform(embeddings)
        # elif type == "scvi":
        #     return scvi.model.SCVI(embeddings)
        else:
            raise ValueError(f"Invalid embedding type: {type}")
    
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
        if self.pct_mask_nodes > 0:
            batch_masked, mask_idx = apply_mask(batch)
        else:
            mask_idx = torch.ones(batch.x.shape[0], dtype=torch.bool, device=batch.x.device)
            batch_masked = batch
        
        embedding = self.create_gex_embedding(batch_masked.x.cpu().numpy(), type="PCA")
        embedding = torch.tensor(embedding, dtype=torch.float32, device=batch_masked.x.device)
        
        padded_emb, src_padding_mask, pad_index_nodes, attention_mask = self.common_step_local_to_global(batch_masked, embedding)
        global_embedding, src_padding_mask = self.forward(padded_emb, src_padding_mask, attention_mask)
        
        y_true = []
        adjusted_mask_idx = []  # Track new positions of masked nodes
        current_offset = 0
        start = 0

        for i in range(batch.batch[-1] + 1):
            mask = batch.batch.eq(i)
            pad_indices = torch.tensor(pad_index_nodes[i], device=batch.x.device)
            end = start + len(pad_indices)

            # can not assume that pad_indices is a subset of mask_idx
            for idx in mask_idx[start:end]:
                if idx in pad_indices:
                    new_idx = torch.where(pad_indices == idx)[0].item()
                    adjusted_mask_idx.append(new_idx + current_offset)
            
            current_offset += len(pad_indices)
            start = end
            
            if 'classification' in prediction_task:
                if 'node' in prediction_level:
                    y_true += batch.y[mask][pad_index_nodes[i]].clone().detach()
                elif 'graph' in prediction_level:
                    y_true.append(batch.y[mask][-1].clone().detach())
            elif 'regression' in prediction_task:
                y_true += batch.x[mask][pad_index_nodes[i]].clone().detach()
            else:
                raise Exception('Choose a valid prediction tasks (graph or node).')
            assert len(mask) >= len(pad_indices) >= len(adjusted_mask_idx), "mask, pad_indices, adjusted_mask_idx are not consistent"
            
        y_true = torch.stack(y_true)
        adjusted_mask_idx = torch.tensor(adjusted_mask_idx, device=y_true.device)
        
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
        return None, global_embedding, y_pred_masked, y_true_masked

    def get_global_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
        