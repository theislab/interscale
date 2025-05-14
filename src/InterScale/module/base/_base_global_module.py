from abc import abstractmethod

import torch

from InterScale.module.base import BaseModuleClass
from InterScale.tl import pad_batch
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
        mask_idx = None
        embedding = self.create_gex_embedding(batch.x.cpu().numpy(), type="PCA")
        embedding = torch.tensor(embedding, dtype=torch.float32, device=batch.x.device)
        
        padded_emb, src_padding_mask, index_nodes, attention_mask = self.common_step_local_to_global(batch, embedding)
        global_embedding, src_padding_mask = self.forward(padded_emb, src_padding_mask, attention_mask)
        
        y_true = []

        for i in range(batch.batch[-1] + 1):
            mask = batch.batch.eq(i)
            if 'classification' in prediction_task:
                if 'node' in prediction_level:
                    y_true += batch.y[mask][index_nodes[i]].clone().detach()
                elif 'graph' in prediction_level:
                    y_true.append(batch.y[mask][-1].clone().detach()) 
            elif 'regression' in prediction_task:
                y_true += batch.x[mask][index_nodes[i]].clone().detach()
            else:
                raise Exception('Choose a valid prediction tasks (graph or node).')
        y_true = torch.stack(y_true)
        
        ## Graph-level prediction: get cls
        if 'graph' in prediction_level:
            cls = global_embedding[-1,:, :] # [B, E]
            y_pred = self.decoder(cls)
            return None, global_embedding, y_pred, y_true

        ## Node-level prediction: remove cls
        elif 'node' in prediction_level: 
            h_graph = global_embedding[:-1] # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node
            masked_output = h_graph[~ src_padding_mask] # [N, E]
            y_pred = self.decoder(masked_output)
            return None, global_embedding, y_pred, y_true
        
        else:
            raise Exception('Choose a valid prediction tasks (graph or node).')

    def get_global_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
        