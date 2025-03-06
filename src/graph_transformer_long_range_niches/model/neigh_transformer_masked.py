import torch
import torch.nn as nn
from torch_scatter import scatter_mean  # For neighborhood aggregation
import pytorch_lightning as L

from graph_transformer_long_range_niches.modules import TransformerNodeEncoder, BaseModule, TransformerNodeEncoderHook
from graph_transformer_long_range_niches.tl.utils import pad_batch, create_transformer_attention_mask_from_edges
from sklearn.decomposition import PCA
from graph_transformer_long_range_niches.tl.masking import apply_mask
from typing import List

class LitNeighTransformerMasked(BaseModule):
    def __init__(self, 
                 cfg, 
                 class_weights: List = None, 
                 **model_kwargs
        ):
        super().__init__(cfg, class_weights, **model_kwargs)
        
        self._cfg = cfg
        
        assert cfg.transformer.d_model, "Transformer embedding dimension must be defined."
        
        self.model_type = 'NeighborhoodAggregation_Transformer'
        self.output_dim = cfg.transformer.d_model

        self.norm_input = nn.LayerNorm(self.output_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.output_dim], requires_grad=True))

        # Transformer encoder initialization
        self.transformer_encoder = TransformerNodeEncoderHook(cfg)

    def aggregate_neighbors(self, x, edge_index):
        """
        Perform simple mean aggregation over neighbors.
        
        Parameters:
        - x: Tensor of shape [N, F] (node features)
        - edge_index: Tensor of shape [2, E] (edges)

        Returns:
        - Aggregated node features of shape [N, F]
        """
        row, col = edge_index  # row: sources, col: targets
        aggregated_x = scatter_mean(x[row], col, dim=0, out=torch.zeros_like(x))
        return aggregated_x

    def forward(self, batched_data):
        """
        Input: 
            batched_data: Pytorch geometric object 
                batched_data.x = [N, F]
                batched_data.edge_index = [2, E]
        """
        # Aggregate neighbor features instead of GCN
        h_node = self.aggregate_neighbors(batched_data.x, batched_data.edge_index)
        
        h_node = self.norm_input(h_node)

        if self.masked_nodes:
            keep_indices = batched_data.mask
        else:
            keep_indices = None

        # Padding for transformer input
        padded_h_node, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, 
            batched_data.batch, 
            self.transformer_encoder.max_seq_len, 
            get_mask=self.masked_nodes,
            keep_indices=keep_indices
        )

        if self._cfg.transformer.long_range_attention:
            attention_mask = create_transformer_attention_mask_from_edges(batched_data.edge_index, len(batched_data.obs_names), batched_data.batch, index_nodes, self.transformer_encoder.n_heads)
        else:
            attention_mask = None
            
        transformer_out = padded_h_node
        transformer_out, src_padding_mask = self.transformer_encoder(transformer_out, src_padding_mask, mask=attention_mask)

        ## Graph-level prediction
        if 'graph' in self.prediction_task:
            cls = transformer_out[-1, :, :]  # [B, E]
            out = self.graph_pred_linear(cls)
            return h_node, out, index_nodes

        ## Node-level prediction
        elif 'node' in self.prediction_task:
            h_graph = transformer_out[:-1]  # [S, B, E]
            h_graph = torch.permute(h_graph, (1, 0, 2))  # [B, S, E]
            src_padding_mask = src_padding_mask[:, :-1]  # [B, S]
            masked_output = h_graph[~src_padding_mask]  # [N, E]
            out = self.graph_pred_linear(masked_output)
            return h_node, out, index_nodes

        else:
            raise Exception('Choose a valid prediction task (graph or node).')
