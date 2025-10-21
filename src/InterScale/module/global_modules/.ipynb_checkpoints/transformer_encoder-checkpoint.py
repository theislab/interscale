import torch
from torch import nn

from InterScale.module.base import GlobalModuleClass
from InterScale.module.global_modules.transformer_encoder_layer import CustomTransformerEncoderLayer
from InterScale.tl import pad_batch, create_transformer_attention_mask_from_edges

class TransformerNodeEncoderHook(GlobalModuleClass):
    """
    Sequence of: Dropout → Layer Norm → FC → nonlinearity → Dropout → FC → Dropout → Layer Norm + residual connections
    """

    def __init__(self,
                 max_seq_len: int,
                 n_heads: int = 4,
                 act_func: nn.Module = nn.ReLU(),
                 num_layers: int = 3,
                 dim_feedforward: int = 2048,
                 dropout_global: float = 0.1,
                 **base_module_kwargs):
        
        super().__init__(*base_module_kwargs) 
        # Save model parameters
        self.model_type = 'TransformerEncoder'
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.act_func = act_func
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_global = dropout_global
        
        # Create Transformer Encoder
        encoder_layer = CustomTransformerEncoderLayer(
            self.n_embed, self.n_heads, self.dim_feedforward, self.dropout_global, self.act_func
        )
        encoder_norm = nn.LayerNorm(self.n_embed)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers, norm=encoder_norm)

        self.norm_input = nn.LayerNorm(self.n_embed)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.n_embed], requires_grad=True))

    def common_step_local_to_global(self, batched_data, emb):
        """
        Convert local node embeddings [N, E] to padded local node embeddings [max_seq_len, E] 
        with N being the number of nodes in the graph and E being the embedding dimension.
        
        Parameters:
            batched_data: Pytorch geometric object 
            emb: torch.Tensor [N, E]
                Embedding of the local model or user-provided embeddings.
        
        Returns:
            padded_emb: torch.Tensor [max_seq_len, E]
                Padded local node embeddings
            src_padding_mask: torch.Tensor [max_seq_len]
                Mask indicating padding nodes
            index_nodes: torch.Tensor [N]
                Indices of the nodes in the original graph
        """
        # Layer normalization
        emb = self.norm_input(emb)
        
        if self.masked_nodes:
            keep_indices = batched_data.mask
        else:
            keep_indices = None

        # Ensure masked nodes are included in padding
        padded_emb, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            emb, 
            batched_data.batch, 
            self.transformer_encoder.max_seq_len, 
            get_mask=self.masked_nodes,
            keep_indices=keep_indices  # Add parameter to ensure masked nodes are kept
        )
        
        if self._cfg.transformer.long_range_attention:
            attention_mask = create_transformer_attention_mask_from_edges(
                batched_data.edge_index, 
                len(batched_data.obs_names), 
                batched_data.batch, 
                index_nodes, 
                self.transformer_encoder.n_heads
            )
            # Convert attention_mask to same dtype as src_padding_mask
            attention_mask = attention_mask.to(dtype=src_padding_mask.dtype)
        else:
            attention_mask = None
            
        return padded_emb, src_padding_mask, index_nodes, attention_mask

    def forward(self, padded_h_node, src_padding_mask, mask = None, register_hook: bool = False):
        """
        Input: 
            padded_h_node: [n_b x B X h_d] with n_b: dimension of batch, B: batch size, h_d: dimension of transformer
            src_padding_mask: [B x n_b] matrix indicating the size of the padding mask to be ignored during calculation 
            mask: [n_b x n_b] matrix indicating the long-range connections (inverse of adjacency matrix). Default: None
        """
        if register_hook:
            for encoder in self.transformer_encoder.layers:
                encoder.register_hook = True

        # append cls embedding
        expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
        padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0)
        # normalize input
        padded_h_node = self.norm_input(padded_h_node)

        zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
        src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)

        transformer_out = self.transformer_encoder(padded_h_node, src_key_padding_mask=src_padding_mask, mask=mask)  # (S, B, h_d)

        if register_hook:
            for encoder in self.transformer_encoder.layers:
                encoder.register_hook = False

        return transformer_out, src_padding_mask