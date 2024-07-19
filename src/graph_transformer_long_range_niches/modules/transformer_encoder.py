import torch
from torch import nn
import torch.nn.functional as F

class TransformerNodeEncoder(nn.Module):
    """
    Sequence of: Dropout → Layer Norm → FC → nonlinearity → Dropout → FC → Dropout → Layer Norm + residual connections
    """
    
    def __init__(self, cfg):

        super().__init__()

        # Save model parameters
        self.model_type = 'TransformerEncoder'
        self.max_seq_len = cfg.transformer.max_seq_len
        self.d_model = cfg.transformer.d_model
        self.n_heads = cfg.transformer.n_heads
        self.dropout = cfg.transformer.dropout
        self.act_func = cfg.transformer.activation_func
        self.num_encoded_layers = cfg.transformer.num_encoder_layers
        self.dim_feedforward = cfg.transformer.dim_feedforward

        ## ToDo print model parameters

        # Create Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.n_heads, self.dim_feedforward, self.dropout, self.act_func
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_encoded_layers, norm=encoder_norm)

        self.norm_input = nn.LayerNorm(self.d_model)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.d_model], requires_grad=True))
        

    def forward(self, padded_h_node, src_padding_mask):
        """
        Input: 
            padded_h_node: [n_b x B X h_d] with n_b: dimension of batch, B: batch size, h_d: dimension of transformer
            padding_mask: [B x n_b] matrix indicating the size of the padding mask to be ignored during calculation 
        """        
        # append cls embedding
        expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
        padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0)
        # normalize input
        padded_h_node = self.norm_input(padded_h_node)

        # print('CLS + Normalized padded_h_node: ', padded_h_node.shape)

        zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
        src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)

        # print("Padding mask: ", src_padding_mask.shape)

        transformer_out = self.transformer_encoder(padded_h_node, src_key_padding_mask=src_padding_mask)  # (S, B, h_d)
        return transformer_out, src_padding_mask