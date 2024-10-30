import torch
from torch import nn

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Overwrite TransformerEncoderLayer from PyTorch to implement attention map and gradient hook in the self-attention block."""

    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.attn_output = None  # To store attention output
        self.attn_output_weights = None
        self.attn_gradients = None  # To store attention weights gradients
        self.register_hook = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attn_output(self, attn_output):
        self.attn_output = attn_output

    def get_attn_output(self):
        return self.attn_output
    
    def save_attn_output_weights(self, attn_output_weights):
        self.attn_output_weights = attn_output_weights

    def get_attn_output_weights(self):
        return self.attn_output_weights
    
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        """Adjusted self-attention block to save gradients when register_hook is True. 

        Output:
            attention output: Tensor 
                when input is unbatched (L, E), or when batched (L,N,E)
            attention_weights: Tensor
                attention weigths between L (target sequence) and S (source sequence)
                unbatched: (L,S) or batched: (N,L,S) where 
        """
        # Set need_weights=True to get attention weights
        attn_output, attn_output_weights = self.self_attn( 
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False, # return weigths per head
            is_causal=is_causal # from nn.TransformerEncoderLayer
        )
        self.save_attn_output(attn_output)
        self.save_attn_output_weights(attn_output_weights)
        if self.register_hook:
            attn_output_weights.register_hook(self.save_attn_gradients) # Note: Source code modification in MultiHeadAttention Module (F.multi_head_attention_forward function) to calculate gradient on weights
        return self.dropout1(attn_output)