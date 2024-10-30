#import pytest

import graph_transformer_long_range_niches
from graph_transformer_long_range_niches.modules.transformer_encoder_hook import TransformerNodeEncoderHook

import torch
import torch.nn as nn
import torch.nn.functional as F

# Step 1: Define a sample configuration object
class Config:
    class Transformer:
        def __init__(self):
            self.max_seq_len = 10
            self.d_model = 32  # Hidden dimension size
            self.n_heads = 4
            self.dropout = 0.1
            self.activation_func = 'relu'
            self.num_layers = 2
            self.dim_feedforward = 64
    def __init__(self):
        self.transformer = self.Transformer()

# Step 2: Instantiate the TransformerNodeEncoder with this configuration
cfg = Config()
model = TransformerNodeEncoderHook(cfg)

# Step 3: Create sample input data for three different classes
batch_size = 6  # We'll simulate a batch of 6 sequences (2 sequences per class)
seq_len = cfg.transformer.max_seq_len
hidden_dim = cfg.transformer.d_model

# Create data patterns for 3 different classes
# Class 1: Random positive values
class_1_data = torch.rand(seq_len, batch_size // 3, hidden_dim, requires_grad=True) * 2
# Class 2: Random negative values
class_2_data = torch.rand(seq_len, batch_size // 3, hidden_dim, requires_grad=True) * -2
# Class 3: Random values centered around zero
class_3_data = torch.randn(seq_len, batch_size // 3, hidden_dim, requires_grad=True)

# Concatenate all class data into a single batch (simulating mixed classes in one batch)
padded_h_node = torch.cat([class_1_data, class_2_data, class_3_data], dim=1)

# Create source padding mask indicating valid tokens (False) and padding tokens (True)
# Assume that sequences of different lengths exist. For simplicity, let's assume sequences
# of length 7 for each class and pad up to length 10.
valid_seq_len = 7
src_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
src_padding_mask[:, valid_seq_len:] = True  # Mark padding positions as True after the valid sequence length

class_labels = torch.tensor([0] * (batch_size // 3) + [1] * (batch_size // 3) + [2] * (batch_size // 3))

# LR Propogation from Transformer-Explainability
# rule 5 from paper
def avg_heads(attn_map, grad):
    """
    Parameters
    ----------
        attn_map: Tensor
            Attention map that defines connections between pairs of tokens, shape: SxBxE (S=sequence, B=batch, E=embedding)
        attn_grad: Tensor

    """
    attn_map = attn_map.reshape(-1, attn_map.shape[-2], attn_map.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    attn_map = grad * attn_map
    attn_map = attn_map.clamp(min=0).mean(dim=0)
    return attn_map

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def generate_relevance(model, padded_h_node, src_padding_mask, category_index=None):
    """
    Parameters
    ----------
        category_index: List[int]
            List of indices that should be considered in a mask, usually from one class. 
            ToDo: Change to indicate class index and retrieve the list of indices???
    """
    output, _ = model(padded_h_node, src_padding_mask, register_hook = True) # [S, B, E]
    category_mask = torch.zeros(output.size())
    category_mask[:, category_index] = 1
    print('Category mask: ', category_mask.shape)
    loss = (output*category_mask).sum()
    model.zero_grad() # ensure all zero elements are None
    loss.backward(retain_graph=True) # backward only works on one elements torch tensor

    num_tokens = model.transformer_encoder.layers[0].get_attn_output().shape[0] # all encoder layers same shape
    print(num_tokens)

    I = torch.eye(num_tokens, num_tokens).cuda()

    for idx, encoder in enumerate(model.transformer_encoder.layers):
        attn_grad = encoder.get_attn_gradients() # [BH, S, S]
        #attn_map = encoder.get_attn_output() # [S, B, E]
        attn_out_weights = encoder.get_attn_output_weights() # [BH, S, S]

        print(f"Layer {idx + 1} attention gradient shape: {attn_grad.shape}")
        print(f"Layer {idx + 1} attention output weights shape: {attn_out_weights.shape}")

        attn_map = avg_heads(attn_out_weights, attn_grad)
        print(f"Average attention map shape: {attn_map.shape}")
        print("I: ", I)
        I += apply_self_attention_rules(I.cuda(), attn_map.cuda()) # (num_tokens X num_tokens)

generate_relevance(model, padded_h_node, src_padding_mask, category_index=1)
