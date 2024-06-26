
import torch

from graph_transformer_long_range_niches.tl.utils import pad_batch

def eval_gnntransformer(model, batched_data):
    h_node, z = model.gnn(batched_data.x, batched_data.edge_index)
    h_node = model.gnn2transformer(h_node) 
    padded_h_node, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, batched_data.batch, model.transformer_encoder.max_input_len, get_mask=True
        )
    transformer_out, src_padding_mask = model.transformer_encoder(padded_h_node, src_padding_mask)
    return padded_h_node, transformer_out, src_padding_mask, index_nodes

def extract_attention(model, x, src_padding_mask):
    """Returns a list of attention maps (Tensor) for each Transformer layer.
    """
    attn_weights_maps = []
    attn_maps = []
        
    num_layers = model.transformer_encoder.num_layers
    num_heads = model.transformer_encoder.layers[0].self_attn.num_heads
    norm_first = model.transformer_encoder.layers[0].norm_first

    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = model.transformer_encoder.layers[i].norm1(h)
            attn_output, attn_output_weights = model.transformer_encoder.layers[i].self_attn(h, h, h, need_weights=True, key_padding_mask=src_padding_mask)
            attn_maps.append(attn_output)
            attn_weights_maps.append(attn_output_weights)
            # forward of layer i
            x = model.transformer_encoder.layers[i](x)
            
        # attention_maps = torch.stack(attention_maps, dim=0)
        # attention_maps = torch.mean(attention_maps, dim=0)
        
    return attn_maps, attn_weights_maps