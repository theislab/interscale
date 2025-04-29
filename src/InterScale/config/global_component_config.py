from yacs.config import CfgNode as CN

def get_global_component_cfg(cfg):
    """
    Defines global component configuration.
    """
    cfg.global_component = CN()
    
    if cfg.model.global_component.name == 'Transformer':
        cfg.global_component.parameters.d_model = 128
        cfg.global_component.parameters.n_heads = 4
        cfg.global_component.parameters.dim_feedforward = 256
        cfg.global_component.parameters.dropout = 0.1
        cfg.global_component.parameters.activation_func = "relu"
        cfg.global_component.parameters.num_layers = 2
        cfg.global_component.parameters.max_seq_len = 2000
        cfg.global_component.parameters.long_range_attention = True # if True, takes inverse of adjacency matrix as long-range attention mask
    
    return cfg