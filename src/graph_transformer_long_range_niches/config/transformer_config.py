from yacs.config import CfgNode as CN

def get_transformer_cfg(cfg):
    """
        
    """
    cfg.transformer = CN()

    cfg.transformer.d_model = 128
    cfg.transformer.n_heads = 4
    cfg.transformer.dim_feedforward = 256
    cfg.transformer.dropout = 0.1
    cfg.transformer.activation_func = "relu"
    cfg.transformer.num_layers = 2
    cfg.transformer.max_seq_len = 2000
    
    return cfg