from yacs.config import CfgNode as CN

def get_transformer_cfg(cfg):
    """
        
    """
    cfg.transformer = CN()

    cfg.transformer.d_model = 128
    cfg.transformer.n_heads = 4
    cfg.transformer.dim_feedforward = 512
    cfg.transformer.dropout = 0.3
    cfg.transformer.activation_func = "relu"
    cfg.transformer.num_encoder_layers = 1
    cfg.transformer.max_input_len = 1000
    
    return cfg