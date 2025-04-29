from yacs.config import CfgNode as CN

def get_local_component_cfg(cfg, local_component_name):
    
    cfg.model.local_component.parameters = CN()
    
    if local_component_name == "GCN":
        cfg.model.local_component.parameters.embed_dim = 32
        cfg.model.local_component.parameters.hidden_dim = 256
        cfg.model.local_component.parameters.num_layers = 2
        cfg.model.local_component.parameters.dropout = 0.1
    
    return cfg