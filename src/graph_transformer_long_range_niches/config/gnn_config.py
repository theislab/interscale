from yacs.config import CfgNode as CN

def get_gnn_cfg(cfg):
    cfg.gnn = CN()

    cfg.gnn.gnn_type = "GCN"
    cfg.gnn.num_layers = 2
    cfg.gnn.hidden_dim = 256
    cfg.gnn.embed_dim = 128
    cfg.gnn.dropout = 0.1
    return cfg