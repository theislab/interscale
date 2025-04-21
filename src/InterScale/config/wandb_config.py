from yacs.config import CfgNode as CN

def get_wandb_cfg(cfg):
    cfg.wandb = CN()

    cfg.wandb.use = False
    cfg.wandb.project_name = ""
    return cfg