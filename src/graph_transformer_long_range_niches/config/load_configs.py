from torch_geometric.graphgym.register import register_config

@register_config('dataloader')
def dataloader_cfg(cfg):
    cfg.data_path = ''
    cfg.radius = 30