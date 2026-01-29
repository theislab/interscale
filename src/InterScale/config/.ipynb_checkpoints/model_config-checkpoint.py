from yacs.config import CfgNode as CN

def get_model_cfg(cfg):
  """ Defines model training optimization parameters:

    model_type: str = select one from [gnn, transformer, gnn-transformer]
  """
  cfg.model = CN()
  cfg.model.n_embed = 16
  
  cfg.model.local_component = CN()
  cfg.model.local_component.name = None
  #cfg.model.local_component.load = None # not needed can be removed
  
  cfg.model.global_component = CN()
  cfg.model.global_component.name = None
  c#fg.model.global_component.load = None # not needed can be removed
  
  cfg.model.save = None
  cfg.model.loss = None
  cfg.model.decoder = CN()
  cfg.model.decoder.type = 'linear' # [linear, nonlinear, linear-lse]
  cfg.model.decoder.hidden_dims = [256, 128]
  cfg.model.decoder.dropout_decoder = 0.1
  cfg.model.decoder.dual_decoder = False # [True, False] If True, use dual decoder for combined module. Both local and global decoders are used.
  
  return cfg