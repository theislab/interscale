from yacs.config import CfgNode as CN

def get_model_cfg(cfg):
  """ Defines model training optimization parameters:

    model_type: str = select one from [gnn, transformer, gnn-transformer]
  """
  cfg.model = CN()

  cfg.model.model_type = "gnn-transformer" # [gnn, transformer, gnn-transformer]
  cfg.model.n_epochs = 100
  cfg.model.save = None
  cfg.model.output_path = './results/'
  cfg.model.decoder = CN()
  cfg.model.decoder.type = 'linear' # [linear, nonlinear]
  cfg.model.decoder.hidden_dims = [256, 128]
  cfg.model.decoder.dropout = 0.1
  
  return cfg