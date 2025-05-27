from yacs.config import CfgNode as CN

def get_optim_cfg(cfg):
  """ Defines model training optimization parameters:

    lr: float = Learning rate 
    wd: float = Weight decay
    warm_up: int = Warm up epochs
    losss: str = Loss function, either CrossEntropy or WeightedCE
    seed: int
  """
  cfg.optim = CN()

  cfg.optim.lr = 0.005 
  cfg.optim.wd = 0.0
  cfg.optim.warmup = 40
  cfg.optim.loss = "CrossEntropy" 
  cfg.optim.seed = 40
  cfg.optim.cross_corr = 'gene' # cross-gene or cross-cell correlation
  return cfg