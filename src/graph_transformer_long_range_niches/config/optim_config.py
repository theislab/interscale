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

  cfg.optim.lr = 0.001 
  cfg.optim.wd = 1e-3 
  cfg.optim.warm_up = 10 
  cfg.optim.loss = "CrossEntropy" 
  cfg.optim.seed = 42
  return cfg