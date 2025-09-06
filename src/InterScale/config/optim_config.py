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
  cfg.optim.lr_scheduler = "ReduceLROnPlateau" # "ReduceLROnPlateau" or "CosineWarmupScheduler"
  cfg.optim.lr_warmup = 20
  cfg.optim.lr_max_epochs = 100
  cfg.optim.wd = 1e-4
  cfg.optim.loss = "CrossEntropy" # regression: [MSELoss, GaussianNLL, SmoothL1, BalancedPearsonCorrelationLoss]
  cfg.optim.seed = 40
  cfg.optim.cross_corr = 'cell' # Currently cell is the only one that really works
  cfg.optim.n_epochs = 100
  return cfg