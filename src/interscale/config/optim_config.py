from yacs.config import CfgNode as CN


def get_optim_cfg(cfg):
    """Defines model training optimization parameters:

    lr: float = Learning rate
    wd: float = Weight decay
    warm_up: int = Warm up epochs
    losss: str =
    seed: int
    """
    cfg.optim = CN()

    cfg.optim.accelerator = "auto"  # can also be "gpu" or "cpu"
    cfg.optim.lr = 0.001
    cfg.optim.lr_scheduler = "CosineWarmupScheduler"  # "ReduceLROnPlateau" or "CosineWarmupScheduler"
    cfg.optim.lr_warmup = 20
    cfg.optim.lr_max_epochs = 100
    cfg.optim.wd = 1e-4
    cfg.optim.loss = "GaussianNLL"  # classification: [CrossEntropy, WeightedCE], regression: [MSELoss, GaussianNLL, SmoothL1, BalancedPearsonCorrelationLoss, SCELoss]
    cfg.optim.seed = 40
    cfg.optim.cross_corr = "cell"  # Currently cell is the only one that really works
    cfg.optim.n_epochs = 100
    cfg.optim.early_stopping = True
    cfg.optim.patience = 5  # EarlyStopping patience in epochs
    cfg.optim.min_delta = 0.0  # EarlyStopping min_delta
    # Floor on training length. MUST stay above lr_warmup: CosineWarmupScheduler ramps the LR
    # linearly over lr_warmup epochs, so a run that stops inside the ramp has only ever seen a
    # fraction of cfg.optim.lr and is measured at (near) initialisation. The default is 2x
    # lr_warmup, matching chen22. `_validate_optim` enforces the invariant at config-load time
    # rather than leaving it to each dataset to remember.
    cfg.optim.min_epochs = 40
    # Metric driving EarlyStopping / ModelCheckpoint / the LR scheduler.
    # "auto" -> val_f1_macro for classification, val_loss for regression.
    cfg.optim.monitor = "auto"
    return cfg
