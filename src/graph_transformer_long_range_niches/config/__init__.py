from os.path import dirname, basename, isfile, join
from yacs.config import CfgNode as CN
import glob
from .wandb_config import get_wandb_cfg
from .gnn_config import get_gnn_cfg
from .dataset_config import get_dataset_cfg
from .model_config import get_model_cfg
from .optim_config import get_optim_cfg
from .transformer_config import get_transformer_cfg

# TODO: load all configs from folder automatically instead of manual definition
# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [
#     basename(f)[:-3] for f in modules
#     if isfile(f) and not f.endswith('__init__.py')
# ]

def get_cfg_defaults():
    """ Loads the default settings from the .py files in the config folder.
    """
    cfg = CN()

    # Load configurations
    cfg = get_wandb_cfg(cfg)
    cfg = get_gnn_cfg(cfg)
    cfg = get_optim_cfg(cfg)
    cfg = get_dataset_cfg(cfg)
    cfg = get_model_cfg(cfg)
    cfg = get_transformer_cfg(cfg)

    return cfg

def load_config(cfg_path=None):
    """Loads and optionally overrides config values. 
    """
    cfg = get_cfg_defaults()

    if cfg_path:
        cfg.merge_from_file(cfg_path)
    
    cfg.freeze()
    return cfg
