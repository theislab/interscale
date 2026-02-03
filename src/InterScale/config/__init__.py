from os.path import dirname, basename, isfile, join
from yacs.config import CfgNode as CN
import glob
from .wandb_config import get_wandb_cfg
from .local_component_config import get_local_component_cfg
from .dataset_config import get_dataset_cfg
from .model_config import get_model_cfg
from .optim_config import get_optim_cfg
from .global_component_config import get_global_component_cfg

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
    cfg = get_model_cfg(cfg)
    cfg = get_optim_cfg(cfg)
    cfg = get_dataset_cfg(cfg)
    
    return cfg

def load_config(cfg_path=None):
    """Loads and optionally overrides config values. 
    
    Parameters
    ----------
    cfg_path : str, optional
        Path to the config file to load. If None, only default values are used.
        
    Returns
    -------
    CN
        Configuration object with all settings loaded.
    """
    # First get all default configs including local component defaults
    cfg = get_cfg_defaults()
    
    if cfg_path:
        # Create a temporary config to load the model type
        temp_cfg = CN.load_cfg(open(cfg_path))
        
        # If model type is specified, load the corresponding local component configs
        if hasattr(temp_cfg, 'model') and hasattr(temp_cfg.model, 'local_component'):
            if temp_cfg.model.local_component.name is not None:
                local_component_name = temp_cfg.model.local_component.name
                if local_component_name:
                    # Ensure local component configs are loaded before merging
                    cfg = get_local_component_cfg(cfg, local_component_name)
                    
        # If model type is specified, load the corresponding global component configs
        if hasattr(temp_cfg, 'model') and hasattr(temp_cfg.model, 'global_component'):
            if temp_cfg.model.global_component.name is not None:
                global_component_name = temp_cfg.model.global_component.name
                if global_component_name:
                    # Ensure global component configs are loaded before merging
                    cfg = get_global_component_cfg(cfg, global_component_name)
        
        # Now merge the full config file
        cfg.merge_from_file(cfg_path)
    
    cfg.freeze()
    return cfg


def _wandb_config_to_nested_dict(config):
    """Convert WandB run config to nested dict. Handles both dotted keys and nested dicts."""
    if not hasattr(config, "items"):
        config = dict(config)
    keys = list(config.keys())
    if any("." in str(k) for k in keys):
        result = {}
        for key, value in config.items():
            if "." not in str(key):
                result[key] = value
                continue
            parts = str(key).split(".")
            d = result
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        return result
    result = {}
    for k, v in config.items():
        if isinstance(v, dict) and v:
            result[k] = _wandb_config_to_nested_dict(v)
        else:
            result[k] = v
    return result


def config_from_wandb_run(run, save_yaml_path=None):
    """Build a full InterScale CfgNode from a WandB run config and optionally save to YAML.

    Uses all variables from InterScale/config: wandb, model, optim, dataset,
    plus local_component and global_component parameters based on run config.

    Parameters
    ----------
    run : wandb.Api.run or object with .config attribute
        A WandB run (e.g. api.run("entity/project/run_id")).
    save_yaml_path : str, optional
        If set, write the merged config to this YAML file.

    Returns
    -------
    CN
        Configuration object with all settings from the run (and defaults where not set).
    """
    raw = dict(run.config) if hasattr(run.config, "items") else run.config
    nested = _wandb_config_to_nested_dict(raw)
    run_cfg = CN(nested)

    cfg = get_cfg_defaults()
    if hasattr(run_cfg, "model") and hasattr(run_cfg.model, "local_component"):
        if getattr(run_cfg.model.local_component, "name", None):
            cfg = get_local_component_cfg(cfg, run_cfg.model.local_component.name)
    if hasattr(run_cfg, "model") and hasattr(run_cfg.model, "global_component"):
        if getattr(run_cfg.model.global_component, "name", None):
            cfg = get_global_component_cfg(cfg, run_cfg.model.global_component.name)

    cfg.set_new_allowed(True)
    cfg.defrost()
    cfg.merge_from_other_cfg(run_cfg)
    cfg.freeze()

    if save_yaml_path:
        with open(save_yaml_path, "w") as f:
            f.write(cfg.dump())
    return cfg


def load_config_from_yaml(cfg_path):
    """Load config from a YAML file with all InterScale config variables applied.

    Uses defaults from InterScale/config (wandb, model, optim, dataset) and
    local/global component configs based on model type in the YAML, then merges
    the file. Use this (or load_config) when training with a config exported from
    a WandB sweep.

    Parameters
    ----------
    cfg_path : str
        Path to the YAML config file.

    Returns
    -------
    CN
        Configuration object.
    """
    return load_config(cfg_path)
