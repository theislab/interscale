from yacs.config import CfgNode as CN

from .dataset_config import get_dataset_cfg
from .global_component_config import get_global_component_cfg
from .local_component_config import get_local_component_cfg
from .model_config import get_model_cfg
from .optim_config import get_optim_cfg
from .wandb_config import get_wandb_cfg


def get_cfg_defaults():
    """Loads the default settings from the .py files in the config folder."""
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

    with cfg_path.open() as f:
        # Create a temporary config to load the model type
        temp_cfg = CN.load_cfg(f)

        # If model type is specified, load the corresponding local component configs
        if hasattr(temp_cfg, "model") and hasattr(temp_cfg.model, "local_component"):
            if temp_cfg.model.local_component.name is not None:
                local_component_name = temp_cfg.model.local_component.name
                if local_component_name:
                    # Ensure local component configs are loaded before merging
                    cfg = get_local_component_cfg(cfg, local_component_name)

        # If model type is specified, load the corresponding global component configs
        if hasattr(temp_cfg, "model") and hasattr(temp_cfg.model, "global_component"):
            if temp_cfg.model.global_component.name is not None:
                global_component_name = temp_cfg.model.global_component.name
                if global_component_name:
                    # Ensure global component configs are loaded before merging
                    cfg = get_global_component_cfg(cfg, global_component_name)

        # Now merge the full config file
        cfg.merge_from_file(cfg_path)

    cfg.freeze()
    return cfg
