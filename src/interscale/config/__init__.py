from pathlib import Path

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


def _normalise_cfg_paths(cfg_path):
    """Coerce the ``cfg_path`` argument into a list of existing ``Path`` objects.

    Accepts a single str/Path (the historical signature) or an iterable of them,
    so callers that layer several files can pass a list.
    """
    if cfg_path is None:
        return []
    if isinstance(cfg_path, str | Path):
        paths = [Path(cfg_path)]
    else:
        paths = [Path(p) for p in cfg_path]

    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("config file(s) not found: " + ", ".join(missing))
    return paths


def _peek_component_names(cfg_paths):
    """Find the local/global component names declared across ``cfg_paths``.

    The component *parameter* schemas (``model.local_component.parameters.*``) only exist
    once ``get_local_component_cfg`` / ``get_global_component_cfg`` have added them, so the
    names have to be read before any file is merged. Files are scanned in merge order and
    the last file naming a component wins, matching what the merge itself would produce.

    Uses ``getattr(..., "name", None)`` rather than attribute access: a file may set
    ``model.global_component.parameters.max_seq_len`` without naming the component (a
    dataset-level file layered on top of a base file that does the naming), and plain
    attribute access raises ``AttributeError`` on the absent ``name`` key.
    """
    local_component_name = None
    global_component_name = None

    for path in cfg_paths:
        with path.open() as f:
            temp_cfg = CN.load_cfg(f)

        model = getattr(temp_cfg, "model", None)
        if model is None:
            continue

        local_component = getattr(model, "local_component", None)
        if local_component is not None and getattr(local_component, "name", None):
            local_component_name = local_component.name

        global_component = getattr(model, "global_component", None)
        if global_component is not None and getattr(global_component, "name", None):
            global_component_name = global_component.name

    return local_component_name, global_component_name


def _coerce_override_values(cfg, overrides):
    """Return ``overrides`` with ints promoted to float where the cfg default is a float.

    ``merge_from_list`` rejects an int for a float key outright (yacs only tolerates a type
    mismatch when one side is ``None``). YAML writes ``0`` as an int, so an override of
    ``dataset.pct_mask_nodes: 0`` against the ``0.2`` default would raise even though the
    value is perfectly valid. Promote it instead of making callers write ``0.0``.

    ``overrides`` is the flat ``[key, value, key, value, ...]`` form ``merge_from_list`` takes.
    """
    coerced = list(overrides)

    for i in range(0, len(coerced) - 1, 2):
        key, value = coerced[i], coerced[i + 1]
        if not isinstance(value, int) or isinstance(value, bool):
            continue

        # Walk the dotted key to the current value; a missing key is left alone so that
        # merge_from_list raises its own (clearer) error about the unknown key.
        node = cfg
        try:
            *parents, leaf = str(key).split(".")
            for part in parents:
                node = node[part]
            current = node[leaf]
        except (KeyError, TypeError):
            continue

        if isinstance(current, float):
            coerced[i + 1] = float(value)

    return coerced


def _validate_optim(cfg):
    """Reject configs whose training-length settings stop a run inside the LR warm-up.

    ``CosineWarmupScheduler`` ramps the learning rate linearly over ``optim.lr_warmup`` epochs
    (interval ``"epoch"``), so a run that early-stops before the ramp finishes has never trained
    at ``optim.lr`` and reports a near-initialisation model -- typically a collapsed constant
    predictor, which is easy to misread as a class-imbalance problem.

    Raises
    ------
    ValueError
        If ``optim.min_epochs`` does not exceed ``optim.lr_warmup`` while the warm-up scheduler
        is in use.
    """
    if cfg.optim.lr_scheduler != "CosineWarmupScheduler":
        return
    if cfg.optim.min_epochs <= cfg.optim.lr_warmup:
        raise ValueError(
            f"optim.min_epochs ({cfg.optim.min_epochs}) must be greater than optim.lr_warmup "
            f"({cfg.optim.lr_warmup}): with CosineWarmupScheduler the LR is still ramping until "
            f"epoch {cfg.optim.lr_warmup}, so EarlyStopping (patience={cfg.optim.patience}) can "
            "end the run before the model has trained at optim.lr. Raise optim.min_epochs (2x "
            "lr_warmup is the convention here) or lower optim.lr_warmup."
        )


def load_config(cfg_path=None, overrides=None):
    """Loads and optionally overrides config values.

    Parameters
    ----------
    cfg_path : str or pathlib.Path or list, optional
        Path to the config file to load, or a list of paths merged left to right so
        later files override earlier ones. If None, only default values are used.
    overrides : list, optional
        Flat ``[key, value, key, value, ...]`` overrides applied after every file, in the
        form ``merge_from_list`` expects (e.g. ``["dataset.prediction_obs", "condition"]``).
        Keys are dotted paths and must already exist in the config, so a typo raises
        rather than silently doing nothing.

    Returns
    -------
    CN
        Configuration object with all settings loaded.
    """
    # First get all default configs including local component defaults
    cfg = get_cfg_defaults()

    cfg_paths = _normalise_cfg_paths(cfg_path)

    # Documented as defaults-only, but the code below dereferences cfg_path
    # unconditionally, so None used to raise AttributeError too.
    if not cfg_paths and not overrides:
        _validate_optim(cfg)
        cfg.freeze()
        return cfg

    local_component_name, global_component_name = _peek_component_names(cfg_paths)
    if local_component_name:
        # Ensure local component configs are loaded before merging
        cfg = get_local_component_cfg(cfg, local_component_name)
    if global_component_name:
        # Ensure global component configs are loaded before merging
        cfg = get_global_component_cfg(cfg, global_component_name)

    for path in cfg_paths:
        cfg.merge_from_file(str(path))

    if overrides:
        cfg.merge_from_list(_coerce_override_values(cfg, overrides))

    _validate_optim(cfg)
    cfg.freeze()
    return cfg
