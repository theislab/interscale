"""Build a wandb sweep config and apply a sampled trial onto a yacs config.

Both halves used to live inline in ``main_sweep.py`` -- one in ``main_sweep()`` between an
``h5ad`` load and a training call, the other in the ``__main__`` block -- which made the
question "does this sweep actually vary the parameters it declares?" impossible to answer
without launching a real run on a GPU. They are pure config transforms, so they live here
and are covered by ``tests/test_sweep_config.py``.

Application is **generic over the dotted key**: whatever ``parameters`` the sweep yaml
declares is written to that exact path in the config, and a path that does not exist raises.
The previous hand-maintained list of assignments produced two silent-no-op bugs that the
config comments still record -- component keys written without the leading ``model.`` (every
one raised ``KeyError``) and a ``dropout`` assignment against a schema whose key is
``dropout_global``, which ``set_new_allowed(True)`` turned into a dead key that nothing read.
Both classes are structurally impossible here: there is no allow-new-keys, and a key nothing
can apply is an error rather than a shrug.
"""

from yacs.config import CfgNode as CN

# Spelled "hyperparmeter" because that is what main_sweep.py has always matched on and what
# existing sweep invocations pass; renaming it would silently break saved commands.
SWEEP_GOALS = ("robustness", "segmentation", "hyperparmeter", "loss")

# Metric each prediction task ranks trials by. val_f1_macro is what EarlyStopping and
# ModelCheckpoint monitor for classification, and trainer.validate() runs after the best
# checkpoint is restored, so the last logged value is the best epoch's score.
TASK_METRICS = {
    "classification": {"name": "val_f1_macro", "goal": "maximize"},
    "regression": {"name": "val_r2", "goal": "maximize"},
}

LOCAL_COMPONENT_PREFIX = "model.local_component."
GLOBAL_COMPONENT_PREFIX = "model.global_component."


def unused_component_prefixes(model_type):
    """Return the dotted key prefixes ``model_type`` has no component for.

    Two independent checks, not if/elif: ``CombinedModel`` has *both* components, so an
    ``elif`` arm mentioning it is unreachable and its transformer would never be swept.
    """
    prefixes = []
    if model_type not in ("LocalModel", "CombinedModel"):
        prefixes.append(LOCAL_COMPONENT_PREFIX)
    if model_type not in ("GlobalModel", "CombinedModel"):
        prefixes.append(GLOBAL_COMPONENT_PREFIX)
    return prefixes


def build_sweep_config(yaml_config, prediction_task=None, model_type=None):
    """Turn a parsed sweep yaml into the dict ``wandb.sweep`` takes.

    Parameters
    ----------
    yaml_config : dict
        The parsed sweep yaml, containing a top-level ``sweep_config`` key.
    prediction_task : str, optional
        ``"classification"`` or ``"regression"``. When given, overrides the yaml's ``metric``
        block with the metric that task actually logs. When omitted the yaml must supply its
        own ``metric``, since wandb has nothing to rank trials by otherwise (and
        ``method: bayes`` cannot run at all).
    model_type : str, optional
        When given, parameters targeting a component this model type does not have are
        dropped, so wandb does not spend trials varying values nothing consumes.

    Returns
    -------
    tuple[dict, list[str]]
        The sweep config, and the sorted list of parameter names it varies.
    """
    if "sweep_config" not in yaml_config:
        raise ValueError("sweep yaml declares no top-level `sweep_config:` block.")

    # Copied rather than mutated in place so callers can reuse the parsed yaml.
    sweep_config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in yaml_config["sweep_config"].items()}

    if prediction_task is not None:
        if prediction_task not in TASK_METRICS:
            raise ValueError(f"unknown prediction_task '{prediction_task}'. Must be one of: {', '.join(TASK_METRICS)}")
        sweep_config["metric"] = dict(TASK_METRICS[prediction_task])

    if "metric" not in sweep_config:
        raise ValueError(
            "Sweep config declares no `metric`, and no prediction_task was given to supply one. "
            "wandb would have nothing to rank trials by (and method: bayes cannot run)."
        )

    if "parameters" not in sweep_config or not sweep_config["parameters"]:
        raise ValueError("sweep config declares no `parameters` to vary.")

    if model_type is not None:
        drop_prefixes = unused_component_prefixes(model_type)
        dropped = [k for k in sweep_config["parameters"] if any(k.startswith(p) for p in drop_prefixes)]
        for key in dropped:
            print(f"dropping sweep parameter not used by {model_type}: {key}")
            del sweep_config["parameters"][key]
        if not sweep_config["parameters"]:
            raise ValueError(
                f"every sweep parameter was dropped as unused by {model_type}; nothing left to vary."
            )

    return sweep_config, sorted(sweep_config["parameters"])


def _set_dotted(cfg, key, value):
    """Assign ``value`` at the dotted ``key`` in ``cfg``, requiring the path to exist."""
    parts = str(key).split(".")
    node = cfg

    for depth, part in enumerate(parts[:-1]):
        if not isinstance(node, CN) or part not in node:
            raise KeyError(
                f"sweep parameter '{key}' does not exist in the config: "
                f"no '{'.'.join(parts[: depth + 1])}'. It would vary between trials with no effect."
            )
        node = node[part]

    leaf = parts[-1]
    if not isinstance(node, CN) or leaf not in node:
        raise KeyError(
            f"sweep parameter '{key}' does not exist in the config. "
            f"It would vary between trials with no effect. "
            f"Available keys under '{'.'.join(parts[:-1])}': {', '.join(sorted(node)) if isinstance(node, CN) else '<not a config node>'}"
        )

    # yacs rejects an int for a float key, and a sweep declaring `values: [0, 0.1, 0.3]`
    # samples a genuine int for 0. Promote instead of failing mid-sweep.
    current = node[leaf]
    if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)

    node[leaf] = value


def apply_sweep_config(cfg, sweep_goal, sweep_config, model_type=None, sweep_params=None):
    """Write a sampled sweep trial onto ``cfg``.

    Parameters
    ----------
    cfg : CN
        The base config. Defrosted, written to, and re-frozen; mutated in place and returned.
    sweep_goal : str
        One of :data:`SWEEP_GOALS`. Validated so a misspelling fails loudly instead of
        skipping every branch and training the unmodified base config on every trial.
    sweep_config : Mapping
        The sampled trial -- ``wandb.config``, or any mapping of dotted key to value.
    model_type : str, optional
        When given, parameters targeting a component this model type lacks are skipped.
    sweep_params : list, optional
        The parameter names the sweep declares, i.e. the keys of the sweep config's
        ``parameters`` block. This is the authoritative list of what to apply, and any
        declared name missing from ``sweep_config`` raises. When omitted it is inferred as
        the dotted keys of ``sweep_config``.

    Returns
    -------
    tuple[CN, list[str]]
        The frozen config, and the sorted list of keys actually applied.
    """
    if sweep_goal not in SWEEP_GOALS:
        raise ValueError(
            f"Unknown sweep_goal '{sweep_goal}'. Must be one of: {', '.join(SWEEP_GOALS)}. "
            "(Nothing would be swept otherwise -- every trial would train the base config.)"
        )

    # main_sweep.py calls wandb.init(config=cfg), so wandb.config carries the whole base
    # config alongside the sampled trial values. Only the sweep's declared parameters may be
    # written back: iterating every key would assign the top-level `dataset` / `model` /
    # `optim` sections over their CfgNodes with plain dicts.
    if sweep_params is None:
        keys = [k for k in sweep_config.keys() if "." in str(k)]
    else:
        keys = list(sweep_params)
        absent = sorted(k for k in keys if k not in sweep_config)
        if absent:
            raise KeyError(
                f"sweep declares parameters that the trial did not sample, so they cannot be "
                f"applied: {absent}"
            )

    # Deliberately NOT set_new_allowed(True): allowing new keys is what let a misspelled
    # parameter create a dead config entry that nothing ever read.
    was_frozen = cfg.is_frozen()
    cfg.defrost()

    skipped_prefixes = unused_component_prefixes(model_type) if model_type is not None else []

    applied = []
    skipped = []
    for key in keys:
        if any(str(key).startswith(p) for p in skipped_prefixes):
            skipped.append(key)
            continue
        _set_dotted(cfg, key, sweep_config[key])
        applied.append(key)

    if was_frozen:
        cfg.freeze()

    if skipped:
        print(f"skipped sweep parameters for components {model_type} does not have: {sorted(skipped)}")

    if sweep_params is not None:
        # A parameter wandb varies but nothing reads makes every trial's difference invisible,
        # which is indistinguishable from the sweep not working. Fail rather than warn.
        unapplied = sorted(set(sweep_params) - set(applied) - set(skipped))
        if unapplied:
            raise KeyError(
                f"sweep declares parameters that were not applied to the config, so they would "
                f"vary between trials with no effect: {unapplied}"
            )

    print(f"applied sweep parameters: {sorted(applied)}")
    return cfg, sorted(applied)
