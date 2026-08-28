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

**Arms** (see :func:`load_arms`) add the one thing a flat dotted-key sweep cannot express: an
ablation axis whose levels each imply *several* config values that must move together. wandb
searches the cartesian product of its declared parameters, so declaring the coupled keys
separately would produce every invalid crossing of them.
"""

from yacs.config import CfgNode as CN

# Spelled "hyperparmeter" because that is what main_sweep.py has always matched on and what
# existing sweep invocations pass; renaming it would silently break saved commands.
SWEEP_GOALS = ("robustness", "segmentation", "hyperparmeter", "loss")

# The one sweep parameter whose values are arm NAMES rather than config values. Reserved: a
# config key called "arm" could never be a sweep parameter anyway, since every real one is
# dotted.
ARM_PARAM = "arm"

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


def build_sweep_config(yaml_config, prediction_task=None, model_type=None, metric=None):
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

    # An explicit metric wins over the task default. The task default answers "what does this
    # task log?", which is not always the same question as "what is this sweep searching for":
    # a regression sweep looking for the best *correlation* must rank on val_pearson_corr, and
    # ranking it on the task default val_r2 would select for calibration instead.
    if metric is not None:
        sweep_config["metric"] = dict(metric)

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

    if ARM_PARAM in sweep_config["parameters"] and "arms" not in yaml_config:
        raise ValueError(
            f"sweep declares the reserved parameter '{ARM_PARAM}' but the yaml has no top-level "
            f"`arms:` block to resolve its values against. Every trial would try to write the arm "
            f"name to a config key called '{ARM_PARAM}', which does not exist."
        )

    return sweep_config, sorted(sweep_config["parameters"])


def load_arms(yaml_config, sweep_config=None):
    """Read and validate a sweep yaml's optional top-level ``arms:`` block.

    An *arm* is one level of an ablation axis that implies several config values at once. The
    sliding-window ablation is the motivating case: a window size fixes the ``obs`` column the
    windows live in, the transformer's ``max_seq_len`` (attention pads to it, so it cannot be
    shared across arms without paying the largest arm's quadratic cost on every one) and
    ``dataset.name`` (it reaches the checkpoint filename, so without it every arm overwrites its
    predecessor). Those three must move together; wandb searches the cartesian product of what it
    is given, so declaring them as three parameters would ask for 6^3 = 216 trials of which 6 are
    meaningful.

    So only the arm *name* is declared to wandb, as the reserved ``arm`` parameter, and the values
    it stands for are looked up here. That the coupled values never round-trip through wandb is a
    second benefit: ``dataset.sample_key`` is a list, and yacs type-checks assignments, so a value
    wandb chose to return as a tuple or a scalar would fail mid-sweep.

    Parameters
    ----------
    yaml_config : dict
        The parsed sweep yaml. ``arms`` is a sibling of ``sweep_config``, not nested inside it,
        because everything under ``sweep_config`` is sent verbatim to ``wandb.sweep``.
    sweep_config : dict, optional
        The built sweep config, checked against the arms so a mismatch fails before wandb
        registers anything.

    Returns
    -------
    dict or None
        ``{arm_name: {dotted_key: value}}``, or None when the yaml declares no arms.
    """
    if "arms" not in yaml_config:
        return None

    arms = yaml_config["arms"]
    if not isinstance(arms, dict) or not arms:
        raise ValueError("sweep yaml's `arms:` block must be a non-empty mapping of arm name -> overrides.")

    for name, overrides in arms.items():
        if not isinstance(overrides, dict) or not overrides:
            raise ValueError(f"arm '{name}' declares no overrides; it would be identical to every other arm.")
        undotted = sorted(k for k in overrides if "." not in str(k))
        if undotted:
            raise ValueError(
                f"arm '{name}' declares non-dotted keys {undotted}. Arm overrides are dotted config "
                f"paths (e.g. dataset.sample_key), applied exactly as sweep parameters are."
            )

    # Every arm must set the SAME keys. An arm that omits one silently leaves it at the base
    # config's value while its siblings override it, so the ablation would compare arms that
    # differ in a way the yaml never states -- the same silent-no-op class this module exists to
    # make impossible.
    key_sets = {name: frozenset(overrides) for name, overrides in arms.items()}
    reference_name, reference_keys = next(iter(key_sets.items()))
    for name, keys in key_sets.items():
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(
                f"arm '{name}' does not declare the same keys as arm '{reference_name}': "
                f"missing {missing}, extra {extra}. Any key not set by every arm stays at the base "
                f"config value for the arms that omit it, so the arms would differ in an unstated way."
            )

    if sweep_config is not None:
        parameters = sweep_config.get("parameters", {})
        if ARM_PARAM not in parameters:
            raise ValueError(
                f"sweep yaml declares an `arms:` block but no '{ARM_PARAM}' parameter to select "
                f"between them, so no arm would ever be applied."
            )

        declared = parameters[ARM_PARAM]
        if not isinstance(declared, dict) or "values" not in declared:
            raise ValueError(
                f"the '{ARM_PARAM}' parameter must declare `values:` naming the arms to run "
                f"(got {declared!r})."
            )

        selected = list(declared["values"])
        unknown = sorted(set(selected) - set(arms))
        if unknown:
            raise ValueError(f"'{ARM_PARAM}' selects arms that the `arms:` block does not define: {unknown}")
        # Not an error -- running a subset of the defined arms is a legitimate way to re-run one
        # of them -- but silence here is how you discover a typo only after the sweep finishes.
        unused = sorted(set(arms) - set(selected))
        if unused:
            print(f"arms defined but not selected by '{ARM_PARAM}': {unused}")

        # An arm key that is ALSO a declared sweep parameter would be written twice per trial with
        # the order deciding which wins.
        collisions = sorted(reference_keys & set(parameters))
        if collisions:
            raise ValueError(
                f"keys are set both by the arms and as sweep parameters, so which value a trial "
                f"gets would depend on application order: {collisions}"
            )

    return arms


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


def apply_sweep_config(cfg, sweep_goal, sweep_config, model_type=None, sweep_params=None, arms=None):
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
    arms : dict, optional
        The yaml's ``arms:`` block, from :func:`load_arms`. When given, the trial's ``arm`` value
        is expanded into that arm's dotted overrides instead of being written to the config.

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

    # The arm name is not a config path, so it is removed from `keys` and replaced by the dotted
    # overrides it stands for. Done before the loop below so an arm override and a plain sweep
    # parameter go through exactly the same _set_dotted validation.
    arm_name = None
    arm_overrides = {}
    if arms is not None:
        if ARM_PARAM not in sweep_config:
            raise KeyError(
                f"arms were given but the trial carries no '{ARM_PARAM}' value, so no arm can be "
                f"applied and every trial would train the base config's windowing."
            )
        arm_name = sweep_config[ARM_PARAM]
        if arm_name not in arms:
            raise KeyError(f"trial selected arm '{arm_name}', which the `arms:` block does not define.")
        arm_overrides = dict(arms[arm_name])
        keys = [k for k in keys if k != ARM_PARAM]
    elif ARM_PARAM in (sweep_params or []):
        raise KeyError(
            f"the sweep declares the reserved parameter '{ARM_PARAM}' but no arms were passed to "
            f"apply_sweep_config, so it would be written to a config key that does not exist."
        )

    # Deliberately NOT set_new_allowed(True): allowing new keys is what let a misspelled
    # parameter create a dead config entry that nothing ever read.
    was_frozen = cfg.is_frozen()
    cfg.defrost()

    skipped_prefixes = unused_component_prefixes(model_type) if model_type is not None else []

    applied = []
    skipped = []
    # Arm overrides go first so a later plain parameter, if the two ever collided, would win --
    # but load_arms rejects that collision outright, so the order is only about determinism.
    for key, value in list(arm_overrides.items()) + [(k, sweep_config[k]) for k in keys]:
        if any(str(key).startswith(p) for p in skipped_prefixes):
            skipped.append(key)
            continue
        _set_dotted(cfg, key, value)
        applied.append(key)

    if was_frozen:
        cfg.freeze()

    if skipped:
        print(f"skipped sweep parameters for components {model_type} does not have: {sorted(skipped)}")

    if sweep_params is not None:
        # A parameter wandb varies but nothing reads makes every trial's difference invisible,
        # which is indistinguishable from the sweep not working. Fail rather than warn.
        # ARM_PARAM is excluded because it is applied as its expansion, not as itself.
        unapplied = sorted(set(sweep_params) - set(applied) - set(skipped) - {ARM_PARAM})
        if unapplied:
            raise KeyError(
                f"sweep declares parameters that were not applied to the config, so they would "
                f"vary between trials with no effect: {unapplied}"
            )

    if arm_name is not None:
        print(f"applied arm '{arm_name}': {sorted(arm_overrides)}")
    print(f"applied sweep parameters: {sorted(applied)}")
    return cfg, sorted(applied)
