"""Tests that a sweep actually varies the hyperparameters it declares.

The failure these guard against is silent: wandb samples a value, nothing in the code reads the
key it was sampled for, every trial trains an identical model, and the sweep report looks like a
legitimate "no effect" result. Two real instances are recorded in the config comments -- component
keys written without the leading ``model.``, and a ``dropout`` assignment against a schema key
actually named ``dropout_global``.

The central test is :func:`test_every_declared_sweep_parameter_changes_the_config`, which walks the
real ``config_files/sweeps/hyperparameters.yaml`` and proves each declared parameter reaches the
config for a real registered dataset/task pair.
"""

from pathlib import Path

import pytest
import yaml
from yacs.config import CfgNode as CN

from interscale.config import load_config
from interscale.config.registry import resolve_config
from interscale.config.sweep import (
    ARM_PARAM,
    SWEEP_GOALS,
    apply_sweep_config,
    build_sweep_config,
    load_arms,
    unused_component_prefixes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config_files"
REGISTRY = CONFIG_DIR / "registry.yaml"
SWEEP_YAML = CONFIG_DIR / "sweeps" / "hyperparameters.yaml"

# A pair that exercises both components, so no parameter is dropped as unused.
REFERENCE_DATASET = "melton25"
REFERENCE_TASK = "graph_clas"


def get_dotted(cfg, key):
    """Read a dotted path out of a config, raising KeyError if any segment is missing."""
    node = cfg
    for part in key.split("."):
        node = node[part]
    return node


@pytest.fixture
def sweep_yaml():
    with SWEEP_YAML.open() as f:
        return yaml.safe_load(f)


@pytest.fixture
def base_cfg():
    return resolve_config(REFERENCE_DATASET, REFERENCE_TASK, registry_path=REGISTRY)


def pick_differing_value(declared_values, current):
    """Pick a declared sweep value that differs from the config's current value.

    Using the sweep's own declared values keeps the test honest: it proves the real candidate
    values land in the config, not just that some arbitrary sentinel can be written.
    """
    for value in declared_values:
        if value != current:
            return value
    return None


def test_sweep_yaml_exists():
    assert SWEEP_YAML.is_file(), f"expected the sweep config at {SWEEP_YAML}"


def test_every_declared_sweep_parameter_changes_the_config(sweep_yaml, base_cfg):
    """Every parameter the sweep declares must actually reach the config.

    This is the test that would have caught both historical silent-no-op bugs.
    """
    sweep_config, sweep_params = build_sweep_config(
        sweep_yaml, prediction_task="classification", model_type="CombinedModel"
    )
    assert sweep_params, "the sweep declares no parameters"

    unverifiable = []

    for key in sweep_params:
        declared = sweep_config["parameters"][key]
        values = declared.get("values", [declared.get("value")])
        current = get_dotted(base_cfg, key)

        target = pick_differing_value(values, current)
        if target is None:
            # Every declared value already equals the base config's value, so applying it
            # cannot be observed. Reported rather than silently passed.
            unverifiable.append(key)
            continue

        cfg = base_cfg.clone()
        cfg, applied = apply_sweep_config(
            cfg, "hyperparmeter", {key: target}, model_type="CombinedModel", sweep_params=[key]
        )

        assert key in applied, f"{key} was declared by the sweep but not applied"
        assert get_dotted(cfg, key) == target, (
            f"sweep parameter {key} did not change the config: "
            f"expected {target!r}, config still holds {get_dotted(cfg, key)!r}"
        )

    assert not unverifiable, (
        "these sweep parameters declare only values identical to the base config, so a trial "
        f"varying them is indistinguishable from no trial at all: {unverifiable}"
    )


def test_all_sweep_parameters_applied_together(sweep_yaml, base_cfg):
    """Applying a full sampled trial writes every parameter, not just the first."""
    sweep_config, sweep_params = build_sweep_config(
        sweep_yaml, prediction_task="classification", model_type="CombinedModel"
    )

    trial = {}
    for key in sweep_params:
        declared = sweep_config["parameters"][key]
        values = declared.get("values", [declared.get("value")])
        target = pick_differing_value(values, get_dotted(base_cfg, key))
        trial[key] = target if target is not None else values[0]

    cfg, applied = apply_sweep_config(
        base_cfg.clone(), "hyperparmeter", trial, model_type="CombinedModel", sweep_params=sweep_params
    )

    assert sorted(applied) == sorted(sweep_params)
    for key, expected in trial.items():
        assert get_dotted(cfg, key) == expected, f"{key} not applied"


def test_sweep_parameters_are_all_known_to_the_config(sweep_yaml, base_cfg):
    """No declared parameter names a config path that does not exist."""
    _, sweep_params = build_sweep_config(
        sweep_yaml, prediction_task="classification", model_type="CombinedModel"
    )
    for key in sweep_params:
        # Raises KeyError with the offending key if the path is absent.
        get_dotted(base_cfg, key)


def test_trial_does_not_leak_into_the_base_config(base_cfg):
    """Applying a trial must not mutate a shared config.

    wandb.agent runs many trials in one process. If they shared a config object, trial N would
    inherit every earlier trial's values and the sweep results would be meaningless.
    """
    original = base_cfg.optim.lr
    other = original + 0.5

    applied_cfg, _ = apply_sweep_config(
        base_cfg.clone(), "hyperparmeter", {"optim.lr": other}, sweep_params=["optim.lr"]
    )

    assert applied_cfg.optim.lr == other
    assert base_cfg.optim.lr == original, "the trial leaked into the shared base config"


def test_base_config_stays_frozen_after_apply(base_cfg):
    """A frozen config comes back frozen, so later accidental writes still raise."""
    cfg = base_cfg.clone()
    assert cfg.is_frozen()
    cfg, _ = apply_sweep_config(cfg, "hyperparmeter", {"optim.lr": 0.123}, sweep_params=["optim.lr"])
    assert cfg.is_frozen()


# --- the two historical silent-no-op bugs, as regression tests ----------------------------


def test_component_key_without_model_prefix_raises(base_cfg):
    """`local_component.parameters.num_layers` (no leading `model.`) must not pass silently."""
    with pytest.raises(KeyError, match="local_component.parameters.num_layers"):
        apply_sweep_config(
            base_cfg.clone(),
            "hyperparmeter",
            {"local_component.parameters.num_layers": 3},
            sweep_params=["local_component.parameters.num_layers"],
        )


def test_misspelled_dropout_key_raises(base_cfg):
    """The transformer key is `dropout_global`; plain `dropout` used to become a dead key."""
    with pytest.raises(KeyError, match="dropout"):
        apply_sweep_config(
            base_cfg.clone(),
            "hyperparmeter",
            {"model.global_component.parameters.dropout": 0.5},
            sweep_params=["model.global_component.parameters.dropout"],
        )

    # ...while the correctly spelled key does land.
    cfg, _ = apply_sweep_config(
        base_cfg.clone(),
        "hyperparmeter",
        {"model.global_component.parameters.dropout_global": 0.5},
        sweep_params=["model.global_component.parameters.dropout_global"],
    )
    assert cfg.model.global_component.parameters.dropout_global == 0.5


def test_unknown_key_is_not_silently_created(base_cfg):
    """A typo must raise rather than create a config entry nothing reads."""
    with pytest.raises(KeyError):
        apply_sweep_config(
            base_cfg.clone(),
            "hyperparmeter",
            {"optim.learning_rate": 0.1},
            sweep_params=["optim.learning_rate"],
        )


def test_declared_but_unsampled_parameter_raises(base_cfg):
    """A parameter the sweep declares but the trial never sampled is an error, not a warning."""
    with pytest.raises(KeyError, match="did not sample"):
        apply_sweep_config(
            base_cfg.clone(),
            "hyperparmeter",
            {"optim.lr": 0.01},
            sweep_params=["optim.lr", "optim.wd"],
        )


def test_whole_config_sections_are_not_clobbered(base_cfg):
    """wandb.config carries the whole base config; only declared parameters may be written.

    main_sweep.py calls wandb.init(config=cfg), so wandb.config contains top-level `dataset` /
    `model` / `optim` entries next to the sampled dotted keys. Writing those back would replace
    CfgNodes with plain dicts.
    """
    wandb_like = {
        "dataset": {"batch_size": 999},  # the section dump, not a sweep parameter
        "model": {"n_embed": 999},
        "optim.lr": 0.007,  # the actual sampled parameter
    }

    cfg, applied = apply_sweep_config(
        base_cfg.clone(), "hyperparmeter", wandb_like, sweep_params=["optim.lr"]
    )

    assert applied == ["optim.lr"]
    assert cfg.optim.lr == 0.007
    assert isinstance(cfg.dataset, CN), "the dataset section was replaced by a plain dict"
    assert cfg.dataset.batch_size == base_cfg.dataset.batch_size
    assert cfg.model.n_embed == base_cfg.model.n_embed


def test_inferred_params_ignore_non_dotted_keys(base_cfg):
    """With no explicit sweep_params, only dotted keys are treated as sweep parameters."""
    cfg, applied = apply_sweep_config(
        base_cfg.clone(),
        "hyperparmeter",
        {"dataset": {"batch_size": 999}, "optim.lr": 0.003},
    )
    assert applied == ["optim.lr"]
    assert isinstance(cfg.dataset, CN)


# --- goal and model_type handling -----------------------------------------------------------


def test_unknown_sweep_goal_raises(base_cfg):
    """A misspelled goal must fail loudly, not train the base config on every trial."""
    with pytest.raises(ValueError, match="Unknown sweep_goal"):
        apply_sweep_config(base_cfg.clone(), "hyperparameter", {"optim.lr": 0.1})  # sic: correct spelling


@pytest.mark.parametrize("goal", SWEEP_GOALS)
def test_every_declared_goal_is_accepted(goal, base_cfg):
    cfg, applied = apply_sweep_config(base_cfg.clone(), goal, {"optim.lr": 0.02}, sweep_params=["optim.lr"])
    assert applied == ["optim.lr"]
    assert cfg.optim.lr == 0.02


def test_robustness_goal_parameters_apply(base_cfg):
    """The robustness sweep's three keys all exist and all land."""
    trial = {
        "dataset.mask_percentage": 0.42,
        "dataset.spatial_neigbors_kwargs.radius": 77,
        "optim.seed": 7,
    }
    cfg, applied = apply_sweep_config(
        base_cfg.clone(), "robustness", trial, sweep_params=sorted(trial)
    )
    assert sorted(applied) == sorted(trial)
    assert cfg.dataset.mask_percentage == 0.42
    assert cfg.dataset.spatial_neigbors_kwargs.radius == 77
    assert cfg.optim.seed == 7


def test_segmentation_goal_parameters_apply(base_cfg):
    cfg, _ = apply_sweep_config(
        base_cfg.clone(),
        "segmentation",
        {"dataset.segmentation_robustness": [0.1, 0.2]},
        sweep_params=["dataset.segmentation_robustness"],
    )
    assert cfg.dataset.segmentation_robustness == [0.1, 0.2]


def test_int_is_promoted_for_a_float_key(base_cfg):
    """`values: [0, 0.1, 0.3]` samples a real int for 0; yacs rejects int for a float key."""
    cfg, _ = apply_sweep_config(
        base_cfg.clone(),
        "hyperparmeter",
        {"model.local_component.parameters.dropout_local": 0},
        sweep_params=["model.local_component.parameters.dropout_local"],
    )
    value = cfg.model.local_component.parameters.dropout_local
    assert value == 0
    assert isinstance(value, float), "an int would break a later merge against this float key"


@pytest.mark.parametrize(
    ("model_type", "expected_prefixes"),
    [
        ("CombinedModel", []),
        ("LocalModel", ["model.global_component."]),
        ("GlobalModel", ["model.local_component."]),
    ],
)
def test_unused_component_prefixes(model_type, expected_prefixes):
    """CombinedModel must keep BOTH components -- an if/elif chain used to drop its transformer."""
    assert unused_component_prefixes(model_type) == expected_prefixes


@pytest.mark.parametrize(
    ("model_type", "dropped_prefix", "kept_prefix"),
    [
        ("LocalModel", "model.global_component.", "model.local_component."),
        ("GlobalModel", "model.local_component.", "model.global_component."),
    ],
)
def test_single_component_model_drops_the_other_components_parameters(
    sweep_yaml, model_type, dropped_prefix, kept_prefix
):
    _, sweep_params = build_sweep_config(
        sweep_yaml, prediction_task="classification", model_type=model_type
    )
    assert not any(k.startswith(dropped_prefix) for k in sweep_params)
    assert any(k.startswith(kept_prefix) for k in sweep_params), (
        f"{model_type} should still sweep its own component's parameters"
    )


def test_combined_model_sweeps_both_components(sweep_yaml):
    _, sweep_params = build_sweep_config(
        sweep_yaml, prediction_task="classification", model_type="CombinedModel"
    )
    assert any(k.startswith("model.local_component.") for k in sweep_params)
    assert any(k.startswith("model.global_component.") for k in sweep_params)


def test_component_parameters_are_skipped_not_applied_for_wrong_model_type(base_cfg):
    """A global parameter reaching a LocalModel trial is skipped rather than written."""
    cfg, applied = apply_sweep_config(
        base_cfg.clone(),
        "hyperparmeter",
        {"model.global_component.parameters.n_heads": 8, "optim.lr": 0.02},
        model_type="LocalModel",
        sweep_params=["model.global_component.parameters.n_heads", "optim.lr"],
    )
    assert applied == ["optim.lr"]
    assert cfg.model.global_component.parameters.n_heads == base_cfg.model.global_component.parameters.n_heads


# --- metric selection ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prediction_task", "expected_metric"),
    [("classification", "val_f1_macro"), ("regression", "val_r2")],
)
def test_metric_follows_prediction_task(sweep_yaml, prediction_task, expected_metric):
    """A regression sweep must not be ranked by an f1 metric it never logs."""
    sweep_config, _ = build_sweep_config(sweep_yaml, prediction_task=prediction_task)
    assert sweep_config["metric"]["name"] == expected_metric
    assert sweep_config["metric"]["goal"] == "maximize"


def test_yaml_metric_is_kept_when_no_prediction_task_given(sweep_yaml):
    sweep_config, _ = build_sweep_config(sweep_yaml)
    assert sweep_config["metric"]["name"] == sweep_yaml["sweep_config"]["metric"]["name"]


def test_missing_metric_without_prediction_task_raises():
    with pytest.raises(ValueError, match="no `metric`"):
        build_sweep_config({"sweep_config": {"method": "random", "parameters": {"optim.lr": {"values": [1]}}}})


def test_unknown_prediction_task_raises(sweep_yaml):
    with pytest.raises(ValueError, match="unknown prediction_task"):
        build_sweep_config(sweep_yaml, prediction_task="clustering")


def test_missing_sweep_config_block_raises():
    with pytest.raises(ValueError, match="sweep_config"):
        build_sweep_config({"parameters": {}})


def test_empty_parameters_raises():
    with pytest.raises(ValueError, match="no `parameters`"):
        build_sweep_config({"sweep_config": {"metric": {"name": "val_loss", "goal": "minimize"}, "parameters": {}}})


def test_build_sweep_config_does_not_mutate_the_parsed_yaml(sweep_yaml):
    """Dropping component parameters must not corrupt a reusable parsed yaml."""
    before = sorted(sweep_yaml["sweep_config"]["parameters"])
    build_sweep_config(sweep_yaml, prediction_task="classification", model_type="LocalModel")
    assert sorted(sweep_yaml["sweep_config"]["parameters"]) == before


def test_sweep_parameters_apply_to_a_bare_default_config():
    """The sweep's keys exist in the plain defaults too, not only in a dataset config."""
    cfg = load_config(CONFIG_DIR / "base.yaml")
    with SWEEP_YAML.open() as f:
        sweep_config, sweep_params = build_sweep_config(
            yaml.safe_load(f), prediction_task="classification", model_type="CombinedModel"
        )
    trial = {k: sweep_config["parameters"][k].get("values", [None])[0] for k in sweep_params}
    _, applied = apply_sweep_config(cfg, "hyperparmeter", trial, model_type="CombinedModel", sweep_params=sweep_params)
    assert sorted(applied) == sorted(sweep_params)


# --------------------------------------------------------------------------------------------
# Arms: one sweep parameter standing for several coupled config keys.
#
# The failure these guard against is the one the flat dotted-key design cannot express at all.
# wandb searches the cartesian product of its parameters, so a window-size ablation whose three
# implied keys were declared separately would enumerate 6^3 = 216 trials, 210 of them invalid
# crossings (e.g. the 400-cell window column with the 3436 max_seq_len). The arms block keeps the
# three coupled, and these tests prove the coupling survives the round trip.
# --------------------------------------------------------------------------------------------

SLIDING_WINDOW_YAML = CONFIG_DIR / "sweeps" / "sliding_window_melton25.yaml"

ARM_DATASET = "melton25_sw"
ARM_TASK = "node_reg"


@pytest.fixture
def arm_yaml():
    with SLIDING_WINDOW_YAML.open() as f:
        return yaml.safe_load(f)


@pytest.fixture
def arm_cfg():
    return resolve_config(ARM_DATASET, ARM_TASK, registry_path=REGISTRY)


def make_arm_trial(sweep_config, sweep_params, arm_name):
    """Build the trial dict wandb would hand back for one arm, first value for everything else."""
    trial = {k: sweep_config["parameters"][k]["values"][0] for k in sweep_params}
    trial[ARM_PARAM] = arm_name
    return trial


def test_sliding_window_sweep_yaml_exists():
    assert SLIDING_WINDOW_YAML.is_file(), f"missing sweep config: {SLIDING_WINDOW_YAML}"


def test_load_arms_returns_none_without_an_arms_block(sweep_yaml):
    """The flat sweeps must be entirely unaffected by the arms machinery."""
    assert load_arms(sweep_yaml) is None


def test_every_arm_applies_all_of_its_coupled_keys(arm_yaml, arm_cfg):
    """Each arm's full set of dotted overrides reaches the config, for every arm."""
    sweep_config, sweep_params = build_sweep_config(
        arm_yaml, prediction_task="regression", model_type="CombinedModel"
    )
    arms = load_arms(arm_yaml, sweep_config)

    for arm_name, overrides in arms.items():
        cfg = arm_cfg.clone()
        trial = make_arm_trial(sweep_config, sweep_params, arm_name)
        cfg, applied = apply_sweep_config(
            cfg, "robustness", trial, model_type="CombinedModel", sweep_params=sweep_params, arms=arms
        )
        for key, expected in overrides.items():
            assert get_dotted(cfg, key) == expected, f"arm {arm_name}: {key} did not reach the config"
            assert key in applied


def test_arm_name_itself_is_never_written_to_the_config(arm_yaml, arm_cfg):
    """`arm` is a selector, not a config path; writing it would need a config key called 'arm'."""
    sweep_config, sweep_params = build_sweep_config(
        arm_yaml, prediction_task="regression", model_type="CombinedModel"
    )
    arms = load_arms(arm_yaml, sweep_config)
    trial = make_arm_trial(sweep_config, sweep_params, "w400")
    cfg, applied = apply_sweep_config(
        arm_cfg, "robustness", trial, model_type="CombinedModel", sweep_params=sweep_params, arms=arms
    )
    assert ARM_PARAM not in applied
    assert ARM_PARAM not in cfg
    assert ARM_PARAM not in cfg.dataset


def test_arms_give_distinct_checkpoint_prefixes(arm_yaml, arm_cfg):
    """dataset.name must differ per arm, or every arm overwrites the previous arm's checkpoint.

    This is not hypothetical: get_model_filename_prefix keys on dataset.name, prediction task,
    level and seed, none of which the window size touches on its own, and
    trainer.save_checkpoint() overwrites unconditionally.
    """
    from interscale.tl.utils import get_model_filename_prefix

    sweep_config, sweep_params = build_sweep_config(
        arm_yaml, prediction_task="regression", model_type="CombinedModel"
    )
    arms = load_arms(arm_yaml, sweep_config)

    prefixes = {}
    for arm_name in arms:
        cfg = arm_cfg.clone()
        cfg, _ = apply_sweep_config(
            cfg,
            "robustness",
            make_arm_trial(sweep_config, sweep_params, arm_name),
            model_type="CombinedModel",
            sweep_params=sweep_params,
            arms=arms,
        )
        prefixes[arm_name] = get_model_filename_prefix(cfg, local_component=True, global_component=True)

    assert len(set(prefixes.values())) == len(arms), f"arms share a checkpoint filename: {prefixes}"


def test_max_seq_len_is_never_below_the_arms_largest_window(arm_yaml):
    """Every arm's max_seq_len must be >= the largest window of the column it selects.

    Below it, pad_batch random-subsamples the window each step and get_model_output stores an
    attention matrix narrower than the window, which makes the downstream net-flow computation
    fail with a shape mismatch rather than a wrong number.
    """
    # Largest window over ALL splits, measured on melton25_sliding_window.h5ad. Inference runs on
    # every cell, so the train-split maximum is not the relevant bound.
    LARGEST_WINDOW = {
        "sliding_window_400": 89,
        "sliding_window_800": 330,
        "sliding_window_1200": 685,
        "sliding_window_1600": 1218,
        "sliding_window_2000": 1720,
        "sliding_window_3000": 3436,
    }
    arms = load_arms(arm_yaml)
    for arm_name, overrides in arms.items():
        (column,) = overrides["dataset.sample_key"]
        max_seq_len = overrides["model.global_component.parameters.max_seq_len"]
        assert column in LARGEST_WINDOW, f"arm {arm_name} selects an unmeasured column {column}"
        assert max_seq_len >= LARGEST_WINDOW[column], (
            f"arm {arm_name}: max_seq_len {max_seq_len} < largest {column} window {LARGEST_WINDOW[column]}"
        )


def test_arm_with_a_missing_key_raises():
    """An arm that omits a key its siblings set would silently keep the base config's value."""
    yaml_config = {
        "sweep_config": {"metric": {"name": "val_r2", "goal": "maximize"}, "parameters": {ARM_PARAM: {"values": ["a", "b"]}}},
        "arms": {
            "a": {"dataset.name": "a", "dataset.batch_size": 8},
            "b": {"dataset.name": "b"},
        },
    }
    with pytest.raises(ValueError, match="does not declare the same keys"):
        load_arms(yaml_config, yaml_config["sweep_config"])


def test_arm_selecting_an_undefined_arm_raises():
    yaml_config = {
        "sweep_config": {"metric": {"name": "val_r2", "goal": "maximize"}, "parameters": {ARM_PARAM: {"values": ["a", "typo"]}}},
        "arms": {"a": {"dataset.name": "a"}},
    }
    with pytest.raises(ValueError, match="does not define"):
        load_arms(yaml_config, yaml_config["sweep_config"])


def test_arm_key_colliding_with_a_sweep_parameter_raises():
    """Set twice per trial, with application order deciding the winner."""
    yaml_config = {
        "sweep_config": {
            "metric": {"name": "val_r2", "goal": "maximize"},
            "parameters": {ARM_PARAM: {"values": ["a"]}, "dataset.batch_size": {"values": [4, 8]}},
        },
        "arms": {"a": {"dataset.batch_size": 16}},
    }
    with pytest.raises(ValueError, match="both by the arms and as sweep parameters"):
        load_arms(yaml_config, yaml_config["sweep_config"])


def test_arms_block_without_an_arm_parameter_raises():
    yaml_config = {
        "sweep_config": {"metric": {"name": "val_r2", "goal": "maximize"}, "parameters": {"optim.seed": {"values": [1]}}},
        "arms": {"a": {"dataset.name": "a"}},
    }
    with pytest.raises(ValueError, match="no 'arm' parameter"):
        load_arms(yaml_config, yaml_config["sweep_config"])


def test_arm_parameter_without_an_arms_block_raises():
    """Otherwise the arm name would be written to a config key called 'arm', which cannot exist."""
    yaml_config = {
        "sweep_config": {
            "metric": {"name": "val_r2", "goal": "maximize"},
            "parameters": {ARM_PARAM: {"values": ["a"]}},
        }
    }
    with pytest.raises(ValueError, match="no top-level `arms:` block"):
        build_sweep_config(yaml_config, prediction_task="regression", model_type="CombinedModel")


def test_arm_overrides_must_be_dotted():
    yaml_config = {
        "sweep_config": {"metric": {"name": "val_r2", "goal": "maximize"}, "parameters": {ARM_PARAM: {"values": ["a"]}}},
        "arms": {"a": {"batch_size": 8}},
    }
    with pytest.raises(ValueError, match="non-dotted keys"):
        load_arms(yaml_config, yaml_config["sweep_config"])


def test_arm_with_an_unknown_dotted_key_raises(arm_cfg):
    """Arm overrides go through the same _set_dotted validation as sweep parameters."""
    arms = {"a": {"dataset.no_such_key": 1}}
    with pytest.raises(KeyError):
        apply_sweep_config(
            arm_cfg, "robustness", {ARM_PARAM: "a"}, model_type="CombinedModel", sweep_params=[ARM_PARAM], arms=arms
        )


def test_reserved_arm_parameter_without_arms_passed_to_apply_raises(arm_cfg):
    """Guards the wiring: forgetting to thread `arms` through must fail, not train the base config."""
    with pytest.raises(KeyError, match="no arms were passed"):
        apply_sweep_config(
            arm_cfg, "robustness", {ARM_PARAM: "w400"}, model_type="CombinedModel", sweep_params=[ARM_PARAM]
        )


def test_arm_trial_does_not_leak_into_the_base_config(arm_cfg, arm_yaml):
    """wandb.agent reuses one process per agent, so a leaked arm would poison later trials."""
    sweep_config, sweep_params = build_sweep_config(
        arm_yaml, prediction_task="regression", model_type="CombinedModel"
    )
    arms = load_arms(arm_yaml, sweep_config)
    before = list(arm_cfg.dataset.sample_key)

    clone = arm_cfg.clone()
    apply_sweep_config(
        clone,
        "robustness",
        make_arm_trial(sweep_config, sweep_params, "w3000"),
        model_type="CombinedModel",
        sweep_params=sweep_params,
        arms=arms,
    )
    assert list(arm_cfg.dataset.sample_key) == before
    assert list(clone.dataset.sample_key) == ["sliding_window_3000"]


# --------------------------------------------------------------------------------------------
# Every arm-bearing sweep yaml in the repo, not just the one this file was written against.
#
# Discovered rather than listed: a new ablation adds a yaml and inherits these checks, which is
# the point. Each is resolved against the (dataset, task) pair its own header names, so the test
# proves the arms apply to the config they will actually be run with.
# --------------------------------------------------------------------------------------------

# sweep yaml -> the registered pair it is written for. Kept explicit because the yaml does not
# name its dataset: --dataset/--task are passed on the command line.
ARM_SWEEP_PAIRS = {
    "sliding_window_melton25.yaml": ("melton25_sw", "node_reg"),
    "overlap_ladder_legnini.yaml": ("legnini23_overlap", "node_reg"),
    # Masking-granularity ablation. Resolved against the CELL-masking pair: the arms set
    # mask_strategy/mask_token themselves, so starting from node_reg proves each arm overrides
    # the baseline rather than relying on the genemask task file to have set it.
    "mask_granularity_legnini.yaml": ("legnini23", "node_reg"),
    # Rate/length follow-up. Same base pair, same reason.
    "mask_rate_and_length_legnini.yaml": ("legnini23", "node_reg"),
    # Cell-masking rate ladder, the counterpart to the gene ladder in mask_granularity.
    "mask_rate_cell_ladder_legnini.yaml": ("legnini23", "node_reg"),
}


def arm_sweep_yamls():
    """Every yaml under config_files/sweeps that declares an `arms:` block."""
    found = []
    for path in sorted((CONFIG_DIR / "sweeps").glob("*.yaml")):
        with path.open() as f:
            if "arms" in (yaml.safe_load(f) or {}):
                found.append(path)
    return found


def test_every_arm_sweep_yaml_is_registered_in_this_test():
    """A new arm sweep must be added to ARM_SWEEP_PAIRS, or it goes untested."""
    undeclared = [p.name for p in arm_sweep_yamls() if p.name not in ARM_SWEEP_PAIRS]
    assert not undeclared, (
        f"arm sweep yaml(s) with no (dataset, task) declared in ARM_SWEEP_PAIRS: {undeclared}. "
        f"Add them so the checks below cover them."
    )


@pytest.mark.parametrize("yaml_name", sorted(ARM_SWEEP_PAIRS))
def test_arm_sweep_applies_to_its_registered_pair(yaml_name):
    """Every arm of every arm sweep resolves and applies against its own dataset/task pair.

    This is the check that would have caught a step of the overlap ladder naming an obs column that
    the registry's dataset file does not point at, or a max_seq_len key that moved.
    """
    dataset, task = ARM_SWEEP_PAIRS[yaml_name]
    with (CONFIG_DIR / "sweeps" / yaml_name).open() as f:
        yaml_config = yaml.safe_load(f)

    base = resolve_config(dataset, task, registry_path=REGISTRY)
    sweep_config, sweep_params = build_sweep_config(
        yaml_config, prediction_task=base.dataset.prediction_task, model_type="CombinedModel"
    )
    arms = load_arms(yaml_config, sweep_config)
    assert arms, f"{yaml_name} declares no arms"

    for arm_name in sweep_config["parameters"][ARM_PARAM]["values"]:
        trial = {k: v["values"][0] for k, v in sweep_config["parameters"].items()}
        trial[ARM_PARAM] = arm_name
        cfg, applied = apply_sweep_config(
            base.clone(), "robustness", trial, model_type="CombinedModel",
            sweep_params=sweep_params, arms=arms,
        )
        for key, expected in arms[arm_name].items():
            assert get_dotted(cfg, key) == expected, f"{yaml_name} {arm_name}: {key} not applied"
        # A sample_key that is empty would train on nothing; one that is a bare string would be
        # iterated character by character by prepare_geome_dataset.
        assert isinstance(cfg.dataset.sample_key, list) and cfg.dataset.sample_key, (
            f"{yaml_name} {arm_name}: dataset.sample_key must be a non-empty list, "
            f"got {cfg.dataset.sample_key!r}"
        )


@pytest.mark.parametrize("yaml_name", sorted(ARM_SWEEP_PAIRS))
def test_arm_sweep_gives_every_trial_its_own_checkpoint(yaml_name):
    """No two trials of an arm sweep may write the same checkpoint filename.

    get_model_filename_prefix keys on dataset.name, task, level and seed. An arm that forgets to
    set dataset.name silently overwrites its predecessor's checkpoint, and the sweep then reports
    metrics for models that no longer exist on disk.
    """
    from interscale.tl.utils import get_model_filename_prefix

    dataset, task = ARM_SWEEP_PAIRS[yaml_name]
    with (CONFIG_DIR / "sweeps" / yaml_name).open() as f:
        yaml_config = yaml.safe_load(f)
    base = resolve_config(dataset, task, registry_path=REGISTRY)
    sweep_config, sweep_params = build_sweep_config(
        yaml_config, prediction_task=base.dataset.prediction_task, model_type="CombinedModel"
    )
    arms = load_arms(yaml_config, sweep_config)

    seeds = sweep_config["parameters"].get("optim.seed", {}).get("values", [None])
    prefixes = {}
    for arm_name in sweep_config["parameters"][ARM_PARAM]["values"]:
        for seed in seeds:
            trial = {k: v["values"][0] for k, v in sweep_config["parameters"].items()}
            trial[ARM_PARAM] = arm_name
            if seed is not None:
                trial["optim.seed"] = seed
            cfg, _ = apply_sweep_config(
                base.clone(), "robustness", trial, model_type="CombinedModel",
                sweep_params=sweep_params, arms=arms,
            )
            prefixes[(arm_name, seed)] = get_model_filename_prefix(cfg, True, True)

    collisions = {v for v in prefixes.values() if list(prefixes.values()).count(v) > 1}
    assert not collisions, f"{yaml_name}: trials sharing a checkpoint filename: {collisions}"
