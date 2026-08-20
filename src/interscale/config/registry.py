"""Resolve a ``(dataset, task)`` pair into the config files that describe that run.

A run's configuration is layered, most general first, so that nothing is stated twice:

1. ``base`` -- architecture choices shared by every run (component names, decoder type).
2. the **dataset** file -- what the data *is*: paths, ``sample_key``, neighbour graph radius,
   ``max_seq_len``, results dir, wandb project.
3. the **task** file -- what is being predicted: ``prediction_task``/``prediction_level``,
   loss, monitored metric.
4. per-pair **overrides** -- the handful of values that belong to neither, most importantly
   ``dataset.prediction_obs`` (chen22+graph is ``stage``, melton25+graph is ``condition``,
   melton25+node is ``cell_type_coarse``, so the label column is a property of the pair).

Dataset comes *before* task on purpose: a dataset that needs its own training length
(chen22 trains for 400 epochs with patience 40, against the task default of 200/20) should
win over the task-level default, because it is the more specific statement.

The registry is deliberately **not** a yacs config. ``CfgNode.merge_from_file`` raises
``KeyError`` for any key absent from the defaults schema, so a nested ``tasks:`` block
could never live inside a config file that also gets merged into ``cfg``.
"""

from pathlib import Path

import yaml

from . import load_config

# Relative to the current working directory. Jobs run from the repo root (scripts/run.sh
# does `cd "$REPO"`), and the package cannot locate the repo itself: the container binds
# src/ to /opt/interscale_src, so a path derived from __file__ would point at /opt.
DEFAULT_REGISTRY_PATH = Path("config_files/registry.yaml")


def load_registry(registry_path=None):
    """Read the dataset/task registry.

    Parameters
    ----------
    registry_path : str or pathlib.Path, optional
        Path to ``registry.yaml``. Defaults to ``config_files/registry.yaml`` relative to
        the current working directory.

    Returns
    -------
    dict
        The parsed registry, with a ``_root`` key holding the directory the registry lives
        in, used to resolve the relative config paths it names.
    """
    path = Path(DEFAULT_REGISTRY_PATH if registry_path is None else registry_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"registry not found: {path} (cwd={Path.cwd()}). Run from the repo root or pass --registry."
        )

    with path.open() as f:
        registry = yaml.safe_load(f) or {}

    if "datasets" not in registry:
        raise ValueError(f"{path} declares no `datasets:` block.")

    registry["_root"] = path.parent
    return registry


def list_datasets(registry):
    """Return the dataset keys the registry defines, sorted."""
    return sorted(registry["datasets"])


def list_tasks(registry, dataset):
    """Return the task keys defined for ``dataset``, sorted."""
    return sorted(_dataset_entry(registry, dataset).get("tasks", {}))


def _dataset_entry(registry, dataset):
    datasets = registry["datasets"]
    if dataset not in datasets:
        raise KeyError(f"unknown dataset '{dataset}'. Known datasets: {', '.join(sorted(datasets))}")
    return datasets[dataset]


def _task_entry(registry, dataset, task):
    tasks = _dataset_entry(registry, dataset).get("tasks", {})
    if task not in tasks:
        raise KeyError(
            f"dataset '{dataset}' defines no task '{task}'. Known tasks for {dataset}: {', '.join(sorted(tasks))}"
        )
    return tasks[task]


def resolve_config_paths(dataset, task, registry_path=None, registry=None):
    """Resolve a ``(dataset, task)`` pair into config file paths plus flat overrides.

    Returns
    -------
    tuple[list[pathlib.Path], list]
        The config files to merge in order, and the ``[key, value, ...]`` override list, both
        ready to hand to :func:`interscale.config.load_config`.
    """
    if registry is None:
        registry = load_registry(registry_path)
    root = Path(registry["_root"])

    dataset_entry = _dataset_entry(registry, dataset)
    task_entry = _task_entry(registry, dataset, task)

    paths = []
    if registry.get("base"):
        paths.append(root / registry["base"])
    if dataset_entry.get("config"):
        paths.append(root / dataset_entry["config"])
    if task_entry.get("task"):
        paths.append(root / task_entry["task"])

    # Dataset-level overrides first so a task can still override them for its own pair.
    overrides = {}
    overrides.update(dataset_entry.get("overrides") or {})
    overrides.update(task_entry.get("overrides") or {})

    flat_overrides = []
    for key, value in overrides.items():
        flat_overrides.extend([key, value])

    return paths, flat_overrides


def resolve_config(dataset, task, registry_path=None, registry=None):
    """Load the fully merged, frozen config for a ``(dataset, task)`` pair."""
    paths, overrides = resolve_config_paths(dataset, task, registry_path=registry_path, registry=registry)
    return load_config(paths, overrides=overrides)


def iter_pairs(registry):
    """Yield every ``(dataset, task)`` pair the registry defines, sorted."""
    for dataset in list_datasets(registry):
        for task in list_tasks(registry, dataset):
            yield dataset, task
