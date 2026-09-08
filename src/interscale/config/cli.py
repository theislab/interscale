"""Shared ``--dataset/--task`` command-line plumbing for the training entrypoints.

``main.py`` and ``main_sweep.py`` both need to turn command-line arguments into one frozen
config. Keeping that in one place is what stops the two entrypoints drifting apart the way the
per-dataset shell scripts did.
"""

import sys

from . import load_config
from .registry import iter_pairs, list_datasets, list_tasks, load_registry, resolve_config


def add_config_args(parser):
    """Add the config-selection arguments shared by every training entrypoint."""
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        default=None,
        help="Dataset key from config_files/registry.yaml (e.g. melton25). Use with --task.",
    )
    parser.add_argument(
        "--task",
        dest="task",
        type=str,
        default=None,
        help="Task key for that dataset (e.g. graph_clas). Use with --dataset.",
    )
    parser.add_argument(
        "--registry",
        dest="registry",
        type=str,
        default=None,
        help="Path to the registry (default: config_files/registry.yaml, relative to the cwd).",
    )
    parser.add_argument(
        "--cfg",
        dest="cfg",
        type=str,
        default=None,
        help="A single config file, bypassing the registry. Mutually exclusive with --dataset/--task.",
    )
    parser.add_argument(
        "--list",
        dest="list_pairs",
        action="store_true",
        help="Print the dataset/task pairs the registry defines, then exit.",
    )
    return parser


def print_registry(registry_path=None, stream=None):
    """Print every ``(dataset, task)`` pair the registry defines."""
    stream = sys.stdout if stream is None else stream
    registry = load_registry(registry_path)

    print("Available --dataset / --task pairs:\n", file=stream)
    for dataset in list_datasets(registry):
        tasks = list_tasks(registry, dataset)
        print(f"  {dataset}", file=stream)
        for task in tasks:
            print(f"      --dataset {dataset} --task {task}", file=stream)
    print("", file=stream)
    return list(iter_pairs(registry))


def resolve_cfg_from_args(args, parser=None):
    """Turn parsed arguments into one frozen config.

    Accepts either ``--dataset``/``--task`` (resolved through the registry) or a single
    ``--cfg`` file, and refuses the ambiguous combination of both.
    """

    def fail(message):
        if parser is not None:
            parser.error(message)
        raise SystemExit(f"error: {message}")

    using_registry = args.dataset is not None or args.task is not None

    if args.cfg is not None and using_registry:
        fail("--cfg cannot be combined with --dataset/--task; the layered registry config would be ignored.")

    if args.cfg is not None:
        return load_config(args.cfg)

    if not using_registry:
        fail("give either --dataset and --task, or --cfg. Use --list to see the registered pairs.")

    if args.dataset is None or args.task is None:
        fail("--dataset and --task must be given together. Use --list to see the registered pairs.")

    # KeyError from the registry carries the list of valid keys; surface it as a clean CLI
    # error rather than a traceback several minutes into a queued job.
    try:
        return resolve_config(args.dataset, args.task, registry_path=args.registry)
    except KeyError as exc:
        fail(str(exc).strip("\"'"))
