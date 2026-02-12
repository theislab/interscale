"""Define and save a W&B workspace layout for InterScale runs.

Uses the wandb-workspaces API (Public Preview) to create a workspace with:
- One chart: train_loss, val_loss, test_loss
- One chart: local vs global loss (val_local_loss, val_global_loss; optional train/test)
- (Classification only) One chart per class: train_f1_CLASS, val_f1_CLASS, test_f1_CLASS

Requires: pip install wandb-workspaces
"""

from __future__ import annotations
from typing import Sequence

TRAIN_COLOR = "#50B953" # green
VAL_COLOR = "#FDC835" # yellow
TEST_COLOR = "#F33B16" # red

# Color scheme for workspace charts (hex). Use in W&B UI or pass as line_colors if supported.
LOSS_COLORS = {
    "train_loss": TRAIN_COLOR,
    "val_loss": VAL_COLOR,
    "test_loss": TEST_COLOR,
}
LOCAL_GLOBAL_COLORS = {
    "train_local_loss": TRAIN_COLOR,
    "train_global_loss": TRAIN_COLOR,
    "val_local_loss": VAL_COLOR,
    "val_global_loss": VAL_COLOR,
    "test_local_loss": TEST_COLOR,
    "test_global_loss": TEST_COLOR,
}


def _line_plot_with_colors(x, y, color_map):
    """Build LinePlot with line_colors if the API supports it; otherwise plain LinePlot."""
    import wandb_workspaces.reports.v2 as wr

    colors = {m: color_map[m] for m in y if m in color_map}
    if colors:
        try:
            return wr.LinePlot(x=x, y=y, line_colors=colors)
        except TypeError:
            pass
    return wr.LinePlot(x=x, y=y)


def _build_class_f1_colors(class_labels: Sequence[str]) -> dict[str, str]:
    """Build a color map for per-class F1 metrics (train/val/test)."""
    colors: dict[str, str] = {}
    for cls in class_labels:
        colors[f"train_f1_{cls}"] = TRAIN_COLOR
        colors[f"val_f1_{cls}"] = VAL_COLOR
        colors[f"test_f1_{cls}"] = TEST_COLOR
    return colors


def get_interscale_workspace_sections(class_labels: Sequence[str] | None = None):
    """Build section config for InterScale metrics (reusable).

    Parameters
    ----------
    class_labels : sequence of str, optional
        Class names used in classification tasks.  When provided an
        additional section is added that groups per-class F1 scores
        (``train_f1_<class>``, ``val_f1_<class>``, ``test_f1_<class>``)
        into a single chart.
    """
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr

    loss_metrics = ["train_loss", "val_loss", "test_loss"]
    local_global_metrics = [
        "train_local_loss",
        "train_global_loss",
        "val_local_loss",
        "val_global_loss",
        "test_local_loss",
        "test_global_loss",
    ]

    loss_plot = _line_plot_with_colors("Step", loss_metrics, LOSS_COLORS)
    dual_plot = _line_plot_with_colors("Step", local_global_metrics, LOCAL_GLOBAL_COLORS)

    sections = [
        ws.Section(
            name="Loss (train / val / test)",
            panels=[loss_plot],
            is_open=True,
        ),
        ws.Section(
            name="Local vs global loss",
            panels=[dual_plot],
            is_open=True,
        ),
    ]

    # ---- Per-class F1 section (classification only) ----
    if class_labels is not None and len(class_labels) > 0:
        f1_metrics = []
        for cls in class_labels:
            f1_metrics.extend([
                f"train_f1_{cls}",
                f"val_f1_{cls}",
                f"test_f1_{cls}",
            ])
        f1_colors = _build_class_f1_colors(class_labels)
        f1_plot = _line_plot_with_colors("Step", f1_metrics, f1_colors)
        sections.append(
            ws.Section(
                name="Per-class F1 (train / val / test)",
                panels=[f1_plot],
                is_open=True,
            ),
        )

    return sections


def setup_wandb_workspace(
    entity: str,
    project: str,
    class_labels: Sequence[str] | None = None,
    name: str = "InterScale Layout",
):
    """Create and save a W&B workspace with InterScale chart layout.

    Call this when wandb is enabled (e.g. after wandb.init()) so that the
    project's workspace shows train/val/test loss in one chart and
    local/global loss in another.

    Parameters
    ----------
    entity : str
        W&B entity (e.g. your username or team).
    project : str
        W&B project name (e.g. cfg.wandb.project_name).
    class_labels : sequence of str, optional
        Class names for classification tasks.  When given, an extra chart
        section is added that groups the per-class F1 scores.
    name : str
        Workspace name shown in the UI.
    """
    import wandb_workspaces.workspaces as ws

    sections = get_interscale_workspace_sections(class_labels=class_labels)
    workspace = ws.Workspace(
        name=name,
        entity=entity,
        project=project,
        sections=sections,
    )
    workspace.save()


def setup_wandb_workspace_if_available(
    entity: str,
    project: str,
    class_labels: Sequence[str] | None = None,
):
    """Call setup_wandb_workspace if wandb-workspaces is installed; no-op otherwise."""
    try:
        setup_wandb_workspace(entity=entity, project=project, class_labels=class_labels)
    except ImportError:
        pass  # wandb-workspaces not installed; skip workspace setup
