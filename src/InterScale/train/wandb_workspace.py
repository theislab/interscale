"""Define and save a W&B workspace layout for InterScale runs.

Uses the wandb-workspaces API (Public Preview) to create a workspace with:
- One chart: train_loss, val_loss, test_loss
- One chart: local vs global loss (val_local_loss, val_global_loss; optional train/test)

Requires: pip install wandb-workspaces
"""

# Color scheme for workspace charts (hex). Use in W&B UI or pass as line_colors if supported.
LOSS_COLORS = {
    "train_loss": "#2E86AB",
    "val_loss": "#A23B72",
    "test_loss": "#F18F01",
}
LOCAL_GLOBAL_COLORS = {
    "train_local_loss": "#2E86AB",
    "train_global_loss": "#A23B72",
    "val_local_loss": "#3B1F2B",
    "val_global_loss": "#C73E1D",
    "test_local_loss": "#6A994E",
    "test_global_loss": "#BC4B51",
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


def get_interscale_workspace_sections():
    """Build section config for InterScale metrics (reusable)."""
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

    return [
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


def setup_wandb_workspace(entity: str, project: str, name: str = "InterScale Layout"):
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
    name : str
        Workspace name shown in the UI.
    """
    import wandb_workspaces.workspaces as ws

    sections = get_interscale_workspace_sections()
    workspace = ws.Workspace(
        name=name,
        entity=entity,
        project=project,
        sections=sections,
    )
    workspace.save()


def setup_wandb_workspace_if_available(entity: str, project: str):
    """Call setup_wandb_workspace if wandb-workspaces is installed; no-op otherwise."""
    try:
        setup_wandb_workspace(entity=entity, project=project)
    except ImportError:
        pass  # wandb-workspaces not installed; skip workspace setup
