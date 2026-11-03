import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.callbacks import Callback

# Lower is better for a loss and for an error -- mse, and the scaled cosine error, which would
# otherwise be maximised when monitored. Everything else logged (r2, correlations, cosine
# similarity, f1, accuracy) is higher-is-better.
MINIMIZED_SUFFIXES = ("loss", "error", "mse")


def metric_direction(metric: str) -> str:
    """Direction a logged metric should be optimised in, inferred from its name.

    Shared by the layers that each pick a direction independently: the in-trial callbacks and
    the ReduceLROnPlateau scheduler via :meth:`NodeMaskingTrainingPlan._resolve_monitor`, and the
    wandb sweep's own ``metric.goal`` in ``main_sweep``. The sweep's goal is registered
    server-side before any trial exists, so it cannot call ``_resolve_monitor`` (a bound method
    needing a model) -- it calls this on the bare metric name instead, which keeps them from
    disagreeing. It lives here rather than in ``_training`` so ``_trainingplans`` can import it
    without a cycle.

    Parameters
    ----------
    metric : str
        Name of a metric the training plan logs, e.g. ``"val_scaled_cosine_error"``.

    Returns
    -------
    str
        ``"min"`` if lower is better, ``"max"`` otherwise.
    """
    return "min" if metric.endswith(MINIMIZED_SUFFIXES) else "max"


class NodeMaskResampleCallback(Callback):
    """Redraws the training node mask at the start of every epoch.

    Without this, `GraphAnnDataModule` only ever samples one fixed set of masked nodes for the
    whole training run (see `GraphAnnDataModule.resample_train_mask`), so nodes outside that
    initial sample never receive direct supervision. Validation/test masks are left untouched so
    monitored metrics (e.g. `val_loss`) stay comparable across epochs.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.resample_train_mask()


class MetricsHistory(Callback):
    def __init__(self):
        super().__init__()
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # Convert tensor values to float
        epoch_dict = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
        epoch_dict["epoch"] = trainer.current_epoch
        self.history.append(epoch_dict)

    def on_test_epoch_end(self, trainer, pl_module):
        print("on_test_epoch_end", trainer.current_epoch)
        metrics = trainer.callback_metrics
        # Convert tensor values to float
        epoch_dict = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
        epoch_dict["epoch"] = trainer.current_epoch

        # Record metrics every 10 epochs or at the end of training
        if trainer.current_epoch % 5 == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            self.history.append(epoch_dict)

    def plot_history(self, subset_term=None):
        history = pd.DataFrame(self.history)
        if subset_term is not None:
            history = history[[c for c in history.columns if subset_term in c]]
        ax = history[[c for c in history.columns if "train" in c]].plot()
        plt.gca().set_prop_cycle(None)
        history[[c for c in history.columns if "val" in c]].plot(style="--", ax=ax)
        # plt.gca().set_prop_cycle(None)
        # history[[c for c in history.columns if 'test' in c]].plot(style=':', ax=ax)
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()
