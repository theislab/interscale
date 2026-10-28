import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint


class PostWarmupModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` that ignores every epoch before ``start_epoch``.

    With ``CosineWarmupScheduler`` the LR ramps linearly over the first ``lr_warmup`` epochs, so
    a checkpoint from inside the ramp is a near-initialisation model. On small, noisy validation
    sets such an epoch can still hit the best monitored score by chance -- in chen22 sweep
    qbmsh55r 77 of 90 runs had their best val_f1_macro during warm-up, the top one at epoch 0 --
    and train() then restores it and reports it as the run's result.

    Parameters
    ----------
    start_epoch
        First (0-indexed) epoch that may be saved. Epochs before it are never candidates.
    **kwargs
        Passed to :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    def __init__(self, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = int(start_epoch)

    # Private Lightning hook (2.x); tests/test_warmup_selection.py fails if it disappears.
    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        return trainer.current_epoch < self.start_epoch or super()._should_skip_saving_checkpoint(trainer)


class PostWarmupEarlyStopping(EarlyStopping):
    """``EarlyStopping`` that neither tracks a best score nor counts patience before ``start_epoch``.

    Must be used together with :class:`PostWarmupModelCheckpoint`. If only the checkpoint ignored
    warm-up, a warm-up peak would still set EarlyStopping's best score: patience would run out
    ``patience`` epochs after that peak, possibly before any post-warm-up epoch beat it.

    Parameters
    ----------
    start_epoch
        First (0-indexed) epoch whose monitored value is considered.
    **kwargs
        Passed to :class:`~lightning.pytorch.callbacks.EarlyStopping`.
    """

    def __init__(self, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = int(start_epoch)

    # Private Lightning hook (2.x); tests/test_warmup_selection.py fails if it disappears.
    def _should_skip_check(self, trainer) -> bool:
        return trainer.current_epoch < self.start_epoch or super()._should_skip_check(trainer)


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
