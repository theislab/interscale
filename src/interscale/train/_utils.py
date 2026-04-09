import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.callbacks import Callback


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
