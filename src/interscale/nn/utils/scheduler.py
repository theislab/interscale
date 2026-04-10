import numpy as np
from torch import optim


class CosineWarmupScheduler(optim.lr_scheduler.LRScheduler):
    """Cosine warmup scheduler."""

    def __init__(self, optimizer, warmup, max_epochs):
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to schedule.
        warmup : int
            The number of warmup epochs.
        max_epochs : int
            The maximum number of epochs.
        """
        self.warmup = warmup
        self.max_num_epochs = max_epochs
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        """Returns the learning rate for the current epoch."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(1e-5, base_lr * lr_factor) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """
        Parameters
        ----------
        epoch : int
            The current epoch.
        """
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epochs))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
