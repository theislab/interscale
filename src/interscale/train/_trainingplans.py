from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

from interscale.module.base._base_module import BaseModule
from interscale.nn import CosineWarmupScheduler
from interscale.tl.masking import masked_loss

from .losses import BalancedPearsonCorrelationLoss, SCE_EntropyATT_Loss, SCELoss

class RunningCosineSimilarity(torchmetrics.Metric):
    """Mean per-cell cosine similarity, with state that does not grow with the dataset.

    ``torchmetrics.CosineSimilarity`` is a *list-state* metric: it keeps every prediction and
    target it is shown and concatenates them at compute time. Every other regression metric here
    holds a few kilobytes of running sums, and this one holds ``n_cells x n_genes x 2`` floats --
    which ``MetricCollection.forward`` then duplicates via ``_copy_state_dict`` on every step.

    On legnini23 (43k cells, 88 genes) that is ~30 MB and invisible. On the CosMx pancreas
    (387k cells, 979 genes) one epoch is ~850 MB before the copy, and it OOMed a 20 GB card
    inside ``_regression_metrics`` on the very first trial, regardless of batch size -- the total
    per epoch is the same however the cells are batched.

    This computes the same quantity (the mean over cells of the per-cell cosine) from a running
    sum and count, so the state is two scalars.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate the summed per-cell cosine similarity and the number of cells."""
        cos = nn.functional.cosine_similarity(preds, target, dim=1)
        self.total = self.total + cos.sum()
        self.count = self.count + cos.numel()

    def compute(self) -> torch.Tensor:
        """Mean per-cell cosine similarity over everything seen since the last reset."""
        return self.total / self.count


def masked_regression_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, entry_mask: torch.Tensor, eps: float = 1e-8
) -> dict[str, torch.Tensor]:
    """The regression metrics of ``_setup_regression_metrics``, restricted to the masked entries.

    Under gene masking the scored rows are mostly entries the model was *given*. Feeding the full
    rows to the ``MetricCollection`` would score the identity map on those and inflate every
    number -- including ``val_r2``, which drives early stopping and checkpoint selection. There
    is no way to express "these entries only" to a torchmetrics per-output metric (the surviving
    entries are ragged across genes), so the same quantities are computed here from masked sums.

    Every entry that is not masked is multiplied by zero before any sum is taken, and zeros
    contribute nothing to a sum, so each moment below is exactly the moment over the masked
    entries -- no approximation.

    Parameters
    ----------
    y_pred, y_true
        ``[N, G]`` predictions and targets for the scored rows.
    entry_mask
        ``[N, G]`` boolean marking the masked entries.
    eps
        Guard for degenerate (zero-variance) genes.

    Returns
    -------
    dict
        Unprefixed metric names mapped to scalar tensors, matching the keys that
        ``_setup_regression_metrics`` produces: ``mse``, ``r2`` (per-gene, uniform average),
        ``pearson_corr``, ``concordance_corr``, ``cosine_similarity`` (per cell).
    """
    m = entry_mask.to(y_pred.dtype)
    p_ = y_pred * m
    t_ = y_true * m

    n_gene = m.sum(dim=0)  # [G] masked cells per gene
    n_total = m.sum()

    mse = ((p_ - t_) ** 2).sum() / n_total.clamp(min=1)

    # Per-gene first and second moments over that gene's masked cells.
    ng = n_gene.clamp(min=1)
    mean_p = p_.sum(dim=0) / ng
    mean_t = t_.sum(dim=0) / ng
    var_p = (p_**2).sum(dim=0) / ng - mean_p**2
    var_t = (t_**2).sum(dim=0) / ng - mean_t**2
    cov = (p_ * t_).sum(dim=0) / ng - mean_p * mean_t

    # A gene with fewer than two masked cells, or with no spread in either vector, has no
    # correlation defined; NaN it out and let nanmean skip it, exactly as the unmasked path
    # already does for constant genes.
    nan = torch.tensor(float("nan"), device=y_pred.device, dtype=y_pred.dtype)
    usable = (n_gene >= 2) & (var_t > eps)

    pearson = torch.where(usable & (var_p > eps), cov / (var_p.clamp(min=eps) * var_t.clamp(min=eps)).sqrt(), nan)
    concordance = torch.where(usable, 2 * cov / (var_p + var_t + (mean_p - mean_t) ** 2 + eps), nan)

    # R2 per gene, then uniform average -- the same reduction torchmetrics'
    # R2Score(multioutput="uniform_average") applies.
    ss_res = ((p_ - t_) ** 2).sum(dim=0)
    ss_tot = var_t * ng
    r2 = torch.where(usable, 1 - ss_res / ss_tot.clamp(min=eps), nan)

    # Per-cell cosine over that cell's masked genes: the zeroed entries drop out of both the dot
    # product and the two norms, so this is the cosine on the masked coordinates.
    cosine = nn.functional.cosine_similarity(p_, t_, dim=1)

    return {
        "mse": mse,
        "r2": torch.nanmean(r2),
        "pearson_corr": torch.nanmean(pearson),
        "concordance_corr": torch.nanmean(concordance),
        "cosine_similarity": cosine.mean(),
    }


CLASSIFICATION_LOSSES = ["CrossEntropy", "WeightedCE"]
REGRESSION_LOSSES = [
    "MSELoss",
    "GaussianNLL",
    "SmoothL1",
    "BalancedPearsonCorrelationLoss",
    "SCELoss",
    "SCE_EntropyATT_Loss",
]


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/train/_trainingplans.py
# accessed on 28 April 2025
class TrainingPlan(pl.LightningModule):
    """Lightning module task to train scvi-tools modules.

    The training plan is a PyTorch Lightning Module that is initialized
    with a scvi-tools module object. It configures the optimizers, defines
    the training step and validation step, and computes metrics to be recorded
    during training. The training step and validation step are functions that
    take data, run it through the model and return the loss, which will then
    be used to optimize the model parameters in the Trainer. Overall, custom
    training plans can be used to develop complex inference schemes on top of
    modules.

    The following developer tutorial will familiarize you more with training plans
    and how to use them: :doc:`/tutorials/notebooks/dev/model_user_guide`.

    Parameters
    ----------
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.

    lr_scheduler: None | Literal["ReduceLROnPlateau", "CosineWarmupScheduler"] = None
        Learning rate scheduler to use. Default is None. CosineWarmupScheduler reduces LR at each step, ReduceLROnPlateau reduces LR with a patience if no improvement is seen.
    """

    def __init__(
        self,
        module: BaseModule,
        prediction_task: str,
        prediction_level: Literal["node", "graph"],
        loss: Literal[CLASSIFICATION_LOSSES, REGRESSION_LOSSES],
        cross_corr: Literal["gene", "cell"],
        batch_size: int,
        class_weights: np.ndarray | None = None,
        class_labels: list[str] | None = None,
        *,
        lr_scheduler: None | Literal["ReduceLROnPlateau", "CosineWarmupScheduler"] = None,
        weight_decay: float = 1e-6,
        lr: float = 1e-3,
        lr_warmup: int = 0,
        lr_max_epochs: int = 100000,
        patience_in_steps: int = 100000,
        **kwargs,
    ):
        super().__init__()
        self.module = module
        self.prediction_task = prediction_task
        self.prediction_level = prediction_level
        self.loss_type = loss
        self.cross_corr = cross_corr
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.class_labels = class_labels
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.patience_in_steps = patience_in_steps
        self.lr_warmup = lr_warmup
        self.lr_max_epochs = lr_max_epochs
        self.lr = lr
        if self.prediction_task == "regression":
            if self.cross_corr == "gene":
                print("cross-gene per cell correlation metrics")
                self.AXIS = 1  # selecting rows / cells
            elif self.cross_corr == "cell":
                print("cross-cell per gene correlation metrics")
                self.AXIS = 0  # selecting columns / genes

        # setup metrics and loss
        if "classification" in self.prediction_task:
            metrics = self._setup_classification_metrics(self.module.n_output)
            self.loss = self._setup_classification_loss(self.loss_type, self.class_weights)
            # Must name a metric that is actually logged -- this is handed to Lightning as the
            # LR-scheduler monitor. "val_f1" never existed; only val_f1_micro/macro/<class> do.
            self.monitor_metric = "val_f1_macro"
        elif "regression" in self.prediction_task:
            metrics = self._setup_regression_metrics(self.module.n_output)
            self.loss = self._setup_regression_loss(self.loss_type)
            self.monitor_metric = "val_r2"
        else:
            raise ValueError("Prediction task must define 'classification' or 'regression'.")

        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    @staticmethod
    def _setup_classification_loss(
        loss: Literal["CrossEntropy", "WeightedCE"], class_weights: torch.Tensor | None = None
    ):
        """Setup loss function based on prediction task and configuration."""
        assert loss in CLASSIFICATION_LOSSES, "Classification must be run with CrossEntropy or WeightedCE loss."
        if loss == "CrossEntropy":
            return nn.CrossEntropyLoss()
        elif loss == "WeightedCE":
            assert class_weights is not None, "Class weights must be provided for WeightedCE loss."
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch tensor"
            # .float() guards against a float64 weight buffer meeting float32 logits.
            return nn.CrossEntropyLoss(weight=class_weights.float())

    def _setup_regression_loss(self, loss: Literal[REGRESSION_LOSSES]):
        """Setup loss function based on prediction task and configuration."""
        assert loss in REGRESSION_LOSSES, (
            f"{loss} not in {REGRESSION_LOSSES}"
        )  # "Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss."
        if loss == "MSELoss":
            return nn.MSELoss()
        elif loss == "GaussianNLL":
            return nn.GaussianNLLLoss()
        elif loss == "SmoothL1":
            return nn.SmoothL1Loss()
        elif loss == "BalancedPearsonCorrelationLoss":
            return BalancedPearsonCorrelationLoss(None)
        elif loss == "SCELoss":
            return SCELoss()
        elif loss == "SCE_EntropyATT_Loss":
            return SCE_EntropyATT_Loss()

    @staticmethod
    def _setup_classification_metrics(num_outputs: int):
        return MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs),
                "f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="micro"),
                "f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="macro"),
                "f1_per_class": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average=None),
            }
        )

    @staticmethod
    def _setup_regression_metrics(num_outputs: int):
        return MetricCollection(
            {
                "mse": torchmetrics.MeanSquaredError(),
                "r2": torchmetrics.R2Score(multioutput="uniform_average"),
                "pearson_corr": torchmetrics.PearsonCorrCoef(num_outputs=num_outputs),
                # Pearson is invariant to any per-gene affine rescaling of the predictions, so a
                # model whose outputs are (say) 11x too spread out still scores a high r while
                # its R2 goes to -113. Writing predictions as k times the true sd with offset d,
                # R2 = 2*r*k - k^2 - d^2/sigma^2, maximised at k = r -- so R2 <= r^2, and the gap
                # between them is purely calibration. Concordance correlation folds that penalty
                # back in, which makes it the metric to select on when both the co-variation
                # structure AND the expression scale have to be usable.
                "concordance_corr": torchmetrics.ConcordanceCorrCoef(num_outputs=num_outputs),
                # Not torchmetrics.CosineSimilarity: see RunningCosineSimilarity for why its
                # list state cannot be used on a dataset this size. Same value, O(1) memory.
                "cosine_similarity": RunningCosineSimilarity(),
            }
        )

    def _classification_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        mask_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate classification metrics."""
        ## TODO: Currently mask_idx is applied in module._common_step. Maybe move to here?
        # if mask_idx is not None:
        #     y_pred = y_pred[mask_idx]
        #     y_true = y_true[mask_idx]

        loss = self.loss(y_pred, y_true)
        metrics = metrics(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        metrics[f"{mode}_loss"] = loss

        return loss, metrics

    def _regression_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        mask_idx: torch.Tensor | None = None,
        attn: torch.Tensor | None = None,
        entry_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate regression metrics.

        Parameters
        ----------
        y_true : torch.Tensor
            True values of shape [N, G], where N is the number of cells and G is the number of genes
        y_pred : torch.Tensor
            Predicted values of shape [N, G], where N is the number of cells and G is the number of genes
            True and predicted values of shape [N, G], where N is the number of cells and G is the number of genes
        mode : str
            The mode of the metrics.
        metrics : MetricCollection
            The metrics to calculate.
        mask_idx : torch.Tensor | None
            The mask indices to apply to the metrics.
        attn : torch.Tensor | None
            The attention weights to apply to the metrics.
        entry_mask : torch.Tensor | None
            [N, G] boolean marking the entries that were actually masked. Set under
            ``mask_strategy="gene"``; ``None`` under cell masking, where the whole row of every
            scored cell was blanked and there is nothing to restrict. When given, BOTH the loss
            and the metrics are computed over those entries only -- see ``masked_regression_metrics``.
        """
        if self.loss_type == "SCE_EntropyATT_Loss":
            # Takes attention as a third argument, so it cannot go through masked_loss. Zeroing
            # the unmasked entries restricts its row-wise cosine to the masked coordinates,
            # which is what the row-structured branch of masked_loss does too.
            if entry_mask is None:
                loss = self.loss(y_pred, y_true, attn)
            else:
                m = entry_mask.to(y_pred.dtype)
                loss = self.loss(y_pred * m, y_true * m, attn)
        else:
            loss = masked_loss(self.loss, self.loss_type, y_pred, y_true, entry_mask)

        if entry_mask is None:
            metrics = metrics(y_pred, y_true)
            # Take mean across pearson correlation
            metrics[f"{mode}_pearson_corr"] = torch.nanmean(metrics[f"{mode}_pearson_corr"].contiguous())
            # Same reduction, same reason: both are per-gene vectors of length n_output, and a
            # constant gene yields NaN rather than a number.
            metrics[f"{mode}_concordance_corr"] = torch.nanmean(metrics[f"{mode}_concordance_corr"].contiguous())
        else:
            # Same metric names, so `optim.monitor`, the sweep `--metric` flag and every existing
            # wandb panel keep working across both strategies -- what changes is only which
            # entries they are computed over.
            metrics = {f"{mode}_{k}": v for k, v in masked_regression_metrics(y_pred, y_true, entry_mask).items()}

        metrics[f"{mode}_loss"] = loss
        return loss, metrics

    def forward(self, *args, **kwargs):
        """Passthrough to the module's forward method."""
        return self.module(
            *args,
            **kwargs,
        )

    # @torch.inference_mode() decorator disables gradient computation. TODO: enable again after calculating loss in module.
    def _compute_and_log_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        attn: torch.Tensor | None,
        entry_mask: torch.Tensor | None = None,
    ):
        """Helper method to log metrics for training, validation, or test steps.

        Parameters
        ----------
        y_true, y_pred: torch.Tensor
            True and predicted values of shape [N, G], where N is the number of cells and G is the number of genes
        mode
            One of 'train', 'val', or 'test'
        metrics: MetricCollection
            Metrics to log
        """
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        # TODO: where is the batch size?

        if "classification" in self.prediction_task:
            loss, metrics = self._classification_metrics(y_pred, y_true, mode, metrics)
            for class_idx, class_score in enumerate(metrics[f"{mode}_f1_per_class"]):
                metrics[f"{mode}_f1_{self.class_labels[class_idx]}"] = class_score
            metrics.pop(f"{mode}_f1_per_class")

        elif "regression" in self.prediction_task:
            loss, metrics = self._regression_metrics(y_pred, y_true, mode, metrics, attn=attn, entry_mask=entry_mask)

        # Set sync_dist=True only for test mode
        sync_dist = mode == "test"
        self.log_dict(metrics, batch_size=int(self.batch_size), on_step=False, on_epoch=True, sync_dist=sync_dist)

        return loss

    def training_step(self, batch):
        """Training step for the model.

        Returns
        -------
            loss: torch.nn.Module
        """
        local_embedding, global_embedding, y_pred, y_true, attn, entry_mask = self.module._common_step(
            batch, self.prediction_task, self.prediction_level
        )

        # Check if module supports separate loss computation (e.g., DualDecoderCombinedModule)
        if hasattr(self.module, "compute_separate_losses"):
            separate_losses = self.module.compute_separate_losses(
                self.loss, self.loss_type, y_pred, y_true, entry_mask
            )

            # Log separate losses (on_step=False, on_epoch=True to match existing pattern)
            if separate_losses.get("local_loss") is not None:
                self.log(
                    "train_local_loss",
                    separate_losses["local_loss"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=False,
                )
            if separate_losses.get("global_loss") is not None:
                self.log(
                    "train_global_loss",
                    separate_losses["global_loss"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=False,
                )
            if separate_losses.get("combined_loss") is not None:
                self.log(
                    "train_combined_loss",
                    separate_losses["combined_loss"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=False,
                )

            #  compute and log metrics using combined predictions
            loss = self._compute_and_log_metrics(y_pred, y_true, "train", self.train_metrics, attn=attn, entry_mask=entry_mask)

            if separate_losses.get("kl_loss") is not None:
                kl_loss = separate_losses["kl_loss"]

                # KL Annealing/Weighting (beta)
                # You can use a fixed weight or a scheduler (e.g., self.current_epoch)
                kl_weight = getattr(self.hparams, "kl_weight", 1.0)
                weighted_kl = kl_weight * kl_loss

                self.log(
                    "train_kl_loss",
                    kl_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=False,
                )

                # Add KL to the final loss to be backpropagated
                loss += weighted_kl

            assert not torch.isnan(loss), "loss is NaN"
            return loss
        else:
            return self._compute_and_log_metrics(y_pred, y_true, "train", self.train_metrics, attn=attn, entry_mask=entry_mask)
        # return self._compute_and_log_metrics(y_pred, y_true, 'train', self.train_metrics, attn=attn)

    def validation_step(self, batch):
        """Validation step for the model."""
        local_embedding, global_embedding, y_pred, y_true, attn, entry_mask = self.module._common_step(
            batch, self.prediction_task, self.prediction_level
        )

        # Check if module supports separate loss computation (e.g., DualDecoderCombinedModule)
        if hasattr(self.module, "compute_separate_losses"):
            separate_losses = self.module.compute_separate_losses(
                self.loss, self.loss_type, y_pred, y_true, entry_mask
            )

            # Log separate losses (on_step=False, on_epoch=True to match existing pattern)
            if separate_losses.get("local_loss") is not None:
                self.log(
                    "val_local_loss",
                    separate_losses["local_loss"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=False,
                )
            if separate_losses.get("global_loss") is not None:
                self.log(
                    "val_global_loss",
                    separate_losses["global_loss"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=False,
                )

            #  compute and log metrics using combined predictions
            loss = self._compute_and_log_metrics(y_pred, y_true, "val", self.valid_metrics, attn=attn, entry_mask=entry_mask)

            if separate_losses.get("kl_loss") is not None:
                kl_loss = separate_losses["kl_loss"]

                # KL Annealing/Weighting (beta)
                # You can use a fixed weight or a scheduler (e.g., self.current_epoch)
                kl_weight = getattr(self.hparams, "kl_weight", 1.0)
                weighted_kl = kl_weight * kl_loss

                self.log(
                    "val_kl_loss",
                    kl_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=False,
                )

                # Add KL to the final loss to be backpropagated
                loss += weighted_kl

            assert not torch.isnan(loss), "loss is NaN"
            return loss
        else:
            return self._compute_and_log_metrics(y_pred, y_true, "val", self.valid_metrics, attn=attn, entry_mask=entry_mask)

        # return self._compute_and_log_metrics(y_pred, y_true, 'val', self.valid_metrics, attn=attn)

    def test_step(self, batch):
        """Test step for the model."""
        local_embedding, global_embedding, y_pred, y_true, attn, entry_mask = self.module._common_step(
            batch, self.prediction_task, self.prediction_level
        )
        # Check if module supports separate loss computation (e.g., DualDecoderCombinedModule)
        if hasattr(self.module, "compute_separate_losses"):
            separate_losses = self.module.compute_separate_losses(
                self.loss, self.loss_type, y_pred, y_true, entry_mask
            )

            # Log separate losses (on_step=False, on_epoch=True to match existing pattern, sync_dist=True for test)
            if separate_losses.get("local_loss") is not None:
                self.log(
                    "test_local_loss",
                    separate_losses["local_loss"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=True,
                )
            if separate_losses.get("global_loss") is not None:
                self.log(
                    "test_global_loss",
                    separate_losses["global_loss"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=True,
                )
            if separate_losses.get("combined_loss") is not None:
                self.log(
                    "test_combined_loss",
                    separate_losses["combined_loss"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=True,
                )

            #  compute and log metrics using combined predictions
            loss = self._compute_and_log_metrics(y_pred, y_true, "test", self.test_metrics, attn=attn, entry_mask=entry_mask)

            if separate_losses.get("kl_loss") is not None:
                kl_loss = separate_losses["kl_loss"]

                # KL Annealing/Weighting (beta)
                # You can use a fixed weight or a scheduler (e.g., self.current_epoch)
                kl_weight = getattr(self.hparams, "kl_weight", 1.0)
                weighted_kl = kl_weight * kl_loss

                self.log(
                    "test_kl_loss",
                    kl_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(self.batch_size),
                    sync_dist=False,
                )

                # Add KL to the final loss to be backpropagated
                loss += weighted_kl

            assert not torch.isnan(loss), "loss is NaN"
            return loss
        else:
            return self._compute_and_log_metrics(y_pred, y_true, "test", self.test_metrics, attn=attn, entry_mask=entry_mask)
        # return self._compute_and_log_metrics(y_pred, y_true, 'test', self.test_metrics,attn=attn)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        params = []
        params.extend(filter(lambda p: p.requires_grad, self.module.parameters()))
        # if self.model.local_component is not None:
        #     params.extend(filter(lambda p: p.requires_grad, self.module.local_component.parameters()))
        # if self.model.global_component is not None:
        #     params.extend(filter(lambda p: p.requires_grad, self.model.global_component.parameters()))
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=self.patience_in_steps, verbose=True
            )
        elif self.lr_scheduler == "CosineWarmupScheduler":
            lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.lr_warmup, max_epochs=self.lr_max_epochs)
        elif self.lr_scheduler is None:
            lr_scheduler = None
        else:
            raise ValueError(
                f"Invalid lr_scheduler: {self.lr_scheduler}. Must be either 'None', 'ReduceLROnPlateau' or 'CosineWarmupScheduler'."
            )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "monitor": self.monitor_metric}]
