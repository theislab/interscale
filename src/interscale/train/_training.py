import math

import lightning as L
import lightning.pytorch as pl
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import seed_everything

from interscale.tl.utils import get_model_filename_prefix
from interscale.train._trainingplans import TrainingPlan
from interscale.train._utils import MetricsHistory

# from interscale.model.base._trainer import TrainRunner


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_training_mixin.py#L31
# accessed on 24 April 2025
class NodeMaskingTrainingPlan:
    """General purpose class for training plans that mask nodes and predict masked nodes."""

    # _data_splitter_cls = DataSplitter
    _training_plan_cls = TrainingPlan

    # @devices_dsp.dedent -TODO: Why is this here in scvi-tools?
    def train(
        self,
        max_epochs: int,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        early_stopping: bool = True,
        patience: int = 5,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        datamodule: L.LightningDataModule | None = None,
        wandb_use: bool | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            The maximum number of epochs to train the model. The actual number of
            epochs may be less if early stopping is enabled.
        shuffle_set_split
            Whether to shuffle indices before splitting. If ``False``, the val,
            train, and test set are split in the sequential order of the data.
        load_sparse_tensor
            Whether to load data as sparse tensors.
        early_stopping
            Perform early stopping. Additional arguments can be passed in through
            ``**trainer_kwargs``.
        patience
            Patience for early stopping: number of epochs to wait for improvement
            before stopping.
        datasplitter_kwargs
            Additional keyword arguments passed into the data splitter. Not used
            if ``datamodule`` is passed in.
        plan_kwargs
            Additional keyword arguments passed into the training plan.
        datamodule
            A :class:`~lightning.pytorch.core.LightningDataModule` instance to use
            for training.
        wandb_use
            Whether to log to Weights & Biases. Defaults to the project config.
        **trainer_kwargs
            Additional keyword arguments passed into the
            :class:`~lightning.pytorch.trainer.trainer.Trainer`.
        """
        # if datamodule is not None and not self._module_init_on_train:
        #     raise ValueError(
        #         "Cannot pass in `datamodule` if the model was initialized with `adata`."
        #     )
        # elif datamodule is None and self._module_init_on_train:
        #     raise ValueError(
        #         "If the model was not initialized with `adata`, a `datamodule` must be passed in."
        #     )

        # if datamodule is None:
        #     datasplitter_kwargs = datasplitter_kwargs or {}
        #     datamodule = self._data_splitter_cls(
        #         self.adata_manager,
        #         train_size=train_size,
        #         validation_size=validation_size,
        #         batch_size=batch_size,
        #         shuffle_set_split=shuffle_set_split,
        #         **datasplitter_kwargs,
        #     )

        plan_kwargs = plan_kwargs or {}

        seed_everything(self._cfg.optim.seed, workers=True)

        self.wandb_use = wandb_use if wandb_use is not None else self._cfg.wandb.use
        self.batch_size = self._cfg.dataset.batch_size
        self.train_size = self._cfg.dataset.train_size
        self.validation_size = self._cfg.dataset.val_size

        # TODO: change steps per epoch to be based on datamodule
        # steps_per_epoch = math.ceil(len(train_ds) / cfg.dataset.batch_size)
        steps_per_epoch = math.ceil(len(datamodule.train_data) / self.batch_size)
        print("Steps per epoch", steps_per_epoch)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        self.history_ = MetricsHistory()
        checkpoint_callback = None
        loss_callback = None
        performance_callback = None

        # defines optimizers, training step, val step, logged metrics
        training_plan = self._training_plan_cls(
            self.module,
            self.prediction_task,
            self.prediction_level,
            self._cfg.optim.loss,
            self._cfg.optim.cross_corr,
            self.batch_size,
            self.class_weights,
            self.class_labels if self.prediction_task == "classification" else None,
            **plan_kwargs,
            lr_scheduler=self._cfg.optim.lr_scheduler,
            weight_decay=self._cfg.optim.wd,
            lr=self._cfg.optim.lr,
            lr_warmup=self._cfg.optim.lr_warmup,
            lr_max_epochs=self._cfg.optim.n_epochs,
            patience_in_steps=steps_per_epoch,
        )

        if early_stopping:
            # TODO: why does the self.history_ stop working when using loss_callback?
            # loss_callback = EarlyStopping(
            #         monitor="val_loss",
            #         min_delta=0.001,
            #         patience=(patience//2)*steps_per_epoch,  # Often shorter patience for loss
            #         verbose=False,
            #         mode="min"
            #     )
            if "classification" in self.prediction_task:
                performance_callback = EarlyStopping(
                    monitor="val_loss", min_delta=0.005, patience=patience, verbose=False, mode="min"
                )
            elif "regression" in self.prediction_task:
                performance_callback = EarlyStopping(
                    monitor="val_loss", min_delta=0.005, patience=patience, verbose=False, mode="min"
                )
            else:
                raise Exception("Training must be classification or regression based.")

        if self._cfg.model.save is not None:
            run_name = get_model_filename_prefix(self._cfg, self.local_component, self.global_component)
            if "classification" in self._cfg.dataset.prediction_task:
                checkpoint_callback = ModelCheckpoint(
                    dirpath=self._cfg.model.save,
                    filename=run_name,
                    monitor="val_loss",
                    mode="min",
                )  # save model if validation accuracy increases
            elif "regression" in self._cfg.dataset.prediction_task:
                if self._cfg.optim.loss == "MSELoss":
                    checkpoint_callback = ModelCheckpoint(
                        dirpath=self._cfg.model.save,
                        filename=run_name,
                        monitor="val_loss",
                        mode="min",
                    )
                elif (
                    self._cfg.optim.loss == "GaussianNLL"
                    or self._cfg.optim.loss == "SmoothL1"
                    or self._cfg.optim.loss == "BalancedPearsonCorrelationLoss"
                    or self._cfg.optim.loss == "SCELoss"
                    or self._cfg.optim.loss == "SCE_EntropyATT_Loss"
                ):
                    checkpoint_callback = ModelCheckpoint(
                        dirpath=self._cfg.model.save,
                        filename=run_name,
                        monitor="val_loss",
                        mode="min",
                    )
                else:
                    raise Exception(
                        f"Regression must be run with MSELoss, GaussianNLL, SmoothL1, BalancedPearsonCorrelationLoss or SCELoss loss. instead of {self._cfg.optim.loss}"
                    )

        # Create list of callbacks and filter out None values
        callbacks = [
            callback
            for callback in [lr_monitor, performance_callback, loss_callback, self.history_, checkpoint_callback]
            if callback is not None
        ]

        # Set up WandB logger if requested
        logger = None
        if self.wandb_use:
            print("Wandb initialize...")
            run_name = get_model_filename_prefix(self._cfg, self.local_component, self.global_component)
            if self._cfg.wandb.project_name is None:
                raise ValueError("wandb_project must be specified when use_wandb is True")
            wandb.init(project=self._cfg.wandb.project_name, config=self._cfg, name=run_name, job_type="model_training")
            logger = WandbLogger(name=run_name, log_model=True)
            total_params = sum(p.numel() for p in self.module.parameters())
            trainable_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
            wandb.log({"total_parameters": total_params, "trainable_parameters": trainable_params})
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

        trainer = pl.Trainer(
            min_epochs=1,
            max_epochs=int(max_epochs),
            # enable_progress_bar=True,
            callbacks=callbacks,
            log_every_n_steps=1,
            logger=logger,
            deterministic=True,  # ensure reproducibility
            accelerator=self._cfg.optim.accelerator,  # Default to CPU (can be overridden via trainer_kwargs)
            **trainer_kwargs,
        )

        trainer.fit(training_plan, datamodule)
        trainer.validate(training_plan, datamodule)
        if self.train_size + self.validation_size < 1:
            trainer.test(training_plan, datamodule)

        # Print early stopping information if it was used
        if early_stopping and loss_callback is not None:
            if loss_callback.stopped_epoch > 0:
                print(f"\nEarly stopping triggered at epoch {loss_callback.stopped_epoch}")
                print(f"Best {loss_callback.monitor}: {loss_callback.best_score:.4f}")
        if early_stopping and performance_callback is not None:
            if performance_callback.stopped_epoch > 0:
                print(f"\nEarly stopping triggered at epoch {performance_callback.stopped_epoch}")
                print(f"Best {performance_callback.monitor}: {performance_callback.best_score:.4f}")

        if self._cfg.model.save is not None:
            print("Model checkpoint will be saved in: ", self._cfg.model.save + run_name + "model.ckpt")
            trainer.save_checkpoint(self._cfg.model.save + run_name + "model.ckpt")
            self.save(self._cfg.model.save)

        # Close WandB logger if it was used
        if self.wandb_use and logger is not None:
            logger.finalize("success")

        self.is_trained_ = True


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/dataloaders/_data_splitting.py#L182
# accessed on 24 April 2025
# class DataSplitter(pl.LightningDataModule):
#     """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

#     If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

#     Parameters
#     ----------
#     adata_manager
#         :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
#     train_size
#         float, or None (default is None, which is practicaly 0.9 and potentially adding small last
#         batch to validation cells)
#     validation_size
#         float, or None (default is None)
#     shuffle_set_split
#         Whether to shuffle indices before splitting. If `False`, the val, train, and test set are
#         split in the sequential order of the data according to `validation_size` and `train_size`
#         percentages.
#     external_indexing
#         A list of data split indices in the order of training, validation, and test sets.
#         Validation and test set are not required and can be left empty.
#     **kwargs
#         Keyword args for data loader. If adata has labeled data, data loader
#         class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
#         else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

#     Examples
#     --------
#     >>> adata = scvi.data.synthetic_iid()
#     >>> scvi.model.SCVI.setup_anndata(adata)
#     >>> adata_manager = scvi.model.SCVI(adata).adata_manager
#     >>> splitter = DataSplitter(adata)
#     >>> splitter.setup()
#     >>> train_dl = splitter.train_dataloader()
#     """

#     data_loader_cls = AnnDataLoader #TODO: change to InterScale.dataloader.GeomeDataLoader

#     def __init__(
#         self,
#         adata_manager: AnnDataManager,
#         train_size: float | None = None,
#         validation_size: float | None = None,
#         shuffle_set_split: bool = True,
#         external_indexing: list[np.array, np.array, np.array] | None = None,
#         **kwargs,
#     ):
#         super().__init__()
#         self.adata_manager = adata_manager
#         self.train_size_is_none = not bool(train_size)
#         self.train_size = 0.9 if self.train_size_is_none else float(train_size)
#         self.validation_size = validation_size
#         self.shuffle_set_split = shuffle_set_split
#         self.drop_last = kwargs.pop("drop_last", False)
#         self.data_loader_kwargs = kwargs
#         self.external_indexing = external_indexing

#         if self.external_indexing is not None:
#             self.n_train, self.n_val = validate_data_split_with_external_indexing(
#                 self.adata_manager.adata.n_obs,
#                 self.external_indexing,
#                 self.data_loader_kwargs.get("batch_size", settings.batch_size),
#                 self.drop_last,
#             )
#         else:
#             self.n_train, self.n_val = validate_data_split(
#                 self.adata_manager.adata.n_obs,
#                 self.train_size,
#                 self.validation_size,
#                 self.data_loader_kwargs.get("batch_size", settings.batch_size),
#                 self.drop_last,
#                 self.train_size_is_none,
#             )

#     def setup(self, stage: str | None = None):
#         """Split indices in train/test/val sets."""
#         if self.external_indexing is not None:
#             # The structure and its order are guaranteed at this stage
#             # (can include missing indexes for some group)
#             self.train_idx = self.external_indexing[0]
#             self.val_idx = self.external_indexing[1]
#             self.test_idx = self.external_indexing[2]
#         else:
#             # just like it used to be w/o external indexing
#             n_train = self.n_train
#             n_val = self.n_val
#             indices = np.arange(self.adata_manager.adata.n_obs)

#             if self.shuffle_set_split:
#                 random_state = np.random.RandomState(seed=settings.seed)
#                 indices = random_state.permutation(indices)

#             self.val_idx = indices[:n_val]
#             self.train_idx = indices[n_val : (n_val + n_train)]
#             self.test_idx = indices[(n_val + n_train) :]

#     def train_dataloader(self):
#         """Create train data loader."""
#         return self.data_loader_cls(
#             self.adata_manager,
#             indices=self.train_idx,
#             shuffle=True,
#             drop_last=self.drop_last,
#             load_sparse_tensor=self.load_sparse_tensor,
#             pin_memory=self.pin_memory,
#             **self.data_loader_kwargs,
#         )

#     def val_dataloader(self):
#         """Create validation data loader."""
#         if len(self.val_idx) > 0:
#             return self.data_loader_cls(
#                 self.adata_manager,
#                 indices=self.val_idx,
#                 shuffle=False,
#                 drop_last=False,
#                 load_sparse_tensor=self.load_sparse_tensor,
#                 pin_memory=self.pin_memory,
#                 **self.data_loader_kwargs,
#             )
#         else:
#             pass

#     def test_dataloader(self):
#         """Create test data loader."""
#         if len(self.test_idx) > 0:
#             return self.data_loader_cls(
#                 self.adata_manager,
#                 indices=self.test_idx,
#                 shuffle=False,
#                 drop_last=False,
#                 load_sparse_tensor=self.load_sparse_tensor,
#                 pin_memory=self.pin_memory,
#                 **self.data_loader_kwargs,
#             )
#         else:
#             pass
