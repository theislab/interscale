import logging
import os
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import scvi
import torch
from anndata import AnnData
from scvi.data import AnnDataManager, fields
from scvi.data._constants import (
    _SCVI_UUID_KEY,
)
from scvi.data._utils import _assign_adata_uuid, _check_if_view
from sklearn.utils.class_weight import compute_class_weight
from yacs.config import CfgNode as CN

from interscale.module.base import GlobalModule, LocalModule
from interscale.module.global_modules import TransformerNodeEncoderHook
from interscale.module.local_modules import GCN
from interscale.tl.utils import get_model_filename_prefix

logger = logging.getLogger(__name__)

from typing import NamedTuple


class _SAVE_KEYS_NT(NamedTuple):
    ADATA_FNAME: str = "adata.h5ad"
    MODEL_FNAME: str = "model.pt"
    MODEL_STATE_DICT_KEY: str = "state_dict"
    VAR_NAMES_KEY: str = "var_names"
    ATTR_DICT_KEY: str = "attr_dict"


SAVE_KEYS = _SAVE_KEYS_NT()


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_base_model.py
# accessed on 22.April 2025
class BaseModelMeta(ABCMeta):
    """Metaclass for :class:`~scvi.model.base.BaseModel`.

    Constructs model class-specific mappings for :class:`~scvi.data.AnnDataManager` instances.
    ``cls._setup_adata_manager_store`` maps from AnnData object UUIDs to
    :class:`~scvi.data.AnnDataManager` instances.

    This mapping is populated everytime ``cls.setup_anndata()`` is called.
    ``cls._per_isntance_manager_store`` maps from model instance UUIDs to AnnData UUID:
    :class:`~scvi.data.AnnDataManager` mappings.
    These :class:`~scvi.data.AnnDataManager` instances are tied to a single model instance and
    populated either
    during model initialization or after running ``self._validate_anndata()``.
    """

    @abstractmethod
    def __init__(cls, name, bases, dct):
        cls._setup_adata_manager_store: dict[
            str, type[AnnDataManager]
        ] = {}  # Maps adata id to AnnDataManager instances.
        cls._per_instance_manager_store: dict[
            str, dict[str, type[AnnDataManager]]
        ] = {}  # Maps model instance id to AnnDataManager mappings.
        super().__init__(name, bases, dct)


class BaseModel(metaclass=BaseModelMeta):
    """Abstract class for InterScale models

    Parameters
    ----------
    adata
        AnnData object
    decoder
        Decoder type either linear or nonlinear
    prediction_obs
        Key in `adata.obs` that contains the prediction information.
    """

    def __init__(
        self,
        adata: AnnData,
        cfg: CN,
    ):
        self.id = str(uuid4())  # Used for cls._manager_store keys.
        self._cfg = cfg

        self.prediction_task = cfg.dataset.prediction_task
        self.prediction_level = cfg.dataset.prediction_level
        assert self.prediction_task in ["classification", "regression"], (
            "Prediction task must be either 'classification' or 'regression'."
        )
        assert self.prediction_level in ["node", "graph"], "Prediction level must be either 'node' or 'graph'."

        # scvi-tools like data handling
        self._adata = adata
        self._adata_manager = self._get_most_recent_anndata_manager(adata, required=True)
        self._register_manager_for_instance(self._adata_manager)
        # Suffix registry instance variable with _ to include it when saving the model.
        self.registry_ = self._adata_manager.registry
        self.summary_stats = self._adata_manager.summary_stats
        # TODO: check that anndata is set up in coherence with the cfg file

        self.n_input = self.summary_stats["n_x"]
        self.n_embed = self._cfg.model.n_embed
        self.n_output = (
            self.summary_stats["n_prediction_obs"]
            if self.prediction_task == "classification"
            else self.summary_stats["n_x"]
        )

        self.is_trained_ = False
        self._model_summary_string = f"{self.prediction_task} model for {self.prediction_level} prediction. \n"
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.history_ = None

        self.local_component = False
        self.global_component = False
        if self.prediction_task == "classification":
            self.class_labels = self._adata.obs[self._cfg.dataset.prediction_obs].cat.categories

        self.class_weights = None
        if self._cfg.optim.loss == "WeightedCE":
            self.class_weights = torch.tensor(
                compute_class_weight(
                    "balanced",
                    classes=np.unique(self._adata.obs[self._cfg.dataset.prediction_obs]),
                    y=self._adata.obs[self._cfg.dataset.prediction_obs],
                )
            )
            print("WeightedCE with class weights: ", self.class_weights)

    @classmethod
    def _setup_anndata(
        cls,
        adata: AnnData,
        *,
        layer_key: str,
        sample_key_list: list[str],
        prediction_task: str = "regression",
        prediction_obs: str = None,
        group_key: str | None = None,
        split_key: str | None = "split",
        view_registry: bool = True,
    ):
        """
        Sets up the AnnDataManager for the model.

        Parameters
        ----------
        cls
            Class of the model. Required for class method.
        adata
            AnnData object
        layer_key
            Key in `adata.layers` that contains the data. If None, uses `adata.X` by default.
        prediction_task
            Prediction task for the model. Either "classification" or "regression". Default is "regression".
        prediction_obs:
            Key in `adata.obs` that contains the prediction information.
        sample_key
            Key in `adata.obs` that contains the sample information. For example, if the data is split by FOV or sliding windows.
        split_key
            Key in `adata.obs` that contains the split information.
        group_key
            Key in `adata.obs` that contains the group information.
            Only required if split should stratify groups of group_key, usually this should be condition. Otherwise random split.
        """
        anndata_fields = [fields.LayerField("x", layer=layer_key)]

        for i, sample_key in enumerate(sample_key_list):
            anndata_fields.append(fields.CategoricalObsField(registry_key=f"sample_key_{i}", attr_key=sample_key))

        if prediction_task == "classification":
            anndata_fields.append(fields.CategoricalObsField(registry_key="prediction_obs", attr_key=prediction_obs))

        if split_key is not None:
            anndata_fields.append(fields.CategoricalObsField(registry_key="split_key", attr_key=split_key))
            # Check that split_key contains required values
            assert {"train", "val"}.issubset(set(adata.obs[split_key].unique())), (
                f"'{split_key}' must contain 'train' and 'val' categories"
            )

        manager = scvi.data.AnnDataManager(anndata_fields)
        manager.register_fields(adata)
        if view_registry:
            manager.view_registry()

        # Store the manager in the class's store
        if _SCVI_UUID_KEY not in adata.uns:
            adata.uns[_SCVI_UUID_KEY] = str(id(adata))
        cls._setup_adata_manager_store[adata.uns[_SCVI_UUID_KEY]] = manager
        cls.sample_key = sample_key

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_base_model.py
    # accessed on 22.April 2025
    @classmethod
    def _get_most_recent_anndata_manager(cls, adata: AnnData, required: bool = False) -> AnnDataManager | None:
        """Retrieves the :class:`~scvi.data.AnnDataManager` for a given AnnData object.

        Checks for the most recent :class:`~scvi.data.AnnDataManager` created for the given AnnData
        object via ``setup_anndata()`` on model initialization. Unlike
        :meth:`scvi.model.base.BaseModel.get_anndata_manager`, this method is not model
        instance specific and can be called before a model is fully initialized.

        Parameters
        ----------
        adata
            AnnData object to find manager instance for.
        required
            If True, errors on missing manager. Otherwise, returns None when manager is missing.
        """
        if _SCVI_UUID_KEY not in adata.uns:
            if required:
                raise ValueError(f"Please set up your AnnData with {cls.__name__}.setup_anndata first.")
            return None

        adata_id = adata.uns[_SCVI_UUID_KEY]

        if adata_id not in cls._setup_adata_manager_store:
            if required:
                raise ValueError(
                    f"Please set up your AnnData with {cls.__name__}.setup_anndata first. "
                    "It appears the AnnData object has been setup with a different model."
                )
            return None

        adata_manager = cls._setup_adata_manager_store[adata_id]
        if adata_manager.adata is not adata:
            raise ValueError(
                "The provided AnnData object does not match the AnnData object "
                "previously provided for setup. Did you make a copy?"
            )

        return adata_manager

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_base_model.py
    # accessed on 22.April 2025
    def _register_manager_for_instance(self, adata_manager: AnnDataManager):
        """Registers an :class:`~scvi.data.AnnDataManager` instance with this model instance.

        Creates a model-instance specific mapping in ``cls._per_instance_manager_store`` for this
        :class:`~scvi.data.AnnDataManager` instance.
        """
        if self.id not in self._per_instance_manager_store:
            self._per_instance_manager_store[self.id] = {}

        adata_id = adata_manager.adata_uuid
        instance_manager_store = self._per_instance_manager_store[self.id]
        instance_manager_store[adata_id] = adata_manager

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_base_model.py
    # accessed on 22.April 2025
    def get_anndata_manager(self, adata: AnnData, required: bool = False) -> AnnDataManager | None:
        """Retrieves the :class:`~scvi.data.AnnDataManager` for a given AnnData object.

        Requires ``self.id`` has been set. Checks for an :class:`~scvi.data.AnnDataManager`
        specific to this model instance.

        Parameters
        ----------
        adata
            AnnData object to find manager instance for.
        required
            If True, errors on missing manager. Otherwise, returns None when manager is missing.
        """
        cls = self.__class__
        if _SCVI_UUID_KEY not in adata.uns:
            if required:
                raise ValueError(
                    f"Please set up your AnnData with {cls.__name__}.setup_anndata'"
                    "or {cls.__name__}.setup_mudata first."
                )
            return None

        adata_id = adata.uns[_SCVI_UUID_KEY]
        if self.id not in cls._per_instance_manager_store:
            if required:
                raise AssertionError(
                    "Unable to find instance specific manager store. "
                    "The model has likely not been initialized with an AnnData object."
                )
            return None
        elif adata_id not in cls._per_instance_manager_store[self.id]:
            if required:
                raise AssertionError("Please call ``self._validate_anndata`` on this AnnData or MuData object.")
            return None

        adata_manager = cls._per_instance_manager_store[self.id][adata_id]
        if adata_manager.adata is not adata:
            logger.info("AnnData object appears to be a copy. Attempting to transfer setup.")
            _assign_adata_uuid(adata, overwrite=True)
            adata_manager = self._adata_manager.transfer_fields(adata)
            self._register_manager_for_instance(adata_manager)

        return adata_manager

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_base_model.py
    # accessed on 22.April 2025
    def _validate_anndata(self, adata: AnnData | None = None, copy_if_view: bool = True) -> AnnData:
        """Validate anndata has been properly registered, transfer if necessary."""
        if adata is None:
            adata = self._adata

        _check_if_view(adata, copy_if_view=copy_if_view)

        adata_manager = self.get_anndata_manager(adata)
        if adata_manager is None:
            logger.info("Input AnnData not setup with scvi-tools. " + "attempting to transfer AnnData setup")
            self._register_manager_for_instance(self._adata_manager.transfer_fields(adata))
        else:
            # Case where correct AnnDataManager is found, replay registration as necessary.
            adata_manager.validate()

        return adata

    def save_evaluation_results(
        self,
        adata: AnnData,
        prefix: str,
        y_pred_local_df: pd.DataFrame,
        y_pred_global_df: pd.DataFrame,
        local_embeddings_df: pd.DataFrame | None = None,
        global_embeddings_df: pd.DataFrame | None = None,
        attention_matrix_df: pd.DataFrame | None = None,
        cls_token_horizontal: np.ndarray | None = None,
        cls_token_vertical: np.ndarray | None = None,
    ):
        """Save the evaluation results in the adata object.

        Parameters
        ----------
        adata: AnnData
        prefix: str
        local_embeddings_df: pd.DataFrame
        global_embeddings_df: pd.DataFrame
        attention_matrix_df: pd.DataFrame
        y_pred_local_df: pd.DataFrame
        y_pred_global_df: pd.DataFrame
        cls_token_horizontal: np.ndarray
        cls_token_vertical: np.ndarray

        Returns
        -------
        adata: AnnData
            AnnData object with the evaluation results saved in the obsm and layers.
        """
        if local_embeddings_df is not None:
            adata.obsm[f"{prefix}_local_emb"] = local_embeddings_df.values
        if global_embeddings_df is not None:
            adata.obsm[f"{prefix}_global_emb"] = global_embeddings_df.values
        if attention_matrix_df is not None:
            adata.obsm[f"{prefix}_attn_matrix"] = attention_matrix_df.values
        if cls_token_horizontal is not None:
            adata.obs[f"{prefix}_cls_horizontal"] = cls_token_horizontal
        if cls_token_vertical is not None:
            adata.obs[f"{prefix}_cls_vertical"] = cls_token_vertical

        if self.prediction_task == "classification" and y_pred_local_df is not None:
            adata.obsm[f"{prefix}_y_pred_local"] = y_pred_local_df.values  # [cells, classes]
        elif y_pred_local_df is not None:
            adata.layers[f"{prefix}_y_pred_local"] = y_pred_local_df.values  # [cells, genes]

        if self.prediction_task == "classification" and y_pred_global_df is not None:
            adata.obsm[f"{prefix}_y_pred_global"] = y_pred_global_df.values  # [cells, classes]
        elif y_pred_global_df is not None:
            adata.layers[f"{prefix}_y_pred_global"] = y_pred_global_df.values  # [cells, genes]

        return adata

    @abstractmethod
    def train(self):
        """Trains the model."""

    def _register_local_component(self) -> LocalModule:
        """Register local component based on name.
        Instance must be defined in InterScale.module.local_components.
        """
        if self._cfg.model.local_component.name == "GCN":
            self._model_summary_string = self._model_summary_string + (
                f"Local component {self._cfg.model.local_component.name}: "
                f"n_layers: {self._cfg.model.local_component.parameters.num_layers},"
                f"n_hidden: {self._cfg.model.local_component.parameters.hidden_dim},"
                f"n_embed: {self.n_embed}, "
                f"dropout_local: {self._cfg.model.local_component.parameters.dropout_local}"
            )
            return GCN(
                n_input=self.n_input,
                n_output=self.n_output,
                n_embed=self.n_embed,
                decoder_type=self._cfg.model.decoder.type,
                dropout_decoder=self._cfg.model.decoder.dropout_decoder,
                pct_mask_nodes=self._cfg.dataset.pct_mask_nodes,
                n_layers=self._cfg.model.local_component.parameters.num_layers,
                hidden_dim=self._cfg.model.local_component.parameters.hidden_dim,
                dropout_local=self._cfg.model.local_component.parameters.dropout_local,
            )
        else:
            raise ValueError(f"Local component {self._cfg.local_component.name} not found.")

    def _register_global_component(self) -> GlobalModule:
        """Register global component based on name.
        Instance must be defined in InterScale.module.global_components.
        """
        if self._cfg.model.global_component.name == "self-attn-transformer":
            self._model_summary_string = self._model_summary_string + (
                f"Global component {self._cfg.model.global_component.name}: "
                f"max_seq_len: {self._cfg.model.global_component.parameters.max_seq_len},"
                f"n_heads: {self._cfg.model.global_component.parameters.n_heads},"
                f"dropout_global: {self._cfg.model.global_component.parameters.dropout_global},"
                f"act_func: {self._cfg.model.global_component.parameters.activation_func},"
                f"num_layers: {self._cfg.model.global_component.parameters.num_layers},"
                f"dim_feedforward: {self._cfg.model.global_component.parameters.dim_feedforward},"
                f"enforce long-range attention: {self._cfg.model.global_component.parameters.long_range_attention}"
            )
            return TransformerNodeEncoderHook(
                n_input=self.n_input,
                n_output=self.n_output,
                n_embed=self.n_embed,
                decoder_type=self._cfg.model.decoder.type,
                dropout_decoder=self._cfg.model.decoder.dropout_decoder,
                pct_mask_nodes=self._cfg.dataset.pct_mask_nodes,
                max_seq_len=self._cfg.model.global_component.parameters.max_seq_len,
                n_heads=self._cfg.model.global_component.parameters.n_heads,
                dropout_global=self._cfg.model.global_component.parameters.dropout_global,
                act_func=self._cfg.model.global_component.parameters.activation_func,
                num_layers=self._cfg.model.global_component.parameters.num_layers,
                dim_feedforward=self._cfg.model.global_component.parameters.dim_feedforward,
                long_range_attention=self._cfg.model.global_component.parameters.long_range_attention,
            )
        else:
            raise ValueError(f"Global component {self._cfg.model.global_component.name} not found.")

    def predict_nodewise(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ):
        """Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via corresponding setup
            method in model class.
        indices
            Indices of the data to predict. If None, all data is predicted.
        """
        if self.prediction_task == "regression":
            raise ValueError("Prediction task is regression. Cannot predict nodewise labels.")
        elif self.prediction_task == "classification" and self.prediction_level == "graph":
            raise ValueError("Prediction level is graph. Cannot predict nodewise labels.")

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        return None

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_base_model.py
    # accessed on 02.May 2025
    def save(
        self,
        dir_path: str | None = None,
        overwrite: bool = False,
        postfix: str | None = None,
        save_kwargs: dict | None = None,
    ):
        """Save the state of the model.
        File is saved as <dataset_name>_<prediction_task[:4]>_<prediction_level>_<local_component_name>_<global_component_name>_<model_state_dict>.pt

        Parameters
        ----------
        dir_path
            Path to a directory or cfg.model.save_path
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_kwargs
            Keyword arguments passed into :func:`~torch.save`.
        """
        import warnings

        if dir_path is None:
            dir_path = self._cfg.model.save

        # Create directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        file_name_prefix = get_model_filename_prefix(self._cfg, self.local_component, self.global_component)

        if postfix is not None:
            file_name_prefix = file_name_prefix + f"{postfix}"

        save_kwargs = save_kwargs or {}

        model_save_path = os.path.join(dir_path, f"{file_name_prefix}{SAVE_KEYS.MODEL_FNAME}")

        # Check if file exists and warn if it does
        if os.path.exists(model_save_path) and not overwrite:
            warnings.warn(
                f"File {model_save_path} already exists. Set overwrite=True to overwrite.",
                UserWarning,
                stacklevel=2,
            )
            return

        # save the model state dict and the trainer state dict only
        model_state_dict = self.module.state_dict()

        torch.save(
            {
                SAVE_KEYS.MODEL_STATE_DICT_KEY: model_state_dict,
            },
            model_save_path,
            **save_kwargs,
        )

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: AnnData,
        cfg: CN,
        model_name: str | None = None,
        local_component: bool = False,
        global_component: bool = False,
        postfix: str | None = None,
        wandb_save: bool = False,
        enable_remapping: bool = True,
    ):
        """Load a saved model.

        Parameters
        ----------
        dir_path
            Path to saved model directory.
        adata
            AnnData object to load the model with.
        cfg
            Configuration object.
        model_name: str | None
            Name of the model to load. If None, the model name is inferred from the config file.
        local_component
            Whether this is a local component model.
        global_component
            Whether this is a global component model.
        wandb_save
            Whether this was saved via wandb.
        enable_remapping
            Whether to enable automatic state dict key remapping.

        Returns
        -------
        model
            Loaded model.
        """
        if model_name is not None:
            file_name_prefix = model_name
        else:
            file_name_prefix = get_model_filename_prefix(cfg, local_component, global_component)

        if postfix is not None:
            file_name_prefix = file_name_prefix + f"{postfix}"

        model_save_path = os.path.join(dir_path, f"{file_name_prefix}{SAVE_KEYS.MODEL_FNAME}")

        # Initialize model
        model = cls(adata, cfg)

        print(f"Loading model from {model_save_path}")

        # Determine map_location based on CUDA availability
        # map_location = 'cpu' if cfg.optim.accelerator == 'cpu' else None

        # Always force CPU when CUDA is unavailable
        if torch.cuda.is_available():
            map_location = None  # load to GPU as usual
        else:
            map_location = torch.device("cpu")

        if os.path.exists(model_save_path):
            state_dict = torch.load(model_save_path, map_location=map_location)[SAVE_KEYS.MODEL_STATE_DICT_KEY]
        else:
            print("Try with .ckpt extension.")
            model_save_path = model_save_path.replace(".pt", ".ckpt")
            if os.path.exists(model_save_path):
                state_dict = torch.load(model_save_path, map_location=map_location)[SAVE_KEYS.MODEL_STATE_DICT_KEY]
            else:
                print(f"Model file {model_save_path} not found.")
                raise FileNotFoundError(f"Model file {model_save_path} not found.")

        # Apply remapping if enabled
        if enable_remapping:
            try:
                from interscale.tl.utils import detect_and_remap_state_dict_keys

                state_dict, source = detect_and_remap_state_dict_keys(state_dict)
                print(f"State dict remapping applied. Source detected: {source}")
            except ImportError:
                print("Warning: Could not import remapping functions. Loading without remapping.")

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Replace 'global_module.module.' with 'global_module.'
            new_key = k.replace("global_module.module.", "global_module.")

            # Also handle generic 'module.' prefix if present at the start (legacy DataParallel)
            if new_key.startswith("module."):
                new_key = new_key.replace("module.", "", 1)

            new_state_dict[new_key] = v
        state_dict = new_state_dict

        # Legacy wandb remapping (kept for backward compatibility)
        if wandb_save:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "", 1)  # only remove the first 'module.'
                new_state_dict[new_key] = v
            state_dict = new_state_dict

        # Load the state dict
        missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys when loading state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading state dict: {unexpected_keys}")

        model.is_trained_ = True

        return model
