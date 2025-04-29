from anndata import AnnData
import scvi
from scvi.data import AnnDataManager, fields
from scvi.data._constants import (
    _ADATA_MINIFY_TYPE_UNS_KEY,
    _MODEL_NAME_KEY,
    _SCVI_UUID_KEY,
    _SETUP_ARGS_KEY,
    _SETUP_METHOD_NAME,
    ADATA_MINIFY_TYPE,
)
from abc import ABC, ABCMeta, abstractmethod
from yacs.config import CfgNode as CN
from uuid import uuid4


from typing import List, Optional, Literal, Dict, Any

import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from InterScale.nn import LinearDecoder, NonLinearDecoder
from InterScale.module.base._base_component import LocalComponent, GlobalComponent  
from InterScale.module.local_components.GCN import GCN
# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_base_model.py
# accessed on 22.April 2025
class BaseModelMetaClass(ABCMeta):
    """Metaclass for :class:`~scvi.model.base.BaseModelClass`.

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


class BaseModelClass(metaclass=BaseModelMetaClass):
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
    
    def __init__(self, 
                 adata: AnnData,
                 prediction_task: str,
                 cfg: CN,
                 ):    
        self.id = str(uuid4())  # Used for cls._manager_store keys.
        self._cfg = cfg
        
        assert prediction_task in ['classification', 'regression'], "Prediction task must be either 'classification' or 'regression'."
        self.prediction_task = prediction_task
        
        # scvi-tools like data handling
        self._adata = adata
        self._adata_manager = self._get_most_recent_anndata_manager(adata, required=True)
        self._register_manager_for_instance(self._adata_manager)
        # Suffix registry instance variable with _ to include it when saving the model.
        self.registry_ = self._adata_manager.registry
        self.summary_stats = self._adata_manager.summary_stats
        self.n_input = self.summary_stats['n_x']
        self.n_embed = self._cfg.model.n_embed
        self.n_output = self.summary_stats['n_prediction_obs'] if self.prediction_task == 'classification' else self.summary_stats['n_x']
        
        self.is_trained_ = False
        self._model_summary_string = ""
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.history_ = None
        
        self.local_component = None
        self.global_component = None
        
        # Initialize loss
        self._setup_loss(self._cfg.model.loss)
        
        decoder_type = self._cfg.model.decoder.type
        if decoder_type == 'linear':
            self.decoder = LinearDecoder(n_input = self.n_embed,
                                        n_output = self.n_output)
        elif decoder_type == 'nonlinear':
            self.decoder = NonLinearDecoder(n_input = self.n_embed,
                                           n_output = self.n_output)
        else:
            raise ValueError(f"Decoder {decoder_type} not found.")
        
    @classmethod
    def _setup_anndata(cls,
                       adata: AnnData,
                       prediction_task: str,
                       layer_key: str,
                       sample_key: str,
                       prediction_obs: str = None,
                       labels_key: str | None = None,
                       group_key: str | None = None):
        
        """
        Sets up the AnnDataManager for the model.

        Parameters
        ----------
        cls
            Class of the model. Required for class method.
        adata
            AnnData object
        layer_key
            Key in `adata.layers` that contains the data.
        prediction_obs:
            Key in `adata.obs` that contains the prediction information.
        sample_key  
            Key in `adata.obs` that contains the sample information. For example, if the data is split by FOV or sliding windows.
        labels_key
            Only required for classification. Key in `adata.obs` that contains the labels.
        group_key
            Only required if split should stratify groups of group_key, usually this should be condition. Otherwise random split.
            
        Returns
        -------
        AnnDataManager object
            AnnDataManager object that contains the data.
        """  
        
        anndata_fields = [fields.LayerField("x", layer = layer_key),
                          fields.CategoricalObsField(registry_key = 'prediction_obs', attr_key = prediction_obs),
                          fields.CategoricalObsField(registry_key = 'sample_key', attr_key = sample_key)]
        
        if labels_key is not None:
            anndata_fields.append(fields.CategoricalObsField(registry_key = 'labels_key', attr_key = labels_key))
        
        if group_key is not None:
            anndata_fields.append(fields.CategoricalObsField(registry_key = 'group_key', attr_key = group_key))    
            
        manager = scvi.data.AnnDataManager(anndata_fields)
        manager.register_fields(adata)
        manager.view_registry()
        
        # Store the manager in the class's store
        if _SCVI_UUID_KEY not in adata.uns:
            adata.uns[_SCVI_UUID_KEY] = str(id(adata))
        cls._setup_adata_manager_store[adata.uns[_SCVI_UUID_KEY]] = manager
        
    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_base_model.py
    # accessed on 22.April 2025
    @classmethod
    def _get_most_recent_anndata_manager(
        cls, adata: AnnData, required: bool = False
    ) -> AnnDataManager | None:
        """Retrieves the :class:`~scvi.data.AnnDataManager` for a given AnnData object.

        Checks for the most recent :class:`~scvi.data.AnnDataManager` created for the given AnnData
        object via ``setup_anndata()`` on model initialization. Unlike
        :meth:`scvi.model.base.BaseModelClass.get_anndata_manager`, this method is not model
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
                raise ValueError(
                    f"Please set up your AnnData with {cls.__name__}.setup_anndata first."
                )
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

    
    def _make_dataloader(self):
        return None
    
    @abstractmethod
    def train(self):
        """Trains the model.""" 
        
    @abstractmethod
    def _common_step(self,
              batch):
        """Shared step between train, val and test."""
        
    def _setup_loss(self, 
                    loss: Literal["CrossEntropy", "WeightedCE", "MSELoss", "GaussianNLL", "SmoothL1"] = None):
        """Setup loss function based on prediction task and configuration."""
        
        loss = loss if self._cfg is None else self._cfg.optim.loss
        
        if 'classification' in self.prediction_task:
            assert loss == 'CrossEntropy' or loss == 'WeightedCE', "Classification must be run with CrossEntropy or WeightedCE loss."
            if loss == 'CrossEntropy':
                self.loss = nn.CrossEntropyLoss()
            elif loss == 'WeightedCE':
                assert self.class_weights is not None, "Class weights must be provided for WeightedCE loss."
                self.loss = nn.CrossEntropyLoss(torch.from_numpy(self.class_weights))
        elif 'regression' in self.prediction_task:
            assert loss == 'MSELoss' or loss == 'GaussianNLL' or loss == 'SmoothL1', "Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss."
            if loss == 'MSELoss':
                self.loss = nn.MSELoss()
            elif loss == 'GaussianNLL':
                self.loss = nn.GaussianNLLLoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
            else:
                raise ValueError("Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss.")
        else:
            raise ValueError("Prediction task must define 'classification' or 'regression'.")
        
    def _classification_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask_idx: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate classification metrics."""
        if mask_idx is not None:
            y_pred = y_pred[mask_idx]
            y_true = y_true[mask_idx]
            
        loss = self.loss(y_pred, y_true)
        metrics = self.metrics(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        metrics['loss'] = loss
        
        return loss, metrics
        
    def _regression_metrics(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            mask_idx: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        """Calculate regression metrics."""
        if mask_idx is not None:
            y_pred = y_pred[mask_idx]
            y_true = y_true[mask_idx]
            
        loss = self.loss(y_pred, y_true)
        metrics = self.metrics(y_pred, y_true)
        metrics['loss'] = loss
        
        return loss, metrics
    
    def _register_local_component(self) -> LocalComponent:
        """Register local component based on name.
        Instance must be defined in InterScale.module.local_components."""
        
        if self._cfg.model.local_component.name == 'GCN':
            self._model_summary_string = (
                f"Local compnent {self._cfg.model.local_component.name}: "
                f"n_layers: {self._cfg.model.local_component.parameters.num_layers},"
                f"n_hidden: {self._cfg.model.local_component.parameters.hidden_dim},"
                f"n_embed: {self._cfg.model.n_embed}, "
                f"dropout_rate: {self._cfg.model.local_component.parameters.dropout}"
            )
            return GCN(n_input = self.n_input,
                       n_output = self.n_output,
                       n_embed = self._cfg.model.n_embed,
                       dropout = self._cfg.model.local_component.parameters.dropout,
                       n_layers = self._cfg.model.local_component.parameters.num_layers,
                       hidden_dim = self._cfg.model.local_component.parameters.hidden_dim)
        else:
            raise ValueError(f"Local component {self._cfg.local_component.name} not found.")
