from anndata import AnnData
import scvi
from scvi.data import AnnDataManager, fields
from abc import ABC, abstractmethod
from InterScale.nn import LinearDecoder, NonLinearDecoder

class BaseModel(ABC):
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
                 adata: AnnData):    
        
        self._adata = adata
        
        self.is_trained_ = False
        self._model_summary_string = ""
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.history_ = None
        
        self.local_component = None
        self.global_component = None
        
        # # TODO: before I load the data I don't know the dimensions of the input and output
        # if decoder == 'linear':
        #     self.decoder = LinearDecoder(n_input = None,
        #                                 n_output = None)
        # elif decoder == 'nonlinear':
        #     self.decoder = NonLinearDecoder(n_input = None,
        #                                    n_output = None)
        
    def _setup_anndata(self,
                       adata: AnnData,
                       layer_key: str,
                       sample_key: str,
                       prediction_obs: str = None,
                       labels_key: str | None = None,
                       group_key: str | None = None):
        
        """
        Sets up the AnnDataManager for the model.

        Parameters
        ----------
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
        
        anndata_fields = [fields.LayerField("x", layer = None),
                          fields.CategoricalObsField(registry_key = 'prediction_obs', attr_key = prediction_obs),
                          fields.CategoricalObsField(registry_key = 'sample_key', attr_key = sample_key)]
        
        if labels_key is not None:
            anndata_fields.append(fields.CategoricalObsField(registry_key = 'labels_key', attr_key = labels_key))
        
        if group_key is not None:
            anndata_fields.append(fields.CategoricalObsField(registry_key = 'group_key', attr_key = group_key))    
            
        manager = scvi.data.AnnDataManager(anndata_fields)
        manager.register_fields(self._adata,
                               layer_key = layer_key,
                               sample_key = sample_key,
                               labels_key = labels_key,
                               group_key = group_key)
        
    
    def _make_dataloader(self):
        return None
    
    def training_step(self,
                      batch,
                      batch_idx):
        """Training step for the module."""
        return None
    
    def validation_step(self,
                        batch,
                        batch_idx):
        """Validation step for the module."""
        return None 
    
    def test_step(self,
                  batch,
                  batch_idx):
        """Test step for the module."""
        return None
    
    @abstractmethod
    def train(self):
        """Trains the model."""