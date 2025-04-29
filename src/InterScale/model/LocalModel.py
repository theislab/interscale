from InterScale.model.base._base_model import BaseModelClass
from InterScale.train._training import NodeMaskingTrainingPlan
from InterScale.module.base import LocalComponent
from anndata import AnnData
from yacs.config import CfgNode as CN


class LocalModel(NodeMaskingTrainingPlan,
                 BaseModelClass):
    
    def __init__(self, 
                 adata: AnnData,
                 prediction_task: str,
                 cfg: CN,):
        super().__init__(adata, prediction_task, cfg)
        
        self.local_component = self._register_local_component()
        
    def _common_step(self,
                     batch):
        """Shared step between train, val and test.
        
        Returns:
            local_embedding: torch.Tensor
            global_embedding: torch.Tensor
            y_pred: torch.Tensor
        """
        mask_idx = None
        
        local_embedding = self.local_component.forward(batch.x, batch.edge_index)
        y_pred = self.decoder.forward(local_embedding)
        
        if 'classification' in self.prediction_task:
            y_true = batch.y
            assert y_true.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true
            
        if 'regression' in self.prediction_task:
            y_true = batch.x
            assert y_true.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true
            
        assert False, "Prediction task not supported"
            
    
        
        
        
       