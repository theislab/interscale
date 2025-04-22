from InterScale.model.base._base_model import BaseModel
from InterScale.module.local_components.GCN import GCN
from anndata import AnnData
from yacs.config import CfgNode as CN

class LocalModel(BaseModel):
    def __init__(self, 
                 adata: AnnData,    
                 cfg: CN,
                 local_component_name: str = 'GCN'):
        super().__init__(adata, cfg)
        
        if local_component_name == 'GCN':
            self.local_component = self._register_local_component(local_component_name)

    
    def _common_step(self,
                     batch):
        """Shared step between train, val and test."""
        mask_idx = None
        
        local_embedding = self.local_component.forward(batch)
        
        y_true = batch.x
        
        if 'classification' in self.prediction_task:
            loss, metrics = self._classification_metrics(local_embedding, y_true, mask_idx)
            
        if 'regression' in self.prediction_task:
            loss, metrics = self._regression_metrics(local_embedding, y_true, mask_idx)
            
    def train(self):
        pass
    
    def validation(self):
        pass
    
    def test(self):
        pass
        
        
        
       