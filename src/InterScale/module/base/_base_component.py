from InterScale.module.base._base_module import BaseModuleClass
from abc import abstractmethod

class LocalModuleClass(BaseModuleClass):
    def __init__(self,
                 **base_module_kwargs):
        
        super().__init__(**base_module_kwargs)
        
        self.registered_local_component = True
        self.registered_global_component = False
        
    @abstractmethod
    def forward(self):
        """Forward pass."""
        
    def _common_step(self,
                     batch,
                     prediction_task: str):
        """Shared step between train, val and test.
        
        Returns:
            local_embedding: torch.Tensor
            global_embedding: torch.Tensor
            y_pred: torch.Tensor
        """
        mask_idx = None
        
        local_embedding = self.forward(batch.x, batch.edge_index)
        y_pred = self.decoder.forward(local_embedding)
        
        if 'classification' in prediction_task:
            y_true = batch.y
            assert y_true.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true
            
        if 'regression' in prediction_task:
            y_true = batch.x
            assert y_true.shape == y_pred.shape
            return local_embedding, None, y_pred, y_true
            
        assert False, "Prediction task not supported"
    
    def get_local_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
    
class GlobalModuleClass(BaseModuleClass):
    def __init__(self,
                 **base_module_kwargs):
        
        super().__init__(**base_module_kwargs)
        
        self.registered_local_component = False
        self.registered_global_component = True
        
    @abstractmethod
    def forward(self, x, edge_index):
        return x

    def get_global_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
        