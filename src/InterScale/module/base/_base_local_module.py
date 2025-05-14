from InterScale.module.base._base_module import BaseModuleClass
from abc import abstractmethod
from typing import Literal
from InterScale.tl import apply_mask

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
                     prediction_task: str,
                     prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.
        
        Returns
        -------
        local_embedding: torch.Tensor 
            Size: [N, E]
        global_embedding: torch.Tensor 
            Size: [N, E]
        y_pred: torch.Tensor 
            Size: [N, C] (classification) or [N, F] (regression)
        y_true: torch.Tensor 
            Size: [N, ] (classification) or [N, F] (regression)
        """
        # Mask nodes 
        if self.pct_mask_nodes > 0:
            input_data_masked, mask_idx = apply_mask(batch)
        else:
            mask_idx = None
            
        if mask_idx is None or len(mask_idx) == 0:
            print('No mask_idx provided, using all data')
            input_data_masked = batch
        
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
