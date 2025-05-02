from InterScale.module.base._base_module import BaseModule
from abc import abstractmethod

class LocalModuleClass(BaseModule):
    def __init__(self,
                 **base_module_kwargs):
        
        super().__init__(**base_module_kwargs)
        
        self.registered_local_component = True
        self.registered_global_component = False
        
    @abstractmethod
    def forward(self):
        """Forward pass."""
    
    def get_local_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
    
class GlobalModuleClass(BaseModule):
    def __init__(self):
        super().__init__()
        
        self.registered_local_component = False
        self.registered_global_component = True
        
    @abstractmethod
    def forward(self, x, edge_index):
        return x

    def get_global_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
        