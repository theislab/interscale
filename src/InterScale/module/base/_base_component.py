from InterScale.module.base._base_module import BaseModule
from abc import abstractmethod

class LocalComponent(BaseModule):
    def __init__(self):
        super().__init__()
        
        self.local_component = True
        self.global_component = False
        
    @abstractmethod
    def forward(self, x, edge_index):
        return x
    
    def get_local_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
    
class GlobalComponent(BaseModule):
    def __init__(self):
        super().__init__()
        
        self.local_component = False
        self.global_component = True
        
    @abstractmethod
    def forward(self, x, edge_index):
        return x

    def get_global_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)
        