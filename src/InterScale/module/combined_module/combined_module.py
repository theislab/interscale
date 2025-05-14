from abc import abstractmethod
import torch
from typing import Optional

from InterScale.module.base import BaseModuleClass

class CombinedModuleClass(BaseModuleClass):
    
    def __init__(self,
                 local_module_args: dict,
                 global_module_args: dict,
                 **base_module_kwargs,):
        super().__init__(**base_module_kwargs)
        
        self.local_module_args = local_module_args
        self.global_module_args = global_module_args
        
        self.registered_local_component = True
        self.registered_global_component = True
        
        self.local_module = self._register_local_module(self.local_module_args)
        self.global_module = self._register_global_module(self.global_module_args)
        
        
    def _common_step(self,
                    batch):
        """Shared step between train, val and test.
        """
        local_embedding = self.local_module.forward(batch.x, batch.edge_index)
        global_embedding = self.global_module.forward(batch.x, batch.edge_index)
        
        return None, None
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model"""
        
        