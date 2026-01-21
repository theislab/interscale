from .base._base_module import BaseModuleClass
from .local_modules import GCN, GIN
from .combined_module import CombinedModuleClass, DualDecoderCombinedModuleClass

__all__ = [
    "BaseModuleClass",
    "GCN",
    "GIN",
    "CombinedModuleClass",
    "DualDecoderCombinedModuleClass"
] 