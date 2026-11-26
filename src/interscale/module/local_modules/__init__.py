from .GAT import GAT
from .GCN import GCN
from .GIN import GIN
from .Precomputed import PrecomputedEmbeddingModule
from .SCVI import SCVILocalModule

__all__ = ["GCN", "GIN", "GAT", "SCVILocalModule", "PrecomputedEmbeddingModule"]
