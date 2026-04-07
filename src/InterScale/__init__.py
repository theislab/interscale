
from importlib.metadata import version, PackageNotFoundError

from . import config, module, tl, model

__all__ = ["config", "module", "tl", "model"]

try:
    __version__ = version("interscale")
except PackageNotFoundError:
    pass