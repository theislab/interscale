from importlib.metadata import PackageNotFoundError, version

from . import config, model, module, tl

__all__ = ["config", "module", "tl", "model"]

try:
    __version__ = version("interscale")
except PackageNotFoundError:
    pass
