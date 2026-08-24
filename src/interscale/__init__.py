from importlib.metadata import PackageNotFoundError, version

from . import config, datasets, evaluation, model, module, tl, pl

__all__ = ["config", "datasets", "evaluation", "module", "tl", "model", "pl"]

try:
    __version__ = version("interscale")
except PackageNotFoundError:
    pass
