# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import shutil
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from sphinxcontrib import katex

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("InterScale")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "theislab",
    "github_repo": "interscale",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_inherit_docstrings = False
autodoc_type_aliases = {
    "AnnData": "anndata.AnnData",
    "pd.DataFrame": "pandas.DataFrame",
    "np.ndarray": "numpy.ndarray",
}
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "scvi": ("https://docs.scvi-tools.org/en/stable/", None),
    # "torch": ("https://pytorch.org/docs/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/img/InterScale_logo.png"

html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"
katex_prerender = shutil.which(katex.NODEJS_BINARY) is not None

nitpick_ignore = [
    ("py:class", "yacs.config.CfgNode"),
    ("py:class", "optional"),
    # Type aliases used in docstring text that napoleon converts to cross-references;
    # these can’t be resolved by intersphinx since the inventory uses fully qualified names.
    ("py:class", "AnnData"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "np.ndarray"),
    # Undefined cross-doc labels in torch’s own documentation
    ("ref", "locally-disable-grad-doc"),
    ("ref", "nn-init-doc"),
]
# Suppress all cross-reference warnings from third-party inherited docstrings
# (torch, pytorch_lightning, lightning_fabric) — these are upstream issues.
nitpick_ignore_regex = [
    (r"py:.*", r"pytorch_lightning\..*"),
    (r"py:.*", r"LightningModule"),
    (r"py:.*", r"torch\..*"),
    (r"py:.*", r"torch"),
    (r"py:.*", r"lightning_fabric\..*"),
    (r"py:.*", r"pandas\.core\..*"),
    # Bare names from inherited torch.nn.Module / Lightning docstrings
    (r"py:class", r"Module"),
    (r"py:class", r"Dropout"),
    (r"py:class", r"BatchNorm"),
    (r"py:class", r"torchmetrics\.Metric"),
    (r"py:meth", r"nn\.Module\.load_state_dict"),
    (r"py:meth", r"forward"),
    (r"py:meth", r"training_step"),
    (r"py:meth", r"toggle_optimizer"),
    (r"py:meth", r"save_hyperparameters"),
    (r"py:meth", r"move_data_to_device"),
    (r"py:meth", r"apply_to_collection"),
    (r"py:func", r"add_module"),
    (r"py:func", r"register_module_.*"),
    (r"py:attr", r"state_dict"),
    (r"py:attr", r"strict"),
    (r"py:attr", r"assign"),
    (r"py:attr", r"persistent"),
    (r"py:attr", r"requires_grad"),
    (r"py:attr", r"grad_input"),
    (r"py:attr", r"grad_output"),
    (r"py:attr", r"checkpoint_path"),
]
suppress_warnings = ["docutils", "intersphinx"]
