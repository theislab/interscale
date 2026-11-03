# Installation

InterScale is available in Python >=3.12. All tutorials can be run with CPU support. However, depending on the dataset size, we recommend to train InterScale models on a device with GPU support.

For the fastest installation experience for unimodal training, use the [uv] package manager within a [python-venv] environment. For example, run:

```
python3 -m venv ${/path/to/new/virtual/environment}
source ${/path/to/new/virtual/environment}/bin/activate
pip install uv
```

where `${/path/to/new/virtual/environment}` should be replaced with the path where you want to install the virtual environment.

## PyPi

Install InterScale via pip:
```
uv pip install interscale
```

## Docker container

The CPU supported Docker container can be found here: [francescadr/interscale](https://hub.docker.com/repository/docker/francescadr/interscale/general).

The [Docker] container was set up with [viash].

## Additional Libraries

To use InterScale, you first need to install some external libraries. These
include:
- [PyTorch]
- [PyTorch Scatter]
- [PyTorch Sparse]
- [PyTorch Cluster]
- [geome]

Install all additional dependencies by:

```
# Create a virtual environment with Python 3.13
uv venv .interscale --python 3.13
source .interscale/bin/activate

# Install PyTorch (CPU/MPS build for macOS)
uv pip install torch torchvision torchaudio

# PyG extensions — [geome] still requires them, and PyPI carries only source
# distributions that have to be compiled against your torch. Take the prebuilt
# wheels matching the torch you just installed instead; swap `+cpu` for your
# CUDA build (e.g. `+cu128`) on a GPU machine.
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
uv pip install torch-scatter torch-sparse torch-cluster \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"

# Core dependencies
uv pip install torch-geometric pytorch-lightning wandb yacs scvi-tools
uv pip install geome
```


[Mambaforge]: https://github.com/conda-forge/miniforge
[python-venv]: https://docs.python.org/3/library/venv.html
[uv]: https://docs.astral.sh/uv/getting-started/installation
[Docker]: https://www.docker.com
[PyTorch]: http://pytorch.org
[PyTorch Scatter]: https://github.com/rusty1s/pytorch_scatter
[PyTorch Sparse]: https://github.com/rusty1s/pytorch_sparse
[PyTorch Cluster]: https://github.com/rusty1s/pytorch_cluster
[geome]: https://github.com/theislab/geome
[viash]: https://viash.io/
