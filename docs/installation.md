# Installation

InterScale is available in Python >3.11. All tutorials can be run with CPU support. However, depending on the dataset size, we recommend to train InterScale models on a device with GPU support.

```
python3 -m venv ${/path/to/new/virtual/environment}
source ${/path/to/new/virtual/environment}/bin/activate
pip install uv
```

where `${/path/to/new/virtual/environment}` should be replaced with the path
where you want to install the virtual environment.

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
- [bedtools]




[Mambaforge]: https://github.com/conda-forge/miniforge
[python-venv]: https://docs.python.org/3/library/venv.html
[uv]: https://docs.astral.sh/uv/getting-started/installation
[Docker]: https://www.docker.com
[PyTorch]: http://pytorch.org
[PyTorch Scatter]: https://github.com/rusty1s/pytorch_scatter
[PyTorch Sparse]: https://github.com/rusty1s/pytorch_sparse
[bedtools]: https://bedtools.readthedocs.io
