# graph-transformer-long-range-niches

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/FrancescaDr/graph-transformer-long-range-niches/test.yaml?branch=main
[link-tests]: https://github.com/theislab/GT-long-range-niches/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/graph-transformer-long-range-niches

graph transformer for non-homogeneous niches at long-range prediction

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install graph-transformer-long-range-niches:

<!--
1) Install the latest release of `graph-transformer-long-range-niches` from `PyPI <https://pypi.org/project/graph-transformer-long-range-niches/>`_:

```bash
pip install graph-transformer-long-range-niches
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/GT-long-range-niches.git@main
```

## Environment installation

```bash
mamba create -n GT_long_range python=3.11
mamba activate GT_long_range
mamba install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install pytorch-lightning
pip install wandb
pip install torch-geometric
pip install -e .
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install git+ https://github.com/theislab/geome.git@main
pip install yacs
```

## Guide on Config files

Each experiment requires a `.yaml` file for the settings. Some configs are required for each experiment (e.i. `dataset.h5ad_data`) and others are set to a default value which are overwritten when included in the `experiment.yaml` file. All config parameters and default setting can be found in the `configs` folder.

Example config file: [example.yaml](/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/example.yaml)
(copy paste the file, remove all defaults )

**Config settings are based on [YACS](https://github.com/rbgirshick/yacs)*

## Workflow

1. Data preperation

We recommend using sliding patterns for the training to detect interaction patterns. 

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/FrancescaDr/graph-transformer-long-range-niches/issues
[changelog]: https://graph-transformer-long-range-niches.readthedocs.io/latest/changelog.html
[link-docs]: https://graph-transformer-long-range-niches.readthedocs.io
[link-api]: https://graph-transformer-long-range-niches.readthedocs.io/latest/api.html
