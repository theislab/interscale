# InterScale

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/FrancescaDr/graph-transformer-long-range-niches/test.yaml?branch=main
[link-tests]: https://github.com/theislab/GT-long-range-niches/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/graph-transformer-long-range-niches

InterScale is a computational model for analysis of intercellular interactions in spatial transcriptomcis across different length-scales. It consists of a workflow that generates  per cell, cell-based attention matrix and several evaluation functions for tissue, cell and gene level communication. 

InterScale folder structure: 

```python
/
└── InterScale/
    └── config/ 
    └── eval/
    └── model/
    └── module/
    └── nn/
    └── tl/
    └── train/
└── config_files/
```

<!--
## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].



1) Install the latest release of `graph-transformer-long-range-niches` from `PyPI <https://pypi.org/project/graph-transformer-long-range-niches/>`_:

```bash
pip install graph-transformer-long-range-niches
```
-->


## Environment installation

### With GPU support

#### Conda/Mamba set up

```bash
mamba create -n GT_long_range python=3.11
mamba activate GT_long_range
mamba install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install pytorch-lightning
pip install wandb
pip install torch-geometric
pip install -e .
pip install git+ https://github.com/theislab/geome.git@main
pip install yacs
```

#### Enroot container set up

Use Nvidia NGC container: [PyG Release 24.09](https://docs.nvidia.com/deeplearning/frameworks/pyg-release-notes/rel-24-09.html#rel-24-09l) with `python=3.10`, `numpy=1.24.4`, `torch=2.5.0.a` and `CUDA=12.6`.

1. Import NVIDIA base container

`enroot import docker://nvcr.io/nvidia/pyg:24.09-py3`

2. Create enroot container and start

```
enroot create --name interscale_container nvidia+pyg+24.09-py3.sqsh
enroot start interscale_container
```

3. Install all dependencies/packages

```bash
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu126.html
pip install pytorch-lightning wandb torch-geometric yacs
pip install git+https://github.com/theislab/geome.git@main 
pip install -e /dss/dsshome1/05/di93tig/1_projects/GT-long-range-niches 
pip install scvi-tools
```

Optional: set up CUDA environment variables

```
echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" >> /etc/environment
echo "NVIDIA_REQUIRE_CUDA=cuda>=12.1" >> /etc/environment
echo "NVIDIA_VISIBLE_DEVICES=all" >> /etc/environment
```

`exit` enroot container

4. Export container for re-use

```
enroot export --output InterScale.sqsh interscale_container
```

## Workflow

There are three stages for InterScale:

1. Config set up and data preparation
2. Model set-up and training
3. Evaluation

### 1. Config set up and data preperation

#### 1.1 InterScale config

The default config settings can be observed in:

``` python
/
└── InterScale/
    └── config/ # default config setttings
        └── dataset_config/ # 
        └── global_component_config/ # 
        └── local_component_config/ # 
        └── model_config/ # 
        └── optim_config/ # training optimization parameters (e.i. learning rate, weight decay,...)
        └── wandb_config/ # 
```

Some parameters can not be loaded as default such as path to h5ad object, results directory etc. An example of a config file with the necessary parameters to set can be found [here](./../src/config_files/InterScale_example.yaml). By default the model is trained for a node regression tasks, meaning prediction of GEX values, with `adata.X`. 

You can customize the model by inluding other parameter from the config folder files. If you set them in your `.yaml` file you will overwrite the default values. 

#### 1.2 Data preperation

For model training we three necessary steps to prepare the data

1. **Normalization** (we recommend log-norm to have counts in a range between 0-3.0) 
2. Calculate **spatial connectivity matrix** (with suidpy.)
3. Optional: Split into **sliding windows**. We recommend creating sliding windows when your tissue slices contain more than 4k cells. The reason for this is the context length of the transformer, for larger context lengths training still works but will take longer.

Check out [this tutorial]() for more instructions to set up and download data. 

#### 2. Model training

The model can either be trained interactively in a notebook (only recommended for small datasets) or via a script. 

In both cases the model will be saved as `.ckpt` and then loaded for the evaluation. 
                                    

#### 3.  Evaluation

@Sara add descriptions

| Evaluation level | Function name | Description |
| ---------------- | ------------- | ----------- |
| Tissue level       | | | 
| Cell level       | | | 
| Gene level | `gene_ranke_analysis` | Requires training on regression task, gene prediction. Ranks which genes from the local vs global model are well predicted. | 

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
