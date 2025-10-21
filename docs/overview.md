# Overview

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
     
## InterScale config

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

## Data preperation

For model training we three necessary steps to prepare the data

1. **Normalization** (we recommend log-norm to have counts in a range between 0-3.0) 
2. Calculate **spatial connectivity matrix** (with suidpy.)
3. Optional: Split into **sliding windows**. We recommend creating sliding windows when your tissue slices contain more than 4k cells. The reason for this is the context length of the transformer, for larger context lengths training still works but will take longer.

Check out [this tutorial]() for more instructions to set up and download data. 

## Model training

The model can either be trained interactively in a notebook (only recommended for small datasets) or via a script. 

In both cases the model will be saved as `.ckpt` and then loaded for the evaluation. 

## Evaluation 

@Sara add descriptions

