# Overview

InterScale is a computational model for analysis of intercellular interactions in spatial transcriptomcis across different length-scales. It consists of a workflow that generates  per cell, cell-based attention matrix and several evaluation functions for tissue, cell and gene level communication. 

InterScale folder structure: 

```python
/
└── InterScale/
    └── config/ 
    └── eval/
    └── model/
    └── xenium_human/module/
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

Some parameters can not be loaded as default such as path to h5ad object, results directory etc. An example of a config file with the necessary parameters to set can be found [here](). Fill out the necessary (and potentially indicate more custom settings) to train your own model. 

Check out [this tutorial]() for more instructions to set up and download data. 


