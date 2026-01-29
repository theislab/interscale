## Config.yaml

Some more explanation about how to set your `config.yaml` file.

### cfg.model

The `cfg.model` contains all necessary parameters for models.

For precomputed models: Set `cfg.model.global_component.parameters.type_gex_embedding: Precomputed`, then indicate which `adata.obsm['X_scVI']` key should be used `cfg.model.global_component.latent_obsm_key: X_scVI` and change the number of latent embeddings `cfg.model.n_embed` to match the embeddings of the precomputed embeddings in `adata.obsm['X_scVI'].

Note: `cfg.model.n_embed` must be dividable by `cfg.model.global_components.n_heads` .
