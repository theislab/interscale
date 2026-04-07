import numpy as np
import torch
import scipy.sparse as sp
import anndata as ad

def gene_loadings(
    adata: ad.AnnData,
    model,
    layer_key: str,
    local_latent_key: str = "_local_emb",
    global_latent_key: str = "_global_emb",
    local_varm_key: str = "_local_std_gene_loadings",
    global_varm_key: str = "_global_std_gene_loadings",
    eps: float = 1e-8,
):
    """
    Compute standardized local + global gene loadings from two linear decoders.

    The model is assumed to decode log-normalized expression from two
    transformer outputs (local + global embedding):

        x_hat_local  = W_local  z_local  + b_local
        x_hat_global = W_global z_global + b_global

    Standardized loading (computed separately for local and global):
        S_gk = W_gk * std(z[k]) / std(x_g)

    Parameters
    ----------
    adata
        AnnData object
    model
        Trained model (possibly DDP-wrapped)
        Decoder weights accessed via:
          - model.module.state_dict()['local_module.decoder.decoder.weight']
          - model.module.state_dict()['global_module.decoder.decoder.weight']
    layer_key
        adata.layers[layer_key] must contain log-normalized expression
    local_latent_key
        adata.obsm key containing local transformer output embeddings
    global_latent_key
        adata.obsm key containing global transformer output embeddings
    local_varm_key
        Key to store standardized LOCAL gene loadings in adata.varm
    global_varm_key
        Key to store standardized GLOBAL gene loadings in adata.varm
    eps
        Small constant for numerical stability
    """

    # ------------------------------------------------------------
    # 1. Retrieve decoder weights W (genes x latent)
    # ------------------------------------------------------------
    core = model.module if hasattr(model, "module") else model
    sd = core.state_dict()

    if "local_module.decoder.decoder.weight" not in sd:
        raise KeyError("local_module.decoder.decoder.weight not found in model.state_dict()")
    if "global_module.decoder.decoder.weight" not in sd:
        raise KeyError("global_module.decoder.decoder.weight not found in model.state_dict()")

    W_local = sd["local_module.decoder.decoder.weight"]
    W_global = sd["global_module.decoder.decoder.weight"]

    if isinstance(W_local, torch.Tensor):
        W_local = W_local.detach().cpu().numpy()
    else:
        W_local = np.asarray(W_local)

    if isinstance(W_global, torch.Tensor):
        W_global = W_global.detach().cpu().numpy()
    else:
        W_global = np.asarray(W_global)

    n_genes_l, n_latent_l = W_local.shape
    n_genes_g, n_latent_g = W_global.shape

    if n_genes_l != n_genes_g:
        raise ValueError(
            f"Gene count mismatch between decoders: local has {n_genes_l}, global has {n_genes_g}"
        )

    # ------------------------------------------------------------
    # 2. Retrieve local + global latent embedding
    # ------------------------------------------------------------
    if local_latent_key not in adata.obsm:
        raise KeyError(f"{local_latent_key} not found in adata.obsm")
    if global_latent_key not in adata.obsm:
        raise KeyError(f"{global_latent_key} not found in adata.obsm")

    Zl = np.asarray(adata.obsm[local_latent_key], dtype=np.float32)
    Zg = np.asarray(adata.obsm[global_latent_key], dtype=np.float32)

    #z_std = Zg.std(axis=0, ddof=1)
    #z_std = np.maximum(z_std, eps)

    Zl = np.asarray(adata.obsm[local_latent_key], dtype=np.float32)
    Zg = np.asarray(adata.obsm[global_latent_key], dtype=np.float32)

    if Zl.ndim == 1:
        Zl = Zl[:, None]
    if Zg.ndim == 1:
        Zg = Zg[:, None]

    if Zl.shape[1] != n_latent_l:
        raise ValueError(
            f"Latent dimension mismatch: local decoder has {n_latent_l}, "
            f"but {local_latent_key} has {Zl.shape[1]}"
        )
    if Zg.shape[1] != n_latent_g:
        raise ValueError(
            f"Latent dimension mismatch: global decoder has {n_latent_g}, "
            f"but {global_latent_key} has {Zg.shape[1]}"
        )

    valid_cells_l = np.isfinite(Zl).all(axis=1)
    if valid_cells_l.sum() == 0:
        raise ValueError("No valid (finite) rows in local embedding")

    valid_cells_g = np.isfinite(Zg).all(axis=1)
    if valid_cells_g.sum() == 0:
        raise ValueError("No valid (finite) rows in global embedding")

    Zl_valid = Zl[valid_cells_l]
    Zg_valid = Zg[valid_cells_g]

    z_std_l = Zl_valid.std(axis=0, ddof=1)
    z_std_l = np.asarray(z_std_l, dtype=np.float32)

    z_std_l[~np.isfinite(z_std_l)] = eps
    z_std_l[z_std_l < eps] = eps

    z_std_g = Zg_valid.std(axis=0, ddof=1)
    z_std_g = np.asarray(z_std_g, dtype=np.float32)

    z_std_g[~np.isfinite(z_std_g)] = eps
    z_std_g[z_std_g < eps] = eps

    # ------------------------------------------------------------
    # 3. Compute gene-wise expression std from chosen layer
    # ------------------------------------------------------------
    if layer_key not in adata.layers:
        raise KeyError(f"{layer_key} not found in adata.layers")

    X = adata.layers[layer_key]

    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        x_std = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    else:
        x_std = X.std(axis=0, ddof=1)

    x_std = np.maximum(np.asarray(x_std).ravel(), eps)

    if len(x_std) != n_genes_l:
        raise ValueError(
            f"Gene count mismatch: decoders have {n_genes_l} genes, "
            f"expression layer has {len(x_std)}"
        )

    # ------------------------------------------------------------
    # 4. Standardize gene loadings (local + global)
    # ------------------------------------------------------------
    local_std_gene_loadings = W_local * z_std_l[None, :] / x_std[:, None]
    global_std_gene_loadings = W_global * z_std_g[None, :] / x_std[:, None]

    # ------------------------------------------------------------
    # 5. Store in AnnData
    # ------------------------------------------------------------
    adata.varm[local_varm_key] = local_std_gene_loadings
    adata.varm[global_varm_key] = global_std_gene_loadings

    return adata
