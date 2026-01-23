import numpy as np
import torch
import scipy.sparse as sp
import anndata as ad


def gene_loadings(
    adata: ad.AnnData,
    model,
    layer_key: str,
    global_latent_key: str = "_global_emb",
    varm_key: str = "_std_gene_loadings",
    eps: float = 1e-8,
):
    """
    Compute standardized global gene loadings from a linear decoder.

    The model is assumed to decode log-normalized expression from a
    transformer output (global embedding):

        x_hat = W z_global + b

    Standardized loading:
        S_gk = W_gk * std(z_global[k]) / std(x_g)

    Parameters
    ----------
    adata
        AnnData object
    model
        Trained model (possibly DDP-wrapped)
        Decoder weights accessed as model.module.decoder.decoder.weight
    layer_key
        adata.layers[layer_key] must contain log-normalized expression
    global_latent_key
        adata.obsm key containing transformer output embeddings
    varm_key
        Key to store standardized gene loadings in adata.varm
    eps
        Small constant for numerical stability
    """

    # ------------------------------------------------------------
    # 1. Retrieve decoder weights W (genes x latent)
    # ------------------------------------------------------------
    core = model.module if hasattr(model, "module") else model
    W = core.decoder.decoder.weight

    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()

    n_genes, n_latent = W.shape

    # ------------------------------------------------------------
    # 2. Retrieve global latent embedding
    # ------------------------------------------------------------
    if global_latent_key not in adata.obsm:
        raise KeyError(f"{global_latent_key} not found in adata.obsm")

    Zg = np.asarray(adata.obsm[global_latent_key], dtype=np.float32)

    if Zg.shape[1] != n_latent:
        raise ValueError(
            f"Latent dimension mismatch: decoder has {n_latent}, "
            f"but {global_latent_key} has {Zg.shape[1]}"
        )

    #z_std = Zg.std(axis=0, ddof=1)
    #z_std = np.maximum(z_std, eps)

    Zg = np.asarray(adata.obsm[global_latent_key], dtype=np.float32)

    valid_cells = np.isfinite(Zg).all(axis=1)
    if valid_cells.sum() == 0:
        raise ValueError("No valid (finite) rows in global embedding")

    Zg_valid = Zg[valid_cells]

    z_std = Zg_valid.std(axis=0, ddof=1)
    z_std = np.asarray(z_std, dtype=np.float32)

    z_std[~np.isfinite(z_std)] = eps
    z_std[z_std < eps] = eps
    

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

    if len(x_std) != n_genes:
        raise ValueError(
            f"Gene count mismatch: decoder has {n_genes} genes, "
            f"expression layer has {len(x_std)}"
        )

    # ------------------------------------------------------------
    # 4. Standardize gene loadings
    # ------------------------------------------------------------
    std_gene_loadings = W * z_std[None, :] / x_std[:, None]

    # ------------------------------------------------------------
    # 5. Store in AnnData
    # ------------------------------------------------------------
    adata.varm[varm_key] = std_gene_loadings

    return adata
