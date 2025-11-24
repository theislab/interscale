import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy import sparse as sp

def _to_torch_tensor(X, device, dtype):
    """Convert numpy / scipy.sparse / pandas / torch -> torch.Tensor on device+dtype."""
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=dtype, non_blocking=True).contiguous()

    if sp.issparse(X):
        X = X.tocsr().astype(np.float32).toarray(order="C")

    if isinstance(X, pd.DataFrame):
        X = X.values

    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.number):
        X = X.astype(np.float32, copy=False)

    t = torch.as_tensor(X)  # no non_blocking kw here (compat with older PyTorch)
    return t.to(device=device, dtype=dtype, non_blocking=True).contiguous()

def _check_nonfinite(X, name):
    """Raise ValueError if X contains NaN or Inf (works for numpy/sparse/pandas/torch)."""
    if isinstance(X, torch.Tensor):
        if not torch.isfinite(X).all():
            n_nan = torch.isnan(X).sum().item()
            n_inf = torch.isinf(X).sum().item()
            raise ValueError(f"{name} contains NaNs={int(n_nan)}, Infs={int(n_inf)}")
        return

    if sp.issparse(X):
        data = np.asarray(X.data, dtype=np.float64)
        n_nan, n_inf = np.isnan(data).sum(), np.isinf(data).sum()
    else:
        arr = np.asarray(X)
        if not np.issubdtype(arr.dtype, np.number):
            arr = arr.astype(np.float64, copy=False)
        n_nan, n_inf = np.isnan(arr).sum(), np.isinf(arr).sum()

    if n_nan or n_inf:
        raise ValueError(f"{name} contains NaNs={int(n_nan)}, Infs={int(n_inf)}")

def gene_loadings(result: ad.AnnData, dtype=np.float32):
    """
    Single-matmul PyTorch implementation.
    - Uses GPU if available, else CPU.
    - Converts inputs to torch tensors on the same device.
    - Pre-checks for NaN/Inf and raises with counts.
    """
    if "_decoder_weight" not in result.obsm:
        raise KeyError("AnnData.obsm must contain '_decoder_weight'")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch_dtype = torch.float32 if dtype == np.float32 else torch.float64

    # Decoder weights (N, n_genes)
    B_raw = result.obsm["_decoder_weight"]
    _check_nonfinite(B_raw, "_decoder_weight")
    B = _to_torch_tensor(B_raw, device, torch_dtype)

    index = result.var_names

    # Local embeddings
    if "_local_emb" in result.obsm:
        A_raw = result.obsm["_local_emb"]
        _check_nonfinite(A_raw, "_local_emb")
        A = _to_torch_tensor(A_raw, device, torch_dtype)

        C = (A.T @ B).T  # (n_genes, p_local)
        C_cpu = C.to("cpu").numpy()
        cols = [f"emb_{i+1}" for i in range(C_cpu.shape[1])]
        result.varm["_local_gene_loadings"] = pd.DataFrame(C_cpu, index=index, columns=cols)

    # Global embeddings
    if "_global_emb" in result.obsm:
        A_raw = result.obsm["_global_emb"]
        _check_nonfinite(A_raw, "_global_emb")
        A = _to_torch_tensor(A_raw, device, torch_dtype)

        C = (A.T @ B).T  # (n_genes, p_global)
        C_cpu = C.to("cpu").numpy()
        cols = [f"emb_{i+1}" for i in range(C_cpu.shape[1])]
        result.varm["_global_gene_loadings"] = pd.DataFrame(C_cpu, index=index, columns=cols)

    return