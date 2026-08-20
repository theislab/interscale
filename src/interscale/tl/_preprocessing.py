import numpy as np
import squidpy as sq
from typing import Literal


def remove_zero_expression_cells(adata):
    zero_expression_cells = np.array(adata.X.sum(axis=1) == 0).flatten()
    print(f"Nr. of zero expression cells: {zero_expression_cells.sum()}")
    if zero_expression_cells.sum() > 0:
        nonzero_cells = np.array(adata.X.sum(axis=1) != 0).flatten()
        adata = adata[nonzero_cells].copy()
    return adata


PIXEL_TO_UM = 0.138  # Resolve MC1 default; override via cfg if instrument differs

def get_average_local_and_global_size(
    adata,
    cfg,
    *,
    coord_units: Literal["pixels", "micrometer"] = "micrometer",
    pixel_size_um: float = PIXEL_TO_UM,
):
    """Average size of the local and the global window, in cells and in micrometers.

    Parameters
    ----------
    coord_units
        Units of ``adata.obsm['spatial']``. If ``"pixels"``, distances are
        converted to micrometers using ``pixel_size_um``.
    pixel_size_um
        Micrometers per pixel. Ignored when ``coord_units='micrometer'``.
    """
    scale = pixel_size_um if coord_units == "pixels" else 1.0

    ## Local: avg direct-neighbor count * n_layers
    if "spatial_connectivities" not in adata.obsp:
        kwargs = dict(cfg.dataset.spatial_neigbors_kwargs)
        # library_key is only filled in by prepare_geome_dataset, so pick the first sample key here
        kwargs["library_key"] = kwargs.get("library_key") or cfg.dataset.sample_key[0]
        adata.obs[kwargs["library_key"]] = adata.obs[kwargs["library_key"]].astype("category")
        sq.gr.spatial_neighbors(adata, **kwargs)
    n_layers = cfg.model.local_component.parameters.num_layers
    conn = adata.obsp["spatial_connectivities"].tocsr()
    local_cells = conn.getnnz(axis=1).mean() * n_layers
    # reach in micrometers: avg edge length walked over n_layers hops
    local_dist_um = adata.obsp["spatial_distances"].data.mean() * n_layers * scale

    ## Global: avg cells and avg extent per sample (or per sliding window, whichever sample_key defines)
    groups = adata.obs.groupby(list(cfg.dataset.sample_key), observed=True)
    coords = adata.obsm["spatial"]
    global_cells = groups.size().mean()
    global_dist_um = np.mean(
        [np.linalg.norm(coords[idx].max(axis=0) - coords[idx].min(axis=0))
         for idx in groups.indices.values()]
    ) * scale

    return {
        "local_cells": local_cells,
        "local_dist_um": local_dist_um,
        "global_cells": global_cells,
        "global_dist_um": global_dist_um,
    }
    
