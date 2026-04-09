import numpy as np
from anndata import AnnData


# adjusted from NCEM
# https://github.com/theislab/ncem/blob/216fd576d1d1842234ceec300435a1dafcdf8b0c/ncem/estimators/base_estimator.py#L250
# accessed on 28 October 2025
def apply_segmentation_noise(
    adata: AnnData, node_fraction: float, overflow_fraction: float, adjacency_key: str = "spatial_connectivities"
) -> AnnData:
    """
    Apply segmentation noise to the AnnData object by redistributing gene expression
    between neighboring cells to simulate segmentation errors.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to apply noise to
    node_fraction : float
        Fraction of nodes to apply noise to (0.0 to 1.0)
    overflow_fraction : float
        Fraction of signal to redistribute to neighbors (0.0 to 1.0)
    adjacency_key : str
        Key in adata.obsp containing the adjacency matrix

    Returns
    -------
    AnnData
        Modified AnnData object with segmentation noise applied
    """
    if node_fraction <= 0 or node_fraction > 1:
        raise ValueError("node_fraction must be between 0 and 1")
    if overflow_fraction <= 0 or overflow_fraction > 1:
        raise ValueError("overflow_fraction must be between 0 and 1")

    # Create a copy to avoid modifying the original
    adata_noisy = adata.copy()

    # Calculate number of nodes to modify
    total_size = int(adata_noisy.shape[0] * node_fraction)

    if total_size == 0:
        print("Warning: node_fraction too small, no nodes will be modified")
        return adata_noisy

    # Get adjacency matrix
    if adjacency_key not in adata_noisy.obsp:
        raise KeyError(f"Adjacency matrix '{adjacency_key}' not found in adata.obsp")

    adj_matrix = adata_noisy.obsp[adjacency_key].toarray()

    # Randomly select nodes to modify
    random_indices = np.random.choice(adata_noisy.shape[0], size=total_size, replace=False)

    # Apply noise to selected nodes
    for idx in random_indices:
        # Get neighbors of this node
        neighbors = np.where(adj_matrix[idx, :] == 1.0)[0]

        if len(neighbors) == 0:
            print(f"No neighbors found for node {idx}")
            continue  # Skip if no neighbors

        # Randomly select one neighbor
        neighbor_idx = np.random.choice(neighbors, size=1, replace=False)[0]

        # Redistribute gene expression
        original_cell_expr = adata.X[idx, :].copy()
        neighbor_expr = adata.X[neighbor_idx, :].copy()

        # Add overflow fraction of neighbor's expression to current cell
        adata_noisy.X[idx, :] = original_cell_expr + overflow_fraction * neighbor_expr

        # Reduce neighbor's expression by overflow fraction
        adata_noisy.X[neighbor_idx, :] = (1.0 - overflow_fraction) * neighbor_expr

    print("\nSegmentation noise applied:")
    print(f"- Modified {total_size} nodes ({node_fraction:.1%} of all nodes)")
    print(f"- Signal overflow fraction: {overflow_fraction:.1%}")
    print(f"- Adjacency matrix key: {adjacency_key}")

    return adata_noisy
