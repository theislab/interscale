import numpy as np
import squidpy as sq


def compute_neighborhood_stats(adata, radii, *, library_key: str | None = None, show=True):
    """
    Compute the average number of neighbors and standard deviation
    for multiple radii in a spatial transcriptomics dataset using Squidpy.

    Parameters
    ----------
    - adata: AnnData object with spatial coordinates stored in adata.obsm["spatial"].
    - radii: List of floats, the radii within which to count neighbors.

    Returns
    -------
    - stats: Dictionary where keys are radii and values are tuples (avg_neighbors, std_neighbors).
    """
    stats = {}

    for radius in radii:
        # Compute spatial neighbors using Squidpy
        sq.gr.spatial_neighbors(adata, coord_type="generic", radius=radius, library_key=library_key)

        # Extract the neighborhood graph
        graph = adata.obsp["spatial_connectivities"]

        # Compute neighbor counts
        neighbor_counts = np.array(graph.sum(axis=1)).flatten()

        # Compute statistics
        avg_neighbors = np.mean(neighbor_counts)
        std_neighbors = np.std(neighbor_counts)

        stats[radius] = (avg_neighbors, std_neighbors)
        if show:
            print(f"Radius: {radius}, Average Neighbors: {avg_neighbors:.2f}, Std Dev: {std_neighbors:.2f}")

    return stats
