import numpy as np
from scipy.sparse import issparse

def add_shh_cluster(adata, 
                    shh_threshold: float = 3.7,
                    cluster_key: str = 'shh_cluster'):
    """
    Add a cluster label based on SHH expression.
    
    Args:
        adata: AnnData object
        shh_threshold: Threshold for SHH expression
    
    Returns:
        AnnData object with the added 'shh_cluster' label
    """
    shh_values = adata[:, 'SHH'].X
    # If sparse, convert to dense
    if issparse(shh_values):
        shh_values = shh_values.toarray()
    # Ensure it's a 1D array
    shh_values = shh_values.flatten()
    
    adata.obs[cluster_key] = np.where(shh_values < shh_threshold, 'No', 'Yes')
    return adata