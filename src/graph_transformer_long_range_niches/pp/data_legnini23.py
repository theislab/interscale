import numpy as np
from scipy.sparse import issparse

def identify_shh_center(adata, shh_threshold=3.7, copy=False):
    """
    Identify SHH center by thresholding SHH expression values.
    
    Args:
        adata: AnnData object containing gene expression data
        shh_threshold: float, threshold value for SHH expression (default: 3.7)
        copy: bool, whether to return a copy of adata or just the shh_cluster dataframe (default: False)
    """
    shh_values = adata[:, 'SHH'].X
    # If sparse, convert to dense
    if issparse(shh_values):
        shh_values = shh_values.toarray()
    # Ensure it's a 1D array
    shh_values = shh_values.flatten()

    shh_cluster = np.where(shh_values < shh_threshold, str(0), str(1))
    
    if copy:
        adata.obs['shh_cluster'] = shh_cluster
        return adata
    else:
        return pd.DataFrame({'shh_cluster': shh_cluster}, index=adata.obs_names)
    