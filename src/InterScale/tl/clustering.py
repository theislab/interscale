import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
import scanpy as sc
from scipy.spatial import cKDTree


def calculate_adaptive_threshold(adata, gene,threshold=95):
    # Extract expression for the specific gene
    expr = adata[:, gene].X
    if not isinstance(expr, np.ndarray):
        expr = expr.toarray().flatten()
    else:
        expr = expr.flatten()
    
    # Filter out absolute zeros to focus on potential signal
    positive_expr = expr[expr > 0]
    
    if len(positive_expr) == 0:
        return None
    
    # Calculate Otsu threshold on non-zero expression
    try:
        thresh = threshold_otsu(positive_expr)
    except:
        # Fallback to a high percentile if Otsu fails
        thresh = np.percentile(positive_expr, threshold)
        
    return thresh

def create_spatial_cluster_distance(adata,gene,sample_col,distance_bins,bin_labels,threshold=95):
        
    assert 'spatial' in adata.obsm, "Spatial coordinates not found in adata.obsm['spatial']"
    assert len(bin_labels)+1==len(distance_bins), "Number of bin labels must be equal to number of distance bins minus one"
    assert gene in adata.var_names, f"Gene {gene} not found in adata.var_names"

    for sample in adata.obs[sample_col].unique():
        sample_mask = adata.obs[sample_col] == sample
        ctrl_test = adata[sample_mask].copy()

        # Extract expression
        gene_expression = ctrl_test[:, gene].X
        if not isinstance(gene_expression, np.ndarray):
            gene_expression = gene_expression.toarray().flatten()
        else:
            gene_expression = gene_expression.flatten()

        expr_threshold=calculate_adaptive_threshold(ctrl_test, gene,threshold)

        is_source = gene_expression >= expr_threshold

    # --- Step B: Handle Missing Source ---
        if np.sum(is_source) == 0:
            print(f"Sample {sample}: No {gene}+ cells. Marking No_Source.")
            adata.obs.loc[sample_mask, f'{gene}_zone'] = 'No_Source'
            continue

    # --- Step C: Spatial Calculation ---
        coords = ctrl_test.obsm["spatial"]
        source_coords = coords[is_source]

        tree = cKDTree(source_coords)
        dists, _ = tree.query(coords, k=1)

    # Apply the bins focused on the "near" field
        zones = pd.cut(
            dists, 
            bins=distance_bins, 
            labels=bin_labels, 
            include_lowest=True
        )

    # --- Step D: Update ---
        adata.obs.loc[sample_mask, f'dist_to_{gene}'] = dists
        adata.obs.loc[sample_mask, f'{gene}_zone'] = zones.astype(str)

    # Final Categorical ordering
    adata.obs[f'{gene}_zone'] = pd.Categorical(
        adata.obs[f'{gene}_zone'], 
        categories=bin_labels + ['No_Source', 'Background'], 
        ordered=True
    )

    return adata
    



def create_spatial_cluster_neighbours(adata,gene,sample_col,distance_bins=[10,20,30],threshold=95):
        
    assert 'spatial' in adata.obsm, "Spatial coordinates not found in adata.obsm['spatial']"
    assert gene in adata.var_names, f"Gene {gene} not found in adata.var_names"


    for sample in adata.obs[sample_col].unique():
    #subset to slide
        sample_mask = adata.obs[sample_col] == sample
        ctrl_test = adata[sample_mask].copy()

    if gene in ctrl_test.var_names:
        gene_expression = ctrl_test[:, gene].X
        if not isinstance(gene_expression, np.ndarray):
            gene_expression = gene_expression.toarray().flatten()  # For sparse matrices
        else:
            gene_expression = gene_expression.flatten()
            
        threshold=calculate_adaptive_threshold(ctrl_test, gene,threshold)
        # Annotate cells in the obs field based on the expression threshold
        ctrl_test.obs[f'{gene}_status'] = np.where(gene_expression >= threshold, f'{gene}+', f'{gene}-')
    else:
        print(f"Gene '{gene}' not found in adata.var_names.")
    ## Coordinates
    coords = ctrl_test.obsm["spatial"]
    gene_coords = coords[ctrl_test.obs[f"{gene}_status"] == f"{gene}+"]

    ## Compute distance to nearest gene+ cell
    tree = cKDTree(gene_coords)
    dist, _ = tree.query(coords, k=1)
    ctrl_test.obs[f"dist_to_{gene}+"] = dist

    ## Cluster cells based on distance
    sc.pp.neighbors(ctrl_test, use_rep=None, n_neighbors=10)
    ctrl_test.obs["distance_cluster"] = np.digitize(ctrl_test.obs[f"dist_to_{gene}+"], bins=distance_bins)  # arbitrary cutoffs

    dist = ctrl_test.obs[f"dist_to_{gene}+"].to_numpy()

    n_bins = len(distance_bins)+1
    edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    labels = [f"ring_{i+1}" for i in range(len(edges)-1)]

    cats = pd.cut(dist, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    # directly ensure it's ordered


    adata.obs.loc[sample_mask, f'{gene}_cats'] = cats.astype(str)

    return(adata)