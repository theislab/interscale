import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from anndata import AnnData

def predict_gene_r2(adata: AnnData, layers_pred: str, top_n: int = 5) -> pd.DataFrame:
    """
    Predict gene R² scores for a given model layer.
    
    Parameters:
        adata: AnnData object containing the data
        layers_pred: str, name of the model layer to predict
        top_n: int, number of top genes to return
    """
    # Convert y_true to a dense array
    y_true = adata.X.toarray().astype(float)
    
    # Convert predictions to NumPy arrays
    y_pred = adata.layers[layers_pred]
    
    # Ensure predictions are also NumPy arrays (if they're tensors)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Ensure predictions are also NumPy arrays of type float
    y_pred = y_pred.astype(float)
    
    # Compute R² scores for each gene
    r2_scores = []
    for i in range(y_true.shape[1]):
        # Mask for non-NaN values in both y_true and y_pred for gene i
        mask = ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred[:, i])
        if np.sum(mask) > 1:  # Need at least 2 points to compute R²
            r2 = r2_score(y_true[mask, i], y_pred[mask, i])
        else:
            r2 = np.nan  # Not enough data to compute R²
        r2_scores.append(r2)
    #r2_scores_log = [np.log(r2 + 1) for r2 in r2_scores if not np.isnan(r2)]
    r2_scores_log = [np.log1p(r2) if not np.isnan(r2) else np.nan for r2 in r2_scores]
    r2_ranked = rankdata(r2_scores, method="average")
    
    # Convert to DataFrame for easy sorting
    genes = adata.var_names  # Gene names
    r2_df = pd.DataFrame({'gene': genes, 'r2': r2_scores, 'r2_log': r2_scores_log, 'r2_rank': r2_ranked})
    
    # Get top 5 genes for each model
    top = r2_df.nlargest(top_n, 'r2')
    
    print(f"Top {top_n} genes for {layers_pred} model:\n", top)
    
    return r2_df

def gene_rank_analysis(adata,
                       layers_local_pred: str = 'layers_local',
                       layers_global_pred: str = 'layers_global',
                       top_n: int = 5,
                       plot_result: bool = True,
                       return_top_genes: bool = False,
                       save_dir: str = None,
                       post_fix: str = None):
    """Ranks how well the local and global predictions capture the gene expression. 
    Plots the top N predicted genes for each model and consensus genes. 

    Args:
        adata (_type_): _description_
        layers_local_pred (str, optional): _description_. Defaults to 'layers_local'.
        layers_global_pred (str, optional): _description_. Defaults to 'layers_global'.
        top_n (int, optional): _description_. Defaults to 5.
        plot_result (bool, optional): _description_. Defaults to True.
        return_top_genes (bool, optional): _description_. Defaults to False.
        save_dir (str, optional): Directory to save the figure. If None, figure is not saved. Defaults to None.
    """
    assert layers_local_pred in adata.layers.keys(), f"layers_local_pred {layers_local_pred} not in adata.layers.keys()"
    assert layers_global_pred in adata.layers.keys(), f"layers_global_pred {layers_global_pred} not in adata.layers.keys()"
    
    local_df = predict_gene_r2(adata, layers_local_pred, top_n)
    global_df = predict_gene_r2(adata, layers_global_pred, top_n)

    # Select relevant columns and rename for clarity
    local_df = local_df[['gene', 'r2_rank']].rename(columns={'r2_rank': 'Local Rank'})
    global_df = global_df[['gene', 'r2_rank']].rename(columns={'r2_rank': 'Global Rank'})
    
    # Merge on 'gene'
    merged_df = pd.merge(local_df, global_df, on='gene', how='inner')
    
    # Compute rank difference
    merged_df['Rank Difference'] = merged_df['Local Rank'] - merged_df['Global Rank']
    
    # Compute overall prediction quality (higher avg rank means better prediction)
    merged_df['Avg Rank'] = (merged_df['Local Rank'] + merged_df['Global Rank']) / 2
    
    # Get top_n genes in each category
    top_local_genes = merged_df.nsmallest(top_n, "Rank Difference")  # More local-driven
    top_global_genes = merged_df.nlargest(top_n, "Rank Difference")  # More global-driven
    top_best_genes = merged_df.nlargest(top_n, "Avg Rank")  # Best overall predicted genes
    
    # Plot all genes
    plt.figure(figsize=(8, 8))
    plt.scatter(merged_df["Local Rank"], merged_df["Global Rank"], alpha=0.6, label="All Genes", color="gray")

    # Plot and label top local genes
    plt.scatter(top_local_genes["Local Rank"], top_local_genes["Global Rank"], color="blue", label="Top Local")
    for _, row in top_local_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=10, color="blue")

    # Plot and label top global genes
    plt.scatter(top_global_genes["Local Rank"], top_global_genes["Global Rank"], color="red", label="Top Global")
    for _, row in top_global_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=10, color="red")
    
    # Plot and label best-predicted genes
    plt.scatter(top_best_genes["Local Rank"], top_best_genes["Global Rank"], color="green", label="Best Predicted")
    for _, row in top_best_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=10, color="green")

    # Reference diagonal
    min_rank, max_rank = merged_df[['Local Rank', 'Global Rank']].values.min(), merged_df[['Local Rank', 'Global Rank']].values.max()
    plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', label="Equal Ranking (y=x)")  
    
    # Labels and legend
    plt.xlabel("Local Model Rank")
    plt.ylabel("Global Model Rank")
    plt.title("Gene Prediction Rank: Local vs. Global")
    plt.legend()
    
    # Save figure if save_dir is provided
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"gene_rank_analysis_{post_fix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')    
        print(f"Figure saved to: {save_path}")
    
    plt.show()

    # Return top genes if requested
    if return_top_genes:
        return top_local_genes, top_global_genes, top_best_genes
    
    