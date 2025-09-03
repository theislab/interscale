import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
import pandas as pd

def leiden_cluster_embeddings(result, seeds=[0, 1, 2, 3, 4], leiden_res=1.0):
    """
    Leiden cluster the local and global embeddings multiple times with different seeds.
    Compare coherence using ARI, AMI, and NMI across seeds.
    
    Parameters
    ----------
    result : AnnData
        AnnData object containing 'combined_local_emb' and 'combined_global_emb' in .obsm
    seeds : list of int
        List of seeds to repeat clustering
    leiden_res : float
        Resolution for Leiden clustering
    
    Returns
    -------
    metrics_dict : dict
        Dictionary with keys 'ARI', 'AMI', 'NMI' and values as lists of scores per seed
    """
    # Remove rows where global embedding is all NaN
    embeddings_global = result.obsm['combined_global_emb'].astype(float)
    nan_rows = np.isnan(embeddings_global).all(axis=1)
    if nan_rows.any():
        print(f"Removing {nan_rows.sum()} observations with all-NaN global embeddings")
        result = result[~nan_rows, :].copy()
        print(f"New shape: {result.shape}")
    
    # Initialize results
    metrics_dict = {'ARI': [], 'AMI': [], 'NMI': []}

    for seed in seeds:
        sc.pp.neighbors(result, n_neighbors=40, use_rep='combined_local_emb', key_added=f'local_{seed}', random_state=seed)
        sc.pp.neighbors(result, n_neighbors=40, use_rep='combined_global_emb', key_added=f'global_{seed}', random_state=seed)
        
        sc.tl.leiden(result, resolution=leiden_res, neighbors_key=f'local_{seed}', key_added=f'leiden_local_{seed}', random_state=seed)
        sc.tl.leiden(result, resolution=leiden_res, neighbors_key=f'global_{seed}', key_added=f'leiden_global_{seed}', random_state=seed)
        
        labels_local = result.obs[f'leiden_local_{seed}']
        labels_global = result.obs[f'leiden_global_{seed}']
        
        ari = adjusted_rand_score(labels_local, labels_global)
        ami = adjusted_mutual_info_score(labels_local, labels_global)
        nmi = normalized_mutual_info_score(labels_local, labels_global)
        
        metrics_dict['ARI'].append(ari)
        metrics_dict['AMI'].append(ami)
        metrics_dict['NMI'].append(nmi)
    
    # Plot results
    plot_clustering_metrics(metrics_dict)
    
    return metrics_dict

def plot_clustering_metrics(metrics_dict):
    """
    Create a barplot with mean and quartiles, showing individual points for each metric.
    """
    data = []
    for metric, values in metrics_dict.items():
        for v in values:
            data.append({'Metric': metric, 'Score': v})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(4, 3))
    sns.barplot(x='Metric', y='Score', data=df, ci='sd', palette='pastel', errorbar='iqr')
    sns.stripplot(x='Metric', y='Score', data=df, color='black', size=5, jitter=True)
    plt.title('Clustering Agreement Across Seeds')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.ylim(0, 0.5)
    plt.show()