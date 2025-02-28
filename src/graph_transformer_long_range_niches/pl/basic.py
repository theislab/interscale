from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

def predict_gene_r2(adata: AnnData, layers_pred: str, top_n: int = 5) -> pd.DataFrame:
    """
    Predict gene R² scores for a given model layer.
    
    Parameters:
        adata: AnnData object containing the data
        layers_pred: str, name of the model layer to predict
        top_n: int, number of top genes to return
    """
    # Convert y_true to a dense array
    y_true = adata.X.toarray()  # Convert sparse matrix to dense NumPy array
    
    # Convert predictions to NumPy arrays
    y_pred = adata.layers[layers_pred]
    
    # Ensure predictions are also NumPy arrays (if they're tensors)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Compute R² scores for each gene
    r2_scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    
    # Convert to DataFrame for easy sorting
    genes = adata.var_names  # Gene names
    r2_df = pd.DataFrame({'gene': genes, 'r2': r2_scores})
    
    # Get top 5 genes for each model
    top = r2_df.nlargest(top_n, 'r2')
    
    print(f"Top {top_n} genes for {layers_pred} model:\n", top)
    
    return r2_df

def compare_model_variance(df1: pd.DataFrame, df2: pd.DataFrame, 
                         model1_name: str = "Model 1", 
                         model2_name: str = "Model 2",
                         top_n: int | None = None) -> pd.DataFrame:
    """
    Compare explained variance between two models using their top R² scores.
    Each model can explain up to 0.5% of total variance.
    
    Parameters:
        df1: DataFrame with gene names and R² scores from first model
        df2: DataFrame with gene names and R² scores from second model
        model1_name: Name of the first model for plotting
        model2_name: Name of the second model for plotting
        top_n: Number of top genes to consider from each model
    
    Returns:
        DataFrame with combined metrics for top genes from both models
    """
    # Calculate explained variance (0.5% maximum per model)
    def calc_explained_variance(df):
        df = df.copy()  # Create a copy to avoid modifying the original
        df['exp'] = df['r2'] / 2
        df['unexp'] = 0.5 - df['exp']
        return df
    
    df1 = calc_explained_variance(df1)
    df2 = calc_explained_variance(df2)

    if top_n is not None:
        # Get top genes from both models
        top1_genes = df1.nlargest(top_n, 'r2').index
        top2_genes = df2.nlargest(top_n, 'r2').index
        
        # Combine unique genes from both models
        all_top_genes = pd.Index(set(top1_genes) | set(top2_genes))
    else:
        all_top_genes = df1.index
    
    # Create combined DataFrame
    combined_df = pd.DataFrame(index=all_top_genes)
    combined_df['gene'] = df1.loc[all_top_genes, 'gene']
    
    # Add metrics for model 1
    combined_df[f'exp_{model1_name}'] = df1.loc[all_top_genes, 'exp']
    combined_df[f'unexp_{model1_name}'] = df1.loc[all_top_genes, 'unexp']
    combined_df[f'r2_{model1_name}'] = df1.loc[all_top_genes, 'r2']
    
    # Add metrics for model 2
    combined_df[f'exp_{model2_name}'] = df2.loc[all_top_genes, 'exp']
    combined_df[f'unexp_{model2_name}'] = df2.loc[all_top_genes, 'unexp']
    combined_df[f'r2_{model2_name}'] = df2.loc[all_top_genes, 'r2']
    
    # Fill NaN values with 0 for exp and 0.5 for unexp
    combined_df = combined_df.fillna({
        f'exp_{model1_name}': 0,
        f'exp_{model2_name}': 0,
        f'unexp_{model1_name}': 0.5,
        f'unexp_{model2_name}': 0.5
    })
    
    # Calculate total unexplained variance (sum of individual unexplained variances)
    combined_df['total_unexp'] = combined_df[f'unexp_{model1_name}'] + combined_df[f'unexp_{model2_name}']
    
    # Sort by maximum explained variance across both models
    max_exp = combined_df[[f'exp_{model1_name}', f'exp_{model2_name}']].max(axis=1)
    combined_df = combined_df.loc[max_exp.sort_values(ascending=False).index]
       
    return combined_df

# Add plotting function
def plot_variance_comparison(combined_df, model1_name, model2_name):
    import matplotlib.pyplot as plt
    
    # Prepare data for plotting
    genes = combined_df['gene']  # Exclude the Total_Unexplained row
    exp1 = combined_df[f'exp_{model1_name}']
    exp2 = combined_df[f'exp_{model2_name}']
    total_unexp = combined_df['total_unexp'] # Divide by number of genes for equal distribution
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set the width of each bar and positions of the bars
    width = 0.25
    x = np.arange(len(genes))
    
    # Create bars
    ax.bar(x - width, exp1, width, label=f'{model1_name} Explained', color='skyblue')
    ax.bar(x, exp2, width, label=f'{model2_name} Explained', color='lightgreen')
    ax.bar(x + width, total_unexp, width, label='Unexplained', color='lightgray')
    
    # Customize the plot
    ax.set_ylabel('Variance')
    ax.set_title('Explained vs Unexplained Variance by Gene')
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on the bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    add_labels(ax.patches)
    
    plt.tight_layout()
    plt.show()
    
def plot_lfc_scatter(df, model1_name, model2_name, metric='r2'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # Two subplots side by side

    # Scatter Plot 1: Full range
    sns.scatterplot(data=df, x=f'{metric}_{model1_name}', y=f'{metric}_{model2_name}', alpha=0.7, ax=axes[0])
    axes[0].set_xlabel(f'{metric} ({model1_name})')
    axes[0].set_ylabel(f'{metric} ({model2_name})')
    axes[0].set_title('Local vs. Global Model Reconstruction (Full)')

    # Scatter Plot 2: Zoomed in to x > 0 and y > 0
    df_filtered = df[(df[f'{metric}_{model1_name}'] > -0.5) | (df[f'{metric}_{model2_name}'] > -.5)]
    sns.scatterplot(data=df_filtered, x=f'{metric}_{model1_name}', y=f'{metric}_{model2_name}', alpha=0.7, ax=axes[1])
    axes[1].set_xlabel(f'{metric} ({model1_name})')
    axes[1].set_ylabel(f'{metric} ({model2_name})')
    axes[1].set_title('Local vs. Global Model Reconstruction (Zoomed In)')

    # Highlight top genes in both plots
    top_genes = df.loc[df[[f'{metric}_{model1_name}', f'{metric}_{model2_name}']].max(axis=1) > 0.1, 'gene']
    print(top_genes)
    for gene in top_genes:
        gene_data = df[df['gene'] == gene]
        axes[0].text(gene_data[f'{metric}_{model1_name}'].values[0], 
                     gene_data[f'{metric}_{model2_name}'].values[0], 
                     gene, fontsize=8, color='red')
        if gene_data[f'{metric}_{model1_name}'].values[0] > 0 or gene_data[f'{metric}_{model2_name}'].values[0] > 0:
            axes[1].text(gene_data[f'{metric}_{model1_name}'].values[0], 
                         gene_data[f'{metric}_{model2_name}'].values[0], 
                         gene, fontsize=8, color='red')

    plt.tight_layout()
    plt.show()
