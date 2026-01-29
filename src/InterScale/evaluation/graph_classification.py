from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, average_precision_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Union
import pandas as pd

def scale_cls_by_sample(adata, 
                        sample_key: str, 
                        cls_columns: list = ['combined_cls_horizontal', 'combined_cls_vertical'], 
                        inplace: bool = True, 
                        suffix: str ='_scaled'):
    """
    Scale CLS token values to [0, 1] within each sample/window to make them comparable
    across windows of different sizes.
    
    Parameters:
        adata: AnnData
            Annotated data object
        sample_key: str
            Column name in adata.obs containing the sample/window identifiers
        window_key: str
            Column name in adata.obs containing the window identifiers. If None, no windowing is performed.
        cls_columns: str | list
            Column name(s) of CLS values to scale (e.g., 'combined_cls_horizontal')
        inplace: bool, default=True
            If True, add scaled columns to adata.obs with suffix
            If False, return DataFrame with scaled values
        suffix: str, default='_scaled'
            Suffix to add to column names when inplace=True
    
    Returns:
        DataFrame or None
            If inplace=False, returns DataFrame with scaled values
            If inplace=True, returns None and modifies adata.obs
    """
    # Convert to list if single string
    if isinstance(cls_columns, str):
        cls_columns = [cls_columns]
    
    # Initialize output DataFrame
    if inplace:
        scaled_data = adata.obs.copy()
    else:
        scaled_data = pd.DataFrame(index=adata.obs.index)
    
    # Scale each CLS column
    for col in cls_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")
        
        scaled_col_name = f"{col}{suffix}" if inplace else col
        scaled_data[scaled_col_name] = np.nan
        
        # Scale within each sample/window
        for sample in adata.obs[sample_key].unique():
            mask = adata.obs[sample_key] == sample
            values = adata.obs.loc[mask, col]
            
            # Skip if no valid values
            if not values.notna().any():
                continue
            
            # Min-Max scaling
            val_min = values.min()
            val_max = values.max()
            
            if val_max > val_min:
                scaled_values = (values - val_min) / (val_max - val_min)
            else:
                # All values are the same
                scaled_values = pd.Series(0.5, index=values.index)
            
            scaled_data.loc[mask, scaled_col_name] = scaled_values
    
    if inplace:
        # Add scaled columns to adata.obs
        for col in cls_columns:
            scaled_col_name = f"{col}{suffix}"
            adata.obs[scaled_col_name] = scaled_data[scaled_col_name]
        return None
    else:
        return scaled_data

def calculate_pr_auc(result: AnnData,
                     plot_curve: bool = True):
    y_pred = result.obsm['combined_y_pred']  # Shape: (n_samples, 2)
    y_true = result.obs['condition']
    classes = y_true.cat.categories.tolist()

    # Convert to proper binary format
    y_true_binary = label_binarize(y_true, classes=classes)

    # Create proper one-hot encoding
    y_true_binary = np.zeros((len(y_true), 2))
    y_true_binary[y_true == classes[0], 0] = 1
    y_true_binary[y_true == classes[1], 1] = 1
    
    # Calculate with 'samples' or 'macro' averaging
    ap_macro = average_precision_score(y_true_binary, y_pred, average='macro')
    ap_micro = average_precision_score(y_true_binary, y_pred, average='micro')
    ap_weighted = average_precision_score(y_true_binary, y_pred, average='weighted')

    print(f"Macro AP: {ap_macro:.4f}")
    print(f"Micro AP: {ap_micro:.4f}")
    print(f"Weighted AP: {ap_weighted:.4f}")
    
    if plot_curve:
        pr_auc_curve(y_true_binary, y_pred, classes)

def pr_auc_curve(y_true_binary: np.ndarray, 
                 y_pred: np.ndarray,
                 classes: list[str]):
    # Calculate class distribution
    class_counts = y_true_binary.sum(axis=0)
    total = len(y_true_binary)
    class_ratios = class_counts / total

    # Plot precision-recall curves
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'red']

    for i, (class_name, color) in enumerate(zip(classes, colors)):
        precision, recall, _ = precision_recall_curve(y_true_binary[:, i], y_pred[:, i])
        ap = average_precision_score(y_true_binary[:, i], y_pred[:, i])
        
        # Add class ratio to label
        ax.plot(recall, precision, color=color, lw=2, 
                label=f'{class_name} (AP={ap:.4f}, n={int(class_counts[i])}, {class_ratios[i]:.1%})')

    # Optional: Add baseline (random classifier performance = class ratio)
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        ax.axhline(y=class_ratios[i], color=color, linestyle=':', alpha=0.5,
                label=f'{class_name} baseline ({class_ratios[i]:.1%})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves with Class Distribution', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_adata_grouped_heatmaps(
    adata_dict: Dict[str, 'AnnData'],
    group_by: str,
    value_cols: List[str] = None,
    labels: List[str] = None,
    split_key: Optional[str] = 'split',
    split_value: Optional[str] = 'test',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    cmap: str = 'viridis_r',
    annot: bool = True,
    fmt: str = '.2f',
    linewidth: float = 0.5,
    shared_colorbar: bool = True,
    agg_func: str = 'mean',
    **kwargs
) -> plt.Figure:
    """
    Create side-by-side heatmaps from multiple AnnData objects with shared color scale.
    
    Parameters
    ----------
    adata_dict : Dict[str, AnnData]
        Dictionary with keys as condition names and values as AnnData objects.
        Order of keys determines subplot order.
    group_by : str
        Column name in adata.obs to group by (e.g., 'CellType', 'cluster').
    value_cols : List[str], optional
        List of column names to aggregate. If None, defaults to:
        ['combined_cls_horizontal_scaled', 'combined_cls_vertical_scaled']
    labels : List[str], optional
        Subplot titles. If None (default), automatically uses keys from adata_dict.
    split_key : str, optional
        Column name for filtering (e.g., 'split'). Set to None to skip filtering.
    split_value : str, optional
        Value to filter on (e.g., 'test'). Only used if split_key is not None.
    figsize : tuple, default (10, 6)
        Figure size (width, height).
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    cmap : str, default 'viridis_r'
        Colormap name.
    annot : bool, default True
        Whether to annotate heatmap cells with values.
    fmt : str, default '.2f'
        Format string for annotations.
    linewidth : float, default 0.5
        Width of lines dividing cells.
    shared_colorbar : bool, default True
        Whether to use a shared colorbar (only on last subplot).
    agg_func : str, default 'mean'
        Aggregation function ('mean', 'median', 'sum', etc.).
    **kwargs
        Additional keyword arguments passed to sns.heatmap.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    
    Examples
    --------
    >>> # Basic usage - subplot titles automatically use dict keys ('ND', 'Onset', 'Long')
    >>> adata_dict = {
    ...     'ND': result_nd,
    ...     'Onset': result_onset,
    ...     'Long': result_long
    ... }
    >>> fig = plot_adata_grouped_heatmaps(
    ...     adata_dict=adata_dict,
    ...     group_by='CellType',
    ...     save_path='output/cls_heatmap.png'
    ... )
    
    >>> # Custom columns and no filtering - titles use dict keys
    >>> fig = plot_adata_grouped_heatmaps(
    ...     adata_dict={'Control': adata1, 'Treatment': adata2},
    ...     group_by='cell_type',
    ...     value_cols=['gene_A', 'gene_B', 'gene_C'],
    ...     split_key=None,
    ...     figsize=(12, 5)
    ... )
    
    >>> # Only use custom labels if you want titles different from dict keys
    >>> fig = plot_adata_grouped_heatmaps(
    ...     adata_dict={'cond1': adata1, 'cond2': adata2},
    ...     group_by='CellType',
    ...     labels=['Control Group', 'Treatment Group']  # Optional override
    ... )
    """
    # Set default value columns if not provided
    if value_cols is None:
        value_cols = ['combined_cls_horizontal_scaled', 'combined_cls_vertical_scaled']
    
    # Use dict keys as labels if not provided (default behavior)
    if labels is None:
        labels = list(adata_dict.keys())
    else:
        # Validate if custom labels are provided
        if len(labels) != len(adata_dict):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of AnnData objects ({len(adata_dict)})")
    
    # Process each AnnData object and compute aggregated data
    processed_data = {}
    for key, adata in adata_dict.items():
        # Filter by split if specified
        if split_key is not None and split_value is not None:
            if split_key not in adata.obs.columns:
                raise KeyError(f"Column '{split_key}' not found in adata.obs for '{key}'")
            adata_filtered = adata[adata.obs[split_key] == split_value]
        else:
            adata_filtered = adata
        
        # Check if group_by column exists
        if group_by not in adata_filtered.obs.columns:
            raise KeyError(f"Column '{group_by}' not found in adata.obs for '{key}'")
        
        # Check if value columns exist
        for col in value_cols:
            if col not in adata_filtered.obs.columns:
                raise KeyError(f"Column '{col}' not found in adata.obs for '{key}'")
        
        # Create aggregation dictionary
        agg_dict = {f'mean_{col}': (col, agg_func) for col in value_cols}
        
        # Group and aggregate
        processed_data[key] = adata_filtered.obs.groupby(group_by).agg(**agg_dict)
    
    # Calculate shared color scale
    all_values = np.concatenate([df.values.flatten() for df in processed_data.values()])
    vmin = all_values.min()
    vmax = all_values.max()
    
    # Create figure with subplots
    n_plots = len(processed_data)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Handle case of single subplot
    if n_plots == 1:
        axes = [axes]
    
    # Plot each heatmap
    for idx, (key, label) in enumerate(zip(adata_dict.keys(), labels)):
        data = processed_data[key]
        
        # Determine if this is the last plot (for colorbar)
        show_cbar = shared_colorbar and (idx == n_plots - 1)
        
        sns.heatmap(
            data,
            annot=annot,
            linewidth=linewidth,
            fmt=fmt,
            ax=axes[idx],
            vmin=vmin,
            vmax=vmax,
            cbar=show_cbar,
            cmap=cmap,
            **kwargs
        )
        axes[idx].set_title(label)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig