from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, average_precision_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

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