from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, average_precision_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

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