import squidpy as sq
import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Mapping, Sequence
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def ligand_receptor_interaction(
    adata: AnnData | Mapping[str, pd.DataFrame],
    ligand: str | Sequence[str],
    receptor: str | Sequence[str],
    ligand_threshold: float = 0.3,
    receptor_threshold: float = 0.3,
) -> None:
    """
    Plot ligand-receptor expression on spatial data with separate color scales in the same plot.

    Parameters
    ----------
    adata
        AnnData object or a dictionary containing the ligand-receptor data.
    ligand
        Gene name for the ligand.
    receptor
        Gene name for the receptor.
    ligand_threshold
        Expression threshold to display ligand genes; below this, cells are colored grey.s
    receptor_threshold
        Expression threshold to display receptor genes; below this, cells are colored grey.s
    """
    # Convert ligand, receptor, and gene names to uppercase
    adata.var_names = adata.var_names.str.upper()
    ligand = ligand.upper()
    receptor = receptor.upper()

    # Validate inputs
    if ligand not in adata.var_names:
        raise ValueError(f"Ligand gene '{ligand}' not found in adata.var_names.")
    if receptor not in adata.var_names:
        raise ValueError(f"Receptor gene '{receptor}' not found in adata.var_names.")
    
    # Extract ligand and receptor expression
    ligand_expression = np.array(adata.X[:, adata.var_names == ligand].todense()).flatten()
    receptor_expression = np.array(adata.X[:, adata.var_names == receptor].todense()).flatten()

    # Apply separate thresholds for ligand and receptor
    ligand_mask = ligand_expression > ligand_threshold #TODO: Add that not ligand + receptor expr. > threshold
    receptor_mask = receptor_expression > receptor_threshold

    # Identify cells expressing both ligand and receptor above thresholds
    both_mask = ligand_mask & receptor_mask

    # Update masks for ligand-only and receptor-only
    ligand_only_mask = ligand_mask & ~both_mask
    receptor_only_mask = receptor_mask & ~both_mask
    
    # Add filtered expression data to `adata.obs`, setting values below threshold to NaN
    adata.obs['ligand_expression'] = np.where(ligand_only_mask, ligand_expression, np.nan)
    adata.obs['receptor_expression'] = np.where(receptor_only_mask, receptor_expression, np.nan)

    # Define a mask for cells below both thresholds
    below_threshold_mask = ~(ligand_mask | receptor_mask)
    adata.obs['below_threshold'] = np.where(below_threshold_mask, 1, np.nan)  # Use 1 for color mapping, NaN otherwise
    adata.obs['both_threshold'] = np.where(both_mask, 1, np.nan)  
    print(sum(both_mask))

    # Set up the plot
    fig, ax = plt.subplots()

    # Plot cells below both thresholds in gray
    sq.pl.spatial_scatter(
        adata,
        color='below_threshold',
        cmap='gray',
        ax=ax,
        shape=None,
        size=10,
        alpha = 0.01,
        colorbar = False
    )

    # Plot cells below both thresholds in gray
    sq.pl.spatial_scatter(
        adata,
        color='both_threshold',
        cmap='Dark2',
        ax=ax,
        shape=None,
        size=10,
        colorbar = False
    )

    # Plot ligand expression (only cells meeting the ligand threshold)
    sq.pl.spatial_scatter(
        adata,
        color='ligand_expression',
        cmap='Reds',
        ax=ax,
        shape=None,
        size=10,
        colorbar=False
    )

    # Add ligand colorbar
    norm_ligand = colors.Normalize(vmin=ligand_expression[ligand_only_mask].min(), vmax=ligand_expression[ligand_only_mask].max())
    cbar_ligand = fig.colorbar(cm.ScalarMappable(norm=norm_ligand, cmap='Reds'), ax=ax, orientation='vertical')
    cbar_ligand.set_label('Ligand Expression')

    # Overlay receptor expression (only cells meeting the receptor threshold)
    sq.pl.spatial_scatter(
        adata,
        color='receptor_expression',
        cmap='Blues',
        ax=ax,
        alpha=0.5,
        shape=None,
        size=10,
        colorbar = False
    )

    # Add receptor colorbar
    norm_receptor = colors.Normalize(vmin=receptor_expression[receptor_only_mask].min(), vmax=receptor_expression[receptor_only_mask].max())
    cbar_receptor = fig.colorbar(cm.ScalarMappable(norm=norm_receptor, cmap='Blues'), ax=ax, orientation='vertical')
    cbar_receptor.set_label('Receptor Expression')

    # Display the plot
    plt.show()