import numpy as np


def remove_zero_expression_cells(adata):
    zero_expression_cells = np.array(adata.X.sum(axis=1) == 0).flatten()
    print(f"Nr. of zero expression cells: {zero_expression_cells.sum()}")
    if zero_expression_cells.sum() > 0:
        nonzero_cells = np.array(adata.X.sum(axis=1) != 0).flatten()
        adata = adata[nonzero_cells].copy()
    return adata
