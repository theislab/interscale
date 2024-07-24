import matplotlib.pyplot as plt
import squidpy as sq
from anndata import AnnData
import numpy as np

def sender_receiver_stream(adata, sender_cell_type, receiver_cell_type, density = None):
    sender_spatial = adata.obsm['spatial'][adata.obs['cell_type'] == sender_cell_type]
    receiver_spatial = adata.obsm['spatial'][adata.obs['cell_type'] == receiver_cell_type] 
    spatial = adata.obsm['spatial'][adata.obs['cell_type'].isin([sender_cell_type, receiver_cell_type])] 
    attention_matrix = adata.obsm['attention_matrix']
    S, n_dim = sender_spatial.shape
    R = receiver_spatial.shape[0]
    density = 1 if density is None else density

    # Determine the range of data
    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(spatial[:, dim_i]), np.max(spatial[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    # Generate grid
    X, Y = np.meshgrid(*grs)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(S):
        for j in range(R):
            # Get sender and receiver coordinates
            sender_x, sender_y = sender_spatial[i]
            receiver_x, receiver_y = receiver_spatial[j]
            
            # Get attention value
            attention_value = attention_matrix.iloc[i, j]
            
            # Compute the vector components
            u = attention_value * (receiver_x - sender_x)
            v = attention_value * (receiver_y - sender_y)
            
            # Add the components to the grid
            U += u / ((X - sender_x)**2 + (Y - sender_y)**2 + 1e-6)
            V += v / ((X - sender_x)**2 + (Y - sender_y)**2 + 1e-6)

    return X, Y, U, V

def plot_attention_sender_receiver(
        adata: AnnData,
        sender_cell_type: str,
        receiver_cell_type: str,
        cell_type: str = 'cell_type',
        attn_matrix_key: str ='attention',
        save_img: str = None,
        add_streamline: bool = False
    ) -> None:
    """
    Plot the spatial scatter plot of sender and receiver cells, highlighting the sender cells based on the sum 
    of their attention values towards the receiver cells.

    Parameters
    ----------
        adata: AnnData
            Annotated data matrix.
        sender_cell_type: str
            The cell type considered as the sender in the attention matrix.
        receiver_cell_type: str
            The cell type considered as the receiver in the attention matrix.
        cell_type_col: str, optional (default: 'cell_type')
            The column in adata.obs that contains the cell type information.
        obsm_key: str, optional (default: 'attention')
            The key in adata.obsm that contains the attention matrix.
    """
    # Extract the attention matrix from .obsm
    attention_matrix = adata.obsm[attn_matrix_key]

    # Extract indices for sender and receiver cells
    sender_indices = adata.obs.index[adata.obs[cell_type] == sender_cell_type]
    receiver_indices = adata.obs.index[adata.obs[cell_type] == receiver_cell_type]

    # Compute the sum of attention values for each sender cell towards all receiver cells
    attention_sums = attention_matrix.loc[sender_cell_type, receiver_cell_type].sum(axis=1)

    subadata = adata[adata.obs[cell_type].isin([sender_cell_type, receiver_cell_type])]

    subadata.obs['attention_values'] = 0
    subadata.obs.loc[subadata.obs[cell_type] == sender_cell_type, 'attention_values'] = attention_sums.values

    # Create a color array initialized to grey for all cells
    cell_colors = [None] * subadata.n_obs
    # Update the colors for sender cells based on their attention sums
    for idx, value in enumerate(subadata.obs['attention_values']):
        if value == 0:
            cell_colors[idx] = (211/255, 211/255, 211/255, 1.0)
        else:
            cell_colors[idx] = plt.cm.summer(value / attention_sums.max())

    subadata.obs['colors'] = cell_colors

    fig, ax = plt.subplots()
    sq.pl.spatial_scatter(subadata, shape=None, color='colors', ax=ax, palette=[cell_colors], img = False, size_key = 2)

    if add_streamline:
        X, Y, U, V = sender_receiver_stream(adata, sender_cell_type, receiver_cell_type, density = None)
        ax.streamplot(X, Y, U, V, density=0.5, linewidth=1, arrowsize=1.5)

    # Save the plot if a path or filename is provided
    if save_img:
        fig.savefig(save_img, bbox_inches='tight')

    plt.show()
