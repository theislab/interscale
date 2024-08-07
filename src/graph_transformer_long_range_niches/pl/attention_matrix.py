import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from anndata import AnnData
import pandas as pd


def sender_receiver_stream(adata, 
                           sender_cell_type, 
                           receiver_cell_type, 
                           cell_type_key: str,
                           density = None):
    sender_spatial = adata.obsm['spatial'][adata.obs[cell_type_key] == sender_cell_type]
    receiver_spatial = adata.obsm['spatial'][adata.obs[cell_type_key] == receiver_cell_type]
    spatial = adata.obsm['spatial'][adata.obs[cell_type_key].isin([sender_cell_type, receiver_cell_type])]
    attention_matrix = adata.obsm['attention_matrix']
    attention_matrix = pd.DataFrame(attention_matrix, index=adata.obs[cell_type_key], columns=adata.obs[cell_type_key])
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
        adata,
        sender: str | list[str],
        receiver: str,
        cell_type_key: str = 'cell_type',
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
            Single or multiple cell type(s) considered as the sender in the attention matrix.
        receiver_cell_type: str
            The cell type considered as the receiver in the attention matrix.
        cell_type_col: str, optional (default: 'cell_type')
            The column in adata.obs that contains the cell type information.
        obsm_key: str, optional (default: 'attention')
            The key in adata.obsm that contains the attention matrix as NumpyArray.
    """
    if isinstance(sender, str):
        sender = [sender]
    if isinstance(receiver, str):
        receiver_type = receiver

    num_senders = len(sender)
    fig, axes = plt.subplots(1, num_senders, figsize=(5 * num_senders, 5), squeeze=False)

    for i, sender_type in enumerate(sender):
        ax = axes[0, i]

        # Extract the attention matrix from .obsm
        attention_matrix = adata.obsm[attn_matrix_key].copy()
        obs_names = adata.obs[cell_type_key]
        attention_matrix = pd.DataFrame(attention_matrix, index=obs_names, columns=obs_names)

        # Compute the sum of attention values for each sender cell towards all receiver cells
        attention_sums = attention_matrix.loc[sender_type, receiver_type].sum(axis=1)

        subadata = adata[adata.obs[cell_type_key].isin([sender_type, receiver_type])]

        subadata.obs['attention_values'] = 0
        subadata.obs.loc[subadata.obs[cell_type_key] == sender_type, 'attention_values'] = attention_sums.values

        attn_min = min(subadata.obs['attention_values'][subadata.obs['attention_values'] != 0])  # minimum that is not 0
        attn_max = max(subadata.obs['attention_values'])

        normalized_attention_values = (subadata.obs['attention_values'] - attn_min) / (attn_max-attn_min)

        # Create a color array initialized to grey for all cells
        cell_colors = [None] * subadata.n_obs
        # Update the colors for sender cells based on their attention sums
        for idx, (value, norm_value) in enumerate(zip(subadata.obs['attention_values'], normalized_attention_values)):
            if value == 0:
                cell_colors[idx] = (211/255, 211/255, 211/255, 1.0)
            else:
                cell_colors[idx] = plt.cm.summer(norm_value)

        subadata.obs['colors'] = cell_colors
        subadata.obs['colors'] = cell_colors

        sq.pl.spatial_scatter(subadata, shape=None, color='colors', ax=ax, palette=[cell_colors], img = False, size_key = 2, vmin = attn_min, vmax = attn_max)

        if add_streamline:
            X, Y, U, V = sender_receiver_stream(adata, sender_type, receiver_type, cell_type_key, density = None)
            ax.streamplot(X, Y, U, V, density=0.5, linewidth=1, arrowsize=1.5)

        ax.set_title(f"Sender: {sender_type} & Receiver: {receiver_type}")

    fig.suptitle('Attention value distribution between sender and grey receiver cells', fontsize=16)
    
    # Save the plot if a path or filename is provided
    if save_img:
        fig.savefig(save_img, bbox_inches='tight')

    plt.show()