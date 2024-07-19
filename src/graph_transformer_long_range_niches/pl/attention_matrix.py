def spatial_attention_distribution(
                adata, 
                sender_cell_class: str, 
                receiver_cell_class: str
    ):
    """
    Plots the attention values over the spatial cell distribution with sender cells colored by the attention value and receiver cells colored in grey.

    Parameters:
    -----------
        attention_matrix: NxN matrix 
        sender_cell_class: adata.obs class of sender cells (e.i. cell type label)
        sender_cell_class: adata.obs class of receiver cells (e.i. cell type label)
    
    """

    # color scheme: sender = attention value gradient, receive = grey

    sq.pl.spatial_scatter(
        sub_adata_ct,
        spatial_key = 'spatial',
        shape=None,
        #library_key=library_key, 
        color=[f"attn_{class_label}"] ,
        #palette = cm.get_colormap_name(),
        size = 50,
        vmin=np.min(sub_adata.obs[f'attn_{class_label}'])
    )