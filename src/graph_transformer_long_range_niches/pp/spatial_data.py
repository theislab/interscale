import numpy as np

def sliding_windows(adata, 
                    window_size, 
                    library_key: str | None = None, 
                    overlap: int = 0):
    """
        Input:
            - nr_windows: number of windows 
            - overlap: 
        TODO: integrate overlap
    """
    if library_key is not None:
        #_assert_categorical_obs(adata, key=library_key) squidpy function
        libs = adata.obs[library_key].cat.categories
        #make_index_unique(adata.obs_names) squidpy function
    else:
        libs = [None]

    adata.obs['sliding_window'] = 'NaN'

    width = adata.obs['x'].max() - adata.obs['x'].min()
    length = adata.obs['y'].max() - adata.obs['y'].min()
    
    nr_windows_per_row = np.ceil(np.divide(width, window_size))
    nr_windows_per_col = np.ceil(np.divide(length, window_size))
    print(nr_windows_per_row, nr_windows_per_col)

    for row in range(1, int(nr_windows_per_row)+1):
        x_min = adata.obs['x'].min() + (row-1)*window_size 
        x_max = adata.obs['x'].min() + row*window_size
        for col in range(1, int(nr_windows_per_col)+1):
            y_min = adata.obs['y'].min() + (col-1)*window_size
            y_max = adata.obs['y'].min() + col*window_size
            mask = (adata.obs['x'] >= x_min) & (adata.obs['x'] < x_max) & (adata.obs['y'] >= y_min) & (adata.obs['y'] < y_max)
            adata.obs.loc[mask, 'sliding_window'] = f'{row}_{col}'

    return adata