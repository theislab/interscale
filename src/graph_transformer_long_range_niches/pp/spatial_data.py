import numpy as np

def sliding_windows(adata, window_size, library_key: str = 'library_key', overlap: int = 0):
    """
        Input:
            - nr_windows: number of windows 
            - overlap: 
    """
    adata.obs['sliding_window'] = 'NaN'
    max_coords_x = adata.obs.groupby(library_key)[['x']].max()
    min_coords_x = adata.obs.groupby(library_key)[['x']].min()    
    max_coords_y = adata.obs.groupby(library_key)[['y']].max()
    min_coords_y = adata.obs.groupby(library_key)[['y']].min()  
    assert len(max_coords_x) == len(min_coords_x)
    width = max_coords_x - min_coords_x
    length = max_coords_y - min_coords_y
    
    nr_windows_per_row = np.ceil(np.divide(width, window_size))
    nr_windows_per_col = np.ceil(np.divide(length, window_size))

    #_assert_categorical_obs(adata, key=library_key) squidpy function
    libs = adata.obs[library_key].cat.categories
    #make_index_unique(adata.obs_names) squidpy function


    for lib in libs:
        for row in range(int(nr_windows_per_row.loc[lib, 'x'])):
            x_min = min_coords_x.loc[lib, 'x'] + (row)*window_size 
            x_max = min_coords_x.loc[lib, 'x'] + (row+1)*window_size 
            for col in range(int(nr_windows_per_col.loc[lib, 'y'])):
                y_min = min_coords_y.loc[lib, 'y'] + (col)*window_size
                y_max = min_coords_y.loc[lib, 'y'] + (col+1)*window_size
                mask = (adata.obs['x'] >= x_min) & (adata.obs['x'] < x_max) & (adata.obs['y'] >= y_min) & (adata.obs['y'] < y_max) & (adata.obs[library_key] == lib)
                adata.obs.loc[mask, 'sliding_window'] = f'{lib}_{row}_{col}'

    return adata