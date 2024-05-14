import numpy as np

def window_splits(adata, 
                  library_key: str = 'library_key',
                  nr_off_cells_per_window: int = 2000):
    
    """ Splits each slice from the spatial adata object into windows with approximatly N number of cells per window. 

    Input: 
        adata: AnnData object with x and y coordinates in .obs
        library_key: String that indicates the slice number in .obs 

    """

    max_coords_x = adata.obs.groupby('library_key')[['x']].max()
    min_coords_x = adata.obs.groupby('library_key')[['x']].min()    
    max_coords_y = adata.obs.groupby('library_key')[['y']].max()
    min_coords_y = adata.obs.groupby('library_key')[['y']].min()  

    x_diff = max_coords_x - min_coords_x
    y_diff = max_coords_y - min_coords_y

    cell_count_per_slice = adata.obs.groupby('library_key').size()
    windows_per_image = np.ceil(np.divide(cell_count_per_slice, nr_off_cells_per_window))
    rows = np.ceil(np.sqrt(windows_per_image))
    cols = np.ceil(np.divide(windows_per_image, rows))
    
    adata.obs['window'] = None
    for image in np.unique(adata.obs['library_key']):
        x_size = x_diff['x'][image] / rows[image]
        y_size = y_diff['y'][image] / cols[image]
        for row in range(int(rows[image])):
            for col in range(int(cols[image])): 
                x_min = min_coords_x['x'][image] + x_size * row
                x_max = min_coords_x['x'][image]  + x_size * (row + 1)
                y_min = min_coords_y['y'][image]  + y_size * col
                y_max = min_coords_y['y'][image] + y_size * (col + 1)
                selected_cells = (adata.obs['x'] > x_min) & (adata.obs['x'] < x_max) & (adata.obs['y'] > y_min) & (adata.obs['y'] < y_max)
                # Update the values of selected cells to 0
                adata.obs.loc[selected_cells, 'window'] = f'{image}_{row}_{col}'

    cells_per_window = adata.obs.groupby('window').size()
    print(f'Cells per window stats: Max: {cells_per_window.max()}, MIN: {cells_per_window.min()}, mean: {cells_per_window.mean()}, nr. of windows: {len(cells_per_window)}')
    return adata