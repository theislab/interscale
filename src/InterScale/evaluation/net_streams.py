import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import squidpy as sq
from matplotlib.colors import to_rgb, to_hex, rgb_to_hsv, hsv_to_rgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from scipy.spatial import KDTree

def plot_all_spatial_net_streams(
	adata, fov_id, fov_key='fov', window_key='sliding_window_assignment',
	cell_type_col='cell_type_coarse', grid_res=50, max_dist=None,
	k_dist=0.05, density=1.5, ax=None,
	additional_embeddings=None,
	return_streams=False,
	**kwargs
):
	# 1. Prepare FOV slice
	slice_mask = adata.obs[fov_key] == fov_id
	adata_slice = adata[slice_mask].copy()
	
	# Extract color mapping for all categories
	categories = list(adata.obs[cell_type_col].cat.categories)
	if f'{cell_type_col}_colors' in adata.uns:
		colors = list(adata.uns[f'{cell_type_col}_colors'])
		color_map = dict(zip(categories, colors))
	else:
		import matplotlib.cm as cm
		colors = [to_hex(c) for c in cm.tab20(np.linspace(0, 1, len(categories)))]
		color_map = dict(zip(categories, colors))

	# Setup spatial grid
	sf = 1.0
	try:
		s_data = adata_slice.uns['spatial'][fov_id]
		sf = s_data['scalefactors'].get('tissue_hires_scalef', 1.0)
	except (KeyError, AttributeError):
		sf = 1.0
		
	spatial_coords = adata_slice.obsm['spatial'] * sf
	x_min, y_min = spatial_coords.min(axis=0)
	x_max, y_max = spatial_coords.max(axis=0)
	
	X_lin = np.linspace(x_min - (5*sf), x_max + (5*sf), grid_res)
	Y_lin = np.linspace(y_min - (5*sf), y_max + (5*sf), grid_res)
	grid_x, grid_y = np.meshgrid(X_lin, Y_lin)
	
	# Initialize dictionaries to store vector fields for each cell type
	U_dict = {cat: np.zeros_like(grid_x) for cat in categories}
	V_dict = {cat: np.zeros_like(grid_x) for cat in categories}
	
	if max_dist is None:
		max_dist = (x_max - x_min) * k_dist

	# 2. Iterate through windows to calculate global net flow
	windows = adata_slice.obs[window_key].unique()
	for win in windows:
		win_mask = adata_slice.obs[window_key] == win
		adata_win = adata_slice[win_mask]
		n_win = len(adata_win.obs)
		if n_win < 2: continue
		
		# Matrix of all-vs-all attention in the window
		M = pd.DataFrame(
			adata_win.obsm['_attn_matrix'][:, :n_win],
			index=adata_win.obs_names,
			columns=adata_win.obs_names
		)
		M_net = M - M.T  # Net flow between all cells
		
		pos_win = adata_win.obsm['spatial'] * sf
		types_win = adata_win.obs[cell_type_col].values
		
		for j, cell_j_name in enumerate(adata_win.obs_names):
			s_coord = pos_win[j]
			cell_type = types_win[j]
			
			# Identify flows incoming to cell j
			net_flows = M_net.iloc[:, j].values
			positive_flows = np.maximum(net_flows, 0)
			
			if np.sum(positive_flows) == 0: continue
			
			# Compute distance-weighted direction vector
			diff = pos_win - s_coord
			dist = np.linalg.norm(diff, axis=1)
			unit_diff = diff / (dist[:, np.newaxis] + 1e-6)
			s_weight = np.exp(-dist**2 / (2 * max_dist**2))
			
			# Calculate local cell flow vector
			v_cell = np.sum(unit_diff * (positive_flows * s_weight)[:, np.newaxis], axis=0)
			
			# Distribute the vector onto the grid for its specific cell type
			g_dist_sq = (grid_x - s_coord[0])**2 + (grid_y - s_coord[1])**2
			kernel = np.exp(-g_dist_sq / (2 * (max_dist/4)**2))
			
			U_dict[cell_type] += v_cell[0] * kernel
			V_dict[cell_type] += v_cell[1] * kernel

	# 3. Plotting
	if ax is None:
		fig, ax = plt.subplots(figsize=(12, 10))
	

	emb= additional_embeddings if additional_embeddings is not None else cell_type_col
	# Background scatter plot
	sq.pl.spatial_scatter(
		adata_slice, color=emb, 
		library_key=fov_key, library_id=[fov_id],
		ax=ax, spatial_key='spatial', 
		img=False,**kwargs
	)

	# 4. Draw streamlines for each cell type individually

	legend_elements = []

	for cat in categories:
		U, V = U_dict[cat], V_dict[cat]
		mag = np.sqrt(U**2 + V**2)
		if np.max(mag) == 0: continue
		
		# Local normalization and thresholding
		thresh = 0.01 * np.max(mag)
		Un = np.divide(U, mag, out=np.zeros_like(U), where=mag > thresh)
		Vn = np.divide(V, mag, out=np.zeros_like(V), where=mag > thresh)
		Un[mag <= thresh] = np.nan
		Vn[mag <= thresh] = np.nan
		
		if not np.all(np.isnan(Un)):
			# Create a darker version of the category color for visibility
			rgb = to_rgb(color_map[cat])
			hsv = rgb_to_hsv(rgb)
			dark_color = to_hex(hsv_to_rgb([hsv[0], hsv[1], hsv[2] * 0.7]))
			
			ax.streamplot(X_lin, Y_lin, Un, Vn, color=dark_color, 
						linewidth=1.2, density=density, arrowsize=1.2)
			
			legend_elements.append(Line2D([0], [0], color=dark_color, lw=2, 
										label=f'Flow: {cat}'))
			
	if legend_elements and additional_embeddings is not None:
		old_legend = ax.get_legend()
		new_legend = ax.legend(
					handles=legend_elements, 
					loc='upper center', 
					ncol=4,
					bbox_to_anchor=(0.5,-0.1), # Posizionata in alto a destra
					title=f"Net Flow Directions - {cell_type_col}"
		)
		if old_legend is not None:
			ax.add_artist(old_legend)

	if return_streams:
		streams = {cat: (U_dict[cat], V_dict[cat]) for cat in categories}
		return streams,X_lin, Y_lin
	else:
		return ax

def calculate_divergence(U, V):
    return np.gradient(U, axis=1) + np.gradient(V, axis=0)	


def cluster_spatial_flows(U_dict, V_dict, n_clusters=5):
	"""
	Performs unsupervised clustering of spatial regions based on multi-type flow vectors.
	
	Parameters:
	- U_dict, V_dict: Dictionaries of U and V grid components from previous functions.
	- n_clusters: Number of spatial domains to identify.
	"""
	
	categories = list(U_dict.keys())
	grid_shape = list(U_dict.values())[0].shape
	n_points = grid_shape[0] * grid_shape[1]
	
	# 1. Feature Engineering: Build a "Flow Signature" for each grid point
	# We concatenate U and V for all cell types: [U_cat1, V_cat1, U_cat2, V_cat2, ...]
	feature_list = []
	for cat in categories:
		div=calculate_divergence(U_dict[cat], V_dict[cat])
		mag=np.sqrt(U_dict[cat]**2 + V_dict[cat]**2)
		feature_list.append(div.flatten())
		feature_list.append(mag.flatten())
		# feature_list.append(U_dict[cat].flatten())
		# feature_list.append(V_dict[cat].flatten())
	
	# Transpose to get (n_points, n_features)
	X = np.array(feature_list).T
	
	# 2. Data Cleaning
	# Replace NaNs (where flow was below threshold) with 0
	X = np.nan_to_num(X)
	
	# 3. Standardization
	# Scale features so that high-intensity flow types don't dominate the clustering
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	
	# 4. Unsupervised Clustering
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	clusters = kmeans.fit_predict(X_scaled)

	#clusters_c = [f"Domain_{i}" for i in clusters]
	
	# Reshape clusters back to grid dimensions
	cluster_grid = clusters.reshape(grid_shape)
	
	return cluster_grid, X_scaled

def plot_flow_clusters(cluster_grid, X_lin, Y_lin, adata_slice, fov_id, fov_key='fov', cell_type_col=None):
	"""
	Visualizes the identified flow domains as a background for the spatial data.
	"""

	unique_clusters = np.unique(cluster_grid)
	n_clusters = len(unique_clusters)

	cmap = plt.get_cmap('Set3', n_clusters)


	fig, ax = plt.subplots(figsize=(10, 8))
	# Overlay the original cell positions
	sq.pl.spatial_scatter(
		adata_slice, color=cell_type_col, 
		library_key=fov_key, library_id=[fov_id],
		ax=ax, spatial_key='spatial', 
		size=2, alpha=0.4, img=False, 
		title=f"Unsupervised Flow Domains (n={len(np.unique(cluster_grid))})"
	)
	# Plot the clusters as a heatmap (Voronoi-like segmentation of flow)
	im = ax.pcolormesh(X_lin, Y_lin, cluster_grid, 
					cmap=cmap, alpha=0.4, shading='auto',
					vmin=unique_clusters.min()-0.5, 
					vmax=unique_clusters.max()+0.5)
	
	legend_handles = []
	for i, cluster_id in enumerate(unique_clusters):
		color = cmap(i)
		patch = mpatches.Patch(color=color, label=f"Domain {int(cluster_id)}")
		legend_handles.append(patch)

	
	ax.legend(handles=legend_handles, title="Flow Domains", 
              loc='center left', bbox_to_anchor=(1, 0.5))
	return fig, ax


def map_clusters_to_cells(cluster_grid, X_lin, Y_lin, adata, fov_id, fov_key='fov'):
	"""
	Assign grid annotation to individual cells based on their spatial coordinates.

	"""
	# Create the grid coordinates for KDTree
	grid_x, grid_y = np.meshgrid(X_lin, Y_lin)
	
	# Flat the coordinates to create a list of (x, y) points
	grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
	
	# 2. Create a KDTree for efficient nearest neighbor search
	tree = KDTree(grid_coords)
	
	adata_slice=adata[adata.obs[fov_key] == fov_id].copy()

	sf = 1.0
	try:
		s_data = adata_slice.uns['spatial'][fov_id]
		sf = s_data['scalefactors'].get('tissue_hires_scalef', 1.0)
	except (KeyError, AttributeError):
		sf = 1.0

	# Get cell coordinates and scale them to match the grid
	cell_coords = adata_slice.obsm['spatial'] * sf
	
	# Search for the nearest grid point for each cell
	dists, indices = tree.query(cell_coords)
	
	# Get the cluster assignment for each cell based on the nearest grid point
	flat_clusters = cluster_grid.ravel()
	cell_clusters = flat_clusters[indices]
	
	# Save
	cluster_key = 'flow_domain'
	adata_slice.obs[cluster_key] = [f"Domain_{int(i)}" for i in cell_clusters]
	adata_slice.obs[cluster_key] = adata_slice.obs[cluster_key].astype('category')

	return adata_slice.obs[cluster_key].values						