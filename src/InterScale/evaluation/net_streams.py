import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import squidpy as sq
import seaborn as sns
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
	cell_list=None,
	inter_only=False,
	**kwargs
):
	# 1. Prepare FOV slice
	slice_mask = adata.obs[fov_key] == fov_id
	adata_slice = adata[slice_mask].copy()
	
	# Extract color mapping for all categories
	categories = list(adata.obs[cell_type_col].cat.categories)
	if cell_list is not None:
		categories = [cat for cat in categories if cat in cell_list]
	if f'{cell_type_col}_colors' in adata.uns:
		colors = list(adata.uns[f'{cell_type_col}_colors'])
		if cell_list is not None:
			colors = [color for cat, color in zip(adata.obs[cell_type_col].cat.categories, colors) if cat in cell_list]
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

		# --- NEW FILTER FOR CELL_LIST ---
		# If cell_list is provided, keep only those cells in the current window
		if cell_list is not None:
			cell_mask = adata_win.obs[cell_type_col].isin(cell_list)
			adata_win = adata_win[cell_mask].copy()
		# --------------------------------
		n_win = len(adata_win.obs)
		if n_win < 2: continue
		




		# Matrix of all-vs-all attention in the window
		M = pd.DataFrame(
			adata_win.obsm['_attn_matrix'][:, :n_win],
			index=adata_win.obs_names,
			columns=adata_win.obs_names
		)
		M_net = M - M.T  # Net flow between all cells

		if inter_only:
			# Create a mask where row type != column type
			# Reshape types to compare every pair
			types = adata_win.obs[cell_type_col].astype(str).values
			type_matrix = types[:, None] != types[None, :]
			M_net_values = M_net.values
			M_net_values[~type_matrix] = 0
			# Apply mask to M_net: set same-type interactions to 0

			M_net = pd.DataFrame(M_net_values, index=M_net.index, columns=M_net.columns)


		max_net_flow = np.max(np.abs(M_net.values))

		if max_net_flow > 0:
			M_net = M_net / max_net_flow

		pos_win = adata_win.obsm['spatial'] * sf
		types_win = adata_win.obs[cell_type_col].values
		
		for j, cell_j_name in enumerate(adata_win.obs_names):
			s_coord = pos_win[j]
			cell_type = types_win[j]
			if cell_type not in categories: continue
			# Identify flows incoming to cell j
			net_flows = M_net.iloc[:, j].values
			positive_flows = np.maximum(net_flows, 0)
			
			if np.sum(positive_flows) == 0: continue
			
			# Compute distance-weighted direction vector
			diff = pos_win - s_coord
			dist = np.linalg.norm(diff, axis=1)
			unit_diff = diff / (dist[:, np.newaxis] + 1e-6)
			s_weight = np.exp(-dist**2 / (2 * max_dist**2))
			#s_weight=1
			# Calculate local cell flow vector
			v_cell = np.sum(unit_diff * (positive_flows * s_weight)[:, np.newaxis], axis=0)
			
			# Distribute the vector onto the grid for its specific cell type
			g_dist_sq = (grid_x - s_coord[0])**2 + (grid_y - s_coord[1])**2
			kernel = np.exp(-g_dist_sq / (2 * (max_dist/4)**2))
			
			U_dict[cell_type] += v_cell[0] * kernel
			V_dict[cell_type] += v_cell[1] * kernel



	if return_streams is False:
		# 3. Plotting
		if ax is None:
			fig, ax = plt.subplots(figsize=(6, 5))
		
		emb = additional_embeddings if additional_embeddings is not None else cell_type_col
		
		# Background scatter plot
		sq.pl.spatial_scatter(
			adata_slice, color=emb, 
			library_key=fov_key, library_id=[fov_id],
			ax=ax, spatial_key='spatial', 
			img=False, **kwargs
		)

		# Grab the existing legend from Squidpy BEFORE adding new streams and legends
		old_legend = ax.get_legend()
		#print(old_legend)
		legend_elements = []

		# 4. Draw streamlines for each cell type individually
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
		
		# Handle legends outside the loop
		if legend_elements:
			# Create the new legend for net flows (bottom center)
			new_legend = ax.legend(
				handles=legend_elements, 
				loc='upper center', 
				ncol=4,
				bbox_to_anchor=(0.5, -0.15), # Pushed slightly lower to avoid overlaps
				title=f"Net Flow Directions - {cell_type_col}"
			)
			
			#Re-attach the old Squidpy legend (center right outside)
			# if old_legend is not None:
			# 	#old_legend.set_bbox_to_anchor((0.5, 0.5))
			# 	old_legend.set_loc('best')
			# 	ax.add_artist(old_legend)
			
		# 	# Force Matplotlib to calculate layout to fit the external legends
		# 	#ax.figure.tight_layout()
		if ax is None:
			plt.tight_layout()
		else:
			return ax

	if return_streams:
		streams = {cat: (U_dict[cat], V_dict[cat]) for cat in categories}
		return streams, X_lin, Y_lin

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

def plot_flow_clusters(cluster_grid, X_lin, Y_lin, adata_slice, fov_id, fov_key='fov', cell_type_col=None, **kwargs):
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
		alpha=0.4, img=False, 
		title=f"Unsupervised Flow Domains (n={len(np.unique(cluster_grid))})",
		**kwargs
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

def characterize_flow_clusters(U_dict, V_dict, cluster_grid,k=2):
	results = []
	categories = list(U_dict.keys())
	k=1
	for cat in categories:
		# Calculate divergence for this category
		div = calculate_divergence(U_dict[cat], V_dict[cat])
		mu = np.mean(div)
		sigma = np.std(div)
		
		# Adaptive thresholds
		source_thresh = mu + (k * sigma)
		sink_thresh = mu - (k * sigma)
		
		# For each cluster ID in the grid
		for cluster_id in np.unique(cluster_grid):
			# Mask the divergence map with the current cluster
			mask = (cluster_grid == cluster_id)
			avg_div = np.mean(div[mask])

			if avg_div > source_thresh:
				role = 'Source'
			elif avg_div < sink_thresh:
				role = 'Sink'
			else:
				role = 'Neutral'
			
			results.append({
				'cell_type': cat,
				'cluster_id': cluster_id,
				'avg_divergence': avg_div,
				'role': role
			})
			
	return pd.DataFrame(results)


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


def compute_hierarchical_net_flow(
	adata, 
	window_key='sliding_window_assignment', 
	sample_key='fov', 
	condition_key=None, 
	cell_type_col='cell_type_coarse',
	compute_net=True
):
	"""
	Computes net flow across hierarchy: Windows -> Samples -> Conditions.
	If condition_key is None, aggregates all samples into 'all_samples'.
	"""
	cell_types = sorted(adata.obs[cell_type_col].unique())
	windows = adata.obs[window_key].unique()
	
	# --- 1. Window Level Flow Calculation ---
	window_results = {}
	for win in windows:
		sub = adata[adata.obs[window_key] == win]
		n_win = len(sub.obs)
		
		# Get attention matrix for current window
		M = pd.DataFrame(
			sub.obsm['_attn_matrix'][:, :n_win],
			index=sub.obs_names, columns=sub.obs_names
		)
		
		agg_matrix = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
		for ct_s in cell_types:
			idx_s = sub.obs_names[sub.obs[cell_type_col] == ct_s]
			if len(idx_s) == 0: continue
			for ct_r in cell_types:
				idx_r = sub.obs_names[sub.obs[cell_type_col] == ct_r]
				if len(idx_r) == 0: continue
				
				# Interaction density: mean value per pair
				agg_matrix.loc[ct_s, ct_r] = M.loc[idx_s, idx_r].values.mean()
		
		if compute_net:
			# Compute net flow (A->B - B->A) ((change sign to consider information as opposite of attention))
			net_matrix = agg_matrix.T - agg_matrix
		else:
			net_matrix = agg_matrix.T
			for ct_r in cell_types:
				net_matrix.loc[ct_r,ct_r]=0
		max_net_flow = np.max(np.abs(net_matrix.values))

		if max_net_flow > 0:
			net_matrix = net_matrix / max_net_flow

		window_results[win] = net_matrix

	# --- 2. Sample Level Aggregation ---
	sample_net_flows = {}
	sample_to_wins = adata.obs.groupby(sample_key)[window_key].unique()
	
	for s, win_ids in sample_to_wins.items():
		flows = [window_results[w].values for w in win_ids if w in window_results]
		if flows:
			sample_net_flows[s] = pd.DataFrame(
				np.nanmean(np.stack(flows), axis=0),
				index=cell_types, columns=cell_types
			)

	# --- 3. Condition Level Aggregation ---
	final_results = {}
	if condition_key is not None:
		sample_to_cond = adata.obs.groupby(sample_key)[condition_key].first()
		for cond in adata.obs[condition_key].unique():
			relevant_samples = sample_to_cond[sample_to_cond == cond].index
			flows = [sample_net_flows[s].values for s in relevant_samples if s in sample_net_flows]
			if flows:
				stacked = np.stack(flows)
				final_results[cond] = {
					'mean': pd.DataFrame(np.nanmean(stacked, axis=0), index=cell_types, columns=cell_types),
					'std': pd.DataFrame(np.nanstd(stacked, axis=0), index=cell_types, columns=cell_types)
				}
	else:
		# Aggregate everything if no condition is provided
		flows = [df.values for df in sample_net_flows.values()]
		if flows:
			stacked = np.stack(flows)
			final_results['all_samples'] = {
				'mean': pd.DataFrame(np.nanmean(stacked, axis=0), index=cell_types, columns=cell_types),
				'std': pd.DataFrame(np.nanstd(stacked, axis=0), index=cell_types, columns=cell_types)
			}

	return final_results


def plot_global_directionality(mean_df, std_df, only_positive=True,title="Net Flow",figsize=(8, 6)):
	# 1. Melt the data
	plot_data = mean_df.reset_index().melt(id_vars='index')
	plot_data.columns = ['Sender', 'Receiver', 'Flow']
	
	# 2. Define and Enforce the same order for both axes
	categories = sorted(mean_df.index.unique())
	
	# Convert to Categorical with a fixed list of categories
	plot_data['Sender'] = pd.Categorical(plot_data['Sender'], categories=categories)
	plot_data['Receiver'] = pd.Categorical(plot_data['Receiver'], categories=categories)
	
	# 3. Calculate Consistency
	std_flat = std_df.values.flatten()
	plot_data['Consistency'] = 1 / (std_flat + 1e-9) 
	
	# Filter only positive flows
	if only_positive:
		plot_data = plot_data[plot_data['Flow'] > 0]
	else:
		plot_data = plot_data[plot_data['Flow'].abs() > 0]
	
	# 4. Plotting
	plt.figure(figsize=figsize)
	
	# Now Seaborn will use the categorical order automatically
	sns.scatterplot(
		data=plot_data, 
		x='Receiver', 
		y='Sender', 
		size='Consistency', 
		hue='Flow', 
		palette='YlOrRd' if only_positive else 'coolwarm', 
		sizes=(20, 500)
	)
	
	# Force axes to show all categories in the right order
	plt.xticks(ticks=range(len(categories)), labels=categories, rotation=45)
	plt.yticks(ticks=range(len(categories)), labels=categories)
	
	# Move legend outside
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	
	plt.title(title)
	plt.tight_layout()
	plt.show()