import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import squidpy as sq
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _prepare_spatial_metadata(adata_slice, fov_id, cell_type_col, cell_list, grid_res):
    """
    Prepare spatial metadata for visualization or analysis, including coordinate
    scaling, grid generation for interpolation, and color mapping for cell types.

    Parameters
    ----------
    adata_slice : AnnData
        An AnnData object containing a single slice or FOV of spatial data.
    fov_id : str
        The unique identifier for the FOV within the spatial
        metadata of the AnnData object.
    cell_type_col : str
        The column name in `adata_slice.obs` that contains cell type annotations.
    cell_list : list or None
        A list of specific cell types to include. If None, all categories in
        `cell_type_col` are used.
    grid_res : int
        The resolution (number of points per axis) for the generated spatial grid.

    Returns
    -------
    sf : float
        The scale factor applied to the spatial coordinates.
    grid_x : numpy.ndarray
        2D meshgrid of X coordinates.
    grid_y : numpy.ndarray
        2D meshgrid of Y coordinates.
    X_lin : numpy.ndarray
        1D array of linearly spaced points along the X axis.
    Y_lin : numpy.ndarray
        1D array of linearly spaced points along the Y axis.
    color_map : dict
        A dictionary mapping cell type names to hex color strings.
    categories : list
        The list of cell type categories processed.
    """
    # Scaling
    sf = 1.0
    try:
        s_data = adata_slice.uns["spatial"][fov_id]
        sf = s_data["scalefactors"].get("tissue_hires_scalef", 1.0)
    except (KeyError, AttributeError):
        pass

    spatial_coords = adata_slice.obsm["spatial"] * sf
    x_min, y_min = spatial_coords.min(axis=0)
    x_max, y_max = spatial_coords.max(axis=0)

    X_lin = np.linspace(x_min - (5 * sf), x_max + (5 * sf), grid_res)
    Y_lin = np.linspace(y_min - (5 * sf), y_max + (5 * sf), grid_res)
    grid_x, grid_y = np.meshgrid(X_lin, Y_lin)

    # Color Mapping
    categories = list(adata_slice.obs[cell_type_col].cat.categories)
    if cell_list is not None:
        categories = [cat for cat in categories if cat in cell_list]

    if f"{cell_type_col}_colors" in adata_slice.uns:
        all_colors = adata_slice.uns[f"{cell_type_col}_colors"]
        color_map = {
            cat: color
            for cat, color in zip(adata_slice.obs[cell_type_col].cat.categories, all_colors)
            if cat in categories
        }
    else:
        import matplotlib.cm as cm

        colors = [to_hex(c) for c in cm.tab20(np.linspace(0, 1, len(categories)))]
        color_map = dict(zip(categories, colors))

    return sf, grid_x, grid_y, X_lin, Y_lin, color_map, categories


def _compute_vector_fields(
    adata_slice, categories, window_key, cell_type_col, cell_list, sf, grid_x, grid_y, max_dist, inter_only
):
    """
    Compute vector fields (U and V components) representing spatial interaction
    flows for each cell type category.

    This function iterates through local windows, processes attention matrices
    to derive net interaction flows, and projects these flows onto a spatial
    grid using Gaussian kernels.

    Parameters
    ----------
    adata_slice : AnnData
        AnnData object containing spatial coordinates and attention matrices.
    categories : list
        List of cell type categories to process.
    window_key : str
        Key in `adata_slice.obs` identifying local spatial windows.
    cell_type_col : str
        Column name for cell type annotations.
    cell_list : list or None
        Optional subset of cell types to consider.
    sf : float
        Scaling factor for coordinates.
    grid_x : numpy.ndarray
        X coordinates of the interpolation grid.
    grid_y : numpy.ndarray
        Y coordinates of the interpolation grid.
    max_dist : float
        Maximum distance for spatial influence and Gaussian kernel bandwidth.
    inter_only : bool
        If True, only consider interactions between different cell types
        (exclude self-interactions).

    Returns
    -------
    U_dict : dict of numpy.ndarray
        X-direction vector components for each cell type, mapped to the grid.
    V_dict : dict of numpy.ndarray
        Y-direction vector components for each cell type, mapped to the grid.
    """
    U_dict = {cat: np.zeros_like(grid_x) for cat in categories}
    V_dict = {cat: np.zeros_like(grid_x) for cat in categories}

    windows = adata_slice.obs[window_key].unique()

    for win in windows:
        win_mask = adata_slice.obs[window_key] == win
        adata_win = adata_slice[win_mask]

        if cell_list is not None:
            cell_mask = adata_win.obs[cell_type_col].isin(cell_list)
            adata_win = adata_win[cell_mask].copy()

        n_win = len(adata_win.obs)
        if n_win < 2:
            continue

        # Attention Matrix processing
        M = pd.DataFrame(
            adata_win.obsm["_attn_matrix"][:, :n_win], index=adata_win.obs_names, columns=adata_win.obs_names
        )
        M_net = M - M.T

        if inter_only:
            types = adata_win.obs[cell_type_col].astype(str).values
            type_matrix = types[:, None] != types[None, :]
            M_net_values = M_net.values
            M_net_values[~type_matrix] = 0
            M_net = pd.DataFrame(M_net_values, index=M_net.index, columns=M_net.columns)

        max_val = np.max(np.abs(M_net.values))
        if max_val > 0:
            M_net /= max_val

        pos_win = adata_win.obsm["spatial"] * sf
        types_win = adata_win.obs[cell_type_col].values

        for j, cell_j_name in enumerate(adata_win.obs_names):
            cell_type = types_win[j]
            if cell_type not in categories:
                continue

            net_flows = M_net.iloc[:, j].values
            positive_flows = np.maximum(net_flows, 0)
            if np.sum(positive_flows) == 0:
                continue

            # Vector calculation
            s_coord = pos_win[j]
            diff = pos_win - s_coord
            dist = np.linalg.norm(diff, axis=1)
            unit_diff = diff / (dist[:, np.newaxis] + 1e-6)
            s_weight = np.exp(-(dist**2) / (2 * max_dist**2))

            v_cell = np.sum(unit_diff * (positive_flows * s_weight)[:, np.newaxis], axis=0)

            # Grid distribution
            g_dist_sq = (grid_x - s_coord[0]) ** 2 + (grid_y - s_coord[1]) ** 2
            kernel = np.exp(-g_dist_sq / (2 * (max_dist / 4) ** 2))

            U_dict[cell_type] += v_cell[0] * kernel
            V_dict[cell_type] += v_cell[1] * kernel

    return U_dict, V_dict


def plot_all_spatial_net_streams(
    adata,
    fov_id,
    fov_key="fov",
    window_key="sliding_window_assignment",
    cell_type_col="cell_type_coarse",
    grid_res=50,
    max_dist=None,
    k_dist=0.05,
    density=1.5,
    ax=None,
    additional_embeddings=None,
    return_streams=False,
    cell_list=None,
    inter_only=False,
    **kwargs,
):
    """
    Visualize spatial interaction flows using streamplots overlaid on a spatial scatter plot.

    This function coordinates the extraction of spatial metadata, computation of vector
    fields based on attention matrices, and the final rendering of directional flows
    (streams) for different cell types.

    Parameters
    ----------
    adata : anndata.AnnData
        The complete AnnData object.
    fov_id : str
        The specific Field of View to visualize.
    fov_key : str
        Key in `adata.obs` identifying FOVs. Defaults to "fov".
    window_key : str
        Key identifying local windows for flow computation.
    cell_type_col : str
        Column in `adata.obs` with cell type labels.
    grid_res : int
        Resolution of the interpolation grid. Defaults to 50.
    max_dist : float
        Maximum influence distance for vectors. If None, calculated via `k_dist`.
    k_dist : float
        Fraction of the FOV width to use as `max_dist` if `max_dist` is None.
    density : float
        Density of the streamplot lines. Defaults to 1.5.
    ax : matplotlib.axes.Axes
        Pre-existing axes for plotting.
    additional_embeddings : str
        Additional color mapping for the background scatter plot.
    return_streams : bool
        If True, returns the computed vector fields instead of plotting.
    cell_list : list
        Subset of cell types to include in the flow analysis.
    inter_only : bool
        If True, excludes self-interactions from the vector fields.
    **kwargs
        Passed to `squidpy.pl.spatial_scatter`.

    Returns
    -------
    ax : matplotlib.axes.Axes or dict
        The plot axes, or a dictionary of vector fields if `return_streams` is True.
    """
    # 1. Slice data
    slice_mask = adata.obs[fov_key] == fov_id
    adata_slice = adata[slice_mask].copy()

    # 2. Setup Metadata & Grid
    sf, grid_x, grid_y, X_lin, Y_lin, color_map, categories = _prepare_spatial_metadata(
        adata_slice, fov_id, cell_type_col, cell_list, grid_res
    )

    # 3. Compute Vectors
    if max_dist is None:
        max_dist = (X_lin.max() - X_lin.min()) * k_dist

    U_dict, V_dict = _compute_vector_fields(
        adata_slice, categories, window_key, cell_type_col, cell_list, sf, grid_x, grid_y, max_dist, inter_only
    )

    if return_streams:
        return {cat: (U_dict[cat], V_dict[cat]) for cat in categories}, X_lin, Y_lin

    # 4. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    emb = additional_embeddings or cell_type_col
    sq.pl.spatial_scatter(
        adata_slice,
        color=emb,
        library_key=fov_key,
        library_id=[fov_id],
        ax=ax,
        spatial_key="spatial",
        img=False,
        **kwargs,
    )

    legend_elements = []
    for cat in categories:
        U, V = U_dict[cat], V_dict[cat]
        mag = np.sqrt(U**2 + V**2)
        if np.max(mag) == 0:
            continue

        thresh = 0.01 * np.max(mag)
        Un = np.divide(U, mag, out=np.full_like(U, np.nan), where=mag > thresh)
        Vn = np.divide(V, mag, out=np.full_like(V, np.nan), where=mag > thresh)

        if not np.all(np.isnan(Un)):
            rgb = to_rgb(color_map[cat])
            hsv = rgb_to_hsv(rgb)
            dark_color = to_hex(hsv_to_rgb([hsv[0], hsv[1], hsv[2] * 0.7]))

            ax.streamplot(X_lin, Y_lin, Un, Vn, color=dark_color, linewidth=1.2, density=density, arrowsize=1.2)
            legend_elements.append(Line2D([0], [0], color=dark_color, lw=2, label=f"Flow: {cat}"))

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.15),
            title=f"Net Flow Directions - {cell_type_col}",
        )

    return ax


def calculate_divergence(U, V):
    return np.gradient(U, axis=1) + np.gradient(V, axis=0)


def cluster_spatial_flows(U_dict, V_dict, n_clusters=5):
    """
    Perform unsupervised clustering of spatial regions based on multi-type flow signatures.

    This function builds a high-dimensional feature matrix for each grid point,
    incorporating flow magnitude and divergence for all cell types, and clusters
    them to identify distinct functional spatial domains.

    Parameters
    ----------
    U_dict : dict of numpy.ndarray
        X-direction grid components per cell type.
    V_dict : dict of numpy.ndarray
        Y-direction grid components per cell type.
    n_clusters : int, optional
        The number of spatial domains to identify. Defaults to 5.

    Returns
    -------
    cluster_grid : numpy.ndarray
        A 2D array (matching the grid shape) where each pixel contains its cluster ID.
    X_scaled : numpy.ndarray
        The standardized feature matrix used for clustering (n_points, n_features).

    """
    categories = list(U_dict.keys())
    grid_shape = list(U_dict.values())[0].shape
    n_points = grid_shape[0] * grid_shape[1]

    # 1. Feature Engineering: Build a "Flow Signature" for each grid point
    # We concatenate U and V for all cell types: [U_cat1, V_cat1, U_cat2, V_cat2, ...]
    feature_list = []
    for cat in categories:
        div = calculate_divergence(U_dict[cat], V_dict[cat])
        mag = np.sqrt(U_dict[cat] ** 2 + V_dict[cat] ** 2)
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

    # clusters_c = [f"Domain_{i}" for i in clusters]

    # Reshape clusters back to grid dimensions
    cluster_grid = clusters.reshape(grid_shape)

    return cluster_grid, X_scaled


def plot_flow_clusters(cluster_grid, X_lin, Y_lin, adata_slice, fov_id, fov_key="fov", cell_type_col=None, **kwargs):
    """
    Visualize identified flow domains as a segmented background for spatial transcriptomics data.

    This function overlays a Voronoi-like heatmap (representing clustered interaction
    patterns) with the actual cell positions to provide spatial context to the
    unsupervised domains.

    Parameters
    ----------
    cluster_grid : numpy.ndarray
        2D array of cluster IDs assigned to each grid point.
    X_lin : numpy.ndarray
        1D array of X-axis coordinates for the grid.
    Y_lin : numpy.ndarray
        1D array of Y-axis coordinates for the grid.
    adata_slice : anndata.AnnData
        The sliced AnnData object for the current FOV.
    fov_id : str
        The specific Field of View identifier.
    fov_key : str
        The key in `adata.obs` for FOVs. Defaults to "fov".
    cell_type_col : str
        The column to use for coloring cells in the scatter plot.
    **kwargs
        Additional arguments passed to `squidpy.pl.spatial_scatter`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    unique_clusters = np.unique(cluster_grid)
    n_clusters = len(unique_clusters)

    cmap = plt.get_cmap("Set3", n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Overlay the original cell positions
    sq.pl.spatial_scatter(
        adata_slice,
        color=cell_type_col,
        library_key=fov_key,
        library_id=[fov_id],
        ax=ax,
        spatial_key="spatial",
        alpha=0.4,
        img=False,
        title=f"Unsupervised Flow Domains (n={len(np.unique(cluster_grid))})",
        **kwargs,
    )
    # Plot the clusters as a heatmap (Voronoi-like segmentation of flow)
    im = ax.pcolormesh(
        X_lin,
        Y_lin,
        cluster_grid,
        cmap=cmap,
        alpha=0.4,
        shading="auto",
        vmin=unique_clusters.min() - 0.5,
        vmax=unique_clusters.max() + 0.5,
    )

    legend_handles = []
    for i, cluster_id in enumerate(unique_clusters):
        color = cmap(i)
        patch = mpatches.Patch(color=color, label=f"Domain {int(cluster_id)}")
        legend_handles.append(patch)

    ax.legend(handles=legend_handles, title="Flow Domains", loc="center left", bbox_to_anchor=(1, 0.5))
    return fig, ax


def characterize_flow_clusters(U_dict, V_dict, cluster_grid, k=1):
    """
    Characterize identified spatial domains by identifying the role of each cell type
    (Source, Sink, or Neutral) based on their divergence values.

    Parameters
    ----------
    U_dict : dict of numpy.ndarray
        X-direction grid components per cell type.
    V_dict : dict of numpy.ndarray
        Y-direction grid components per cell type.
    cluster_grid : numpy.ndarray
        2D array of cluster IDs assigned to each grid point.
    k : float, optional
        Sensitivity multiplier for the standard deviation threshold.
        Higher values make role assignment more stringent. Defaults to 1.

    Returns
    -------
    pd.DataFrame
        A summary table containing the cell type, cluster ID, average divergence,
        and the assigned functional role for each domain.
    """
    results = []
    categories = list(U_dict.keys())
    k = 1
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
            mask = cluster_grid == cluster_id
            avg_div = np.mean(div[mask])

            if avg_div > source_thresh:
                role = "Source"
            elif avg_div < sink_thresh:
                role = "Sink"
            else:
                role = "Neutral"

            results.append({"cell_type": cat, "cluster_id": cluster_id, "avg_divergence": avg_div, "role": role})

    return pd.DataFrame(results)


def map_clusters_to_cells(cluster_grid, X_lin, Y_lin, adata, fov_id, fov_key="fov"):
    """
    Assign grid-based flow domain annotations to individual cells using
    nearest-neighbor matching.

    Parameters
    ----------
    cluster_grid : numpy.ndarray
        2D array of cluster IDs assigned to grid points.
    X_lin : numpy.ndarray
        1D array of X coordinates for the grid.
    Y_lin : numpy.ndarray
        1D array of Y coordinates for the grid.
    adata : AnnData
        The original AnnData object.
    fov_id : str
        The Field of View identifier to process.
    fov_key : str, optional
        Key in `adata.obs` identifying FOVs. Defaults to "fov".

    Returns
    -------
    numpy.ndarray
        A categorical array of domain assignments for each cell in the slice.
    """
    # Create the grid coordinates for KDTree
    grid_x, grid_y = np.meshgrid(X_lin, Y_lin)

    # Flat the coordinates to create a list of (x, y) points
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # 2. Create a KDTree for efficient nearest neighbor search
    tree = KDTree(grid_coords)

    adata_slice = adata[adata.obs[fov_key] == fov_id].copy()

    sf = 1.0
    try:
        s_data = adata_slice.uns["spatial"][fov_id]
        sf = s_data["scalefactors"].get("tissue_hires_scalef", 1.0)
    except (KeyError, AttributeError):
        sf = 1.0

    # Get cell coordinates and scale them to match the grid
    cell_coords = adata_slice.obsm["spatial"] * sf

    # Search for the nearest grid point for each cell
    dists, indices = tree.query(cell_coords)

    # Get the cluster assignment for each cell based on the nearest grid point
    flat_clusters = cluster_grid.ravel()
    cell_clusters = flat_clusters[indices]

    # Save
    cluster_key = "flow_domain"
    adata_slice.obs[cluster_key] = [f"Domain_{int(i)}" for i in cell_clusters]
    adata_slice.obs[cluster_key] = adata_slice.obs[cluster_key].astype("category")

    return adata_slice.obs[cluster_key].values


def compute_hierarchical_net_flow(
    adata,
    window_key="sliding_window_assignment",
    sample_key="fov",
    condition_key=None,
    cell_type_col="cell_type_coarse",
    compute_net=True,
):
    """
    Compute net communication flows across a hierarchy:
    Windows -> Samples (FOVs) -> Experimental Conditions.

    This function aggregates attention-based interactions into a directed flow
    matrix, allowing for statistical comparison between different experimental groups.

    Parameters
    ----------
    adata : AnnData
        The annotated data object containing attention matrices in `.obsm`.
    window_key : str, optional
        Key in `.obs` identifying the sliding windows.
    sample_key : str, optional
        Key in `.obs` identifying the samples or Fields of View (FOV).
    condition_key : str, optional
        Key in `.obs` for experimental conditions (e.g., 'Control' vs 'Treated').
        If None, all samples are aggregated together.
    cell_type_col : str, optional
        Key in `.obs` for cell type annotations.
    compute_net : bool, optional
        If True, computes net flow (A->B minus B->A). If False, returns raw
        aggregated attention directed from sender to receiver.

    Returns
    -------
    dict
        A dictionary where keys are conditions and values are dictionaries
        containing 'mean' and 'std' DataFrames of the flows.
    """
    cell_types = sorted(adata.obs[cell_type_col].unique())
    windows = adata.obs[window_key].unique()

    # --- 1. Window Level Flow Calculation ---
    window_results = {}
    for win in windows:
        sub = adata[adata.obs[window_key] == win]
        n_win = len(sub.obs)

        # Get attention matrix for current window
        M = pd.DataFrame(sub.obsm["_attn_matrix"][:, :n_win], index=sub.obs_names, columns=sub.obs_names)

        agg_matrix = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
        for ct_s in cell_types:
            idx_s = sub.obs_names[sub.obs[cell_type_col] == ct_s]
            if len(idx_s) == 0:
                continue
            for ct_r in cell_types:
                idx_r = sub.obs_names[sub.obs[cell_type_col] == ct_r]
                if len(idx_r) == 0:
                    continue

                # Interaction density: mean value per pair
                agg_matrix.loc[ct_s, ct_r] = M.loc[idx_s, idx_r].values.mean()

        if compute_net:
            # Compute net flow (A->B - B->A) ((change sign to consider information as opposite of attention))
            net_matrix = agg_matrix.T - agg_matrix
        else:
            net_matrix = agg_matrix.T
            for ct_r in cell_types:
                net_matrix.loc[ct_r, ct_r] = 0
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
                np.nanmean(np.stack(flows), axis=0), index=cell_types, columns=cell_types
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
                    "mean": pd.DataFrame(np.nanmean(stacked, axis=0), index=cell_types, columns=cell_types),
                    "std": pd.DataFrame(np.nanstd(stacked, axis=0), index=cell_types, columns=cell_types),
                }
    else:
        # Aggregate everything if no condition is provided
        flows = [df.values for df in sample_net_flows.values()]
        if flows:
            stacked = np.stack(flows)
            final_results["all_samples"] = {
                "mean": pd.DataFrame(np.nanmean(stacked, axis=0), index=cell_types, columns=cell_types),
                "std": pd.DataFrame(np.nanstd(stacked, axis=0), index=cell_types, columns=cell_types),
            }

    return final_results


def plot_global_directionality(mean_df, std_df, only_positive=True, title="Net Flow", figsize=(8, 6)):
    """
    Visualize aggregated net flow between cell types using a dot plot.

    The plot represents the flow from a 'Sender' to a 'Receiver'.
    - The color intensity represents the average flow magnitude.
    - The size of the dot represents the consistency (inverse of standard deviation)
      across samples/windows.

    Parameters
    ----------
    mean_df : pd.DataFrame
        Matrix of mean net flows (Sender as rows, Receiver as columns).
    std_df : pd.DataFrame
        Matrix of flow standard deviations across samples.
    only_positive : bool, optional
        If True, only displays positive flow values (A -> B). Defaults to True.
    title : str, optional
        Title of the plot.
    figsize : tuple, optional
        Dimensions of the figure.
    """
    # 1. Melt the data
    plot_data = mean_df.reset_index().melt(id_vars="index")
    plot_data.columns = ["Sender", "Receiver", "Flow"]

    # 2. Define and Enforce the same order for both axes
    categories = sorted(mean_df.index.unique())

    # Convert to Categorical with a fixed list of categories
    plot_data["Sender"] = pd.Categorical(plot_data["Sender"], categories=categories)
    plot_data["Receiver"] = pd.Categorical(plot_data["Receiver"], categories=categories)

    # 3. Calculate Consistency
    std_flat = std_df.values.flatten()
    plot_data["Consistency"] = 1 / (std_flat + 1e-9)

    # Filter only positive flows
    if only_positive:
        plot_data = plot_data[plot_data["Flow"] > 0]
    else:
        plot_data = plot_data[plot_data["Flow"].abs() > 0]

    # 4. Plotting
    plt.figure(figsize=figsize)

    # Now Seaborn will use the categorical order automatically
    sns.scatterplot(
        data=plot_data,
        x="Receiver",
        y="Sender",
        size="Consistency",
        hue="Flow",
        palette="YlOrRd" if only_positive else "coolwarm",
        sizes=(20, 500),
    )

    # Force axes to show all categories in the right order
    plt.xticks(ticks=range(len(categories)), labels=categories, rotation=45)
    plt.yticks(ticks=range(len(categories)), labels=categories)

    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.title(title)
    plt.tight_layout()
    plt.show()
