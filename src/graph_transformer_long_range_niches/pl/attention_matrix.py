import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from anndata import AnnData
import pandas as pd

import torch

from graph_transformer_long_range_niches.pp import prepare_geome_dataset
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import MinMaxScaler

class SelfAttentionRelevance:
    """ Chefer, H., Gur, S. & Wolf, L. Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers. 
    Preprint at https://doi.org/10.48550/arXiv.2103.15679 (2021).
    """

    def __init__(self, model):
        """
        Initializes the SelfAttentionRelevance class with a TransformerNodeEncoder.

        Parameters
        ----------
            model: torch.nn.Module
                The model to be used for generating relevance.
        """
        self.model = model

    @staticmethod
    def avg_heads(attn_map, grad):
        """
        Rule 5 from Chefer et al.: Averages the heads in the attention map after applying gradients.

        Parameters
        ----------
            attn_map: Tensor
                Attention weight (Key and Query dependent), shape: BHxSxS
            grad: Tensor
                Gradients to apply to the attention map, shape: BHxSxS

        Returns
        -------
            Tensor
                Averaged attention map after applying gradients.
        """
        attn_map = attn_map.reshape(-1, attn_map.shape[-2], attn_map.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        attn_map = grad * attn_map
        attn_map = attn_map.clamp(min=0).mean(dim=0)
        return attn_map

    @staticmethod
    def apply_self_attention_rules(R_ss, cam_ss):
        """
        Rule 6 from Chefer et al.: Applies self-attention rules to update the relevance score.

        Parameters
        ----------
            R_ss: Tensor
                Relevance score matrix.
            cam_ss: Tensor
                Attention map.

        Returns
        -------
            Tensor
                Updated relevance score matrix.
        """
        return torch.matmul(cam_ss, R_ss)

    def generate_relevance(self, padded_h_node, src_padding_mask, category_index=None):
        """
        Generates the relevance score for a given input and category index.

        Parameters
        ----------
            padded_h_node: Tensor
                Padded input node representations.
            src_padding_mask: Tensor
                Padding mask for the input.
            category_index: List[int], optional
                List of indices to consider in the category mask, default is None.
        """
        output, _ = self.model(padded_h_node, src_padding_mask, register_hook=True)
        category_mask = torch.zeros(output.size())
        if category_index is not None:
            category_mask[category_index, :, :] = 1
        loss = (output * category_mask).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        num_tokens = self.model.transformer_encoder.layers[0].get_attn_output().shape[0]
        print(num_tokens)

        #I = torch.eye(num_tokens, num_tokens).cuda()
        I = torch.eye(num_tokens, num_tokens)

        for idx, encoder in enumerate(self.model.transformer_encoder.layers):
            attn_out_weights = encoder.get_attn_output_weights()
            attn_grad = encoder.get_attn_gradients()

            attn_map = self.avg_heads(attn_out_weights, attn_grad)
            #I += self.apply_self_attention_rules(I.cuda(), attn_map.cuda())
            I += self.apply_self_attention_rules(I, attn_map)
        
        return I

def calculate_attention(adata, cfg, model_transformer, obs_col, class_name, attention_obs=None, attention_class=None, library_key=None, split_key = 'split'):
    """
    Parameters
    ----------
        adata: AnnData
        obs_metadata: pandas.Dataframe
            .obs metadata from object of interest
        obs_col: str
            Name of annotation column in .obs where the observation used as classes to 
        class_name: str 
            Name of class in .obs[obs_col] that we are interested in plotting the attention for
        attention_class: str
            If None, all classes in attention_obs are considered
        library_key: Optional[str]
    """
    self_attention_relevance = SelfAttentionRelevance(model_transformer.transformer_encoder)
    
    # subset relevant data
    sub_adata = adata[adata.obs[obs_col] == class_name]
    assert split_key in adata.obs

    cfg.set_new_allowed(True)
    cfg.defrost()
    cfg.dataset.library_key = [library_key]
    cfg.freeze()

    if library_key:  
        assert library_key in sub_adata.obs
        cfg.set_new_allowed(True)
        cfg.defrost()
        cfg.dataset.library_key = [library_key]
        cfg.freeze()
        library_key_list = np.unique(sub_adata.obs[library_key])
    else: 
        library_key_list = [None]
    print(library_key_list)
    
    # load PyG objects for evaluation
    pyg_datas, _ = prepare_geome_dataset(sub_adata, cfg, split_key=split_key) # datas = [datas_train, datas_test]
    datas = [pyg for datas in pyg_datas for pyg in datas]
    data_loader = DataLoader(datas)
    
    sub_adata.obs['cls'] = np.nan
    attention_matrix_dict = {}
    transformer_in_dict = {}
    transformer_out_dict = {}

    
    for batch, library_id in zip(data_loader, library_key_list):
        print('batch: ', batch, 'library_id', library_id)
        transformer_in, transformer_out, src_padding_mask, index_nodes, dec_out = model_transformer.evaluation(batch)
        if not attention_obs:
            attention_index = np.arange(0, len(index_nodes))
        I = self_attention_relevance.generate_relevance(transformer_in, src_padding_mask, category_index=attention_index)
        cls = I[:1, 1:].cpu().detach().numpy() 
        print('cls', len(cls[0]))
        # Create a pandas DataFrame for the attention matrix with obs_names as row and column indices
        if library_key:
            library_mask = (sub_adata.obs[library_key] == library_id)
            library_indices = sub_adata.obs[library_mask].index
            print('library_mask: ', len(library_mask), ' library_indices: ' , len(library_indices), ' index_nodes :', len(index_nodes[0]))
            
            # Then, use these indices to select only the ones in index_nodes[0]
            final_indices = library_indices[index_nodes[0]]
            
            # Now assign the values using these indices
            sub_adata.obs.loc[final_indices, 'cls'] = cls[0]
            attention_matrix_df = pd.DataFrame(
                I[1:, 1:].cpu().detach().numpy(),
                # TODO: check if this is correct
                # index=sub_adata.obs_names[sub_adata.obs[library_key]==library_id][index_nodes[0]],
                # columns=sub_adata.obs_names[sub_adata.obs[library_key]==library_id][index_nodes[0]]
                index = final_indices,
                columns = final_indices
            )
        else:
            sub_adata.obs['cls'][sub_adata.obs_names[index_nodes[0]]] = cls[0] # TODO change
            attention_matrix_df = pd.DataFrame(
                I[1:, 1:].cpu().detach().numpy(),
                index=sub_adata.obs_names[index_nodes[0]],
                columns=sub_adata.obs_names[index_nodes[0]]
            )
        attention_matrix_dict[str(library_id)] = attention_matrix_df
        transformer_in_dict[str(library_id)] = transformer_in
        transformer_out_dict[str(library_id)] = transformer_out
        
    return sub_adata, attention_matrix_dict, transformer_in_dict, transformer_out_dict, dec_out

def normalized_attention(attention_matrix, clamp = 0.05):
    np.fill_diagonal(attention_matrix.values, 0)
    
    # Clamp and scale attention matrix
    scores = torch.tensor(attention_matrix.values)
    if clamp:
        q05, q95 = torch.quantile(scores, clamp), torch.quantile(scores, 1-clamp)
        scores = np.clip(scores, a_min=q05, a_max=q95)
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores)
    return scores

def normalized_class_attention(attention_matrix, clamp: int = 0.05):
    """
    Returns the normalized attention for each class to class in the attention matrix
    Parameters
    ----------
        attention_matrix: PandasDataframe
            Matrix of size NxN with column and row indices belonging to either of K classes
    Returns
    -------
        attn_norm: 
            KxK, where 
    """
    scores = normalized_attention(attention_matrix, clamp)
    attention_matrix = pd.DataFrame(scores, index = attention_matrix.index, columns = attention_matrix.columns)
    
    # Create an empty KxK DataFrame to store the summed and normalized attention values
    class_names = np.unique(attention_matrix.columns)
    K = len(class_names)
    attn_norm = pd.DataFrame(np.zeros((K, K)), index=class_names, columns=class_names)

    # Iterate over each unique cell type combination
    for i, class_i in enumerate(class_names):
        for j, class_j in enumerate(class_names):
            # Find the indices in the original CxC DataFrame that correspond to the given cell types
            indices_i = (attention_matrix.index == class_i)
            indices_j = (attention_matrix.columns == class_j)
            norm_value = attention_matrix.loc[indices_i, indices_j].sum() / len(np.argwhere(indices_i==True))
            summed_value = norm_value.sum() / len(np.argwhere(indices_j==True))
            attn_norm.at[class_i, class_j] = summed_value

    return attn_norm

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
        attn_matrix_key: str ='attention_matrix',
        save_img: str = None,
        add_streamline: bool = False,
        discrete_values: bool = False,
        show_index: bool = False,
    ) -> None:
    """
    Plot the spatial scatter plot of sender and receiver cells, highlighting the sender cells based on the sum 
    of their attention values towards the receiver cells, and vice versa.

    Parameters
    ----------
        adata: AnnData
            Annotated data matrix.
        sender: str or list of str
            Single or multiple cell type(s) considered as the sender in the attention matrix.
        receiver: str
            The cell type considered as the receiver in the attention matrix.
        cell_type_key: str, optional (default: 'cell_type')
            The column in adata.obs that contains the cell type information.
        attn_matrix_key: str, optional (default: 'attention')
            The key in adata.obsm that contains the attention matrix as NumpyArray.
        save_img: str, optional
            Path to save the plot image.
        add_streamline: bool, optional
            If True, adds streamlines to the plot to indicate directionality.
        discrete_values: bool, optional
            If false, plots normalized attention values, if true uses discretized values.
        show_index: bool, optional
            If True, adds an index to the x-axis to indicate position.
    """
    if isinstance(sender, str):
        sender = [sender]

    num_senders = len(sender)
    fig, axes = plt.subplots(num_senders, 2, figsize=(10, 5 * num_senders))

    for i, sender_type in enumerate(sender):
        for j, (current_sender, current_receiver) in enumerate([(sender_type, receiver), (receiver, sender_type)]):
            ax = axes[i, j]

            # Extract the attention matrix from .obsm
            attention_matrix = adata.obsm[attn_matrix_key].copy()
            obs_names = adata.obs[cell_type_key]
            attention_matrix = pd.DataFrame(attention_matrix, index=obs_names, columns=obs_names)

            # Compute the sum of attention values for each sender cell towards all receiver cells
            attention_sums = attention_matrix.loc[current_sender, current_receiver].sum(axis=1)

            subadata = adata[adata.obs[cell_type_key].isin([current_sender, current_receiver])]

            subadata.obs['attention_values'] = 0
            subadata.obs.loc[subadata.obs[cell_type_key] == current_sender, 'attention_values'] = attention_sums.values

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

            sq.pl.spatial_scatter(subadata, shape=None, color='colors', ax=ax, palette=[cell_colors], img=False, size_key=2, vmin=attn_min, vmax=attn_max)

            # Calculate total x and y distance
            x_coords = subadata.obsm['spatial'][:, 0]
            y_coords = subadata.obsm['spatial'][:, 1]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)

            # Set axis labels with distances
            ax.set_xlabel(f'X-axis (Range: {x_range:.0f})')
            ax.set_ylabel(f'Y-axis (Range: {y_range:.0f})')

            if show_index:
                x_indices_to_show = np.linspace(min(x_coords), max(x_coords), num=4, dtype=int)
                ax.set_xticks(x_indices_to_show)
                ax.set_xticklabels(x_indices_to_show, rotation=0)
                y_indices_to_show = np.linspace(min(y_coords), max(y_coords), num=4, dtype=int)
                ax.set_yticks(y_indices_to_show)
                ax.set_yticklabels(y_indices_to_show, rotation=0)

            if add_streamline:
                X, Y, U, V = sender_receiver_stream(adata, current_sender, current_receiver, cell_type_key, density=None)
                ax.streamplot(X, Y, U, V, density=0.5, linewidth=1, arrowsize=1.5)

            ax.set_title(f"{'Sender: ' if j == 0 else 'Receiver: '}{current_sender} & {'Receiver: ' if j == 0 else 'Sender: '}{current_receiver}")

    fig.suptitle('Attention value distribution between sender and receiver cells', fontsize=16)
    
    # Save the plot if a path or filename is provided
    if save_img:
        fig.savefig(save_img, bbox_inches='tight')

    plt.show()