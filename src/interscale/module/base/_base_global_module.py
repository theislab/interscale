from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from sklearn.decomposition import NMF, PCA

from ._base_module import BaseModule


class GlobalModule(BaseModule):
    def __init__(self, **base_module_kwargs):

        super().__init__(**base_module_kwargs)

        self.registered_local_component = False
        self.registered_global_component = True

        if self.type_gex_embedding == "PCA":
            self.pca = PCA(n_components=self.n_embed)
            # A fitted sklearn estimator is not part of `module.state_dict()`, and
            # `BaseModel.save` persists nothing else -- so a GlobalModel reloaded for inference
            # would arrive with an UNFITTED pca and silently refit it on the first evaluation
            # batch. The transformer's weights were learned on the basis fitted to the first
            # TRAINING batch; a basis refitted elsewhere differs by rotation and by component
            # sign, so the reloaded model would decode a different space than it was trained on
            # and produce attention that means nothing. These buffers carry the fit through the
            # checkpoint.
            #
            # Registered unconditionally for the PCA branch (not lazily on first fit) because a
            # buffer that does not exist at construction time cannot receive a value from
            # `load_state_dict`, which is exactly when it is needed. Older checkpoints simply
            # have no entry for them; `BaseModel.load` uses strict=False, so they stay zeroed
            # and `pca_fitted_` stays False, reproducing the previous refit-on-load behaviour
            # rather than failing.
            self.register_buffer("pca_mean_", torch.zeros(self.n_input))
            self.register_buffer("pca_components_", torch.zeros(self.n_embed, self.n_input))
            self.register_buffer("pca_fitted_", torch.zeros(1, dtype=torch.bool))
        elif self.type_gex_embedding == "NMF":
            self.nmf = NMF(n_components=self.n_embed, init="random", random_state=0)
        elif self.type_gex_embedding == "Precomputed":
            pass
        elif self.type_gex_embedding is None:
            # No GEX embedding needed when using CombinedModule (local module provides embeddings)
            pass
        else:
            raise ValueError(f"Invalid embedding type: {self.type_gex_embedding}")

    @abstractmethod
    def forward(self, embeddings: torch.Tensor):
        """
        Input:
            embeddings: torch.Tensor
                Size: [N, E], either output of local component or user-provided embeddings.
        """

    def create_gex_embedding(self, embeddings: torch.Tensor, type: Literal["PCA", "NMF", "scvi"]):
        """Generate embeddings for GEX if no local component is used.

        Parameters
        ----------
        batch.x: torch.Tensor
            Size: [N, F]
        type: Literal["PCA", "scvi"]
            Type of embedding to generate.

        Returns
        -------
        gex_embedding: torch.Tensor
            Size: [N, E]
        """
        if type == "PCA":
            # Fit PCA only once (on the first batch that arrives with no fit available), then
            # project every later batch through the stored basis. Two sources of a fit, in
            # order: the buffers restored from a checkpoint, then this process's own first
            # batch. Checking the buffers FIRST is what makes a reloaded model reproduce the
            # basis it was trained on instead of refitting on evaluation data.
            # `_common_step` hands this a numpy array but `GlobalModel.get_model_output` hands
            # it `batch.x` straight off the batch, which is a (possibly CUDA) tensor.
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            if not bool(self.pca_fitted_):
                self.pca.fit(embeddings)
                self.pca_mean_.copy_(torch.as_tensor(self.pca.mean_, dtype=self.pca_mean_.dtype))
                self.pca_components_.copy_(
                    torch.as_tensor(self.pca.components_, dtype=self.pca_components_.dtype)
                )
                self.pca_fitted_.fill_(True)
            return self._pca_transform(embeddings)
        elif type == "NMF":
            if not hasattr(self.nmf, "components_"):
                return self.nmf.fit_transform(embeddings)
            else:
                return self.nmf.transform(embeddings)
        else:
            raise ValueError(f"Invalid embedding type: {type}")

    def _pca_transform(self, embeddings):
        """Project onto the stored PCA basis: ``(X - mean_) @ components_.T``.

        Written out rather than delegated to ``self.pca.transform`` so that the projection
        depends only on the two buffers, which are the only part of the fit that survives a
        checkpoint round trip. ``sklearn``'s own ``transform`` would additionally require the
        estimator's private fitted attributes to be present, which after a reload they are not.
        Equivalent to it for ``whiten=False``, which is the default this module constructs.
        """
        mean = self.pca_mean_.detach().cpu().numpy()
        components = self.pca_components_.detach().cpu().numpy()
        return (np.asarray(embeddings, dtype=np.float64) - mean) @ components.T

    def _process_batch_for_metrics(self, batch, prediction_task, prediction_level, pad_index_nodes, mask_idx_tensor):
        """Process batch to extract y_true and adjusted_mask_idx for metrics calculation.

        mask_idx = torch.tensor([0, 2, 3, 7, 8])
        pad_index_nodes = [[0, 1, 2, 3], [0, 1], [0, 1, 2, 3]]

        Parameters
        ----------
        batch
            Input batch
        prediction_task: str
            Type of prediction task ('classification' or 'regression')
        prediction_level: str
            Level of prediction ('node' or 'graph')
        pad_index_nodes: List[List[int]]
            List of padded node indices: [B, S] or [B,N] if number of nodes in graph are smaller than max_seq_len (S)
        mask_idx_tensor: torch.Tensor
            Indices of masked nodes of shape [N_masked_nodes] with range [0, N_nodes-1]

        Returns
        -------
        y_true: torch.Tensor [N_included_nodes, C] (classification) or [N_included_nodes, F] (regression)
            Ground truth values
        adjusted_mask_idx: torch.Tensor [N_masked nodes]
            Adjusted indices for masked nodes
        entry_mask: torch.Tensor [N_included_nodes, F] | None
            The batch's `gene_mask` gathered and reordered exactly like `y_true`, so that
            `entry_mask[adjusted_mask_idx]` lines up entry-for-entry with the scored predictions.
            `None` whenever the batch carries no gene mask (cell masking, or classification).
        """
        assert prediction_level == "node", "Node specific retrieval only necessary for node-level prediction."

        nr_batches = batch.batch[-1] + 1
        device = batch.x.device

        # Pre-compute batch boundaries (O(B×N) total)
        batch_sizes = torch.tensor([batch.batch.eq(i).sum().item() for i in range(nr_batches)], device=device)
        batch_starts = torch.cat([torch.tensor([0], device=device), batch_sizes.cumsum(0)[:-1]])
        batch_ends = batch_starts + batch_sizes

        # Pre-compute cumulative offsets for adjusted indices
        pad_lengths = torch.tensor([len(pad) for pad in pad_index_nodes], device=device)
        cumulative_offsets = torch.cat([torch.tensor([0], device=device), pad_lengths.cumsum(0)[:-1]])

        adjusted_mask_idx_list = []
        y_true_list = []

        # Gathered in lockstep with y_true below. Gated on the configured strategy, not merely on
        # the attribute being present, so a stale `gene_mask` on a reused Data object cannot turn
        # a cell-masking run into a gene-masking one. Only regression has entries to mask; a
        # classification target is a label per cell, not a gene vector.
        use_gene_mask = self.mask_strategy == "gene" and "regression" in prediction_task
        gene_mask = getattr(batch, "gene_mask", None) if use_gene_mask else None
        entry_mask_list = [] if gene_mask is not None else None

        for i in range(nr_batches):
            batch_start = batch_starts[i].item()
            batch_end = batch_ends[i].item()

            # Find masked indices in this batch range (vectorized)
            mask_in_batch = (mask_idx_tensor >= batch_start) & (mask_idx_tensor < batch_end)
            batch_mask_idx = mask_idx_tensor[mask_in_batch]

            if len(batch_mask_idx) == 0:
                # Extract y_true even if no masked nodes
                mask = batch.batch.eq(i)
                if "classification" in prediction_task:
                    y_true_list.append(batch.y[mask][pad_index_nodes[i]])
                elif "regression" in prediction_task:
                    y_true_list.append(batch.x[mask][pad_index_nodes[i]])
                if entry_mask_list is not None:
                    entry_mask_list.append(gene_mask[mask][pad_index_nodes[i]])
                continue

            # Create pad_indices tensor once
            pad_indices = torch.tensor(pad_index_nodes[i], device=device) + batch_start

            # Vectorized intersection and position finding
            # Use broadcasting: [M, 1] == [1, P] creates [M, P] boolean matrix
            matches = batch_mask_idx.unsqueeze(1) == pad_indices.unsqueeze(0)  # [M, P]
            is_in_pad = matches.any(dim=1)  # [M] - which masked nodes are in pad_indices

            if is_in_pad.any():
                # Get positions of matches in pad_indices (first occurrence)
                positions_in_pad = matches.long().argmax(dim=1)[is_in_pad]  # [M_valid]

                # Adjust indices with cumulative offset
                adjusted_indices = positions_in_pad + cumulative_offsets[i]
                adjusted_mask_idx_list.append(adjusted_indices)

            # Extract y_true for included nodes
            mask = batch.batch.eq(i)
            if "classification" in prediction_task:
                y_true_list.append(batch.y[mask][pad_index_nodes[i]])
            elif "regression" in prediction_task:
                y_true_list.append(batch.x[mask][pad_index_nodes[i]])
            else:
                raise Exception("Choose a valid prediction task (classification or regression).")
            if entry_mask_list is not None:
                entry_mask_list.append(gene_mask[mask][pad_index_nodes[i]])

        # Concatenate results
        y_true = torch.cat(y_true_list, dim=0)
        entry_mask = torch.cat(entry_mask_list, dim=0) if entry_mask_list else None
        adjusted_mask_idx = (
            torch.cat(adjusted_mask_idx_list, dim=0)
            if adjusted_mask_idx_list
            else torch.tensor([], device=device, dtype=torch.long)
        )

        # Every adjusted index must address a valid row of y_true. This is the invariant the
        # offset arithmetic can actually violate, so it stays.
        #
        # A second assertion used to sit here requiring `adjusted_mask_idx.max() >
        # len(pad_index_nodes[0])` whenever nr_batches > 1, i.e. that masked nodes came from more
        # than just the first graph. It was wrong twice over. Graph i's indices start at
        # cumulative_offsets[i], so a masked node that is the *first kept node* of graph 1 gets
        # index exactly len(pad_index_nodes[0]) -- a valid position that `>` rejected. That is
        # reachable whenever the last batch of an epoch holds two graphs and the second is small
        # enough for its single masked node to be node 0, which killed a legnini23 run at epoch 22.
        # It also asserted a property that is not invariant: if a graph's masked set exceeds
        # max_seq_len, _select_masked_nodes legitimately drops some, and that case is caught with
        # an accurate message by the length checks in the combined modules' _common_step, where
        # the local branch's masked-node count is compared against the global branch's.
        if len(adjusted_mask_idx) > 0:
            assert adjusted_mask_idx.max() < len(y_true), (
                f"Mismatch: max(adjusted_mask_idx): {adjusted_mask_idx.max()}, len(y_true): {len(y_true)}"
            )

        if entry_mask is not None:
            assert entry_mask.shape == y_true.shape, (
                f"Mismatch: entry_mask.shape: {tuple(entry_mask.shape)}, y_true.shape: {tuple(y_true.shape)}"
            )

        return y_true, adjusted_mask_idx, entry_mask

    # def _process_batch_for_metrics(self, batch, prediction_task, prediction_level, pad_index_nodes, mask_idx_tensor):
    #     """Process batch to extract y_true and adjusted_mask_idx for metrics calculation.

    #     mask_idx = torch.tensor([0, 2, 3, 7, 8])
    #     pad_index_nodes = [[0, 1, 2, 3], [0, 1], [0, 1, 2, 3]]

    #     Parameters
    #     ----------
    #     batch
    #         Input batch
    #     prediction_task: str
    #         Type of prediction task ('classification' or 'regression')
    #     prediction_level: str
    #         Level of prediction ('node' or 'graph')
    #     pad_index_nodes: List[List[int]]
    #         List of padded node indices: [B, S] or [B,N] if number of nodes in graph are smaller than max_seq_len (S)
    #     mask_idx_tensor: torch.Tensor
    #         Indices of masked nodes of shape [N_masked_nodes] with range [0, N_nodes-1]

    #     Returns
    #     -------
    #     y_true: torch.Tensor [N_included_nodes, C] (classification) or [N_included_nodes, F] (regression)
    #         Ground truth values
    #     adjusted_mask_idx: torch.Tensor [N_masked nodes]
    #         Adjusted indices for masked nodes
    #     """
    #     assert prediction_level == "node", "Node specific retrieval only necessary for node-level prediction."

    #     y_true = []
    #     adjusted_mask_idx = []  # Track new positions of masked nodes
    #     current_offset = 0
    #     start = 0
    #     mask_j = 0
    #     nr_batches = batch.batch[-1] + 1

    #     for i in range(nr_batches):
    #         mask = batch.batch.eq(i)
    #         pad_indices = torch.tensor(pad_index_nodes[i], device=batch.x.device) + start
    #         end = start + torch.sum(mask)

    #         # can not assume that pad_indices is a subset of mask_idx
    #         #TODO: use stack and pop instead
    #         for j, mask_idx in enumerate(mask_idx_tensor[mask_j:]):
    #             if mask_idx > end:
    #                 break
    #             if mask_idx in pad_indices:
    #                 new_idx = torch.where(pad_indices == mask_idx)[0].item()
    #                 adjusted_mask_idx.append(new_idx + current_offset)

    #         current_offset += len(pad_indices)
    #         start = end
    #         mask_j = j

    #         # only return y_true for included nodes
    #         if 'classification' in prediction_task:
    #             y_true += batch.y[mask][pad_index_nodes[i]].clone().detach()
    #         elif 'regression' in prediction_task:
    #             y_true += batch.x[mask][pad_index_nodes[i]].clone().detach()
    #         else:
    #             raise Exception('Choose a valid prediction tasks (graph or node).')
    #         assert len(mask) >= len(pad_indices) >= len(adjusted_mask_idx), "mask, pad_indices, adjusted_mask_idx are not consistent"

    #     y_true = torch.stack(y_true)
    #     adjusted_mask_idx = torch.tensor(adjusted_mask_idx, device=y_true.device)

    #     assert max(adjusted_mask_idx) < len(y_true), f"Mismatch: max(adjusted_mask_idx): {max(adjusted_mask_idx)}, len(y_true): {len(y_true)}"
    #     if nr_batches > 1:
    #         assert  max(adjusted_mask_idx) > len(pad_index_nodes[0]), f"No masked node included from all batches: first batch has {len(pad_index_nodes[0])} nodes, but {max(adjusted_mask_idx)} nodes were included"

    #     assert torch.equal(y_true_new, y_true), "y_true_new and y_true are not consistent"
    #     assert torch.equal(adjusted_mask_idx_new, adjusted_mask_idx), "adjusted_mask_idx_new and adjusted_mask_idx are not consistent"

    #     return y_true, adjusted_mask_idx

    def predict(self, global_embedding, src_padding_mask, prediction_level):
        """Predict with the decoder.

        Parameters
        ----------
        global_embedding: torch.Tensor
            Size: [N, E]
        prediction_level: Literal["node", "graph"]
            Level of prediction
        """
        ## Graph-level prediction: get cls_token from last position
        if "graph" in prediction_level:
            cls_token = global_embedding[-1, :, :]  # [B, E]
            return self.decoder(cls_token)
        ## Node-level prediction: remove cls_token from last position
        elif "node" in prediction_level:
            h_graph = global_embedding[:-1]  # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2))  # [B, S, E]
            src_padding_mask = src_padding_mask[:, :-1]  # True = Pad, False = Node
            masked_output = h_graph[~src_padding_mask]  # [N, E]
            return self.decoder(masked_output)
        else:
            raise Exception("Choose a valid prediction tasks (graph or node).")

    def _common_step(self, batch, prediction_task: str, prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.

        Returns
        -------
        local_embedding: torch.Tensor
            Size: [N, E]
        global_embedding: torch.Tensor
            Size: [N, E] with SEQ_LEN_MASK for padding nodes.
        y_pred: torch.Tensor
            Size: [N, C] (classification) or [N, F] (regression) with SEQ_LEN_MASK for padding nodes.
        y_true: torch.Tensor
            Size: [N, C] (classification) or [N, F] (regression) with SEQ_LEN_MASK for padding nodes.
        attn_matrix: torch.Tensor
            Stacked per-layer attention weights.
        entry_mask: torch.Tensor | None
            Size: [N, F] under gene masking, marking the entries the loss is scored on.
        """
        # Mask nodes  - before GEX embedding because otherwise embedding contains information about masked nodes
        batch_masked, mask_idx, _ = self._common_step_masking(batch)
        if hasattr(batch_masked, "embeddings"):
            embedding = batch_masked.embeddings
        else:
            embedding = self.create_gex_embedding(batch_masked.x.cpu().numpy(), type=self.type_gex_embedding)

        embedding = torch.tensor(embedding, dtype=torch.float32, device=batch_masked.x.device)
        assert embedding.shape == (batch_masked.x.shape[0], self.n_embed), (
            f"Mismatch: embedding.shape: {embedding.shape}, batch_masked.x.shape: {batch_masked.x.shape}"
        )
        assert not torch.any(torch.isnan(embedding)), "embedding contains NaN values"

        padded_emb, src_padding_mask, pad_index_nodes, attention_mask = self.common_step_local_to_global(
            batch_masked, embedding
        )
        assert not torch.any(torch.isnan(padded_emb)), "padded_emb contains NaN values"

        global_embedding, src_padding_mask, attn_matrix = self.forward(padded_emb, src_padding_mask, attention_mask)
        # global_embedding, src_padding_mask = self.forward(padded_emb, src_padding_mask, attention_mask)
        assert not torch.any(torch.isnan(global_embedding)), "global_embedding contains NaN values"

        y_pred = self.predict(global_embedding, src_padding_mask, prediction_level)

        if prediction_task == "classification" and prediction_level == "graph":
            y_true = batch.y[batch.ptr[:-1]]
            entry_mask = None
        else:
            y_true, adjusted_mask_idx, entry_mask = self._process_batch_for_metrics(
                batch, prediction_task, prediction_level, pad_index_nodes, mask_idx
            )
            y_pred = y_pred[adjusted_mask_idx]
            y_true = y_true[adjusted_mask_idx]
            if entry_mask is not None:
                entry_mask = entry_mask[adjusted_mask_idx]

        assert len(y_pred) == len(y_true), "y_pred and y_true are not consistent"
        assert not torch.any(torch.isnan(y_pred)), "y_pred contains NaN values"
        assert not torch.any(torch.isnan(y_true)), "y_true contains NaN values"

        return None, global_embedding, y_pred, y_true, attn_matrix, entry_mask

    def get_global_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)

    # acts as a factory method to create a module from a config
    @staticmethod
    def from_config(cfg, **kwargs):
        module_name = cfg.model.global_component.name
        params = cfg.model.global_component.parameters.copy()  # Make a copy to avoid modifying the original

        if module_name == "self-attn-transformer":
            from interscale.module.global_modules import TransformerNodeEncoderHook

            return TransformerNodeEncoderHook(
                max_seq_len=params["max_seq_len"],
                n_heads=params["n_heads"],
                dropout_global=params["dropout_global"],
                act_func=params["activation_func"],
                num_layers=params["num_layers"],
                dim_feedforward=params["dim_feedforward"],
                long_range_attention=params["long_range_attention"],
                **kwargs,
            )
        # Add more elifs for other modules
        else:
            raise ValueError(f"Unknown local module name: {module_name}")
