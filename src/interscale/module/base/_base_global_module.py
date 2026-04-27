from abc import abstractmethod
from typing import Literal

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
            # Fit PCA only once (on first batch), then use transform for subsequent batches
            # This avoids expensive refitting on every batch during training
            if not hasattr(self.pca, "components_"):
                return self.pca.fit_transform(embeddings)
            else:
                return self.pca.transform(embeddings)
        elif type == "NMF":
            if not hasattr(self.nmf, "components_"):
                return self.nmf.fit_transform(embeddings)
            else:
                return self.nmf.transform(embeddings)
        else:
            raise ValueError(f"Invalid embedding type: {type}")

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

        # Concatenate results
        y_true = torch.cat(y_true_list, dim=0)
        adjusted_mask_idx = (
            torch.cat(adjusted_mask_idx_list, dim=0)
            if adjusted_mask_idx_list
            else torch.tensor([], device=device, dtype=torch.long)
        )

        # Assertions
        if len(adjusted_mask_idx) > 0:
            assert adjusted_mask_idx.max() < len(y_true), (
                f"Mismatch: max(adjusted_mask_idx): {adjusted_mask_idx.max()}, len(y_true): {len(y_true)}"
            )
            if nr_batches > 1:
                assert adjusted_mask_idx.max() > len(pad_index_nodes[0]), "No masked node included from all batches"

        return y_true, adjusted_mask_idx

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
        """
        # Mask nodes  - before GEX embedding because otherwise embedding contains information about masked nodes
        batch_masked, mask_idx = self._common_step_masking(batch)
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
        else:
            y_true, adjusted_mask_idx = self._process_batch_for_metrics(
                batch, prediction_task, prediction_level, pad_index_nodes, mask_idx
            )
            y_pred = y_pred[adjusted_mask_idx]
            y_true = y_true[adjusted_mask_idx]

        assert len(y_pred) == len(y_true), "y_pred and y_true are not consistent"
        assert not torch.any(torch.isnan(y_pred)), "y_pred contains NaN values"
        assert not torch.any(torch.isnan(y_true)), "y_true contains NaN values"

        return None, global_embedding, y_pred, y_true, attn_matrix

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
