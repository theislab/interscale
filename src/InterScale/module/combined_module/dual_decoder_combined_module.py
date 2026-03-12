import torch
from typing import Literal

from InterScale.module.base import BaseModuleClass, LocalModuleClass, GlobalModuleClass
from InterScale.tl import aggregate_node_embeddings_to_graph, aggregate_node_values_to_graph
from yacs.config import CfgNode as CN



class DualDecoderCombinedModuleClass(BaseModuleClass):
    """Combined module with decoders for both local and global modules.
    
    This class uses decoders from both the local and global modules to predict
    from their respective embeddings. The loss function combines predictions
    from both decoders on masked tokens.
    """
    
    def __init__(self,
                 cfg: CN,
                 **base_module_kwargs):
        super().__init__(**base_module_kwargs)
        
        self.local_module_args = cfg.model.local_component
        self.global_module_args = cfg.model.global_component
        
        # module_name = cfg.model.local_component.name
        # local_class = MODULE_REGISTRY.get(module_name)

        # if local_class is None:
        #     raise ValueError(f"Module {module_name} not found in MODULE_REGISTRY")
        
        self.registered_local_component = True
        self.registered_global_component = True
        
        # Local module with decoder
        self.local_module = LocalModuleClass.from_config(cfg,
                                                         n_input=self.n_input,
                                                         n_output=self.n_output,
                                                         n_embed=self.n_embed,
                                                         decoder_type=cfg.model.decoder.type,
                                                         dropout_decoder=cfg.model.decoder.dropout_decoder,
                                                         decoder_hidden_dims=cfg.model.decoder.hidden_dims,
                                                         pct_mask_nodes=self.pct_mask_nodes)
        # Global module with decoder
        self.global_module = GlobalModuleClass.from_config(cfg,
                                                           n_input=self.n_input,
                                                           n_output=self.n_output,
                                                           n_embed=self.n_embed,
                                                           decoder_type=cfg.model.decoder.type,
                                                           dropout_decoder=cfg.model.decoder.dropout_decoder,
                                                           decoder_hidden_dims=cfg.model.decoder.hidden_dims,
                                                           pct_mask_nodes=self.pct_mask_nodes)
        
        # Store split point for separating concatenated predictions
        # For node-level: first half is local, second half is global
        # For graph-level: only global predictions (no split)
        self._n_masked_nodes = None
        self._is_graph_level = False
        
    def predict_local(self,
                     local_embedding,
                     prediction_level: Literal["node", "graph"] = "node",
                     batch_ptr: torch.Tensor | None = None):
        """Predict with the local decoder.

        For node-level: one prediction per masked node [N_masked, C].
        For graph-level: aggregate embeddings per graph (mean), then one prediction per graph [B, C].

        Parameters
        ----------
        local_embedding : torch.Tensor
            Size: [N, E]
        mask_idx : torch.Tensor
            Indices of masked nodes. Ignored when prediction_level == "graph".
        prediction_level : "node" | "graph"
            If "graph", return one prediction per graph; requires batch_ptr.
        batch_ptr : torch.Tensor, optional
            Batch.ptr from PyG Batch. Required when prediction_level == "graph".

        Returns
        -------
        y_pred_local : torch.Tensor
            [N_masked_nodes, C] for node-level, [B, C] for graph-level.
        """
        return self.local_module.predict(local_embedding, prediction_level, batch_ptr)
    
    def predict_global(self,
                      global_embedding,
                      src_padding_mask,
                      prediction_level):
        """Predict with the global decoder.
        
        Parameters
        ----------
        global_embedding: torch.Tensor
            Size: [S+1, B, E] where S is sequence length, B is batch size
        src_padding_mask: torch.Tensor
            Padding mask. Size: [B, S+1]
        prediction_level: Literal["node", "graph"]
            Level of prediction
            
        Returns
        -------
        y_pred_global: torch.Tensor
            Size: [N, C] or [N, F] where N depends on prediction_level
        """
        # Get predictions from global decoder
        y_pred = self.global_module.predict(global_embedding, src_padding_mask, prediction_level)
        return y_pred
    
    def forward(
        self,
        batch_masked):
        """Forward pass through the model"""
        
        local_out = self.local_module.forward(batch_masked.x, batch_masked.edge_index)

        if isinstance(local_out, dict): # neede for scVI component
            local_embedding = local_out['embedding']
            self._current_local_latent_params = local_out 
        else:
            local_embedding = local_out
            self._current_local_latent_params = None
        
        padded_emb, src_padding_mask, pad_index_nodes, attention_mask = self.global_module.common_step_local_to_global(batch_masked, local_embedding)
        assert not torch.any(torch.isnan(padded_emb)), "padded_emb contains NaN values"
        global_embedding, src_padding_mask, attn = self.global_module.forward(padded_emb, src_padding_mask, attention_mask)
        assert not torch.any(torch.isnan(global_embedding)), "global_embedding contains NaN values"
        
        return local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn
        
    def _common_step(self,
                    batch, 
                    prediction_task, 
                    prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.
        
        Returns predictions and ground truth for both local and global decoders
        on masked tokens, which can be combined in the loss function.
        """
        batch_masked, mask_idx = self._common_step_masking(batch)
            
        local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn = self.forward(batch_masked)
        
        # Predict from local embedding (per-node or per-graph depending on prediction_level)
        y_pred_local = self.predict_local(
            local_embedding, 
            prediction_level=prediction_level,
            batch_ptr=batch.ptr if prediction_level == "graph" else None,
        )
        # Predict from global embedding
        y_pred_global = self.predict_global(
            global_embedding, src_padding_mask, prediction_level
        )

        if prediction_task == "classification" and prediction_level == "graph":
            y_true = aggregate_node_values_to_graph(batch.y, batch.ptr, reduce="mean")
            y_pred_combined = torch.cat([y_pred_local, y_pred_global], dim=0)
            y_true_combined = torch.cat([y_true, y_true], dim=0)
            assert len(y_pred_combined) == len(y_true_combined), "y_pred and y_true are not consistent"
            self._n_masked_nodes = len(y_true)
            self._is_graph_level = True

        elif prediction_level == 'node':
            # For node-level predictions, get ground truth for masked nodes
            y_true, adjusted_mask_idx = self.global_module._process_batch_for_metrics(
                batch, prediction_task, prediction_level, pad_index_nodes, mask_idx
            )
            y_true_masked = y_true[adjusted_mask_idx]
            
            # Filter global predictions to masked nodes (same indices as y_true)
            y_pred_global_masked = y_pred_global[adjusted_mask_idx]
            
            # Store split point: first half is local, second half is global
            n_masked = len(y_pred_local)
            self._n_masked_nodes = n_masked
            self._is_graph_level = False
            
            # Combine local and global predictions
            # Both should have the same number of masked nodes
            assert len(y_pred_local) == len(y_pred_global_masked), \
                f"Local and global predictions have different lengths: {len(y_pred_local)} vs {len(y_pred_global_masked)}"
            assert len(y_true_masked) == len(y_pred_local), \
                f"Ground truth and local predictions have different lengths: {len(y_true_masked)} vs {len(y_pred_local)}"
            assert len(y_true_masked) == len(y_pred_global_masked), \
                f"Ground truth and global predictions have different lengths: {len(y_true_masked)} vs {len(y_pred_global_masked)}"
            
            # Concatenate predictions: [N_masked, C] + [N_masked, C] -> [2*N_masked, C]
            # This allows the loss function to compute loss on both predictions
            y_pred_combined = torch.cat([y_pred_local, y_pred_global_masked], dim=0)
            y_true_combined = torch.cat([y_true_masked, y_true_masked], dim=0)
        
        else:
            raise ValueError(f"Invalid prediction level: {prediction_level}")
            
            
        assert len(y_pred_combined) == len(y_true_combined), "y_pred and y_true are not consistent"
        assert not torch.any(torch.isnan(y_pred_combined)), "y_pred contains NaN values"
        assert not torch.any(torch.isnan(y_true_combined)), "y_true contains NaN values"

        return local_embedding, global_embedding, y_pred_combined, y_true_combined, attn
    
    def compute_separate_losses(self, loss_fn, loss_type, y_pred_combined, y_true_combined):
        """Compute separate losses for local and global predictions.
        
        Parameters
        ----------
        loss_fn: callable
            Loss function that takes (y_pred, y_true) and returns a scalar loss.
            Should be compatible with the prediction task (classification or regression).
        loss_type: Literal["GaussianNLL", "MSELoss", "CrossEntropy", "WeightedCE"]
            Type of loss function to use.
        y_pred_combined: torch.Tensor
            Combined predictions from _common_step. For node-level: [2*N_masked, C],
            for graph-level: [B, C] where B is batch size.
        y_true_combined: torch.Tensor
            Combined ground truth from _common_step. Same shape as y_pred_combined.
        
        Returns
        -------
        dict: Dictionary with keys:
            - 'local_loss': scalar loss for local predictions or None
            - 'global_loss': scalar loss for global predictions or None
            - 'combined_loss': combined loss (average of local and global) or None
        """
        losses = {
            'local_loss': None,
            'global_loss': None,
            'combined_loss': None,
            'kl_loss': None
        }
        
        local_loss = None
        global_loss = None
        
        if self._n_masked_nodes is not None:
            # For node level, split the concatenated predictions
            # First half is local, second half is global
            y_pred_local = y_pred_combined[:self._n_masked_nodes]
            y_pred_global = y_pred_combined[self._n_masked_nodes:]
            y_true_local = y_true_combined[:self._n_masked_nodes]
            y_true_global = y_true_combined[self._n_masked_nodes:]
            if loss_type == 'GaussianNLL':
                sd_local = torch.std(y_true_local, dim=1, keepdim=True)
                sd_global = torch.std(y_true_global, dim=1, keepdim=True)
                local_loss = loss_fn(y_pred_local, y_true_local, sd_local)
                global_loss = loss_fn(y_pred_global, y_true_global, sd_global)
            else:
                local_loss = loss_fn(y_pred_local, y_true_local)
                global_loss = loss_fn(y_pred_global, y_true_global)
            losses['local_loss'] = local_loss
            losses['global_loss'] = global_loss
        else:
            raise ValueError(
                "No masked nodes found; _n_masked_nodes was not set in _common_step."
            )
            
        if self._current_local_latent_params is not None:
            losses['kl_loss'] = self.local_module.loss_kl(self._current_local_latent_params)
            self._current_local_latent_params = None

        return losses
    
    def get_model_summary(self) -> str:
        """Returns a string containing the model's parameters summary.

        Returns:
            str: Summary string with model parameters
        """
        summary = (
            f"Dual Decoder Combined Module: \n"
            f"Local Module: {self.local_module.get_model_summary()}\n"
            f"Global Module: {self.global_module.get_model_summary()}\n"
        )
        return summary

