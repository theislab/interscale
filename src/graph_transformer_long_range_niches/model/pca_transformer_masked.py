# Standard libraries

# PyTorch
# PyTorch Lightning
import pytorch_lightning as L
import torch
import torchmetrics
from torch import nn

from graph_transformer_long_range_niches.modules import TransformerNodeEncoder, BaseModule, TransformerNodeEncoderHook
from graph_transformer_long_range_niches.tl.utils import pad_batch, create_transformer_attention_mask_from_edges
from sklearn.decomposition import PCA
from graph_transformer_long_range_niches.tl.masking import apply_mask

from typing import List


class LitPCATransformerMasked(BaseModule):
    def __init__(self,
                 cfg, 
                 class_weights: List = None, 
                 pretrained_gnn_path: str = None, 
                 **model_kwargs
        ):
        super().__init__(cfg, class_weights, **model_kwargs)
        
        self._cfg = cfg
        
        self.model_type = 'PCA_Transformer'
        self.output_dim = cfg.transformer.d_model
        
        self.norm_input = nn.LayerNorm(self.output_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.output_dim], requires_grad = True))
        
        # Initialize PCA
        self.pca = PCA(n_components=self.output_dim)
        
        self.transformer_encoder = TransformerNodeEncoderHook(cfg)
    
    def forward(self, batched_data):
        """
        Input: 
            batched_data: Pytorch geometric object 
                batched_data.x = [N, F]
        """
        # Apply PCA to reduce the dimensionality of node features
        h_node = self.pca.fit_transform(batched_data.x.cpu().numpy())  # Convert to numpy for PCA
        h_node = torch.tensor(h_node, dtype=torch.float32, device=batched_data.x.device)  # Convert back to tensor
        
        h_node = self.norm_input(h_node)
        
        if self.masked_nodes:
            keep_indices = batched_data.mask
        else:
            keep_indices = None

        # Ensure masked nodes are included in padding
        padded_h_node, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, 
            batched_data.batch, 
            self.transformer_encoder.max_seq_len, 
            get_mask=self.masked_nodes,
            keep_indices=keep_indices  # Add parameter to ensure masked nodes are kept
        )
        
        if self._cfg.transformer.long_range_attention:
            attention_mask = create_transformer_attention_mask_from_edges(batched_data.edge_index, len(batched_data.obs_names), batched_data.batch, index_nodes, self.transformer_encoder.n_heads)
        else:
            attention_mask = None
            
        transformer_out = padded_h_node
        transformer_out, src_padding_mask = self.transformer_encoder(transformer_out, src_padding_mask, mask = attention_mask)  # [S+1, B, E], [B, s]
        
        # ## Graph-level prediction: get cls
        if 'graph' in self.prediction_task:
            cls = transformer_out[-1,:, :] # [B, E]
            out = self.graph_pred_linear(cls)
            return out, index_nodes

        ## Node-level prediction: remove cls
        elif 'node' in self.prediction_task: #TODO: I don't think I need to differentiate between regression and classification here.
            h_graph = transformer_out[:-1] # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node
            masked_output = h_graph[~src_padding_mask] # [N, E]
            out = self.graph_pred_linear(masked_output)
            return out, index_nodes

        else:
            raise Exception('Choose a valid prediction tasks (graph or node).')
        
        
    def configure_optimizers(self):
        return self.common_configure_optimizers()

    def training_step(self, batch, batch_idx):
        return self.common_training_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.common_validation_step(batch)

    def test_step(self, batch):
        return self.common_test_step(batch)
    
    def _common_step(self, batch):
        """Shared step between train, val and test.
        """
        # Mask nodes 
        input_data_masked, mask_idx = apply_mask(batch)
        # Run forward pass on masked data
        out_transformer, pad_index_nodes = self.forward(input_data_masked)
        
        y_true = []
        adjusted_mask_idx = []  # Track new positions of masked nodes
        current_offset = 0
        
        start = 0
        
        for i in range(batch.batch[-1] + 1):
            mask = batch.batch.eq(i)
            pad_indices = torch.tensor(pad_index_nodes[i], device=batch.x.device)
            end = start + len(pad_indices)

            for idx in mask_idx[start:end]:
                # Convert to tensor if not already
                if idx in pad_indices:
                    new_idx = torch.where(pad_indices == idx)[0].item()
                    adjusted_mask_idx.append(new_idx + current_offset)
            
            current_offset += len(pad_indices)
            start = end
            
            if 'classification' in self.prediction_task:
                if 'node' in self.prediction_task:
                    y_true += batch.y[mask][pad_index_nodes[i]].clone().detach()
                elif 'graph' in self.prediction_task:
                    y_true.append(batch.y[mask][-1].clone().detach())
            elif 'regression' in self.prediction_task:
                y_true += batch.x[mask][pad_index_nodes[i]].clone().detach()
            else:
                raise Exception('Choose a valid prediction tasks (graph or node).')
                    
        y_true = torch.stack(y_true)
        adjusted_mask_idx = torch.tensor(adjusted_mask_idx, device=y_true.device)

        if 'classification' in self.prediction_task:
            if 'node' in self.prediction_task:
                return self._common_step_classification_metrics(out_transformer, y_true, adjusted_mask_idx)
            elif 'graph' in self.prediction_task:
                return self._common_step_classification_metrics(out_transformer, y_true, None)

        if 'regression' in self.prediction_task:
            return self._common_step_regression_metrics(out_transformer, y_true, adjusted_mask_idx)
    
