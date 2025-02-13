# Standard libraries

# PyTorch
# PyTorch Lightning
import pytorch_lightning as L
import torch
from torch import nn
from graph_transformer_long_range_niches.tl import apply_mask

from typing import List

from graph_transformer_long_range_niches.modules import TransformerNodeEncoderHook, BaseModule, LitGCN
from graph_transformer_long_range_niches.tl import pad_batch, apply_mask

class LitGNNTransformerMasked(BaseModule):
    def __init__(self, 
                 cfg, 
                 class_weights: List = None, 
                 pretrained_gnn_path: str = None, 
                 **model_kwargs
        ):
        # initialize all hyperparameters from BaseModule
        super().__init__(cfg, class_weights, **model_kwargs)
        
        assert cfg.gnn.embed_dim == cfg.transformer.d_model, "GNN and Transformer must have the same embedding dimension."
        
        if cfg.dataset.pct_mask_nodes > 0:
            self.masked_nodes = True
        else:
            self.masked_nodes = False
        
        self.model_type = 'GNN_Transformer'
        self.output_dim = cfg.transformer.d_model

        self.norm_input = nn.LayerNorm(self.output_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.output_dim], requires_grad = True))

        # GNN initialization
        self.gnn = LitGCN(cfg)
        if pretrained_gnn_path:
            # Load pre-trained GNN weights
            pretrained_state_dict = torch.load(pretrained_gnn_path)
            self.gnn.load_state_dict(pretrained_state_dict)
            
            # Freeze GNN parameters
            for param in self.gnn.parameters():
                param.requires_grad = False
            
            print("Loaded pre-trained GNN from:", pretrained_gnn_path)
            print("GNN parameters have been frozen")
        
        # Transformer encoder initialization
        self.transformer_encoder = TransformerNodeEncoderHook(cfg)

        # ## Prediction units
        # self.graph_pred_linear_list = torch.nn.ModuleList()
        # if 'classification' in self.prediction_task:
        #     self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_classes)
        # elif 'regression' in self.prediction_task:
        #     print('num features:', self.num_features)
        #     self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_features)

    def forward(self, batched_data):
        """
        Input: 
            batched_data: Pytorch geometric object 
                batched_data.x = [N, F]
        """
        h_node, z = self.gnn(batched_data.x, batched_data.edge_index)
        
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
        transformer_out = padded_h_node
        transformer_out, src_padding_mask = self.transformer_encoder(transformer_out, src_padding_mask)  # [S+1, B, E], [B, s]
        
        # ## Graph-level prediction: get cls
        if 'graph' in self.prediction_task:
            cls = transformer_out[-1,:, :] # [B, E]
            out = self.graph_pred_linear(cls)
            return z, out, index_nodes

        ## Node-level prediction: remove cls
        elif 'node' in self.prediction_task: #TODO: I don't think I need to differentiate between regression and classification here.
            h_graph = transformer_out[:-1] # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node
            masked_output = h_graph[~src_padding_mask] # [N, E]
            out = self.graph_pred_linear(masked_output)
            return z, out, index_nodes

        else:
            raise Exception('Choose a valid prediction tasks (graph or node).')


    def configure_optimizers(self):
        return self.common_configure_optimizers()

    def training_step(self, batch, batch_idx):
        return self.common_training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.common_validation_step(batch, batch_idx)

    def test_step(self, batch):
        return self.common_test_step(batch)

    def _common_step(self, batch):
        """Shared step between train, val and test.
        """
        # Mask nodes 
        input_data_masked, mask_idx = apply_mask(batch)
        # Run forward pass on masked data
        out_gnn, out_transformer, pad_index_nodes = self.forward(input_data_masked)
        
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
                    y_true += torch.tensor(batch.y[mask][pad_index_nodes[i]])
                elif 'graph' in self.prediction_task:
                    y_true.append(torch.tensor(batch.y[mask][-1]))
            elif 'regression' in self.prediction_task:
                y_true += torch.tensor(batch.x[mask][pad_index_nodes[i]])
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

    def extract_attention(self, x, src_padding_mask, average_attn_heads = True):
        """
        Returns a list of attention maps (Tensor) for each Transformer layer.

        More info at .self_attn in MultiHeadAttention() class (PyTorch)

        Return: 
            attn_maps: unbatched (L, E) or batched (L, N, E) 
            attn_weight_maps: attention weights averaged across heads (L, S) or (N, L, S) or per head (num_heads, L, S) or (N, num_heads, L, S)
            attention_maps:
        """
        attn_weights_maps = []
        attn_maps = []

        num_layers = self.transformer_encoder.num_layers
        num_heads = self.transformer_encoder.layers[0].self_attn.num_heads
        print("num heads: ", num_heads)
        norm_first = self.transformer_encoder.layers[0].norm_first

        with torch.no_grad():
            for i in range(num_layers):
                # compute attention of layer i
                h = x.clone()
                if norm_first:
                    h = self.transformer_encoder.layers[i].norm1(h)
                attn_output, attn_output_weights = self.transformer_encoder.layers[i].self_attn(h, h, h, need_weights=True, key_padding_mask=src_padding_mask, average_attn_weights=average_attn_heads)
                attn_maps.append(attn_output)
                attn_weights_maps.append(attn_output_weights)
                # forward of layer i
                x = self.transformer_encoder.layers[i](x)

            attention_maps = torch.stack(attn_maps, dim=0)
            attention_maps = torch.mean(attention_maps, dim=0)

        return attn_maps, attn_weights_maps, attention_maps
    
    def extract_attention_new(self, x, src_padding_mask, average_attn_heads=True):
        """
        Returns a list of attention maps (Tensor) for each Transformer layer.

        Return: 
            attn_maps: List[Tensor], attention outputs from each layer
            attn_weights_maps: List[Tensor], attention weights from each layer
            attention_maps: Tensor, mean of attention outputs across layers
        """
        attn_weights_maps = []
        attn_maps = []

        num_layers = self.transformer_encoder.num_layers
        num_heads = self.transformer_encoder.layers[0].self_attn.num_heads
        print("num heads: ", num_heads)
        norm_first = self.transformer_encoder.layers[0].norm_first

        # Remove 'with torch.no_grad()' to enable gradient computation
        for i in range(num_layers):
            # Compute attention of layer i
            h = x.clone()
            if norm_first:
                h = self.transformer_encoder.layers[i].norm1(h)
            
            # Set 'need_weights=True' and 'average_attn_weights=False' to get per-head weights
            attn_output, attn_output_weights = self.transformer_encoder.layers[i].self_attn(
                h, h, h,
                need_weights=True,
                key_padding_mask=src_padding_mask,
                average_attn_weights=average_attn_heads
            )
            
            # Ensure attention weights require gradients
            attn_output_weights = attn_output_weights.requires_grad_(True)
            # Retain gradients for attention weights
            attn_output_weights.retain_grad()
            
            attn_maps.append(attn_output)
            attn_weights_maps.append(attn_output_weights)
            
            # Forward of layer i
            x = self.transformer_encoder.layers[i](x)

        # Stack and average attention outputs
        attention_maps = torch.stack(attn_maps, dim=0)
        attention_maps = torch.mean(attention_maps, dim=0)

        # Return attention outputs, attention weights, and mean attention map
        return attn_maps, attn_weights_maps, attention_maps

    def evaluation(self, batched_data):
        h_node, z = self.gnn(batched_data.x, batched_data.edge_index)

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
        transformer_out, src_padding_mask = self.transformer_encoder(padded_h_node, src_padding_mask)

        src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node

        if 'graph' in self.prediction_task:
            cls = transformer_out[-1,:, :] # [B, E]
            dec_out = self.graph_pred_linear(cls)

        ## Node-level prediction: remove cls
        elif 'node' in self.prediction_task: #TODO: I don't think I need to differentiate between regression and classification here.
            h_graph = transformer_out[:-1] # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            masked_output = h_graph[~ src_padding_mask] # [N, E]
            dec_out = self.graph_pred_linear(masked_output)

        return padded_h_node, transformer_out, src_padding_mask, index_nodes, dec_out