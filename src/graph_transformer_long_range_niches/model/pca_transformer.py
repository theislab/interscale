# Standard libraries

# PyTorch
# PyTorch Lightning
import pytorch_lightning as L
import torch
import torchmetrics
from torch import nn

from graph_transformer_long_range_niches.modules.transformer_encoder import TransformerNodeEncoder
from graph_transformer_long_range_niches.modules.transformer_encoder_hook import TransformerNodeEncoderHook
from graph_transformer_long_range_niches.tl import pad_batch, CosineWarmupScheduler
from sklearn.decomposition import PCA

from typing import List


class LitPCATransformer(L.LightningModule):
    def __init__(self, cfg, class_weights: List = None, **model_kwargs):
        super().__init__(cfg, class_weights, **model_kwargs)
        
        self.model_type = 'PCA_Transformer'

        self.output_dim = cfg.transformer.d_model
         # Initialize PCA
        self.pca = PCA(n_components=self.output_dim)

        # Input normalization and transformer encoder
        self.norm_input = nn.LayerNorm(self.output_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.output_dim], requires_grad = True))

        # Transformer encoder initialization
        self.transformer_encoder = TransformerNodeEncoderHook(cfg)

        ## Prediction units
        self.graph_pred_linear_list = torch.nn.ModuleList()
        if 'classification' in self.prediction_task:
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_classes)
        elif 'regression' in self.prediction_task:
            print('num features:', self.num_features)
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_features)

    def forward(self, batched_data):
        """
        Input: 
            batched_data: Pytorch geometric object 
                batched_data.x = [N, F]
        """
        # Apply PCA to reduce the dimensionality of node features
        h_node = self.pca.fit_transform(batched_data.x.cpu().numpy())  # Convert to numpy for PCA
        h_node = torch.tensor(h_node, dtype=torch.float32, device=batched_data.x.device)  # Convert back to tensor

        padded_h_node, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, batched_data.batch, self.transformer_encoder.max_seq_len, get_mask=True
        )  # Pad in the front batched_data.batch before

        transformer_out = padded_h_node
        transformer_out, src_padding_mask = self.transformer_encoder(transformer_out, src_padding_mask)  # [S+1, B, E], [B, s]

        # ## Graph-level prediction: get cls
        if 'graph' in self.prediction_task:
            cls = transformer_out[-1,:, :] # [B, E]
            out = self.graph_pred_linear(cls)
            return None, out, index_nodes

        ## Node-level prediction: remove cls
        elif 'node' in self.prediction_task: #TODO: I don't think I need to differentiate between regression and classification here.
            h_graph = transformer_out[:-1] # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node
            masked_output = h_graph[~ src_padding_mask] # [N, E]
            out = self.graph_pred_linear(masked_output)
            return None, out, index_nodes

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
        out_gnn, out_transformer, index_nodes = self.forward(batch) # batch: [B, C] with C being the number of tasks to predict, e.i.
        # Calculate loss function
        y_true = []

        for i in range(batch.batch[-1] + 1):
            mask = batch.batch.eq(i)
            if 'classification' in self.prediction_task:
                if 'node' in self.prediction_task:
                    y_true += torch.tensor(batch.y[mask][index_nodes[i]])
                elif 'graph' in self.prediction_task:
                    y_true.append(torch.tensor(batch.y[mask][-1])) # assume same label on graph level. ToDo: check if graph level == same label
            elif 'regression' in self.prediction_task:
                y_true += torch.tensor(batch.x[mask][index_nodes[i]])
            else:
                raise Exception('Choose a valid prediction tasks (graph or node).')
        y_true = torch.stack(y_true)

        #print('predicted and true: ', out_transformer[:10].argmax(dim=1), y_true[:10].argmax(dim=1))
        if 'classification' in self.prediction_task:
            print(out_transformer, y_true)
            loss = self.loss(out_transformer, y_true.argmax(dim=1).to(torch.long))
            acc = self.accurary(out_transformer.argmax(dim=1), y_true.argmax(dim=1))
            f1_score_micro = self.f1_score_micro(out_transformer.argmax(dim=1), y_true.argmax(dim=1))
            f1_score_macro = self.f1_score_macro(out_transformer.argmax(dim=1), y_true.argmax(dim=1))
            f1_score_per_class = self.f1_score_per_class(out_transformer.argmax(dim=1), y_true.argmax(dim=1))

            return loss, [acc, f1_score_micro, f1_score_macro, f1_score_per_class]

        if 'regression' in self.prediction_task:
            # GaussianNLLoss -> var (NCEM: used variance per gene)
            # Estimate variance based on the true values (e.g., using batch variance)
            y_var = torch.var(y_true, dim=1, keepdim=True)  # You can adjust the estimation method
            # Ensure variance is non-zero and positive
            y_var = y_var.clamp(min=1e-6)
            loss = self.loss(out_transformer, y_true, y_var)
            mse = self.mse(out_transformer, y_true)
            r2 = self.r2(out_transformer, y_true)
            pearson_corr = torch.mean(self.pearson_corr(out_transformer, y_true))
            return loss, [mse, r2, pearson_corr]


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
        if self._cfg.gnn.embed_dim != self.output_dim:
            h_node = self.gnn2transformer(h_node)
        padded_h_node, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
                h_node, batched_data.batch, self.transformer_encoder.max_seq_len, get_mask=True
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
