# Standard libraries

# PyTorch
# PyTorch Lightning
import pytorch_lightning as L
import torch
import torchmetrics
from torch import nn

from typing import List

from graph_transformer_long_range_niches.modules.gcn import LitGCN
from graph_transformer_long_range_niches.modules.transformer_encoder import TransformerNodeEncoder
from graph_transformer_long_range_niches.modules.transformer_encoder_hook import TransformerNodeEncoderHook
from graph_transformer_long_range_niches.tl.scheduler import CosineWarmupScheduler
from graph_transformer_long_range_niches.tl.utils import pad_batch

class LitGNNTransformer(L.LightningModule):
    def __init__(self, cfg, class_weights: List = None, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self._cfg = cfg
        self.lr = float(self._cfg.optim.lr)
        self.wd = float(self._cfg.optim.wd)
        self.class_weights = class_weights

        self.model_type = 'GNN_Transformer'
        self.prediction_task = cfg.dataset.prediction_task

        self.output_dim = cfg.transformer.d_model
        if cfg.gnn.embed_dim != self.output_dim:
            self.gnn2transformer = nn.Linear(cfg.gnn.embed_dim, self.output_dim)

        self.norm_input = nn.LayerNorm(self.output_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.output_dim], requires_grad = True))
        self.num_classes = cfg.dataset.num_classes
        self.num_features = cfg.dataset.num_features
        self.max_seq_len = cfg.transformer.max_seq_len
        # ToDOo: refer to weighted loss
        # if cfg.get('optim/loss') == 'CrossEntropy' or cfg.get('optim/loss') == 'WeightedCE':
        #     self.loss = torch.nn.CrossEntropyLoss()
        # else:
        #     raise ValueError(f"Invalid loss function specified: {cfg.get('optim/loss')}. Please choose 'CrossEntropy' or 'WeightedCE'.")
        if 'classification' in self.prediction_task:
            self.loss = torch.nn.CrossEntropyLoss()
        elif 'regression' in self.prediction_task:
            #self.loss = torch.nn.MSELoss()
            self.loss = torch.nn.GaussianNLLLoss()
            #self.loss = torch.nn.SmoothL1Loss()
        else:
            raise Exception("Prediction task must define 'classification' or 'regression'.")

        # Define metrics
        if 'classification' in self.prediction_task:
            self.accurary = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.f1_score_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="micro")
            self.f1_score_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
            self.f1_score_per_class = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average=None)
        elif 'regression' in self.prediction_task:
            self.mse = torchmetrics.MeanSquaredError()
            self.r2 = torchmetrics.R2Score(num_outputs=self.num_features, multioutput = 'uniform_average')
            self.pearson_corr = torchmetrics.PearsonCorrCoef(num_outputs=self.num_features)

        # GNN initialization
        self.gnn = LitGCN(cfg)
        # Transformer encoder initialization
        self.transformer_encoder = TransformerNodeEncoderHook(cfg)

        ## Prediction units
        self.graph_pred_linear_list = torch.nn.ModuleList()
        if 'classification' in self.prediction_task:
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_classes)
        elif 'regression' in self.prediction_task:
            print('num features:', self.num_features)
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_features)

        # if self.max_seq_len is None:
        #     self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_classes)
        # else:
        #     for i in range(self.max_seq_len):
        #         self.graph_pred_linear_list.append(torch.nn.Linear(self.output_dim, self.num_classes))

    def forward(self, batched_data):
        """
        Input: 
            batched_data: Pytorch geometric object 
                batched_data.x = [N, F]
        """
        h_node, z = self.gnn(batched_data.x, batched_data.edge_index)
        
        if self._cfg.gnn.embed_dim != self.output_dim:
            h_node = self.gnn2transformer(h_node)  # [s, d_model]
        h_node = self.norm_input(h_node)

        padded_h_node, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, batched_data.batch, self.transformer_encoder.max_seq_len, get_mask=True
        )  # Pad in the front batched_data.batch before

        transformer_out = padded_h_node
        transformer_out, src_padding_mask = self.transformer_encoder(transformer_out, src_padding_mask)  # [S+1, B, E], [B, s]

        # ## Graph-level prediction: get cls
        if 'graph' in self.prediction_task:
            cls = transformer_out[-1,:, :] # [B, E]
            out = self.graph_pred_linear(cls)
            return z, out, index_nodes
            # if self.max_seq_len is None:
            #     out = self.graph_pred_linear(cls)
            #     return z, out, index_nodes
            # pred_list = []
            # for i in range(self.max_seq_len):
            #     pred_list.append(self.graph_pred_linear_list[i](cls))
            #     return z, pred_list, index_nodes

        ## Node-level prediction: remove cls
        elif 'node' in self.prediction_task: #TODO: I don't think I need to differentiate between regression and classification here.
            h_graph = transformer_out[:-1] # [E, B, C]
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node
            masked_output = h_graph[~ src_padding_mask] # [N, E]
            out = self.graph_pred_linear(masked_output)
            return z, out, index_nodes

        else:
            raise Exception('Choose a valid prediction tasks (graph or node).')


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=int(self._cfg.optim.warm_up),
                                             max_epochs=100000)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        #loss, acc, f1_score_micro, f1_score_macro, f1_score_per_class = self._common_step(batch)
        loss, metric_list = self._common_step(batch)
        if 'classification' in self.prediction_task:
            acc, f1_score_micro, f1_score_macro, f1_score_per_class = metric_list
            log_dict = {
                'train_loss': loss,
                'train_acc': acc,
                'train_f1_micro/avg': f1_score_micro,
                'train_f1_macro/avg': f1_score_macro,
            }
            for class_idx in range(self.num_classes):
                log_dict[f'train_f1/class_{class_idx}'] = f1_score_per_class[class_idx]
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        elif 'regression' in self.prediction_task:
            mse, r2, pearson_corr = metric_list
            log_dict = {
                'train_mse': mse,
                'train_r2': r2,
                'train_pearson_corr': pearson_corr,
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric_list = self._common_step(batch)
        if 'classification' in self.prediction_task:
            acc, f1_score_micro, f1_score_macro, f1_score_per_class = metric_list
            log_dict = {
                'val_loss': loss,
                'val_acc': acc,
                'val_f1_micro/avg': f1_score_micro,
                'val_f1_macro/avg': f1_score_macro,
            }
            for class_idx in range(self.num_classes):
                log_dict[f'val_f1/class_{class_idx}'] = f1_score_per_class[class_idx]
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        elif 'regression' in self.prediction_task:
            mse, r2, pearson_corr = metric_list
            log_dict = {
                'val_mse': mse,
                'val_r2': r2,
                'val_pearson_corr': pearson_corr,
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch):
        loss, metric_list = self._common_step(batch)
        if 'classification' in self.prediction_task:
            acc, f1_score_micro, f1_score_macro, f1_score_per_class = metric_list
            log_dict = {
                'test_loss': loss,
                'test_acc': acc,
                'test_f1_micro/avg': f1_score_micro,
                'test_f1_macro/avg': f1_score_macro,
            }
            for class_idx in range(self.num_classes):
                log_dict[f'test_f1/class_{class_idx}'] = f1_score_per_class[class_idx]
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        elif 'regression' in self.prediction_task:
            mse, r2, pearson_corr = metric_list
            log_dict = {
                'test_mse': mse,
                'test_r2': r2,
                'test_pearson_corr': pearson_corr,
            }
            self.log_dict(log_dict, batch_size=int(self._cfg.dataset.batch_size), on_step=False, on_epoch=True)
        return loss


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
            if self._cfg.optim.loss == 'WeightedCE':
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fn(out_transformer, y_true.argmax(dim=1),)
            else:
                loss = self.loss(out_transformer, y_true.argmax(dim=1))
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
