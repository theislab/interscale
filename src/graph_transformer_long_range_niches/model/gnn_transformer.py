# Standard libraries
import numpy as np

# PyTorch 
import torch
from torch import nn

import torchmetrics
from torchmetrics import Metric

# PyTorch Lightning
import pytorch_lightning as L

from graph_transformer_long_range_niches.modules.gcn import LitGCN
from graph_transformer_long_range_niches.modules.transformer_encoder import TransformerNodeEncoder
from graph_transformer_long_range_niches.tl.utils import pad_batch
from graph_transformer_long_range_niches.tl.loss import weighted_cross_entropy
from graph_transformer_long_range_niches.tl.scheduler import CosineWarmupScheduler

class LitGNNTransformer(L.LightningModule):
    def __init__(self, cfg, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self._cfg = cfg
        self.lr = float(self._cfg.get('optim/lr'))
        self.wd = float(self._cfg.get('optim/wd'))

        self.model_type = 'GNN_Transformer'
        self.prediction_task = cfg.get('dataset/prediction_task')

        self.output_dim = cfg.get('transformer/d_model')

        self.gnn2transformer = nn.Linear(cfg.get('gnn/embed_dim'), self.output_dim)
        self.norm_input = nn.LayerNorm(self.output_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.output_dim], requires_grad = True))
        self.num_classes = cfg.get('dataset/num_classes')
        self.max_seq_len = cfg.get('dataset/max_seq_len')
        # ToDOo: refer to weighted loss
        if cfg.get('optim/loss') == 'CrossEntropy':
            self.loss = torch.nn.CrossEntropyLoss()
        elif cfg.get('optim/loss') == 'WeightedCE':
            self.loss = weighted_cross_entropy()
        else:
            raise ValueError(f"Invalid loss function specified: {cfg.get('optim/loss')}. Please choose 'CrossEntropy' or 'WeightedCE'.")

        # Define metrics
        self.accurary = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="micro") 
        self.f1_score_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro") 
        self.f1_score_per_class = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average=None)
        
        # GNN initialization
        self.gnn = LitGCN(cfg)
        # Transformer encoder initialization
        self.transformer_encoder = TransformerNodeEncoder(cfg)

        ## Prediction units
        self.graph_pred_linear_list = torch.nn.ModuleList()
        if self.max_seq_len is None:
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_classes)
        else:
            for i in range(self.max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(self.output_dim, self.num_classes))

    def forward(self, batched_data):
        """
        Input: 
            batched_data: Pytorch geometric object 
                batched_data.x = [N, F]
        """
        h_node, z = self.gnn(batched_data.x, batched_data.edge_index)
        #print('GNN out: ',h_node.shape, 'z', z.shape)
        #print('GNN predicted node label accuracy: ', (z.argmax(dim=1) == batched_data.y).sum() / len(batched_data.y))
        h_node = self.gnn2transformer(h_node)  # [s, d_model]
        #print('After gnn2transformer: ', h_node.shape)

        padded_h_node, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, batched_data.batch, self.transformer_encoder.max_input_len, get_mask=True
        )  # Pad in the front batched_data.batch before

        #print("After Pad: ", padded_h_node.shape)

        transformer_out = padded_h_node
        transformer_out, src_padding_mask = self.transformer_encoder(transformer_out, src_padding_mask)  # [s, B, h], [B, s]
        #print('TransformerEncoder output: ', transformer_out.shape)

        # ## Graph-level prediction: get cls 
        if self.prediction_task == 'graph':
            cls = transformer_out[-1] # [B, C]
            if self.max_seq_len is None:
                out = self.graph_pred_linear(cls)
                return z, out
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.graph_pred_linear_list[i](cls))
                return z, pred_list
            
        ## Node-level prediction: remove cls
        elif self.prediction_task == 'node':
            h_graph = transformer_out[:-1] # [E, B, C]
            #print('hgraph output: ', h_graph.shape) 
            h_graph = torch.permute(h_graph, (1, 0, 2)) #[B, S, E]
            #print('Permuted h_graph:', h_graph.shape)
            src_padding_mask = src_padding_mask[:,:-1] # True = Pad, False = Node
            #print('Padding mask:', src_padding_mask.shape)
            masked_output = h_graph[~ src_padding_mask] # [N, E]
            #print('Masked output:', masked_output.shape)
            out = self.graph_pred_linear(masked_output)
            return z, out, index_nodes
            
        else:
            raise Exception('Choose a valid prediction tasks (graph or node).')

        return None, z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=int(self._cfg.get('optim/warm_up')),
                                             #max_epochs=int(self._cfg.get('model/n_epochs')))
                                             max_epochs=100000)
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        loss, acc, f1_score_micro, f1_score_macro, f1_score_per_class = self._common_step(batch)
        log_dict = {
            'train_loss': loss,
            'train_acc': acc,
            'train_f1_micro/avg': f1_score_micro,
            'train_f1_macro/avg': f1_score_macro,
        }
        for class_idx in range(self.num_classes):
            log_dict[f'train_f1/class_{class_idx}'] = f1_score_per_class[class_idx]
        self.log_dict(log_dict, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1_score_micro, f1_score_macro, f1_score_per_class = self._common_step(batch)
        log_dict = {
            'val_loss': loss,
            'val_acc': acc,
            'val_f1_micro/avg': f1_score_micro,
            'val_f1_macro/avg': f1_score_macro,
        }
        for class_idx in range(self.num_classes):
            log_dict[f'val_f1/class_{class_idx}'] = f1_score_per_class[class_idx]
        self.log_dict(log_dict, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch):
        loss, acc, f1_score_micro, f1_score_macro, f1_score_per_class = self._common_step(batch)
        self.log_dict({'test_loss': loss, 'test_acc': acc, 'test_f1_micro': f1_score_micro, 'test_f1_score_macro': f1_score_macro}, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
        return loss

    def _common_step(self, batch):
        """Shared step between train, val and test.
        """
        out_gnn, out_transformer, index_nodes = self.forward(batch) # batch: [B, C] with C being the number of tasks to predict, e.i. 
        # Calculate loss function
        y_true = []
        for i in range(batch.batch[-1] + 1):
            mask = batch.batch.eq(i)
            y_true += batch.y[mask][index_nodes[i]]
        y_true = torch.stack(y_true)
        #print('predicted and true: ', out_transformer[:10].argmax(dim=1), y_true[:10].argmax(dim=1))
        if self._cfg.get('optim/loss') == 'CrossEntropy':
            weight = weighted_cross_entropy(out_transformer, torch.LongTensor(y_true))
            print('weight: ', weight)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fn(out_transformer, torch.LongTensor(y_true).argmax(dim=1),)
        else:
            loss = self.loss(out_transformer, y_true.argmax(dim=1))
        acc = self.accurary(out_transformer.argmax(dim=1), y_true.argmax(dim=1))
        f1_score_micro = self.f1_score_micro(out_transformer.argmax(dim=1), y_true.argmax(dim=1))
        f1_score_macro = self.f1_score_macro(out_transformer.argmax(dim=1), y_true.argmax(dim=1))
        f1_score_per_class = self.f1_score_per_class(out_transformer.argmax(dim=1), y_true.argmax(dim=1))
        #print(f'acc: {acc}, f1_score: {f1_score}, loss: {loss}')

        return loss, acc, f1_score_micro, f1_score_macro, f1_score_per_class