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
from graph_transformer_long_range_niches.tl.evaluation import accuracy  
from graph_transformer_long_range_niches.tl.loss import weighted_cross_entropy
from graph_transformer_long_range_niches.tl.scheduler import CosineWarmupScheduler

class LitGNNTransformer(L.LightningModule):
    def __init__(self, cfg, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self._cfg = cfg

        self.model_type = 'GNN_Transformer'
        self.prediction_task = cfg.get('dataset/prediction_task')

        self.output_dim = cfg.get('transformer/d_model')

        self.gnn2transformer = nn.Linear(cfg.get('gnn/embed_dim'), self.output_dim)
        self.norm_input = nn.LayerNorm(self.output_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.output_dim], requires_grad = True))
        self.num_classes = cfg.get('dataset/num_classes')
        self.max_seq_len = cfg.get('dataset/max_seq_len')
        self.loss = torch.nn.CrossEntropyLoss()
        # Define metrics
        self.accurary = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes) 
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self._cfg.get('optim/lr')), weight_decay=float(self._cfg.get('optim/wd')))
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=int(self._cfg.get('optim/warm_up')),
                                             max_epochs=int(self._cfg.get('model/n_epochs')))
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch):
        loss, acc, f1_score = self._common_step(batch)
        self.log_dict({'train_loss': loss, 'train_acc': acc, 'train_f1': f1_score}, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        loss, acc, f1_score = self._common_step(batch)
        self.log_dict({'val_loss': loss, 'val_acc': acc, 'val_f1': f1_score}, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch):
        loss, acc, f1_score = self._common_step(batch)
        self.log_dict({'test_loss': loss, 'test_acc': acc, 'test_f1': f1_score}, batch_size=int(self._cfg.get('dataset/batch_size')), on_step=False, on_epoch=True)
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
        if self._cfg.get('optim/loss') == 'weighted-cross-entropy':
            weight = weighted_cross_entropy(out_transformer, torch.LongTensor(y_true))
            print('weight: ', weight)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fn(out_transformer, torch.LongTensor(y_true),)
        else:
            loss = self.loss(out_transformer, torch.LongTensor(y_true))
        print('predicted and true: ', out_transformer.argmax(dim=1)[:10], y_true[:10])
        acc = self.accurary(out_transformer, torch.LongTensor(y_true))
        f1_score = self.f1_score(out_transformer, torch.LongTensor(y_true))

        return loss, acc, f1_score