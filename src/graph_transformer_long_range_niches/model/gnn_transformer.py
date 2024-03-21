import torch
from torch import nn
import numpy as np

from graph_transformer_long_range_niches.modules.gcn import GCN
from graph_transformer_long_range_niches.modules.transformer_encoder import TransformerNodeEncoder
from graph_transformer_long_range_niches.tl.utils import pad_batch

class GNNTransformer(nn.Module):
    """
    Sequence of: Dropout → Layer Norm → FC → nonlinearity → Dropout → FC → Dropout → Layer Norm + residual connections
    """
    
    def __init__(self, cfg):

        super().__init__()

        self.model_type = 'GNN_Transformer'
        self.prediction_task = cfg["dataset"]["prediction_task"]

        self.output_dim = cfg["transformer"]["d_model"]

        self.gnn2transformer = nn.Linear(cfg['gnn']['embed_dim'], cfg['transformer']['d_model'])
        self.norm_input = nn.LayerNorm(cfg['transformer']['d_model'])
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, cfg['transformer']['d_model']], requires_grad = True))
        self.num_tasks = cfg["dataset"]["num_classes"]
        self.max_seq_len = None
        
        # GNN initialization
        self.gnn = GCN(cfg["gnn"], num_classes=self.num_tasks)
        # Transformer encoder initialization
        self.transformer_encoder = TransformerNodeEncoder(cfg["transformer"])

        ## Prediction units
        self.graph_pred_linear_list = torch.nn.ModuleList()
        if self.max_seq_len is None:
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, self.num_tasks)
        else:
            for i in range(self.max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(self.output_dim, self.num_tasks))
        

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