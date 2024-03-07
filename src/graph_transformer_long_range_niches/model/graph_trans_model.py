import torch
from torch_geometric.models import GCN

class GraphTrans(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        ## GNN
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if cfg.gnn.gnn_type == "GCN":
                self.gcn = GCN(
                        in_channels = cfg.gnn.in_dim, 
                        hidden_channels = cfg.gnn.hidden_dim,
                        num_layers = cfg.gnn.num_layers
                    )
        else:
            ValueError("Undefined GNN type called {}".format(cfg.gnn.gnn_type))

        ## Transformer

    def forward(self, batch):
        self.gcn.forward(batch.x, batch.edge_index)
        return None