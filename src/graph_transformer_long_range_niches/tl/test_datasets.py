import torch
from torch_geometric.data import Data, Dataset
import random

class CustomGraphDataset_graphLabel(Dataset):
    def __init__(self, num_graphs, max_nodes, max_edges, nr_node_features, nr_classes):
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nr_node_features = nr_node_features
        self.data_list = []

        for _ in range(num_graphs):
            num_nodes = random.randint(1, max_nodes)
            num_edges = random.randint(0, min(num_nodes * (num_nodes - 1) // 2, max_edges))
            
            edge_index = torch.zeros((2, num_edges), dtype=torch.long)
            for i in range(num_edges):
                edge_index[0, i] = random.randint(0, num_nodes - 1)
                edge_index[1, i] = random.randint(0, num_nodes - 1)
            
            x = torch.rand((num_nodes, nr_node_features), dtype=torch.float)  # Node features (random for example)
            y = torch.tensor([random.randint(0, nr_classes)], dtype=torch.long)  # Graph label (binary for example)

            data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(data)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data_list[idx]
    
class CustomGraphDataset_nodeLabel(Dataset):
    def __init__(self, num_graphs, max_nodes, max_edges, nr_node_features, nr_classes):
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nr_node_features = nr_node_features
        self.nr_classes = nr_classes
        self.data_list = []

        for _ in range(num_graphs):
            num_nodes = random.randint(1, max_nodes)
            num_edges = random.randint(0, min(num_nodes * (num_nodes - 1) // 2, max_edges))
            
            edge_index = torch.zeros((2, num_edges), dtype=torch.long)
            for i in range(num_edges):
                edge_index[0, i] = random.randint(0, num_nodes - 1)
                edge_index[1, i] = random.randint(0, num_nodes - 1)
            
            x = torch.rand((num_nodes, nr_node_features), dtype=torch.float)  # Node features (random for example)
            y = torch.randint(0, nr_classes, (num_nodes,), dtype=torch.long)  # Node labels

            data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(data)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data_list[idx]