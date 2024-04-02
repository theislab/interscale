import torch
from torch_geometric.data import Data, Dataset
import random
from sklearn.model_selection import train_test_split


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
    
def prepare_dataset_split(data, slices, train_size=0.8, val_size=0.2):
    """Create a PyTorch Geometric dataset with slice-level train/validation split."""
    datasets = []
    num_slices = len(slices['x']) - 1
    
    # Split slice indices into train and validation sets
    slice_indices = list(range(num_slices))
    train_slices, val_slices = train_test_split(slice_indices, train_size=train_size, test_size=val_size, random_state=42)
    
    # Iterate over each slice in lung_slices
    for i in range(num_slices):
        slice_data = {}
        start_index_node = slices['x'][i].item()
        end_index_node = slices['x'][i + 1].item()
        start_index_edge = slices['edge_index'][i].item()
        end_index_edge = slices['edge_index'][i + 1].item()
    
        # Extract relevant data from lung_data for this slice
        slice_data['x'] = data.x[start_index_node:end_index_node]
        slice_data['edge_index'] = data.edge_index[:, start_index_edge:end_index_edge]
        slice_data['edge_attr'] = data.edge_attr[start_index_edge:end_index_edge]
        slice_data['y'] = data.y[start_index_node:end_index_node]
    
        # Create PyTorch Geometric Data object for this slice
        data_slice = Data(**slice_data)
        
        # Assign slice to train or validation set
        if i in train_slices:
            data_slice.train_mask = torch.tensor(True)  # Mark as train
            data_slice.val_mask = torch.tensor(False)  # Mark as not validation
        elif i in val_slices:
            data_slice.train_mask = torch.tensor(False)  # Mark as not train
            data_slice.val_mask = torch.tensor(True)  # Mark as validation
        else:
            raise ValueError("Invalid slice index")
        
        datasets.append(data_slice)
        
    return datasets

def prepare_dataset(data, slices):
    """Create a pytorch geometric dataset."""
    datasets = []
    # Iterate over each slice in lung_slices
    for i in range(len(slices['x']) - 1):
        slice_data = {}
        start_index_node = slices['x'][i].item()
        end_index_node = slices['x'][i + 1].item()
        start_index_edge = slices['edge_index'][i].item()
        end_index_edge = slices['edge_index'][i + 1].item()
    
        # Extract relevant data from lung_data for this slice
        slice_data['x'] = data.x[start_index_node:end_index_node]
        slice_data['edge_index'] = data.edge_index[:,start_index_edge:end_index_edge]
        slice_data['edge_attr'] = data.edge_attr[start_index_edge:end_index_edge]
        slice_data['y'] = data.y[start_index_node:end_index_node]
    
        # Create PyTorch Geometric Data object for this slice
        data_slice = Data(**slice_data)
        datasets.append(data_slice)
        
    return datasets
