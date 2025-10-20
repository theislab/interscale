import torch
from torch_geometric.data import Data, Batch
from InterScale.tl import pad_batch
from InterScale.config import load_config
from InterScale.tl import remove_zero_expression_cells, prepare_geome_dataset
from InterScale.geome_dataloader import GraphAnnDataModule
import scanpy as sc


"""The goal of pad batch is to either (1) pad the batch to a fixed length or (2) select nodes to include in the batch (this should be a subset of the nodes from the mask)."""

LEGNINI_H5AD = "/Users/francesca.drummer/Documents/1_Projects/A3-LongRange/data/legnini23.h5ad"

def prepare_batch():
    cfg = load_config("src/config_files/legnini_test.yaml")
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    adata = remove_zero_expression_cells(adata)
    pyg_data_list, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(datas=pyg_data_list, 
                           num_workers=0,  # Set to 0 to avoid multiprocessing issues
                           batch_size=int(cfg.dataset.batch_size), 
                           pct_mask_nodes=0.2,
                           learning_type="node")
    return dm

def test_masked_nodes_priority():
    """Test that when get_mask=True and seq_len is smaller than context, masked nodes are prioritized."""
    # Create toy dataset with known masked nodes
    x1 = torch.randn(5, 10)  # 5 nodes, 10 features
    x2 = torch.randn(7, 10)  # 7 nodes, 10 features
    
    # Create batch indices [0,0,0,0,0, 1,1,1,1,1,1,1]
    batch = torch.cat([torch.zeros(5), torch.ones(7)])
    
    # Create mask where we know exactly which nodes are masked
    # For batch 0: nodes [1,3] are masked
    # For batch 1: nodes [5,8,10] are masked
    mask = torch.zeros_like(batch, dtype=torch.bool)
    mask[torch.tensor([1,3,5+5,8+5,10+5])] = True  # +5 offset for second batch
    
    # Combine features
    x = torch.cat([x1, x2], dim=0)
    
    # Set sequence length smaller than both batches
    seq_len = 3
    
    # Get padded batch with masking
    padded_x, src_padding_mask, index_nodes, num_nodes, masks, max_num_nodes = pad_batch(
        x,
        batch,
        seq_len,
        get_mask=True,
        keep_indices=mask
    )
    
    # Check first batch
    batch0_indices = index_nodes[0]
    assert 1 in batch0_indices, "Masked node 1 should be in first batch"
    assert 3 in batch0_indices, "Masked node 3 should be in first batch"
    assert len(batch0_indices) == seq_len, f"Batch should have {seq_len} nodes"
    
    # Check second batch
    batch1_indices = index_nodes[1]
    # Need to subtract 5 since these are local indices within batch 1
    batch1_indices_local = [idx-5 for idx in batch1_indices]
    assert 0 in batch1_indices_local, "Masked node 5 should be in second batch"
    assert 3 in batch1_indices_local, "Masked node 8 should be in second batch"
    assert 5 in batch1_indices_local, "Masked node 10 should be in second batch"
    assert len(batch1_indices) == seq_len, f"Batch should have {seq_len} nodes"
    
    print("Passed: masked nodes priority")



def test_fill_batch(batch: Batch):
    """Test that the batch is filled correctly if the number of nodes in the batch is less than the maximum sequence length."""
    curr_max_len = 0
    num_batch = batch.batch.max().item() + 1
    for i in range(num_batch):
        mask = batch.batch == i
        length = mask.sum().item()
        print(i, ': length', length)
        curr_max_len = max(curr_max_len, length)
    seq_len = curr_max_len + 123
    padded_h_node, src_padding_mask, index_nodes, num_nodes, masks, max_num_nodes = pad_batch(batch.x, batch.batch, seq_len)
    # Check all batches have same sequence length
    assert padded_h_node.shape[0] == curr_max_len, f"Expected all batches to have length {curr_max_len}, but got {padded_h_node.shape[0]}"
    assert src_padding_mask.shape[1] == curr_max_len, f"Expected padding mask to have length {curr_max_len}, but got {src_padding_mask.shape[1]}"
        
    print("Passed: fill batch")
        
if __name__ == '__main__':
    dm = prepare_batch()
    dm.setup(stage="fit")
    for batch in dm.train_dataloader():
        test_fill_batch(batch)
        test_masked_nodes_priority()
        break
