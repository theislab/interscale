import graph_transformer_long_range_niches.tl as MaskedNodeLightningDataset
from ..conftests import create_sample_graph


def test_add_pyg_mask_nodes(sample_config):
    """Tests that the MaskedNodeLightningDataset adds a mask to the batch, if pct_mask_nodes is set (and >0)."""
    cfg = sample_config
    graph = create_sample_graph(cfg)
    
    # Case 1: Add mask if pct_mask_nodes > 0
    dm = MaskedNodeLightningDataset(train_dataset=graph, pct_mask_nodes=0.8)
    assert len(dm.train_dataloader()) > 0
    for i, batch in enumerate(dm.train_dataloader()):
        print(batch.mask)
        assert batch.mask is not None
        
    # Case 2: No mask if pct_mask_nodes == 0
    dm = MaskedNodeLightningDataset(train_dataset=graph, pct_mask_nodes=0)
    assert len(dm.train_dataloader()) == 1
    for i, batch in enumerate(dm.train_dataloader()):
        assert batch.mask is None
    