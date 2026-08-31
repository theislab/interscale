import torch
from torch_geometric.data import Data

from interscale.geome_dataloader import GraphAnnDataModule
from interscale.train._utils import NodeMaskResampleCallback


def _make_data(num_nodes: int, num_features: int = 3) -> Data:
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def _build_datamodule(mask_percentage: float = 0.5) -> GraphAnnDataModule:
    train_data = [_make_data(20), _make_data(20), _make_data(20)]
    val_data = [_make_data(10)]
    test_data = [_make_data(10)]
    dm = GraphAnnDataModule(
        datas=[train_data, val_data, test_data],
        batch_size=1,
        num_workers=0,
        mask_percentage=mask_percentage,
        learning_type="node",
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")
    return dm


def test_resample_train_mask_changes_masked_nodes_without_replacing_objects():
    """The masked node set must differ across calls, without cloning/replacing the `Data` objects."""
    dm = _build_datamodule()
    original_masks = [data.mask.clone() for data in dm.train_data]
    original_ids = [id(data) for data in dm.train_data]
    original_list_id = id(dm.train_data)

    changed = False
    for _ in range(20):
        dm.resample_train_mask()
        if any(not torch.equal(orig, data.mask) for orig, data in zip(original_masks, dm.train_data)):
            changed = True
            break

    assert changed, "resample_train_mask never produced a different mask across 20 redraws"
    assert [id(data) for data in dm.train_data] == original_ids, "Data objects must be mutated in place, not replaced"
    assert id(dm.train_data) == original_list_id, "train_data list must not be rebuilt"


def test_resample_train_mask_leaves_val_and_test_data_untouched():
    dm = _build_datamodule()
    original_val_masks = [data.mask.clone() for data in dm.val_data]
    original_test_masks = [data.mask.clone() for data in dm.test_data]

    for _ in range(5):
        dm.resample_train_mask()

    for orig, data in zip(original_val_masks, dm.val_data):
        assert torch.equal(orig, data.mask)
    for orig, data in zip(original_test_masks, dm.test_data):
        assert torch.equal(orig, data.mask)


def test_resample_train_mask_is_noop_before_setup():
    """Calling resample_train_mask before setup() must not raise (no dataloader exists yet)."""
    dm = GraphAnnDataModule(datas=[[_make_data(20)], [_make_data(10)]], num_workers=0)

    dm.resample_train_mask()


def test_node_mask_resample_callback_delegates_to_datamodule():
    dm = _build_datamodule()
    calls = []
    dm.resample_train_mask = lambda: calls.append(1)

    class _StubTrainer:
        datamodule = dm

    NodeMaskResampleCallback().on_train_epoch_start(_StubTrainer(), pl_module=None)

    assert calls == [1]
