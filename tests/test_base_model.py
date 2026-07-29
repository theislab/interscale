import numpy as np
import pandas as pd
from anndata import AnnData

from interscale.model.local_model import LocalModel


def _make_adata(sample_key_columns: dict[str, list[str]], split: list[str], n_var: int = 4) -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(len(split), n_var)).astype(np.float32))
    for key, values in sample_key_columns.items():
        adata.obs[key] = pd.Categorical(values)
    adata.obs["split"] = pd.Categorical(split)
    return adata


def test_setup_anndata_stores_full_sample_key_list():
    """`_setup_anndata` must keep every key from `sample_key_list`, not just the last one seen in its
    registration loop."""
    adata = _make_adata(
        sample_key_columns={
            "sample_a": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "sample_b": ["fov1", "fov2", "fov1", "fov2", "fov1", "fov2"],
        },
        split=["train", "train", "val", "train", "val", "val"],
    )
    sample_key_list = ["sample_a", "sample_b"]

    LocalModel._setup_anndata(
        adata=adata,
        layer_key=None,
        sample_key_list=sample_key_list,
        prediction_task="regression",
        view_registry=False,
    )

    assert LocalModel.sample_key_list == sample_key_list


def test_setup_anndata_with_empty_sample_key_list_does_not_raise():
    """An empty `sample_key_list` used to raise `NameError`, since the loop variable it was read from
    was never assigned."""
    adata = _make_adata(sample_key_columns={}, split=["train", "train", "val", "val"])

    LocalModel._setup_anndata(
        adata=adata,
        layer_key=None,
        sample_key_list=[],
        prediction_task="regression",
        view_registry=False,
    )

    assert LocalModel.sample_key_list == []
