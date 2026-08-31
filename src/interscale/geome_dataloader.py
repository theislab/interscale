from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataListLoader, DataLoader

VALID_STAGE = {"fit", "test", "validate", None}
VALID_SPLIT = {"node", "graph"}

# TODO: Fix dataloader
import torch

from interscale.tl.masking import MASK_STRATEGIES, sample_gene_mask, sample_node_mask


class GraphAnnDataModule(pl.LightningDataModule):
    """Lightning DataModule for graph data."""

    def __init__(
        self,
        datas: Sequence[Sequence[Data]] | None = None,
        batch_size: int = 1,
        num_workers: int = 1,
        mask_percentage: float = 0.5,
        mask_strategy: Literal["node", "gene"] = "node",
        learning_type: Literal["node", "graph"] = "node",
    ):
        """Manages loading and sampling schemes before loading to GPU.

        Args:
        ----
        datas (Sequence[Sequence[Data]], optional):
            List of train, val (and test) data to be loaded. Defaults to None.
        batch_size (int, optional): The batch size. Defaults to 1.
        num_workers (int, optional): The number of workers. Defaults to 1.
        mask_percentage (float, optional): Bernoulli masking probability, per cell under
            `mask_strategy="node"` and per (cell, gene) entry under `"gene"`. One argument rather
            than one per strategy: the strategy already says what a unit is. Defaults to 0.5.
        mask_strategy (Literal["node", "gene"], optional): Granularity of the corruption.
            "node" blanks whole cells (loss over all G genes of the masked cells); "gene" blanks
            individual (cell, gene) entries in every cell (loss over those entries only). See
            `interscale.tl.masking` for why the two behave so differently. Defaults to "node".
        learning_type (Literal["node", "graph"], optional): The type of learning to be performed.
            If "graph" is selected, `batch_size` means the number of graphs and `datas` is expected to be a list of Data.
            If "node" is selected, `batch_size` means the number of nodes and `datas` is expected to be a list of Data objects
            with an edge_index attribute. Defaults to "node".

        Raises
        ------
            ValueError: If `learning_type` is not one of {"node", "graph"}.
        """
        super().__init__()
        self.setup_called = False
        self.batch_size = batch_size
        self.num_workers = num_workers
        if len(datas) == 2 or len(datas) == 3:
            self.train_data = datas[0]
            self.val_data = datas[1]
            if len(datas) == 3:
                self.test_data = datas[2]
        else:
            raise ValueError("datas must be list of list with at least train and validation set.")
        if learning_type not in VALID_SPLIT:
            raise ValueError("Learning type must be one of %r." % VALID_SPLIT)
        self.learning_type = learning_type
        if mask_strategy not in MASK_STRATEGIES:
            raise ValueError("mask_strategy must be one of %r." % (MASK_STRATEGIES,))
        self.mask_strategy = mask_strategy
        self.mask_percentage = mask_percentage
        self.first_time = True

    def _nodewise_setup(self, stage: str | None) -> None:
        """Sets up the data loaders for node-wise learning.

        Args:
        ----
        stage (Optional[str]): The stage of training to set up the data loader for. Defaults to None.

        Returns
        -------
            None
        """
        # For node-level learning the masked nodes *are* the supervision/evaluation targets
        # (see `_common_step`: y_pred/y_true are indexed by mask_idx), so every split must be
        # masked. For graph-level learning mask_idx is discarded and the graph label is used
        # instead, making masking pure input augmentation -- so it belongs to train only,
        # otherwise val/test inputs are corrupted for no benefit.
        eval_mask = self.learning_type == "node"

        if stage == "fit" or stage is None:
            self._train_dataloader = self._spatial_node_loader(data_list=self.train_data, shuffle=True)
            self._val_dataloader = self._spatial_node_loader(data_list=self.val_data, shuffle=False, mask=eval_mask)
        if stage == "test" or stage is None:
            self._test_dataloader = self._spatial_node_loader(data_list=self.test_data, shuffle=False, mask=eval_mask)

    def _graphwise_setup(self, stage: str | None) -> None:
        """Sets up the data loaders for graph-wise learning.

        Args:
        ----
        stage (Optional[str]): The stage of training to set up the data loader for. Defaults to None.

        Returns
        -------
            None
        """
        # ToTo: return unmasked object

    def setup(self, stage: str | None = None):
        """Setup function to be called at the beginning of training, validation or testing.

        Args:
        ----
        stage (str, optional): the stage of the training, either 'train', 'val' or 'test'. Defaults to None.
        """
        if stage not in VALID_STAGE:
            raise ValueError("Stage must be one of %r." % VALID_STAGE)

        self._nodewise_setup(stage)
        self.setup_called = True

    def train_dataloader(self):
        """Returns the training dataloader."""
        return self._get_dataloader(self._train_dataloader)

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return self._get_dataloader(self._val_dataloader)

    def test_dataloader(self):
        """Returns the test dataloader."""
        return self._get_dataloader(self._test_dataloader)

    def _get_dataloader(self, dataloader):
        if not self.setup_called:
            raise RuntimeError("setup method should be called before getting dataloaders")
        return dataloader

    def _assign_random_mask(self, data: BaseData) -> None:
        """Overwrites the mask attributes of `data` in place with a fresh Bernoulli draw.

        Under `mask_strategy="node"` only `data.mask` `[N]` is written: each cell is masked
        independently with probability `mask_percentage`.

        Under `mask_strategy="gene"` a second attribute `data.gene_mask` `[N, G]` is written,
        with each (cell, gene) entry drawn independently at `mask_percentage`. `data.mask` is then
        the row-wise OR of it -- i.e. "this cell is a supervision target" -- which is what the
        rest of the pipeline (padding, `_process_batch_for_metrics`) keys on. 

        `gene_mask` is a node-level attribute of shape `[num_nodes, ...]`, so PyG collates it by
        concatenating along dim 0 exactly like `x` -- no custom `__cat_dim__` needed. 
        """
        if self.mask_strategy == "gene":
            gene_mask = sample_gene_mask(data.num_nodes, data.x.shape[1], self.mask_percentage)
            data.gene_mask = gene_mask
            data.mask = gene_mask.any(dim=1)
        else:
            data.mask = sample_node_mask(data.num_nodes, self.mask_percentage)

    def resample_train_mask(self) -> None:
        """Redraws the masked node set for every training graph, in place.

        `_spatial_node_loader` only ever assigns `.mask` once (when the train dataloader is first
        built in `setup()`), so without this the same fixed subset of nodes is masked for the
        entire training run. Call this once per epoch (see `NodeMaskResampleCallback` in
        `interscale.train`) so every node eventually gets used as a supervision target.

        Mutates the `Data` objects already held in `self.train_data` in place — no cloning, no
        dataloader rebuild, so this adds no meaningful memory overhead beyond the `.mask` boolean
        tensor that is already allocated.
        """
        if not self.setup_called:
            return
        for data in self.train_data:
            self._assign_random_mask(data)

    def _spatial_node_loader(
        self, data_list: list[BaseData], shuffle: bool = False, mask: bool = True, **kwargs
    ) -> DataListLoader:
        """Adds a node mask to each Data object.

        Args:
        ----
        data: PyTorch geometric.Batch
        shuffle (bool, optional): whether to shuffle the data. Defaults to False.
        mask (bool, optional): whether to mask nodes at all. Graph-level evaluation splits pass
            False so that val/test inputs are not corrupted. Defaults to True.
        kwargs: arguments passed to the pyg.NeighborLoader

        Returns
        -------
            NeighborLoader: the node dataloader
        """
        for data in data_list:
            if mask:
                self._assign_random_mask(data)
            else:
                data.mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        return DataLoader(
            dataset=data_list,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )
