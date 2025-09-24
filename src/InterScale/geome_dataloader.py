from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, List

import lightning.pytorch as pl
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader, DataListLoader

VALID_STAGE = {"fit", "test", "validate", None}
VALID_SPLIT = {"node", "graph"}

# TODO: Fix dataloader
import torch
import random
import numpy as np

class GraphAnnDataModule(pl.LightningDataModule):
    """Lightning DataModule for graph data."""

    def __init__(
        self,
        datas: Sequence[Sequence[Data]] | None = None,
        batch_size: int = 1,
        num_workers: int = 1,
        pct_mask_nodes: float = 0.5,
        learning_type: Literal["node", "graph"] = "node",
    ):
        """Manages loading and sampling schemes before loading to GPU.

        Args:
        ----
        datas (Sequence[Sequence[Data]], optional): 
            List of train, val (and test) data to be loaded. Defaults to None.
        batch_size (int, optional): The batch size. Defaults to 1.
        num_workers (int, optional): The number of workers. Defaults to 1.
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
        self.pct_mask_nodes = pct_mask_nodes
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

        if stage == "fit" or stage is None:
            self._train_dataloader = self._spatial_node_loader(data_list=self.train_data, shuffle=True)
            self._val_dataloader = self._spatial_node_loader(data_list=self.val_data, shuffle=False)
        if stage == "test" or stage is None:
            self._test_dataloader = self._spatial_node_loader(data_list=self.test_data, shuffle=False)

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

    def _smallest_data_batch_length(self, data_list: List['BaseData']):
        """Returns the number of nodes in the smallest graph from the list of BaseData."""
        lengths = [data.num_nodes for data in data_list]
        return min(lengths)

    def _spatial_node_loader(self, 
                             data_list: List[BaseData], 
                             shuffle: bool = False, 
                             **kwargs) -> DataListLoader:
        """Adds a one-node mask to each Data object. TODO: load each graph multiple times with a different mask.

        Args:
        ----
        data: PyTorch geometric.Batch
        shuffle (bool, optional): whether to shuffle the data. Defaults to False.
        kwargs: arguments passed to the pyg.NeighborLoader

        Returns
        -------
            NeighborLoader: the node dataloader
        """
        smallest_length = self._smallest_data_batch_length(data_list)
        num_nodes_to_mask = int(smallest_length * self.pct_mask_nodes)
        if num_nodes_to_mask == 0: # must mask at least one node
            num_nodes_to_mask = 1
        
        for data in data_list:
            if data.num_nodes < num_nodes_to_mask:
                raise ValueError("Cannot sample more nodes than available in any graph.")

            # Randomly select a ndoe to mask
            mask_indices = random.sample(range(data.num_nodes), num_nodes_to_mask)
            data.mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.mask[mask_indices] = True

        return DataLoader(
            dataset=data_list,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )    