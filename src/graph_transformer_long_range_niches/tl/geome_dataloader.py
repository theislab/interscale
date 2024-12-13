from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pytorch_lightning as pl
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataListLoader, NeighborLoader
from torch_geometric.transforms import RandomNodeSplit

VALID_STAGE = {"fit", "test", None}
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
        # TODO: Fill the docstring

        super().__init__()
        self.setup_called = False
        self.batch_size = batch_size
        self.num_workers = num_workers
        if len(datas) == 2 or len(datas) == 3:
            self.train_data = datas[0]
            self.val_data = datas[1]
            if len(datas) == 3:
                self.test_data = datas[1]
        else:
            raise ValueError("datas must be list of list with at least train and validation set.")
        if learning_type not in VALID_SPLIT:
            raise ValueError("Learning type must be one of %r." % VALID_SPLIT)
        self.learning_type = learning_type
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
        if self.first_time:
            # Batch: Merged datas objects and adds .batch index 
            # datas = [Data(), Data(), .., Data()] -> Data(batch = [N])
            self.train_data = Batch.from_data_list(self.train_data)
            self.val_data = Batch.from_data_list(self.val_data)
            if self.test_data:
                self.test_data = Batch.from_data_list(self.test_data)

            self.first_time = False

        if stage == "fit" or stage is None:
            self._train_dataloader = self._spatial_node_loader(data=self.train_data, shuffle=True)
            self._val_dataloader = self._spatial_node_loader(data=self.val_data, shuffle=True)
        if stage == "test" or stage is None:
            self._test_dataloader = self._spatial_node_loader(data=self.test_data, shuffle=True)

    def _graphwise_setup(self, stage: str | None) -> None:
        """Sets up the data loaders for graph-wise learning.

        Args:
        ----
        stage (Optional[str]): The stage of training to set up the data loader for. Defaults to None.

        Returns
        -------
            None
        """
        num_val = int(len(self.data) * 0.05 + 1)
        num_test = int(len(self.data) * 0.01 + 1)

        if stage == "fit" or stage is None:
            self._train_dataloader = self._graph_loader(
                data=self.data[num_val + num_test :],
                shuffle=True,
            )
            self._val_dataloader = self._graph_loader(data=self.data[:num_val])
        if stage == "test" or stage is None:
            self._test_dataloader = self._graph_loader(data=self.data[num_val : num_val + num_test])

    def setup(self, stage: str | None = None):
        """Setup function to be called at the beginning of training, validation or testing.

        Args:
        ----
        stage (str, optional): the stage of the training, either 'train', 'val' or 'test'. Defaults to None.
        """
        # TODO: Implement each case
        # TODO: Splitting
        # stage = "train" if not stage else stage

        if stage not in VALID_STAGE:
            raise ValueError("Stage must be one of %r." % VALID_STAGE)

        if self.learning_type == "graph":
            if len(self.data) <= 3:
                raise RuntimeError("Not enough graphs in data to do graph-wise learning")
            self._graphwise_setup(stage)

        else:
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

    def _graph_loader(self, data: list, shuffle: bool = False, **kwargs) -> DataListLoader:
        """Loads the data in the form of graphs.

        Args:
        ----
        data (List): list of data to be loaded
        shuffle (bool, optional): whether to shuffle the data. Defaults to False.
        kwargs: arguments passed to the pyg.DataListLoader

        Returns
        -------
        DataListLoader: the graph dataloader
        """
        return DataListLoader(
            dataset=data, shuffle=shuffle, batch_size=self.batch_size, num_workers=self.num_workers, **kwargs
        )
    
    def smallest_data_batch_length(self, data_batch):
        batch_size = data_batch.batch.max().item() + 1  # Number of graphs
        lengths = [(data_batch.batch == i).sum().item() for i in range(batch_size)]
        return min(lengths)

    def _spatial_node_loader(self, 
                             data: Batch, 
                             shuffle: bool = False, 
                             **kwargs) -> DataListLoader:
        """Loads the data in the form of nodes.

        Args:
        ----
        data: PyTorch geometric.Batch
        shuffle (bool, optional): whether to shuffle the data. Defaults to False.
        kwargs: arguments passed to the pyg.NeighborLoader

        Returns
        -------
            NeighborLoader: the node dataloader
        """
        print(self.smallest_data_batch_length(data))
        num_nodes = int(self.smallest_data_batch_length(data) * self.pct_mask_nodes)
        index_list = []
        for batch_idx in np.unique(data.batch):
            mask = data.batch.eq(batch_idx)
            indices = np.where(mask)[0]
            start = indices[0] if len(indices) > 0 else None
            end = indices[-1] if len(indices) > 0 else None
            assert (end - start) > num_nodes # don't sample more nodes than available in a batch
            index_list.append(random.sample(range(start, end), num_nodes))
        for idx in range(num_nodes):
            masked_data = data.clone()
            masked_data.mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            mask_index = [sublist[idx] for sublist in index_list]
            masked_data.mask[mask_index] = True
            return DataListLoader(
                dataset=masked_data, shuffle=shuffle, batch_size=self.batch_size, num_workers=self.num_workers, **kwargs
            )