from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, List

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataListLoader, NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from torch.utils.data import DataLoader

VALID_STAGE = {"fit", "test", None}
VALID_SPLIT = {"node", "graph"}

def custom_collate(batch: List[Data]):
    """Custom collate function for PyG data objects."""
    return batch


class GraphAnnDataModule(pl.LightningDataModule):
    """Lightning DataModule for PyG data"""

    def __init__(self,  
                 datas: Sequence[Data] | None = None,
                 train_size: float=0.8,
                 val_size: float=0.2,
                 batch_size: int = 1,
                 num_workers: int = 1
        ):
        """Manages loading and sampling schemes before loading to GPU.

        Args:
        ----
        datas (Sequence[Data], optional): The data to be loaded, list of PyG data objects. Defaults to None.
        batch_size (int, optional): The batch size. Defaults to 1.
        num_workers (int, optional): The number of workers. Defaults to 1.
        learning_type (Literal["node", "graph"], optional): The type of learning to be performed.
            If "graph" is selected, `batch_size` means the number of graphs and `datas` is expected to be a list of Data.
            If "node" is selected, `batch_size` means the number of nodes and `datas` is expected to be a list of Data objects
            with an edge_index attribute. Defaults to "node".
        """
        super().__init__()
        self.datas = datas
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        if self.train_size + self.val_size < 1.0:
            self.test_size = 1 - self.train_size + self.val_size
        else:
            self.test_size = 0.0

    def prepare_data(self):
        # single gpu, e.i. download data to have it to disk. Do I need to do that? 
        pass

    def setup(self, stage):
        # split data
        self.train_ds, self.val_ds = train_test_split(self.datas, train_size=self.train_size, test_size=self.val_size+self.test_size, random_state=42)
        if self.test_size > 0.0:
            self.val_ds, self.test_ds = train_test_split(self.val_ds, train_size=1-self.test_size, test_size=self.test_size, random_state=42)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
            collate_fn = custom_collate 
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
            collate_fn = custom_collate 
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
            collate_fn = custom_collate 
        )