import pytest
import torch
from torch_geometric.data import Data
import numpy as np
from InterScale.geome_dataloader import GraphAnnDataModule
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from tests._model_test_utils import create_test_pyg_data

def split_data(data_list: List[Data], 
               train_size: float = 0.7, 
               val_size: float = 0.15, 
               test_size: float = 0.15,
               seed: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Split data list into train, validation and test sets using sklearn's train_test_split.
    
    Parameters:
    -----------
    data_list : List[Data]
        List of PyG data objects
    train_size : float
        Proportion of data for training
    val_size : float
        Proportion of data for validation
    test_size : float
        Proportion of data for testing
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[List[Data], List[Data], List[Data]]
        Train, validation and test data lists
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1"
    
    # First split: separate out the test set
    test_ratio = test_size / (1 - train_size)
    train_val_data, test_data = train_test_split(
        data_list,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )
    
    # Second split: split remaining data into train and validation
    val_ratio = val_size / (train_size + val_size)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True
    )
    
    return train_data, val_data, test_data

def check_class_distribution(data_list: List[Data], 
                           expected_distribution: dict,
                           tolerance: float = 0.1) -> bool:
    """
    Check if the class distribution in the data list matches the expected distribution.
    
    Parameters:
    -----------
    data_list : List[Data]
        List of PyG data objects
    expected_distribution : dict
        Expected class distribution
    tolerance : float
        Allowed deviation from expected distribution
        
    Returns:
    --------
    bool
        True if distribution matches within tolerance
    """
    all_labels = []
    for data in data_list:
        all_labels.extend(data.y.tolist())
    
    unique_classes, counts = np.unique(all_labels, return_counts=True)
    total_nodes = len(all_labels)
    
    actual_distribution = {cls: count / total_nodes 
                         for cls, count in zip(unique_classes, counts)}
    
    for cls, expected_prob in expected_distribution.items():
        if cls in actual_distribution:
            actual_prob = actual_distribution[cls]
            if abs(actual_prob - expected_prob) > tolerance:
                return False
    return True

def is_approximately_equal(a: int, b: int, tolerance: int = 1) -> bool:
    """
    Check if two integers are approximately equal within a tolerance.
    
    Parameters:
    -----------
    a : int
        First integer
    b : int
        Second integer
    tolerance : int
        Maximum allowed difference
        
    Returns:
    --------
    bool
        True if integers are approximately equal
    """
    return abs(a - b) <= tolerance

def test_data_splitting():
    # Create test data
    data_list = create_test_pyg_data()
    print("Original data list:", data_list)
    
    # Split data
    train_data, val_data, test_data = split_data(
        data_list,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        seed=42
    )
    print("Split sizes:", len(train_data), len(val_data), len(test_data))
    
    # Check split sizes approximately
    total_size = len(data_list)
    expected_train = int(total_size * 0.7)
    expected_val = int(total_size * 0.15)
    expected_test = int(total_size * 0.15)
    
    assert is_approximately_equal(len(train_data), expected_train), \
        f"Train split size {len(train_data)} is not approximately equal to expected {expected_train}"
    assert is_approximately_equal(len(val_data), expected_val), \
        f"Validation split size {len(val_data)} is not approximately equal to expected {expected_val}"
    assert is_approximately_equal(len(test_data), expected_test), \
        f"Test split size {len(test_data)} is not approximately equal to expected {expected_test}"
    
    # Check that all data points are accounted for
    assert len(train_data) + len(val_data) + len(test_data) == total_size, \
        "Total number of data points in splits does not match original data size"
    
    # Expected class distribution
    expected_distribution = {0: 0.50, 1: 0.25, 2: 0.20, 3: 0.05}
    
    # Check class distribution in each split
    assert check_class_distribution(train_data, expected_distribution), \
        "Train set class distribution is incorrect"
    assert check_class_distribution(val_data, expected_distribution), \
        "Validation set class distribution is incorrect"
    assert check_class_distribution(test_data, expected_distribution), \
        "Test set class distribution is incorrect"

def test_geome_dataloader():
    # Create test data
    data_list = create_test_pyg_data()
    
    # Split data
    train_data, val_data, test_data = split_data(data_list)
    
    # Initialize dataloader with 20% masking
    dataloader = GraphAnnDataModule(
        datas=[train_data, val_data, test_data],  # Use only training data
        batch_size=2,
        pct_mask_nodes=0.2,
        num_workers=0
    )
    
    # Get the spatial node loader
    loader = dataloader._spatial_node_loader(train_data, shuffle=False)
    
    # Test a few batches
    for batch in loader:
        # Check that train_mask exists and has correct shape
        assert hasattr(batch, 'train_mask')
        assert batch.train_mask.shape == (batch.num_nodes,)
        
        # Check that train_mask is boolean
        assert batch.train_mask.dtype == torch.bool
        
        # Check that the number of masked nodes is correct
        # Should be at least 1 node masked (as per the code)
        num_masked = batch.train_mask.sum().item()
        assert num_masked >= 1
        
        # Check that the data structure is preserved
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        assert hasattr(batch, 'y')
        
        # Check that features and labels have correct shapes
        assert batch.x.shape[0] == batch.num_nodes
        assert batch.y.shape[0] == batch.num_nodes
        
        # Check class distribution in the batch
        unique_classes, counts = torch.unique(batch.y, return_counts=True)
        total_nodes = batch.num_nodes
        
        # Calculate actual distribution
        actual_distribution = {cls.item(): count.item() / total_nodes 
                             for cls, count in zip(unique_classes, counts)}
        
        # Check if distribution is roughly correct (within 10% tolerance)
        expected_distribution = {0: 0.50, 1: 0.25, 2: 0.20, 3: 0.05}
        for cls, expected_prob in expected_distribution.items():
            if cls in actual_distribution:
                actual_prob = actual_distribution[cls]
                assert abs(actual_prob - expected_prob) <= 0.1, \
                    f"Class {cls} distribution is off. Expected {expected_prob}, got {actual_prob}"

if __name__ == "__main__":
    test_data_splitting()
    test_geome_dataloader()
    print("All tests passed!") 