import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock

from InterScale.train import TrainingPlan

def create_toy_module(n_output=5, n_input=10):
    """Create a mock module with required attributes."""
    class ToyModule(nn.Module):
        def __init__(self, n_output, n_input):
            super().__init__()
            self.n_output = n_output
            self.n_input = n_input
            self.linear = nn.Linear(n_input, n_output)
            
        def forward(self, x):
            return self.linear(x)
            
        def _common_step(self, batch, prediction_task, prediction_level):
            # Mock implementation for testing
            batch_size = 32
            if 'classification' in prediction_task:
                y_pred = torch.randn(batch_size, self.n_output)
                y_true = torch.randint(0, self.n_output, (batch_size,))
                # Convert to one-hot for classification
                y_true_onehot = torch.zeros(batch_size, self.n_output)
                y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1)
            else:  # regression
                y_pred = torch.randn(batch_size, self.n_input)
                y_true = torch.randn(batch_size, self.n_input)
            
            return torch.randn(batch_size, 64), torch.randn(batch_size, 128), y_pred, y_true_onehot if 'classification' in prediction_task else y_true
    
    return ToyModule(n_output, n_input)

@pytest.mark.parametrize("loss", ["CrossEntropy", "WeightedCE"])
@pytest.mark.parametrize("cross_corr", ["gene", "cell"])
@pytest.mark.parametrize("prediction_level", ["node", "graph"])
def test_classification_setup(loss, cross_corr, prediction_level):
    """Test that classification metrics and loss are set up correctly."""
    module = create_toy_module(n_output=3, n_input=10)
    
    # Test classification setup
    training_plan = TrainingPlan(
        module=module,
        prediction_task="classification",
        prediction_level=prediction_level,
        loss=loss,
        cross_corr=cross_corr,
        class_weights=torch.tensor([0.5, 1.0, 2.0]),
        batch_size=32
    )
    
    # Check that loss is set up correctly
    if loss == "CrossEntropy":
        assert isinstance(training_plan.loss, nn.CrossEntropyLoss)
    elif loss == "WeightedCE":
        assert isinstance(training_plan.loss, nn.CrossEntropyLoss)
        assert training_plan.class_weights is not None
    
    # Check that metrics are set up correctly
    assert hasattr(training_plan, 'train_metrics')
    assert hasattr(training_plan, 'valid_metrics')
    assert hasattr(training_plan, 'test_metrics')
    
    # Check classification metrics
    train_metrics = training_plan.train_metrics
    assert 'accuracy' in train_metrics
    assert 'f1_micro' in train_metrics
    assert 'f1_macro' in train_metrics
    assert 'f1_per_class' in train_metrics
    
    # Check that metrics have correct number of classes
    assert train_metrics['accuracy'].num_classes == 3
    assert train_metrics['f1_micro'].num_classes == 3
    assert train_metrics['f1_macro'].num_classes == 3
    assert train_metrics['f1_per_class'].num_classes == 3

@pytest.mark.parametrize("loss", ["MSELoss", "GaussianNLL", "SmoothL1"])
def test_classification_invalid_loss(loss):
    """Test that classification with invalid loss types raises an error."""
    module = create_toy_module(n_output=3, n_input=10)
    
    # Test that regression losses raise an error for classification
    with pytest.raises(AssertionError, match="Classification must be run with CrossEntropy or WeightedCE loss."):
        TrainingPlan(
            module=module,
            prediction_task="classification",
            prediction_level="node",
            loss=loss,  # Invalid for classification
            cross_corr="gene",
            batch_size=32
        )
    
@pytest.mark.parametrize("loss", ["MSELoss", "GaussianNLL", "SmoothL1"])
@pytest.mark.parametrize("cross_corr", ["gene", "cell"])
@pytest.mark.parametrize("prediction_level", ["node", "graph"]) #TODO: shouldnt be able to set up regression for graph level
def test_regression_setup(loss, cross_corr, prediction_level):
    """Test that regression metrics and loss are set up correctly."""
    module = create_toy_module(n_output=10, n_input=10)
    
    # Test regression setup
    training_plan = TrainingPlan(
        module=module,
        prediction_task="regression",
        prediction_level=prediction_level,
        loss=loss,
        cross_corr=cross_corr,
        batch_size=32
    )
    
    # Check that loss is set up correctly
    if loss == "MSELoss":
        assert isinstance(training_plan.loss, nn.MSELoss)
    elif loss == "GaussianNLL":
        assert isinstance(training_plan.loss, nn.GaussianNLLLoss)
    elif loss == "SmoothL1":
        assert isinstance(training_plan.loss, nn.SmoothL1Loss)
    
    # Check that metrics are set up correctly
    assert hasattr(training_plan, 'train_metrics')
    assert hasattr(training_plan, 'valid_metrics')
    assert hasattr(training_plan, 'test_metrics')
    
    # Check regression metrics
    train_metrics = training_plan.train_metrics
    assert 'mse' in train_metrics
    assert 'r2' in train_metrics
    
    # TODO: check dimension for cross_corr gene vs cell

@pytest.mark.parametrize("loss", ["CrossEntropy", "WeightedCE"])
def test_regression_invalid_loss(loss):
    """Test that regression with invalid loss types raises an error."""
    module = create_toy_module(n_output=10, n_input=10)
    
    # Test that classification losses raise an error for regression
    with pytest.raises(AssertionError, match="Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss."):
        TrainingPlan(
            module=module,
            prediction_task="regression",
            prediction_level="node",
            loss=loss,  # Invalid for regression
            cross_corr="gene",
            batch_size=32
        )

@pytest.mark.parametrize("loss", ["CrossEntropy", "WeightedCE"])
@pytest.mark.parametrize("prediction_level", ["node", "graph"])
@pytest.mark.parametrize("n_cells", [32, 100])
def test_classification_metrics_computation(loss, n_cells, prediction_level):
    """Test that classification metrics are computed correctly."""
    module = create_toy_module(n_output=3, n_input=10)
    
    class_weights = torch.tensor([0.5, 1.0, 2.0])
    
    training_plan = TrainingPlan(
        module=module,
        prediction_task="classification",
        prediction_level=prediction_level,
        loss=loss,
        cross_corr="gene",
        batch_size=32,
        class_weights=class_weights
    )
    
    # Create test data
    y_pred = torch.randn(n_cells, 3)
    y_true = torch.randint(0, 3, (n_cells,))
    y_true_onehot = torch.zeros(n_cells, 3)
    y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1)
    
    # Test metrics computation
    metrics = training_plan._classification_metrics(
        y_pred, y_true_onehot, 'train', training_plan.train_metrics
    )
    
    assert 'train_accuracy' in metrics
    assert 'train_f1_micro' in metrics
    assert 'train_f1_macro' in metrics
    assert 'train_f1_per_class' in metrics
    assert 'train_loss' in metrics
    
    assert metrics['train_accuracy'].unsqueeze(0).shape == torch.Size([1]), f"Train accuracy expected shape (1,), got {metrics['train_accuracy'].unsqueeze(0).shape}"
    assert metrics['train_f1_micro'].unsqueeze(0).shape == torch.Size([1]), f"Train f1_micro expected shape (1,), got {metrics['train_f1_micro'].unsqueeze(0).shape}"
    assert metrics['train_f1_macro'].unsqueeze(0).shape == torch.Size([1]), f"Train f1_macro expected shape (1,), got {metrics['train_f1_macro'].unsqueeze(0).shape}"
    assert metrics['train_f1_per_class'].unsqueeze(0).shape == torch.Size([1,3]), f"Train f1_per_class expected shape (3,), got {metrics['train_f1_per_class'].unsqueeze(0).shape}"
    assert metrics['train_loss'].unsqueeze(0).shape == torch.Size([1]), f"Train loss expected shape (1), got {metrics['train_loss'].unsqueeze(0).shape}" 
    
def test_mask_idx_difference():
    """Test that mask_idx makes a difference in metrics computation."""
    module = create_toy_module(n_output=3, n_input=10)
    
    training_plan = TrainingPlan(
        module=module,
        prediction_task="classification",
        prediction_level="node",
        loss="CrossEntropy",
        cross_corr="gene",
        batch_size=32
    )
    
    # Create test data with some bad predictions
    n_cells = 100
    y_pred = torch.randn(n_cells, 3)
    y_true = torch.randint(0, 3, (n_cells,))
    y_true_onehot = torch.zeros(n_cells, 3)
    y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1)
    
    # Compute metrics without mask
    metrics_no_mask = training_plan._classification_metrics(
        y_pred, y_true_onehot, 'train', training_plan.train_metrics
    )
    
    # Create a mask that excludes some cells (e.g., first 20 cells)
    mask_idx = torch.arange(20, n_cells)
    
    # Compute metrics with mask
    metrics_with_mask = training_plan._classification_metrics(
        y_pred, y_true_onehot, 'train', training_plan.train_metrics, mask_idx=mask_idx
    )
    
    # The metrics should be different because we're using different subsets of data
    print(f"Metrics without mask: {metrics_no_mask}")
    print(f"Metrics with mask: {metrics_with_mask}")
    
    # Check that the loss values are different (they should be since we're using different data)
    assert not torch.allclose(
        metrics_no_mask['train_loss'], 
        metrics_with_mask['train_loss'], 
        atol=1e-6
    ), "Loss should be different with and without mask"
    
    # Check that accuracy is different
    assert not torch.allclose(
        metrics_no_mask['train_accuracy'], 
        metrics_with_mask['train_accuracy'], 
        atol=1e-6
    ), "Accuracy should be different with and without mask"
    
#     def test_regression_metrics_computation(self):
#         """Test that regression metrics are computed correctly."""
#         module = create_toy_module(n_output=10, n_input=10)
        
#         training_plan = TrainingPlan(
#             module=module,
#             prediction_task="regression",
#             prediction_level="node",
#             loss="MSELoss",
#             cross_corr="gene",
#             batch_size=32
#         )
        
#         # Create test data
#         batch_size = 32
#         y_pred = torch.randn(batch_size, 10)
#         y_true = torch.randn(batch_size, 10)
        
#         # Test metrics computation
#         loss, metrics = training_plan._regression_metrics(
#             y_pred, y_true, training_plan.train_metrics
#         )
        
#         assert isinstance(loss, torch.Tensor)
#         assert 'mse' in metrics
#         assert 'r2' in metrics
#         assert 'loss' in metrics
#         assert 'pearson_corr' in metrics


# if __name__ == "__main__":
#     pytest.main([__file__])

