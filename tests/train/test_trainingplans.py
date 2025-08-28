import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

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
        class_weights=torch.tensor([0.5, 1.0, 2.0]) if loss == "WeightedCE" else None,
        class_labels=["class_0", "class_1", "class_2"],
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
    
    class_weights = torch.tensor([0.5, 1.0, 2.0]) if loss == "WeightedCE" else None
    
    training_plan = TrainingPlan(
        module=module,
        prediction_task="classification",
        prediction_level=prediction_level,
        loss=loss,
        cross_corr="gene",
        batch_size=32,
        class_weights=class_weights,
        class_labels=["class_0", "class_1", "class_2"]
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
    


@pytest.mark.parametrize("mode", ["train", "val", "test"])
@pytest.mark.parametrize("prediction_task", ["classification", "regression"])
@pytest.mark.parametrize("cross_corr", ["gene", "cell"])
def test_compute_and_log_metrics_classification(mode, prediction_task, cross_corr):
    """Test the _compute_and_log_metrics method for classification tasks."""
    if prediction_task == "classification":
        module = create_toy_module(n_output=3, n_input=10)
        class_labels = ["class_0", "class_1", "class_2"]
        loss = "CrossEntropy"
    else:
        module = create_toy_module(n_output=10, n_input=10)
        class_labels = None
        loss = "MSELoss"
    
    training_plan = TrainingPlan(
        module=module,
        prediction_task=prediction_task,
        prediction_level="node",
        loss=loss,
        cross_corr=cross_corr,
        batch_size=32,
        class_labels=class_labels
    )
    
    # Create test data
    batch_size = 32
    if prediction_task == "classification":
        y_pred = torch.randn(batch_size, 3)
        y_true = torch.randint(0, 3, (batch_size,))
        y_true_onehot = torch.zeros(batch_size, 3)
        y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1)
    else:
        y_pred = torch.randn(batch_size, 10)
        y_true = torch.randn(batch_size, 10)
    
    if mode == "train":
        metrics = training_plan.train_metrics
    elif mode == "val":
        metrics = training_plan.valid_metrics
    elif mode == "test":
        metrics = training_plan.test_metrics
    
    # Mock the log_dict method to avoid actual logging during tests
    with patch.object(training_plan, 'log_dict') as mock_log_dict:
        # Test metrics computation and logging
        loss_value = training_plan._compute_and_log_metrics(
            y_pred, y_true_onehot if prediction_task == "classification" else y_true, 
            mode, metrics
        )
        
        # Check that log_dict was called
        mock_log_dict.assert_called_once()
        
        # Check the arguments passed to log_dict
        call_args = mock_log_dict.call_args
        logged_metrics = call_args[0][0]  # First argument is the metrics dict
        log_kwargs = call_args[1]  # Keyword arguments
        
        # Check that loss is returned
        assert isinstance(loss_value, torch.Tensor)
        
        # Check that appropriate metrics are logged
        if prediction_task == "classification":
            assert f'{mode}_loss' in logged_metrics
            assert f'{mode}_accuracy' in logged_metrics
            assert f'{mode}_f1_micro' in logged_metrics
            assert f'{mode}_f1_macro' in logged_metrics
            # Check that per-class F1 scores are logged
            for class_name in class_labels:
                assert f'{mode}_f1_{class_name}' in logged_metrics
        else:
            assert f'{mode}_loss' in logged_metrics
            assert f'{mode}_mse' in logged_metrics
            assert f'{mode}_r2' in logged_metrics
            assert f'{mode}_pearson_corr' in logged_metrics
        
        # Check log_kwargs
        assert log_kwargs['batch_size'] == 32
        assert log_kwargs['on_step'] == False
        assert log_kwargs['on_epoch'] == True
        # sync_dist should be True only for test mode
        assert log_kwargs['sync_dist'] == (mode == 'test')


# if __name__ == "__main__":
#     pytest.main([__file__])

