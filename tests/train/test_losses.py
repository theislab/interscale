import pytest
import torch
from InterScale.train.losses import BalancedPearsonCorrelationLoss, GaussianLoss

def setup_loss(loss_type, cross_corr):
    if loss_type == "GaussianLoss":
        return GaussianLoss(cross_corr=cross_corr)
    elif loss_type == "BalancedPearsonCorrelationLoss":
        return BalancedPearsonCorrelationLoss(cross_corr=cross_corr)
    
def get_test_case(test_case: str, nr_cells: int, nr_genes: int = 10):
    """Generate test cases for training plan tests.
    
    Args:
        test_case: The type of test case to generate
        nr_cells: Number of cells to use in the test data

    Returns:
        Dictionary containing y_pred and y_true tensors for the specified test case
    """
    test_cases = {
        "normal": {
            "y_pred": torch.randn(nr_cells, nr_genes),
            "y_true": torch.randn(nr_cells, nr_genes)
        },
        "constant_cell": {
            "y_pred": torch.ones(nr_cells, nr_genes) * torch.randn(nr_cells, 1),
            "y_true": torch.ones(nr_cells, nr_genes) * torch.randn(nr_cells, 1)
        },
        "constant_gene": {
            "y_pred": torch.transpose(torch.ones(nr_genes, nr_cells) * torch.randn(nr_genes, 1), 0, 1),
            "y_true": torch.transpose(torch.ones(nr_genes, nr_cells) * torch.randn(nr_genes, 1), 0, 1)
        },
        "zero_cell": {
            "y_pred": torch.where(torch.rand(nr_cells, nr_genes) > 0.5, torch.randn(nr_cells, nr_genes), torch.zeros(nr_cells, nr_genes)),
            "y_true": torch.where(torch.rand(nr_cells, nr_genes) > 0.5, torch.randn(nr_cells, nr_genes), torch.zeros(nr_cells, nr_genes))
        },
        "zero_gene": {
            "y_pred": torch.where(torch.rand(nr_cells, nr_genes) > 0.5, torch.randn(nr_cells, nr_genes), torch.zeros(nr_cells, nr_genes)),
            "y_true": torch.where(torch.rand(nr_cells, nr_genes) > 0.5, torch.randn(nr_cells, nr_genes), torch.zeros(nr_cells, nr_genes))
        }
    }
    
    if test_case not in test_cases:
        raise ValueError(f"Unknown test case: {test_case}. Available cases: {list(test_cases.keys())}")
    
    return test_cases[test_case]

# Test basic functionality
@pytest.mark.parametrize("loss_type", ["GaussianLoss", "BalancedPearsonCorrelationLoss"])
@pytest.mark.parametrize("cross_corr", ["gene", "cell"])
def test_basic_GaussianLoss(loss_type, cross_corr):
    """Test basic forward pass with different cross_corr modes."""
    N, F = 10, 5
    y_true = torch.randn(N, F)
    y_pred = torch.randn(N, F)
    
    loss = setup_loss(loss_type, cross_corr)
    result = loss(y_true, y_pred)
    
    assert result.dim() == 0, "Loss should return scalar"
    assert torch.isfinite(result), "Loss should be finite"
    
@pytest.mark.parametrize("cross_corr", ["gene", "cell"])
@pytest.mark.parametrize("loss_type", ["GaussianLoss", "BalancedPearsonCorrelationLoss"])
@pytest.mark.parametrize("test_case", ["normal", "constant_cell", "zero_cell", "zero_gene"])
def test_perfect_prediction(cross_corr, loss_type, test_case):
    """Test loss when predictions are perfect."""
    y_true = get_test_case(test_case, 20, 8)["y_true"]
    y_pred = y_true.clone()  # Perfect prediction
    
    loss_fn = setup_loss(loss_type, cross_corr)
    result = loss_fn(y_true, y_pred)
    
    # Loss should be finite (the log term will still contribute)
    assert torch.isfinite(result), "Perfect prediction should give finite loss"
    
    if loss_type == "GaussianLoss":
        assert result == 0.0, "Perfect prediction should give zero loss"
    elif loss_type == "BalancedPearsonCorrelationLoss":
        assert result == 1.0, "Perfect prediction should give zero loss"
    

    
