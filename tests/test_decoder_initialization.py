import pytest

from InterScale.model import LocalModel, GlobalModel, CombinedModel
from tests._model_test_utils import create_minimal_adata, sample_config



def has_decoder(module):
    """Check if a module has a decoder."""
    return hasattr(module, 'decoder') and module.decoder is not None


def test_local_model_decoder():
    """Test that LocalModel only has a decoder in the local module."""
    adata = create_minimal_adata()
    cfg = sample_config(local_component_name="GCN")
    
    # Setup anndata
    LocalModel._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
        group_key=cfg.dataset.group_label
    )
    
    # Create model
    model = LocalModel(adata, cfg)
    print(list(model.module.state_dict().keys()))
    
    # Check decoders
    assert has_decoder(model.module), "LocalModel.module should have a decoder"
    assert model.module.decoder is not None, "LocalModel.module.decoder should not be None"
    
    # Verify it's the local module (not a container)
    assert not hasattr(model.module, 'local_module'), "LocalModel should not have a local_module submodule"
    assert not hasattr(model.module, 'global_module'), "LocalModel should not have a global_module submodule"


def test_global_model_decoder():
    """Test that GlobalModel only has a decoder in the global module."""
    adata = create_minimal_adata()
    cfg = sample_config(global_component_name="self-attn-transformer")
    
    # Setup anndata
    GlobalModel._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
        group_key=cfg.dataset.group_label
    )
    
    # Create model
    model = GlobalModel(adata, cfg)
    print(list(model.module.state_dict().keys()))
    
    # Check decoders
    assert has_decoder(model.module), "GlobalModel.module should have a decoder"
    assert model.module.decoder is not None, "GlobalModel.module.decoder should not be None"
    
    # Verify it's the global module (not a container)
    assert not hasattr(model.module, 'local_module'), "GlobalModel should not have a local_module submodule"
    assert not hasattr(model.module, 'global_module'), "GlobalModel should not have a global_module submodule"


def test_combined_model_single_decoder():
    """Test that CombinedModel (dual_decoder=False) only has a decoder in the global module."""
    adata = create_minimal_adata()
    cfg = sample_config(local_component_name="GCN", global_component_name="self-attn-transformer", dual_decoder = False)
    
    # Setup anndata
    CombinedModel._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
        group_key=cfg.dataset.group_label
    )
    
    # Create model
    model = CombinedModel(adata, cfg)
    print(list(model.module.state_dict().keys()))
    
    # Check that container module does NOT have a decoder
    assert not has_decoder(model.module), "CombinedModel.module (container) should NOT have a decoder"
    assert model.module.decoder_type is None, "CombinedModel.module.decoder_type should be None"
    
    # Check that local module does NOT have a decoder
    assert not has_decoder(model.module.local_module), "CombinedModel.module.local_module should NOT have a decoder"
    assert model.module.local_module.decoder_type is None, "CombinedModel.module.local_module.decoder_type should be None"
    
    # Check that global module DOES have a decoder
    assert has_decoder(model.module.global_module), "CombinedModel.module.global_module should have a decoder"
    assert model.module.global_module.decoder is not None, "CombinedModel.module.global_module.decoder should not be None"
    assert model.module.global_module.decoder_type == cfg.model.decoder.type, "CombinedModel.module.global_module.decoder_type should match config"


def test_combined_model_dual_decoder():
    """Test that CombinedModel (dual_decoder=True) has decoders in both local and global modules."""
    adata = create_minimal_adata()
    cfg = sample_config(local_component_name="GCN", global_component_name="self-attn-transformer", dual_decoder=True)
    
    # Setup anndata
    CombinedModel._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
        group_key=cfg.dataset.group_label
    )
    
    # Create model
    model = CombinedModel(adata, cfg)
    print(list(model.module.state_dict().keys()))
    
    # Check that container module does NOT have a decoder
    assert not has_decoder(model.module), "CombinedModel.module (container) should NOT have a decoder"
    assert model.module.decoder_type is None, "CombinedModel.module.decoder_type should be None"
    
    # Check that local module DOES have a decoder
    assert has_decoder(model.module.local_module), "CombinedModel.module.local_module should have a decoder"
    assert model.module.local_module.decoder is not None, "CombinedModel.module.local_module.decoder should not be None"
    assert model.module.local_module.decoder_type == cfg.model.decoder.type, "CombinedModel.module.local_module.decoder_type should match config"
    
    # Check that global module DOES have a decoder
    assert has_decoder(model.module.global_module), "CombinedModel.module.global_module should have a decoder"
    assert model.module.global_module.decoder is not None, "CombinedModel.module.global_module.decoder should not be None"
    assert model.module.global_module.decoder_type == cfg.model.decoder.type, "CombinedModel.module.global_module.decoder_type should match config"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
