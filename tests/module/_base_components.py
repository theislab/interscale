import scanpy as sc
from pathlib import Path
import pytest
import torch
from InterScale.nn import LinearLSEDecoder, LinearDecoder, NonLinearDecoder

HERE: Path = Path(__file__).parent

_adata = sc.read_h5ad(HERE / "_data" / "test_data.h5ad")
_adata.raw = _adata.copy()

@pytest.mark.parametrize("n_input", [10])
@pytest.mark.parametrize("n_output", [10])
@pytest.mark.parametrize("decoder_type", ["linear-lse", "linear", "nonlinear"])


class TestBaseComponents:
    
    def _latent_space(self, n_input, n_output):
        return torch.randn(n_input, n_output)
    
    def test_linear_lse_decoder(self, n_input, n_output):
        latent_space = self._latent_space(n_input, n_output)
        decoder = LinearLSEDecoder(n_input = n_input, n_output = n_output)
        assert decoder.weight.shape == (n_output, n_input)
        assert decoder.bias.shape == (n_output,)
        output = decoder(latent_space)
        assert output.shape == (n_input, n_output)

    def test_linear_decoder(self, n_input, n_output):
        latent_space = self._latent_space(n_input, n_output)
        decoder = LinearDecoder(n_input = n_input, n_output = n_output)
        assert decoder.weight.shape == (n_output, n_input)
        assert decoder.bias.shape == (n_output,)
        output = decoder(latent_space)
        assert output.shape == (n_input, n_output)

    def test_nonlinear_decoder(self, n_input, n_output):
        latent_space = self._latent_space(n_input, n_output)
        decoder = NonLinearDecoder(n_input = n_input, n_output = n_output)
        assert decoder.weight.shape == (n_output, n_input)
        assert decoder.bias.shape == (n_output,)
        output = decoder(latent_space)
        assert output.shape == (n_input, n_output)
    
