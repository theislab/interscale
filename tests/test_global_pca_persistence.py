"""The PCA front-end of a GlobalModel must survive a checkpoint round trip.

A `GlobalModel` has no local component, so `type_gex_embedding="PCA"` supplies the transformer's
input: a `sklearn` PCA is fitted to the first batch and every later batch is projected through it.
`BaseModel.save` persists only `module.state_dict()`, and a `sklearn` estimator is not part of
one -- so before `pca_mean_`/`pca_components_`/`pca_fitted_` were registered as buffers, a
reloaded model arrived unfitted and refit the PCA on the first *evaluation* batch. The transformer
was trained on the basis fitted to the first training batch; a basis refitted on other data
differs by rotation and by component sign, so the reloaded model decoded a different space than it
was trained on, silently and without an error.

That matters for any analysis that loads a global-only checkpoint back -- the component ablation
reads attention and CLS tokens out of exactly such a reload.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from interscale.module.global_modules import TransformerNodeEncoderHook  # noqa: E402

N_INPUT, N_EMBED = 120, 16


def build_module():
    return TransformerNodeEncoderHook(
        max_seq_len=64,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=32,
        long_range_attention=False,
        n_input=N_INPUT,
        n_output=N_INPUT,
        n_embed=N_EMBED,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[32],
        mask_percentage=0.1,
        mask_strategy="node",
        type_gex_embedding="PCA",
    )


@pytest.fixture
def batches():
    rng = np.random.default_rng(0)
    return (
        rng.normal(size=(200, N_INPUT)).astype(np.float32),  # "training" batch, fits the PCA
        rng.normal(size=(90, N_INPUT)).astype(np.float32),  # "evaluation" batch
    )


def test_fit_is_recorded_in_the_state_dict(batches):
    train_batch, _ = batches
    module = build_module()
    assert not bool(module.pca_fitted_)

    module.create_gex_embedding(train_batch, type="PCA")

    assert bool(module.pca_fitted_)
    state = module.state_dict()
    assert {"pca_mean_", "pca_components_", "pca_fitted_"} <= set(state)
    assert state["pca_components_"].shape == (N_EMBED, N_INPUT)


def test_reload_projects_through_the_trained_basis(batches):
    """The whole point: a reloaded module must not refit on the batch it is handed."""
    train_batch, eval_batch = batches
    module = build_module()
    module.create_gex_embedding(train_batch, type="PCA")
    reference = module.create_gex_embedding(eval_batch, type="PCA")

    reloaded = build_module()
    _, unexpected = reloaded.load_state_dict(module.state_dict(), strict=False)
    assert not unexpected

    np.testing.assert_allclose(reloaded.create_gex_embedding(eval_batch, type="PCA"), reference, atol=1e-6)


def test_projection_matches_sklearn(batches):
    """Equivalent to `PCA.transform` for whiten=False, which is what this module constructs.

    Compared against the module's own fitted estimator rather than a freshly fitted one: at
    n_components well below min(n_samples, n_features), sklearn's `svd_solver="auto"` selects the
    randomized solver, whose `random_state` is None, so two fits of the same matrix do not give
    the same basis.
    """
    train_batch, eval_batch = batches
    module = build_module()
    module.create_gex_embedding(train_batch, type="PCA")

    np.testing.assert_allclose(
        module.create_gex_embedding(eval_batch, type="PCA"),
        module.pca.transform(eval_batch),
        atol=1e-5,
    )


def test_checkpoint_without_pca_buffers_still_loads(batches):
    """Backwards compatibility: checkpoints written before the buffers existed have no entry for
    them. `BaseModel.load` uses strict=False, so those keep the old refit-on-load behaviour rather
    than failing to load at all.
    """
    train_batch, eval_batch = batches
    module = build_module()
    module.create_gex_embedding(train_batch, type="PCA")
    reference = module.create_gex_embedding(eval_batch, type="PCA")

    legacy_state = {k: v for k, v in module.state_dict().items() if not k.startswith("pca_")}
    legacy = build_module()
    legacy.load_state_dict(legacy_state, strict=False)

    assert not bool(legacy.pca_fitted_)
    refit = legacy.create_gex_embedding(eval_batch, type="PCA")
    assert not np.allclose(refit, reference, atol=1e-6)
