def test_import():
    import interscale

    assert hasattr(interscale, "__version__")
