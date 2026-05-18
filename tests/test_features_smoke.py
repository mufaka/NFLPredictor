def test_features_package_imports():
    import nflpredictor.features  # noqa: F401


def test_features_main_module_imports():
    from nflpredictor.features import __main__  # noqa: F401
