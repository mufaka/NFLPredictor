def test_splits_package_imports():
    import nflpredictor.splits  # noqa: F401


def test_splits_main_module_imports():
    from nflpredictor.splits import __main__  # noqa: F401
