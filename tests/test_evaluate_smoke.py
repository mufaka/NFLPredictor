"""EV-TEST-10 prereq: confirm the evaluate package imports cleanly."""

from __future__ import annotations


def test_import_evaluate_package() -> None:
    import nflpredictor.evaluate  # noqa: F401
    import nflpredictor.evaluate.__main__  # noqa: F401
    import nflpredictor.evaluate.config  # noqa: F401
    import nflpredictor.evaluate.sources  # noqa: F401
