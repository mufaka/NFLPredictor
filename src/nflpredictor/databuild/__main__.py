"""Entry point for `python -m nflpredictor.databuild`."""

from __future__ import annotations

import pathlib
import sys
import traceback

from .pipeline import run_build


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    raw_dir = repo_root / "Data" / "raw"
    processed_dir = repo_root / "Data" / "processed"
    try:
        run_build(raw_dir, processed_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
