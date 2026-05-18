import pathlib
import sys

from .pipeline import run_split_build


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    raw_dir = repo_root / "Data" / "raw"
    processed_dir = repo_root / "Data" / "processed"
    try:
        run_split_build(raw_dir, processed_dir, repo_dir=repo_root)
    except Exception as exc:  # noqa: BLE001 — surface any failure (SP-NF-05)
        print(f"split build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
