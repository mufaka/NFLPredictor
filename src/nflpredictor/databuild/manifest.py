"""Build manifest construction (§3.11)."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import subprocess
from typing import Optional

from .normalization import NORMALIZATION_VERSION


def compute_sha256(path: pathlib.Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_get_git_commit(repo_dir: pathlib.Path) -> Optional[str]:
    """Return ``HEAD``'s commit hash, or ``None`` outside a git repo.

    Swallows the common "no git here" failure modes across OSes:
    ``FileNotFoundError`` (git not on PATH, or ``cwd`` missing on POSIX),
    ``NotADirectoryError`` (``cwd`` missing on Windows), ``OSError`` (catch-all
    for other ``cwd`` problems), and ``CalledProcessError`` (e.g., ``repo_dir``
    exists but is not a git work tree).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError, OSError):
        return None
    return out.stdout.strip() or None


def utc_timestamp(now: Optional[_dt.datetime] = None) -> str:
    """Return an ISO 8601 UTC timestamp formatted as ``...Z`` (DB-MAN-03)."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    *,
    raw_inputs: dict[str, pathlib.Path],
    outputs: dict[str, pathlib.Path],
    counts: dict,
    repo_dir: pathlib.Path,
    now: Optional[_dt.datetime] = None,
) -> dict:
    """Assemble the manifest dict per DB-MAN-01/DB-MAN-02.

    ``counts`` is the nested ``{"total": {...}, "by_season": {...}}`` object.
    """
    return {
        "build_timestamp_utc": utc_timestamp(now),
        "normalization_version": NORMALIZATION_VERSION,
        "source_sha256": {
            name: compute_sha256(path) for name, path in raw_inputs.items()
        },
        "output_sha256": {
            name: compute_sha256(path) for name, path in outputs.items()
        },
        "git_commit": try_get_git_commit(repo_dir),
        "counts": counts,
    }


def write_manifest(manifest_dict: dict, path: pathlib.Path) -> None:
    """Write the manifest with sorted keys + trailing newline (DB-MAN-04).

    ``newline="\\n"`` disables Windows' text-mode CRLF translation so the
    manifest is byte-identical across OSes — required because downstream
    phases SHA-pin against these bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_dict, f, sort_keys=True, indent=2)
        f.write("\n")
