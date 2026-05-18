"""Feature build pipeline (§5)."""

from __future__ import annotations

import json
import pathlib

from nflpredictor.databuild.manifest import compute_sha256


PHASE1_MADDEN_BASENAME = "madden_2024.csv"
PHASE1_BOX_SCORES_BASENAME = "box_scores_2024.csv"
PHASE1_MANIFEST_BASENAME = "build_manifest.json"


class Phase1OutputMismatchError(ValueError):
    """Raised when an on-disk Phase 1 output diverges from the manifest hash (FE-IN-04)."""


def verify_phase1_outputs(processed_dir: pathlib.Path) -> dict:
    """Verify Phase 1 outputs against ``build_manifest.json`` (FE-IN-04).

    Returns the parsed manifest dict so the caller can propagate provenance into
    Phase 2's own manifest.
    """
    manifest_path = processed_dir / PHASE1_MANIFEST_BASENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Phase 1 manifest not found at {manifest_path}; "
            "run `python -m nflpredictor.databuild` first"
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected = manifest.get("output_sha256")
    if not isinstance(expected, dict):
        raise Phase1OutputMismatchError(
            "build_manifest.json is missing the 'output_sha256' map"
        )

    for basename in (PHASE1_MADDEN_BASENAME, PHASE1_BOX_SCORES_BASENAME):
        on_disk = processed_dir / basename
        if not on_disk.exists():
            raise FileNotFoundError(
                f"Phase 1 output {on_disk} not found; "
                "run `python -m nflpredictor.databuild` first"
            )
        # Phase 1 records output keys as repo-relative paths.
        manifest_key = f"Data/processed/{basename}"
        if manifest_key not in expected:
            raise Phase1OutputMismatchError(
                f"build_manifest.json output_sha256 does not record {manifest_key}"
            )
        actual = compute_sha256(on_disk)
        if actual != expected[manifest_key]:
            raise Phase1OutputMismatchError(
                f"Phase 1 output {manifest_key} hash mismatch: "
                f"manifest={expected[manifest_key]!r}, disk={actual!r}. "
                "Re-run `python -m nflpredictor.databuild` to regenerate."
            )
    return manifest
