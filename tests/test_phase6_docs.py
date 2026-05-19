"""Completeness checks for the Phase 6 prose deliverables and notebook exports.

DD-TEST-07: walkthrough export exists and is non-empty.
DD-WT-02: the walkthrough has the six required top-level sections in order.

Filled out incrementally across plan-phases 5–7; tests for plan-phases 6/7
land alongside their respective deliverables.
"""

from __future__ import annotations

import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "Docs"

WALKTHROUGH_PATH = DOCS_DIR / "Phase6-Walkthrough.md"
WALKTHROUGH_NOTEBOOK = REPO_ROOT / "notebooks" / "phase6_walkthrough.ipynb"

# DD-WT-02: the six top-level sections required by Spec-Phase6 §3. These are
# the exact ``## N. ...`` headers committed to the notebook; the regex is
# loose on trailing punctuation so light prose edits to a header don't break
# the test.
WALKTHROUGH_SECTION_PATTERNS = (
    r"^## 1\. Game selection and rationale\b",
    r"^## 2\. Phase 1 — raw box-score row \+ Madden join\b",
    r"^## 3\. Phase 2 — feature encoding\b",
    r"^## 4\. Phase 3 — split assignment\b",
    r"^## 5\. Phase 4 — predictions per learned combination\b",
    r"^## 6\. Phase 5 — locate the game on plots and breakdowns\b",
)


def test_walkthrough_export_exists() -> None:
    """DD-TEST-07: the markdown export is present and non-trivial."""
    assert WALKTHROUGH_PATH.exists(), f"{WALKTHROUGH_PATH} missing"
    content = WALKTHROUGH_PATH.read_text(encoding="utf-8")
    assert len(content) > 500, (
        f"{WALKTHROUGH_PATH} is too short ({len(content)} chars); "
        f"likely the nbconvert export ran on an empty notebook"
    )
    # Sanity-check that the source notebook is also present — the export is
    # generated, but a stale export without its source is a real risk.
    assert WALKTHROUGH_NOTEBOOK.exists(), f"{WALKTHROUGH_NOTEBOOK} missing"


def test_walkthrough_has_required_sections() -> None:
    """DD-WT-02: the six numbered top-level sections appear in order."""
    content = WALKTHROUGH_PATH.read_text(encoding="utf-8")
    last_pos = -1
    for pattern in WALKTHROUGH_SECTION_PATTERNS:
        match = re.search(pattern, content, flags=re.MULTILINE)
        assert match is not None, (
            f"{WALKTHROUGH_PATH} missing required section header matching "
            f"pattern {pattern!r}"
        )
        assert match.start() > last_pos, (
            f"section {pattern!r} appears before the previous section "
            f"(positions out of order)"
        )
        last_pos = match.start()
