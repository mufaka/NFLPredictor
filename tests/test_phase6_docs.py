"""Completeness checks for the Phase 6 prose deliverables and notebook exports.

DD-TEST-07: walkthrough export exists and is non-empty.
DD-WT-02: the walkthrough has the six required top-level sections in order.
DD-TEST-08: the reading-outputs guide has one entry per required artifact.

Filled out incrementally across plan-phases 5–7; the training-dynamics tests
land alongside their deliverable in plan-phase 7.
"""

from __future__ import annotations

import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "Docs"

WALKTHROUGH_PATH = DOCS_DIR / "Phase6-Walkthrough.md"
WALKTHROUGH_NOTEBOOK = REPO_ROOT / "notebooks" / "phase6_walkthrough.ipynb"
READING_OUTPUTS_PATH = DOCS_DIR / "Phase6-ReadingTheOutputs.md"

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


# DD-RG-02: the 10 required artifact entries in the Reading-the-Outputs guide.
# These are the exact ``### N. ...`` headers committed to the doc. Regex is
# loose on the angle-bracket-encoded placeholders inside the plot filenames so
# the test stays robust to "<...>" vs "&lt;...&gt;" HTML escaping.
READING_OUTPUTS_ARTIFACT_PATTERNS = (
    r"^### 1\. metrics_headline\.json\b",
    r"^### 2\. breakdowns/by_team\.parquet\b",
    r"^### 3\. breakdowns/by_week\.parquet\b",
    r"^### 4\. breakdowns/by_home_away\.parquet\b",
    r"^### 5\. breakdowns/by_surface\.parquet\b",
    r"^### 6\. breakdowns/by_roof\.parquet\b",
    r"^### 7\. plots/.+scatter\.png\b",
    r"^### 8\. plots/.+residuals\.png\b",
    r"^### 9\. plots/.+by_week\.png\b",
    r"^### 10\. plots/ladder_summary__\{val,test,pooled\}\.png\b",
)


def test_reading_outputs_has_all_entries() -> None:
    """DD-TEST-08: the 10 artifact section headers appear in order."""
    assert READING_OUTPUTS_PATH.exists(), f"{READING_OUTPUTS_PATH} missing"
    content = READING_OUTPUTS_PATH.read_text(encoding="utf-8")
    last_pos = -1
    for pattern in READING_OUTPUTS_ARTIFACT_PATTERNS:
        match = re.search(pattern, content, flags=re.MULTILINE)
        assert match is not None, (
            f"{READING_OUTPUTS_PATH} missing required artifact entry "
            f"matching pattern {pattern!r}"
        )
        assert match.start() > last_pos, (
            f"artifact entry {pattern!r} appears before the previous entry "
            f"(positions out of order)"
        )
        last_pos = match.start()


def test_reading_outputs_has_required_sections() -> None:
    """DD-RG-01 / DD-RG-05 / DD-RG-06: intro, cross-combinations, vocabulary."""
    content = READING_OUTPUTS_PATH.read_text(encoding="utf-8")
    assert re.search(r"^## How to use this document\b", content, flags=re.MULTILINE), (
        "Reading-outputs guide missing 'How to use this document' section (DD-RG-01)"
    )
    assert re.search(r"^## How to read across combinations\b", content, flags=re.MULTILINE), (
        "Reading-outputs guide missing 'How to read across combinations' section (DD-RG-05)"
    )
    assert re.search(r"^## Vocabulary\b", content, flags=re.MULTILINE), (
        "Reading-outputs guide missing 'Vocabulary' appendix (DD-RG-06)"
    )
