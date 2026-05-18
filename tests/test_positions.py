import pytest

from nflpredictor.databuild.positions import POSITION_EQUIVALENCES, positions_compatible


@pytest.mark.parametrize(
    ("box", "madden"),
    [
        ("QB", "QB"),
        ("WR", "WR"),
        ("TE", "TE"),
        ("OL", "C"),
        ("OL", "LT"),
        ("OL", "RG"),
        ("T", "LT"),
        ("OT", "RT"),
        ("G", "LG"),
        ("OG", "RG"),
        ("DL", "DT"),
        ("DL", "RE"),
        ("DE", "LE"),
        ("NT", "DT"),
        ("DB", "CB"),
        ("DB", "FS"),
        ("S", "SS"),
        ("LB", "MLB"),
        ("LB", "LOLB"),
        ("OLB", "ROLB"),
        ("RB", "HB"),
        ("RB", "FB"),
    ],
)
def test_compatible_pairs(box: str, madden: str):
    assert positions_compatible(box, madden)


@pytest.mark.parametrize(
    ("box", "madden"),
    [
        ("QB", "WR"),
        ("OL", "CB"),
        ("DL", "MLB"),
        ("RB", "QB"),
        ("LB", "FS"),
        ("WR", "TE"),  # both real positions, neither in the equivalence table; not equal -> incompatible
    ],
)
def test_incompatible_pairs(box: str, madden: str):
    assert not positions_compatible(box, madden)


def test_table_does_not_include_redundant_equal_entries():
    """An equivalence set should never list its own key value — equality handles that."""
    for box, madden_set in POSITION_EQUIVALENCES.items():
        assert box not in madden_set, f"{box} listed itself as an equivalent"
