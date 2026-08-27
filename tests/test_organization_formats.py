from __future__ import annotations

from meddeid_data.organization_formats import (
    format_healthcare_organization,
    natural_organization_case,
)


def test_natural_organization_case_preserves_small_words_and_acronyms() -> None:
    assert (
        natural_organization_case("KING'S DAUGHTERS' MEDICAL CENTER")
        == "King's Daughters' Medical Center"
    )
    assert (
        natural_organization_case("UCLA HOSPITAL OF THE WEST")
        == "UCLA Hospital of the West"
    )
    assert natural_organization_case("LBJ TROPICAL MEDICAL CENTER") == (
        "LBJ Tropical Medical Center"
    )
    assert natural_organization_case("MCDERMOTT MEDICAL CENTRE") == (
        "McDermott Medical Centre"
    )
    assert natural_organization_case("HENRY FORD II HOSPITAL") == (
        "Henry Ford II Hospital"
    )


def test_healthcare_organization_case_is_balanced_and_deterministic() -> None:
    for profile_id in ("en-GB", "en-US"):
        values = [
            format_healthcare_organization(
                profile_id,
                "GEORGE WASHINGTON UNIVERSITY HOSPITAL",
                str(index),
            )
            for index in range(1_000)
        ]
        uppercase_share = sum(value.isupper() for value in values) / len(values)
        assert 0.25 <= uppercase_share <= 0.35
        assert any(value == "George Washington University Hospital" for value in values)
        assert format_healthcare_organization(
            profile_id, "GEORGE WASHINGTON UNIVERSITY HOSPITAL", "42"
        ) == format_healthcare_organization(
            profile_id, "GEORGE WASHINGTON UNIVERSITY HOSPITAL", "42"
        )
