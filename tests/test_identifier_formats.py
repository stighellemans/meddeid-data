from __future__ import annotations

from meddeid_data.identifier_formats import (
    format_english_identifier,
    synthetic_national_identifier,
)
from meddeid_language_en import is_approved_synthetic_identifier, is_valid_identifier


def test_identifier_formats_are_mostly_numeric_and_structurally_diverse() -> None:
    for profile_id in ("en-GB", "en-US"):
        values = [
            format_english_identifier(profile_id, slot, str(index))
            for index in range(200)
            for slot in (
                "patient.mrn",
                "patient.report_id",
                "caregiver.professional_id",
            )
        ]
        numeric_share = sum(value.isdigit() for value in values) / len(values)
        shapes = {
            "numeric" if value.isdigit() else "spaced" if " " in value else
            "slash" if "/" in value else "hyphen" if "-" in value else
            "compact"
            for value in values
        }

        assert numeric_share >= 0.65
        assert shapes == {"numeric", "spaced", "slash", "hyphen", "compact"}
        assert len(set(values)) >= 590


def test_identifier_formatting_is_deterministic() -> None:
    assert format_english_identifier("en-GB", "patient.mrn", "42") == format_english_identifier(
        "en-GB", "patient.mrn", "42"
    )


def test_national_identifiers_are_diverse_and_deliberately_non_assignable() -> None:
    for profile_id in ("en-GB", "en-US"):
        rows = [synthetic_national_identifier(profile_id, str(index)) for index in range(500)]
        assert len({row["value"] for row in rows}) >= 495
        assert all(
            is_approved_synthetic_identifier(row["value"], profile_id=profile_id)
            for row in rows
        )
        assert all(
            not is_valid_identifier(
                row["value"], profile_id=profile_id, hint=row["kind"]
            )
            for row in rows
        )
        shapes = {
            "numeric" if row["value"].isdigit() else
            "spaced" if " " in row["value"] else
            "hyphen" if "-" in row["value"] else
            "compact"
            for row in rows
        }
        assert ({"numeric", "spaced", "compact"} if profile_id == "en-GB" else
                {"numeric", "spaced", "hyphen"}) <= shapes

    assert {row["kind"] for row in (
        synthetic_national_identifier("en-GB", str(index)) for index in range(100)
    )} == {"nhs_number", "national_insurance_number"}
    assert {
        synthetic_national_identifier("en-US", str(index))["kind"]
        for index in range(100)
    } == {"ssn"}
