from __future__ import annotations

from copy import deepcopy

from meddeid_data.corpus_quality import (
    CorpusDiversityContract,
    audit_corpus_diversity,
)


def _record(
    document_id: str,
    profile: str,
    family: str,
    name: str,
    rendered_date: str,
    mrn: str,
    style: str,
    hard_negative: str,
) -> dict:
    text = (
        f"Patient: {name}\nDate: {rendered_date}\nMRN: {mrn}\n"
        f"{document_id} independently records a distinct follow-up observation."
    )
    spans = []
    for value, label, source_slot in (
        (name, "Name:Patient", "patient.name"),
        (rendered_date, "Date", "encounter.date"),
        (mrn, "ID:Patient", "patient.mrn"),
    ):
        begin = text.index(value)
        spans.append(
            {
                "begin": begin,
                "end": begin + len(value),
                "text": value,
                "label": label,
                "source_slot": source_slot,
            }
        )
    return {
        "document_id": document_id,
        "text": text,
        "spans": spans,
        "metadata": {
            "lang": profile,
            "document_type": family,
            "document_creation_date": "2025-05-01",
            "patient": {"birth_date": "2017-04-01"},
            "style_profile": {"name": style},
            "hard_negative_targets": [{"category": hard_negative, "value": "08:30"}],
        },
    }


def _contract() -> CorpusDiversityContract:
    return CorpusDiversityContract(
        expected_documents=4,
        profiles=("en-GB", "en-US"),
        document_families=("clinic_note", "laboratory_report"),
        allowed_labels=("Name:Patient", "Date", "ID:Patient"),
        forbidden_labels=("Anonymize_Other",),
        required_hard_negative_categories=("time", "measurement"),
        require_balanced_profile_family_cells=True,
        min_styles_per_profile_family=1,
        near_duplicate_distance=None,
        reference_year=2025,
        min_date_values_per_period=1,
        pediatric_minimum_by_profile={"en-GB": 1, "en-US": 1},
    )


def test_shared_diversity_contract_covers_profiles_formats_semantics_and_dates() -> None:
    records = [
        _record("gb-clinic", "en-GB", "clinic_note", "Ada North", "3 April 1998", "GB-101", "compact", "time"),
        _record("gb-lab", "en-GB", "laboratory_report", "Ben West", "2020-04-04", "GB-102", "fields", "measurement"),
        _record("us-clinic", "en-US", "clinic_note", "Cara East", "April 5, 2031", "US-201", "compact", "time"),
        _record("us-lab", "en-US", "laboratory_report", "Dion South", "04/06/2024", "US-202", "fields", "measurement"),
    ]
    cases = [
        {"case_id": "gb", "language": "en-GB", "age_years": 8, "age_unit": "years"},
        {"case_id": "us", "language": "en-US", "age_years": 7, "age_unit": "years"},
    ]

    report = audit_corpus_diversity(records, contract=_contract(), case_records=cases)

    assert report["passed"] is True
    assert report["counts"]["date_periods"] == {
        "contemporary": 2,
        "future_shifted": 1,
        "historical": 1,
    }
    assert report["counts"]["semantic_types"]["patient_mrn"] == 4
    assert report["counts"]["documentation_shapes"]["field_rows"] == 4
    assert set(report["formatting"]["styles_by_profile_document_family"]) == {
        "en-GB:clinic_note",
        "en-GB:laboratory_report",
        "en-US:clinic_note",
        "en-US:laboratory_report",
    }


def test_shared_diversity_contract_blocks_duplicates_and_forbidden_labels() -> None:
    first = _record(
        "first", "en-GB", "clinic_note", "Ada North", "3 April 1998", "GB-101", "compact", "time"
    )
    first["spans"][0]["label"] = "Anonymize_Other"
    duplicate = deepcopy(first)
    duplicate["document_id"] = "duplicate"

    report = audit_corpus_diversity(
        [first, duplicate],
        contract=CorpusDiversityContract(
            expected_documents=2,
            profiles=("en-GB",),
            document_families=("clinic_note",),
            allowed_labels=("Name:Patient", "Date", "ID:Patient"),
            forbidden_labels=("Anonymize_Other",),
            near_duplicate_distance=3,
        ),
    )

    assert report["passed"] is False
    assert any("forbidden generated labels" in failure for failure in report["failures"])
    assert any("exact duplicate" in failure for failure in report["failures"])
    assert any("near-duplicate" in failure for failure in report["failures"])
