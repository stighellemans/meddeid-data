from __future__ import annotations

import json

import pytest

from meddeid_data.english_production import (
    _batch_paths,
    _exclusive_generation_lock,
    diversify_english_email_domains,
    diversify_english_identifier_formats,
    invalidate_document,
    inject_demographic_hard_negative,
    inject_national_patient_identifier,
    migrate_case_age,
    migrate_case_department,
    migrate_case_hospital,
    normalize_english_age_birthdate_boundaries,
    normalize_english_mrn_values,
    record_document_review,
    rebalance_healthcare_organization_case,
    sign_off_batch,
    validate_unversioned_english_generation_profiles,
    write_batch_reports,
)


def test_english_generation_profiles_are_unversioned_and_match_language() -> None:
    docs = [
        {"metadata": {"lang": "en-GB", "generation_profile": "en-GB"}},
        {"metadata": {"lang": "en-US", "generation_profile": "en-US"}},
    ]
    assert validate_unversioned_english_generation_profiles(docs) == {
        "en-GB": 1,
        "en-US": 1,
    }
    with pytest.raises(ValueError, match="must be unversioned"):
        validate_unversioned_english_generation_profiles(
            [{"metadata": {"lang": "en-GB", "generation_profile": "en-GB@1"}}]
        )


def test_email_domain_diversification_updates_metadata_and_later_offsets() -> None:
    text = "Email: alex.smith42@example.test; clinician Jane Smith"
    email = "alex.smith42@example.test"
    name = "Jane Smith"
    record = {
        "text": text,
        "spans": [
            {
                "begin": text.index(email),
                "end": text.index(email) + len(email),
                "text": email,
                "label": "Contactdetails",
            },
            {
                "begin": text.index(name),
                "end": text.index(name) + len(name),
                "text": name,
                "label": "Name:Caregiver",
            },
        ],
        "metadata": {
            "lang": "en-US",
            "required_pii_targets": [{"value": email, "label": "Contactdetails"}],
        },
    }

    assert diversify_english_email_domains(record) == 1
    assert "@example.test" not in record["text"]
    assert "@example.test" not in record["metadata"]["required_pii_targets"][0]["value"]
    assert record["metadata"]["email_domain_diversification"]["replacement_count"] == 1
    for span in record["spans"]:
        assert record["text"][span["begin"] : span["end"]] == span["text"]
    assert diversify_english_email_domains(record) == 0


def test_identifier_diversification_updates_metadata_and_later_offsets() -> None:
    identifier = "REPORT-US-123456"
    name = "Jane Smith"
    text = f"Accession: {identifier}; clinician {name}"
    record = {
        "document_id": "en-us-synthetic-00042",
        "text": text,
        "spans": [
            {
                "begin": text.index(identifier),
                "end": text.index(identifier) + len(identifier),
                "text": identifier,
                "label": "ID:Patient",
                "source_slot": "patient.report_id",
            },
            {
                "begin": text.index(name),
                "end": text.index(name) + len(name),
                "text": name,
                "label": "Name:Caregiver",
            },
        ],
        "metadata": {
            "lang": "en-US",
            "production": {"profile_index": 41},
            "required_pii_targets": [
                {
                    "slot": "patient.report_id",
                    "value": identifier,
                    "label": "ID:Patient",
                }
            ],
        },
    }

    assert diversify_english_identifier_formats(record) == 1
    assert identifier not in record["text"]
    assert record["metadata"]["required_pii_targets"][0]["value"] == record["spans"][0]["text"]
    assert record["metadata"]["identifier_format_diversification"]["replacement_count"] == 1
    for span in record["spans"]:
        assert record["text"][span["begin"] : span["end"]] == span["text"]
    assert diversify_english_identifier_formats(record) == 0


def test_national_identifier_injection_adds_distinct_patient_id_provenance() -> None:
    text = "Clinic note\nPatient Jane Smith was reviewed."
    name = "Jane Smith"
    record = {
        "document_id": "en-us-synthetic-00042",
        "text": text,
        "spans": [
            {
                "begin": text.index(name),
                "end": text.index(name) + len(name),
                "text": name,
                "label": "Name:Patient",
            }
        ],
        "metadata": {
            "lang": "en-US",
            "production": {"profile_index": 41},
            "required_pii_targets": [],
        },
    }

    assert inject_national_patient_identifier(record, prevalence_percent=100) == 1
    national = next(
        span for span in record["spans"]
        if span.get("source_slot") == "patient.national_id"
    )
    assert "SSN:" in record["text"]
    assert national["label"] == "ID:Patient"
    assert national["semantic_type"] == "patient_identifier.national"
    assert record["text"][national["begin"]:national["end"]] == national["text"]
    assert record["metadata"]["required_pii_targets"][-1]["slot"] == "patient.national_id"
    assert all(
        record["text"][span["begin"]:span["end"]] == span["text"]
        for span in record["spans"]
    )
    assert inject_national_patient_identifier(record, prevalence_percent=100) == 0


def test_healthcare_organization_case_rebalancing_preserves_offsets_and_metadata() -> None:
    organization = "GEORGE WASHINGTON UNIVERSITY HOSPITAL"
    caregiver = "Jane Smith"
    text = f"Organization: {organization}\nClinician: {caregiver}"
    record = {
        "document_id": "en-us-synthetic-00042",
        "text": text,
        "spans": [
            {
                "begin": text.index(organization),
                "end": text.index(organization) + len(organization),
                "text": organization,
                "label": "Organization:Healthcare",
                "source_slot": "healthcare.organization",
            },
            {
                "begin": text.index(caregiver),
                "end": text.index(caregiver) + len(caregiver),
                "text": caregiver,
                "label": "Name:Caregiver",
            },
        ],
        "metadata": {
            "lang": "en-US",
            "production": {"profile_index": 41},
            "required_pii_targets": [
                {
                    "slot": "healthcare.organization",
                    "value": organization,
                    "label": "Organization:Healthcare",
                }
            ],
        },
    }

    assert rebalance_healthcare_organization_case(record) == 1
    organization_span = record["spans"][0]
    assert organization_span["text"] == "George Washington University Hospital"
    assert record["metadata"]["required_pii_targets"][0]["value"] == organization_span["text"]
    assert all(
        record["text"][span["begin"]:span["end"]] == span["text"]
        for span in record["spans"]
    )
    assert rebalance_healthcare_organization_case(record) == 0


def test_hyphenated_age_boundary_excludes_old_and_demographics() -> None:
    text = "A 50-year-old African American woman and an 8 y/o child."
    record = {
        "text": text,
        "spans": [
            {
                "begin": 2,
                "end": 36,
                "text": "50-year-old African American woman",
                "label": "Age_Birthdate",
                "source_slot": "patient.age_hyphenated",
            },
            {
                "begin": 44,
                "end": 49,
                "text": "8 y/o",
                "label": "Age_Birthdate",
                "source_slot": "patient.age_abbreviated",
            },
        ],
    }

    assert normalize_english_age_birthdate_boundaries(record) == 1
    assert record["spans"][0]["text"] == "50-year"
    assert record["spans"][0]["end"] == 9
    assert record["spans"][1]["text"] == "8 y/o"


def test_contextual_age_boundary_excludes_age_cue() -> None:
    text = "The patient, aged 43, was reviewed; another is age 12 years."
    record = {
        "text": text,
        "spans": [
            {
                "begin": text.index("aged 43"),
                "end": text.index("aged 43") + len("aged 43"),
                "text": "aged 43",
                "label": "Age_Birthdate",
            },
            {
                "begin": text.index("age 12 years"),
                "end": text.index("age 12 years") + len("age 12 years"),
                "text": "age 12 years",
                "label": "Age_Birthdate",
            },
        ],
    }

    assert normalize_english_age_birthdate_boundaries(record) == 2
    assert [span["text"] for span in record["spans"]] == ["43", "12 years"]


def test_mrn_normalization_rewrites_text_and_shifts_later_spans() -> None:
    text = "MRN: MRN-US-538265; clinician Jane Smith"
    name_begin = text.index("Jane Smith")
    record = {
        "text": text,
        "spans": [
            {
                "begin": text.index("MRN-US-538265"),
                "end": text.index("MRN-US-538265") + len("MRN-US-538265"),
                "text": "MRN-US-538265",
                "label": "ID:Patient",
                "source_slot": "patient.mrn",
            },
            {
                "begin": name_begin,
                "end": name_begin + len("Jane Smith"),
                "text": "Jane Smith",
                "label": "Name:Caregiver",
            },
        ],
    }

    assert normalize_english_mrn_values(record) == 1
    assert record["text"] == "MRN: 538265; clinician Jane Smith"
    assert record["spans"][0]["text"] == "538265"
    for span in record["spans"]:
        assert record["text"][span["begin"] : span["end"]] == span["text"]


def test_demographic_injection_is_unlabeled_and_shifts_later_spans() -> None:
    text = "A 50-year-old patient was seen by Jane Smith."
    name_begin = text.index("Jane Smith")
    record = {
        "text": text,
        "spans": [
            {
                "begin": text.index("50-year"),
                "end": text.index("50-year") + len("50-year"),
                "text": "50-year",
                "label": "Age_Birthdate",
                "source_slot": "patient.age_hyphenated",
            },
            {
                "begin": name_begin,
                "end": name_begin + len("Jane Smith"),
                "text": "Jane Smith",
                "label": "Name:Caregiver",
            },
        ],
        "metadata": {},
    }

    assert inject_demographic_hard_negative(record, 0) == 1
    assert "50-year-old African American male patient" in record["text"]
    assert len(record["spans"]) == 2
    assert record["metadata"]["demographic_hard_negative"] == {
        "ethnicity": "African American",
        "sex": "male",
        "annotated": False,
    }
    for span in record["spans"]:
        assert record["text"][span["begin"] : span["end"]] == span["text"]
    assert inject_demographic_hard_negative(record, 0) == 0


def test_generation_lock_rejects_an_overlapping_batch_writer(tmp_path) -> None:
    with _exclusive_generation_lock(tmp_path, 3):
        with pytest.raises(RuntimeError, match="already has an active generation process"):
            with _exclusive_generation_lock(tmp_path, 3):
                pass

    with _exclusive_generation_lock(tmp_path, 3):
        pass


def test_batch_signoff_requires_a_pass_decision_for_every_document(tmp_path) -> None:
    paths = _batch_paths(tmp_path, 0)
    paths["directory"].mkdir(parents=True)
    paths["report_json"].write_text(
        json.dumps({"documents": 2, "gate_passed": True}), encoding="utf-8"
    )
    paths["docs"].write_text(
        '\n'.join(
            [
                json.dumps({"document_id": "doc-1"}),
                json.dumps({"document_id": "doc-2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["review"].write_text("complete review packet", encoding="utf-8")
    paths["review_decisions"].write_text(
        '\n'.join(
            [
                json.dumps({"document_id": "doc-1", "decision": "pending"}),
                json.dumps({"document_id": "doc-2", "decision": "pending"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="2 non-passing"):
        sign_off_batch(
            output_dir=tmp_path,
            batch_index=0,
            reviewer="Codex",
            decision="pass",
            notes="reviewed all",
        )

    for document_id in ("doc-1", "doc-2"):
        record_document_review(
            output_dir=tmp_path,
            batch_index=0,
            document_id=document_id,
            decision="pass",
            notes="direct review passed",
        )
    signoff = sign_off_batch(
        output_dir=tmp_path,
        batch_index=0,
        reviewer="Codex",
        decision="pass",
        notes="reviewed all",
    )

    assert signoff["decision"] == "pass"
    assert signoff["review_decisions_sha256"]


def test_report_refresh_adds_new_pending_documents_and_preserves_decisions(tmp_path) -> None:
    paths = _batch_paths(tmp_path, 0)
    paths["directory"].mkdir(parents=True)
    paths["review_decisions"].write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "decision": "pass",
                "notes": "already reviewed",
                "reviewed_at": "2026-08-20T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    docs = [
        {"document_id": "doc-1", "text": "First note", "spans": [], "metadata": {}},
        {"document_id": "doc-2", "text": "Second note", "spans": [], "metadata": {}},
    ]
    report = {
        "batch_index": 0,
        "gate_passed": True,
        "documents": 2,
        "expected_documents": 2,
        "word_counts": {"min": 2, "median": 2, "mean": 2, "max": 2},
        "usage": {"estimated_usd_at_2026_08_20_list_price": 0.0},
        "gate_failures": [],
        "counts": {},
        "diversity": {},
        "validation_failures": [],
    }

    write_batch_reports(paths=paths, docs=docs, report=report)
    decisions = [
        json.loads(line)
        for line in paths["review_decisions"].read_text(encoding="utf-8").splitlines()
    ]

    assert decisions[0]["decision"] == "pass"
    assert decisions[1]["document_id"] == "doc-2"
    assert decisions[1]["decision"] == "pending"
    assert decisions[1]["notes"] == ""
    assert decisions[1]["reviewed_at"] is None
    assert decisions[1]["contract_version"] == "meddeid.review-decision.v1"
    assert len(decisions[1]["document_sha256"]) == 64


def test_invalidate_and_migrate_age_preserve_recovery_history(tmp_path) -> None:
    paths = _batch_paths(tmp_path, 0)
    paths["directory"].mkdir(parents=True)
    case = {
        "case_id": "case-00001",
        "language": "en-GB",
        "age_years": 19,
        "birth_date": "2001-12-20",
        "encounter_date": "2021-06-10",
        "hospital": "EALING HOSPITAL CHILD HEALTH",
        "department": "paediatrics",
        "production": {},
    }
    document_id = "en-gb-synthetic-00001"
    paths["cases"].write_text(json.dumps(case) + "\n", encoding="utf-8")
    paths["docs"].write_text(
        json.dumps({"document_id": document_id, "text": "A note", "spans": [], "metadata": {}})
        + "\n",
        encoding="utf-8",
    )
    paths["marked"].write_text(
        json.dumps({"document_id": document_id, "marked_text": "A note"}) + "\n",
        encoding="utf-8",
    )
    paths["review_decisions"].write_text(
        json.dumps({"document_id": document_id, "decision": "fail"}) + "\n",
        encoding="utf-8",
    )

    invalidate_document(
        output_dir=tmp_path,
        batch_index=0,
        document_id=document_id,
        reason="COPD age is implausible",
    )
    migration = migrate_case_age(
        output_dir=tmp_path,
        batch_index=0,
        document_id=document_id,
        age_years=55,
        reason="Use an older adult COPD case",
    )
    hospital_migration = migrate_case_hospital(
        output_dir=tmp_path,
        batch_index=0,
        document_id=document_id,
        hospital="EALING HOSPITAL",
        reason="Use an adult-compatible facility",
    )
    department_migration = migrate_case_department(
        output_dir=tmp_path,
        batch_index=0,
        document_id=document_id,
        department="general medicine",
        reason="Use an adult-compatible department",
    )

    assert paths["docs"].read_text(encoding="utf-8") == ""
    assert paths["marked"].read_text(encoding="utf-8") == ""
    assert paths["review_decisions"].read_text(encoding="utf-8") == ""
    assert len(paths["invalidated"].read_text(encoding="utf-8").splitlines()) == 1
    repaired = json.loads(paths["cases"].read_text(encoding="utf-8"))
    assert repaired["age_years"] == 55
    assert repaired["birth_date"] == "1965-12-20"
    assert repaired["hospital"] == "EALING HOSPITAL"
    assert repaired["department"] == "general medicine"
    assert migration["from"]["age_years"] == 19
    assert hospital_migration["from"] == "EALING HOSPITAL CHILD HEALTH"
    assert department_migration["from"] == "paediatrics"
