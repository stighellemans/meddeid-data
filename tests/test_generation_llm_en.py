from __future__ import annotations

import json

import pytest
from meddeid_core import BERT_ENTITY_LABELS, validate_record
from meddeid_language_en import parse_date_text, pseudonymize_date_text

from meddeid_data.english_production import build_batch_cases
from meddeid_data.generation_llm_en import (
    EnglishLLMRenderError,
    build_english_llm_prompt,
    coverage_targets,
    hard_negative_targets,
    marked_text_to_english_doc,
    pii_slots,
    style_profile,
    validate_english_llm_document,
)


def _valid_marked_text(case: dict) -> str:
    targets = coverage_targets(case)
    pii = "\n".join(
        f"Field {index}: [[PII|{target['label']}|{target['slot']}]]."
        for index, target in enumerate(targets, start=1)
    )
    hard_negatives = " ".join(
        target["value"] + "." for target in hard_negative_targets(case)
    )
    clinical = " ".join(
        [
            "The clinical review considered symptoms, examination findings, investigations, "
            "medication response, functional status, risk, safety-netting, and follow-up."
        ]
        * 12
    )
    return f"Clinical note\n{pii}\n{hard_negatives}\n{clinical}"


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_english_luna_marker_contract_round_trips_exact_offsets(profile_id: str) -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=0, batch_size=2, seed=17)
        if row["language"] == profile_id
    )
    marked = _valid_marked_text(case)
    doc = marked_text_to_english_doc(
        record=case,
        marked_text=marked,
        model="gpt-5.6-luna",
    )

    assert validate_record(doc, strict_taxonomy=True) == []
    assert validate_english_llm_document(record=case, marked_text=marked, doc=doc) == []
    assert all(doc["text"][span["begin"] : span["end"]] == span["text"] for span in doc["spans"])
    assert all(span["label"] != "Anonymize_Other" for span in doc["spans"])


def test_luna_prompt_declares_exact_14_label_allowlist_and_independent_authorship() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=19)[0]
    prompt = json.loads(build_english_llm_prompt(case))
    assert prompt["allowed_labels"] == list(BERT_ENTITY_LABELS)
    assert "Anonymize_Other" not in prompt["allowed_labels"]
    assert "fresh composition" in prompt["task"]
    assert len(prompt["hard_negative_targets"]) == 1
    assert "structural_contract" in prompt["style_profile"] or prompt["style_profile"]["name"] in {
        "narrative-scribe",
        "specialist-report",
        "longitudinal-summary",
        "patient-facing-letter",
    }
    assert any(
        "corpus is for PII detection" in instruction
        for instruction in prompt["output_contract"]
    )
    assert [target["placeholder"] for target in prompt["required_pii_targets"]] == [
        f"[[PII_{index:02d}]]"
        for index in range(1, len(prompt["required_pii_targets"]) + 1)
    ]
    assert all(target["semantic_type"] for target in prompt["required_pii_targets"])


def test_national_patient_identifier_is_available_to_future_llm_generation() -> None:
    cases = build_batch_cases(batch_index=0, batch_size=500, seed=20260820)
    assert all(any(slot["slot"] == "patient.national_id" for slot in pii_slots(case)) for case in cases)
    selected = [
        target
        for case in cases
        for target in coverage_targets(case)
        if target["slot"] == "patient.national_id"
    ]
    assert selected
    assert {target["semantic_type"] for target in selected} == {
        "patient_identifier.national"
    }


def test_hyphenated_age_prompt_forbids_standalone_age_field() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=8, batch_size=500, seed=20260820)
        if any(target["slot"] == "patient.age_hyphenated" for target in coverage_targets(row))
    )
    prompt = json.loads(build_english_llm_prompt(case))
    target = next(
        row
        for row in prompt["required_pii_targets"]
        if row["slot"] == "patient.age_hyphenated"
    )

    assert "Never use the age phrase itself as the value of a Patient field" in target["usage_hint"]
    assert any("never write 'Age: <placeholder>'" in row for row in prompt["output_contract"])


def test_ambiguous_may_hard_negative_prompt_forbids_redundant_subject() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=8, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["value"] == "may improve with hydration"
    )
    target = json.loads(build_english_llm_prompt(case))["hard_negative_targets"][0]

    assert "Never make hydration" in target["instruction"]


def test_medication_hard_negative_is_mandatory_in_minimal_note_prompt() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=9, batch_size=500, seed=20260820)
        if style_profile(row)["name"] == "minimal-ehr"
        and hard_negative_targets(row)[0]["category"] == "medication"
    )
    prompt = json.loads(build_english_llm_prompt(case))

    assert "mandatory even in a compact or minimal note" in prompt["hard_negative_targets"][0]["instruction"]
    assert any("remains mandatory in compact and minimal-EHR styles" in row for row in prompt["output_contract"])


def test_pediatric_other_organization_prompt_preserves_the_organization_kind() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=5, batch_size=500, seed=20260820)
        if int(row["age_years"]) < 18
        and any(target["slot"] == "other.organization" for target in coverage_targets(row))
    )
    case["other_organisation"] = "Example Transit Authority"
    prompt = json.loads(build_english_llm_prompt(case))
    target = next(
        row for row in prompt["required_pii_targets"] if row["slot"] == "other.organization"
    )

    assert "Never call it the child's school" in target["usage_hint"]
    assert "place the child attends" in target["usage_hint"]


def test_pediatric_generic_other_organization_prompt_does_not_invent_school_kind() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=5, batch_size=500, seed=20260820)
        if int(row["age_years"]) < 18
        and any(target["slot"] == "other.organization" for target in coverage_targets(row))
    )
    case["other_organisation"] = "Example Voluntary Service"
    prompt = json.loads(build_english_llm_prompt(case))
    target = next(
        row for row in prompt["required_pii_targets"] if row["slot"] == "other.organization"
    )

    assert "no school marker" in target["usage_hint"]
    assert "never call it the child's school" in target["usage_hint"]


def test_pediatric_college_prompt_requires_explicit_adult_affiliation() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=9, batch_size=500, seed=20260820)
        if int(row["age_years"]) < 16
        and any(target["slot"] == "other.organization" for target in coverage_targets(row))
    )
    case["other_organisation"] = "Example Community College"
    prompt = json.loads(build_english_llm_prompt(case))
    target = next(
        row for row in prompt["required_pii_targets"] if row["slot"] == "other.organization"
    )

    assert "not age-appropriate as the patient's own school" in target["usage_hint"]
    assert "Mother's employer" in target["usage_hint"]
    assert "Do not add an explanatory disclaimer" in target["usage_hint"]


def test_young_adult_relative_prompt_excludes_impossible_adult_child_role() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=6, batch_size=500, seed=20260820)
        if 18 <= int(row["age_years"]) < 31
        and any(target["slot"].startswith("relative.") for target in coverage_targets(row))
    )
    case["relative_role"] = "adult child"
    prompt = json.loads(build_english_llm_prompt(case))
    target = next(
        row for row in prompt["required_pii_targets"] if row["slot"].startswith("relative.")
    )

    assert prompt["case"]["relative_role"] == "sibling"
    assert "Do not describe this person as the patient's adult child" in target["usage_hint"]


def test_small_balanced_case_batch_rotates_across_all_14_labels() -> None:
    cases = build_batch_cases(batch_index=0, batch_size=12, seed=23)
    assert {row["language"] for row in cases} == {"en-GB", "en-US"}
    assert {target["label"] for row in cases for target in coverage_targets(row)} == set(
        BERT_ENTITY_LABELS
    )
    assert all(
        slot["label"] != "Anonymize_Other" for row in cases for slot in pii_slots(row)
    )
    assert all(
        row["profession"] != "Physiotherapist"
        for row in cases
        if row["language"] == "en-US"
    )


def test_young_adult_cases_use_age_plausible_professions() -> None:
    cases = build_batch_cases(batch_index=7, batch_size=500, seed=20260820)
    advanced_terms = {
        "architect",
        "civil engineer",
        "clinical pharmacist",
        "geriatric physical therapist",
        "hospital pharmacist",
        "lawyer",
        "pharmacist",
        "physiotherapist",
        "police officer",
        "teacher",
        "university lecturer",
    }
    assert all(
        not any(term in row["profession"].casefold() for term in advanced_terms)
        for row in cases
        if int(row["age_years"]) < 21
    )


def test_pediatric_cases_never_request_a_profession_target() -> None:
    cases = build_batch_cases(batch_index=4, batch_size=500, seed=20260820)
    assert all(
        not any(target["label"] == "Profession" for target in coverage_targets(case))
        for case in cases
        if int(case["age_years"]) < 18
    )


def test_future_batches_are_weighted_toward_documentation_native_styles() -> None:
    cases = build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
    styles = [style_profile(case)["name"] for case in cases]
    polished = sum(
        name in {"narrative-scribe", "patient-facing-letter"} for name in styles
    )

    assert polished <= 100
    assert sum(name == "minimal-ehr" for name in styles) >= 60
    assert sum(name == "compact-clinician" for name in styles) >= 100
    raw_profiles = [
        style_profile(case)
        for case in cases
        if style_profile(case)["name"]
        in {"compact-clinician", "structured-record", "handover-note", "minimal-ehr"}
    ]
    assert len(raw_profiles) >= 350
    assert all("structural_contract" in profile for profile in raw_profiles)
    assert {profile["length"] for profile in raw_profiles} <= {
        "100-180 words",
        "110-210 words",
        "120-210 words",
        "120-220 words",
        "130-220 words",
        "150-250 words",
    }
    assert {
        hard_negative_targets(case)[0]["category"] for case in cases
    } >= {
        "ambiguous_month_word",
        "caregiver_role",
        "clinical_scale",
        "device_model",
        "duration",
        "eponym",
        "gene_variant",
        "gestational_age",
        "impossible_date",
        "laboratory_code",
        "measurement",
        "medication",
        "pain_score",
        "relative_time",
        "terminology_code",
        "time",
        "vital_sign",
    }


def test_parser_rejects_anonymize_other_instead_of_remapping_it() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=29)[0]
    with pytest.raises(EnglishLLMRenderError, match="Anonymize_Other is forbidden"):
        marked_text_to_english_doc(
            record=case,
            marked_text="[[PII|Anonymize_Other|forbidden]]",
            model="gpt-5.6-luna",
        )


def test_parser_annotates_every_repeated_typed_placeholder_slot() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=31)[0]
    target = coverage_targets(case)[0]
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    doc = marked_text_to_english_doc(
        record=case,
        marked_text=f"Patient: {marker}. Repeated in footer: {marker}.",
        model="gpt-5.6-luna",
    )
    matching = [span for span in doc["spans"] if span["source_slot"] == target["slot"]]
    assert len(matching) == 2
    assert all(doc["text"][span["begin"] : span["end"]] == span["text"] for span in matching)


def test_parser_expands_compact_author_facing_placeholders() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=35)[0]
    targets = coverage_targets(case)
    marked = "\n".join(
        f"Field {index}: [[PII_{index:02d}]]."
        for index in range(1, len(targets) + 1)
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert [span["source_slot"] for span in doc["spans"]] == [
        target["slot"] for target in targets
    ]


def test_validation_rejects_report_identifier_described_as_mrn() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=0, batch_size=120, seed=37)
        if any(target["slot"] == "patient.report_id" for target in coverage_targets(row))
    )
    marked = _valid_marked_text(case)
    report_marker = "[[PII|ID:Patient|patient.report_id]]"
    marked = marked.replace(
        next(line for line in marked.splitlines() if report_marker in line),
        f"Patient ID/MRN: {report_marker}.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "report/accession identifier was described as an MRN or patient identifier" in (
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_lab_validator_used_as_ordering_clinician() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=0, batch_size=120, seed=41)
        if row["document_type"] == "laboratory_report"
    )
    target = next(
        target for target in coverage_targets(case) if target["label"] == "Name:Caregiver"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    marked = _valid_marked_text(case)
    marked = marked.replace(
        next(line for line in marked.splitlines() if marker in line),
        f"Ordering clinician: {marker}.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "laboratory validating clinician was also presented as the ordering clinician" in (
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_compound_medication_hard_negative_is_not_reduced_to_substring_drug() -> None:
    cases = build_batch_cases(batch_index=0, batch_size=240, seed=43)
    medication_targets = []
    for original in cases:
        case = {**original, "condition": {**original["condition"]}}
        case["condition"]["medications"] = ["amoxicillin-clavulanate"]
        medication_targets.extend(
            target
            for target in hard_negative_targets(case)
            if target["category"] == "medication"
        )

    assert medication_targets
    assert {target["value"] for target in medication_targets} == {
        "amoxicillin-clavulanate 875/125 mg twice daily"
    }


def test_every_laboratory_case_supplies_date_and_patient_identifier_targets() -> None:
    cases = build_batch_cases(batch_index=0, batch_size=240, seed=47)
    laboratory_cases = [case for case in cases if case["document_type"] == "laboratory_report"]

    assert laboratory_cases
    for case in laboratory_cases:
        labels = {target["label"] for target in coverage_targets(case)}
        assert {"Date", "ID:Patient", "Name:Caregiver"} <= labels


def test_training_cases_cover_name_capitalization_and_format_variants() -> None:
    cases = build_batch_cases(batch_index=0, batch_size=500, seed=48)
    selected_slots = {
        target["slot"] for case in cases for target in coverage_targets(case)
    }

    assert {
        "patient.name",
        "patient.name_upper",
        "patient.name_lower",
        "patient.given_name",
        "patient.initial_surname",
        "patient.given_family_initial",
    } <= selected_slots
    assert {"caregiver.credentialed_name", "caregiver.initial_surname"} <= selected_slots
    assert {"relative.name_upper", "relative.name_lower", "relative.initial_surname"} <= selected_slots


def test_every_pediatric_case_requires_age_signal_and_age_appropriate_labels() -> None:
    cases = build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
    pediatric = [case for case in cases if case["age_years"] < 18]

    assert len(pediatric) == 82
    for case in pediatric:
        targets = coverage_targets(case)
        assert any(target["label"] == "Age_Birthdate" for target in targets)
        if case["age_years"] < 16:
            assert not any(target["label"] == "Profession" for target in targets)
        if case["document_type"] != "laboratory_report":
            assert any(target["label"] == "Name:Other" for target in targets)


@pytest.mark.parametrize(
    "profile_id,expected",
    [
        ("en-GB", {"3-day-old", "aged 3 days", "day 3 of life", "18-month-old", "aged 18 months", "18 m/o"}),
        ("en-US", {"3-day-old", "age 3 days", "day 3 of life", "18-month-old", "age 18 months", "18 m/o"}),
    ],
)
def test_pediatric_age_slots_cover_day_week_month_and_year_formats(
    profile_id: str, expected: set[str]
) -> None:
    cases = [
        case
        for case in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if case["language"] == profile_id and case["age_years"] < 18
    ]
    values = {
        slot["value"]
        for case in cases
        for slot in pii_slots(case)
        if slot["slot"].startswith("patient.age_")
    }

    assert expected <= values
    assert any("-week-old" in value for value in values)
    assert any(" y/o" in value for value in values)


def test_every_generated_age_surface_is_supported_by_pseudonymization() -> None:
    cases = build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
    checked: set[tuple[str, str]] = set()
    for case in cases:
        profile_id = case["language"]
        for slot in pii_slots(case):
            if not slot["slot"].startswith("patient.age_"):
                continue
            value = slot["value"]
            checked.add((profile_id, value))
            assert parse_date_text(
                value,
                profile_id=profile_id,
                label="Age_Birthdate",
            ) is not None, (profile_id, slot["slot"], value)
            assert pseudonymize_date_text(
                value,
                profile_id=profile_id,
                label="Age_Birthdate",
                date_shift_days=0,
            ) is not None, (profile_id, slot["slot"], value)

    assert len(checked) >= 100


def test_validation_rejects_an_invented_unmarked_full_date() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=49)[0]
    marked = _valid_marked_text(case) + "\nSpecimen collected on 14 February 2025."
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "potential invented or unmarked full date: '14 February 2025'" in (
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_an_invented_unmarked_explicit_age() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=50)[0]
    marked = _valid_marked_text(case) + "\nPatient seen privately, 16 years."
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "potential invented or unmarked explicit age: ', 16 years'" in (
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_an_invented_unmarked_birth_year() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=50)[0]
    marked = _valid_marked_text(case) + "\nThe patient was born in 1955."
    doc = marked_text_to_english_doc(
        record=case, marked_text=marked, model="gpt-5.6-luna"
    )

    assert "potential invented or unmarked birth year: '1955'" in (
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_an_invented_unmarked_jurisdiction() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=0, batch_size=2, seed=50)
        if row["language"] == "en-US"
    )
    marked = _valid_marked_text(case) + "\nThe specimen was processed in Puerto Rico."
    doc = marked_text_to_english_doc(
        record=case, marked_text=marked, model="gpt-5.6-luna"
    )

    assert "potential invented or unmarked jurisdiction: 'Puerto Rico'" in (
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_invented_labeled_report_identifier() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=50)[0]
    marked = _valid_marked_text(case) + "\nReport and accession identifier: LAB-GLY-2047."
    doc = marked_text_to_english_doc(
        record=case, marked_text=marked, model="gpt-5.6-luna"
    )

    assert "potential invented or unmarked report identifier: 'LAB-GLY-2047'" in (
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_uses_targets_stored_with_original_generation_attempt() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=50)[0]
    marked = _valid_marked_text(case)
    doc = marked_text_to_english_doc(
        record=case, marked_text=marked, model="gpt-5.6-luna"
    )

    # Simulate a later sampler revision that would select paediatric targets
    # and hard negatives for this case. The accepted artifact must remain
    # reproducibly valid under its stored authoring contract.
    case["age_years"] = 4
    case["age_group"] = "early_childhood"
    assert validate_english_llm_document(
        record=case, marked_text=marked, doc=doc
    ) == []


def test_validation_rejects_unresolved_template_placeholder_and_wrong_fena() -> None:
    case = build_batch_cases(batch_index=0, batch_size=2, seed=51)[0]
    marked = _valid_marked_text(case) + """
Date: [day of admission]
Serum creatinine: 2.7 mg/dL
Urine sodium: 18 mmol/L
Serum sodium: 136 mmol/L
Urine creatinine: 112 mg/dL
Fractional excretion of sodium (FeNa): approximately 0.03%
"""
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")
    issues = validate_english_llm_document(record=case, marked_text=marked, doc=doc)

    assert "unresolved template placeholder: '[day of admission]'" in issues
    assert any(issue.startswith("incorrect FeNa calculation:") for issue in issues)


def test_validation_rejects_caregiver_credentials_outside_gold_span() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=0, batch_size=12, seed=53)
        if any(target["label"] == "Name:Caregiver" for target in coverage_targets(row))
    )
    targets = coverage_targets(case)
    caregiver = next(target for target in targets if target["label"] == "Name:Caregiver")
    for suffix in (", RN", " RN"):
        marked = _valid_marked_text(case).replace(
            f"[[PII|Name:Caregiver|{caregiver['slot']}]]",
            f"[[PII|Name:Caregiver|{caregiver['slot']}]]{suffix}",
            1,
        )
        doc = marked_text_to_english_doc(
            record=case, marked_text=marked, model="gpt-5.6-luna"
        )

        assert "caregiver credential was added outside the annotated name span" in (
            validate_english_llm_document(record=case, marked_text=marked, doc=doc)
        )


def test_validation_rejects_caregiver_honorific_outside_gold_span() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=0, batch_size=12, seed=53)
        if any(target["label"] == "Name:Caregiver" for target in coverage_targets(row))
    )
    caregiver = next(
        target for target in coverage_targets(case) if target["label"] == "Name:Caregiver"
    )
    marker = f"[[PII|Name:Caregiver|{caregiver['slot']}]]"
    marked = _valid_marked_text(case).replace(marker, f"Dr {marker}", 1)
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "caregiver honorific was added outside the annotated name span" in (
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_wrong_article_before_hyphenated_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(target["value"] == "18-year-old" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"] == "18-year-old"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"A {marker} patient was reviewed.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "incorrect indefinite article before age" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_wrong_article_before_month_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=2, batch_size=500, seed=20260820)
        if any(target["value"] == "18-month-old" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"] == "18-month-old"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"The patient is a {marker} child.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "incorrect indefinite article before age" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_an_before_consonant_sound_numeric_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=2, batch_size=500, seed=20260820)
        if any(target["value"] == "3-year-old" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"] == "3-year-old"
    )
    marked = _valid_marked_text(case)
    marker = f"[[PII|Age_Birthdate|{target['slot']}]]"
    marked = marked.replace(
        next(line for line in marked.splitlines() if marker in line),
        f"The patient is an {marker} patient.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "incorrect indefinite article before age" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_missing_article_before_hyphenated_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=2, batch_size=500, seed=20260820)
        if any(target["value"] == "18-month-old" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"] == "18-month-old"
    )
    marker = f"[[PII|Age_Birthdate|{target['slot']}]]"
    for replacement in (
        f"Presentation is consistent with the diagnosis in {marker} child.",
        f"Age: Patient is {marker}; pediatric infant.",
    ):
        marked = _valid_marked_text(case)
        marked = marked.replace(
            next(line for line in marked.splitlines() if marker in line),
            replacement,
        )
        doc = marked_text_to_english_doc(
            record=case, marked_text=marked, model="gpt-5.6-luna"
        )

        assert "missing indefinite article before age" in "\n".join(
            validate_english_llm_document(record=case, marked_text=marked, doc=doc)
        )


def test_validation_rejects_subjectless_y_o_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(target["value"].endswith(" y/o") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].endswith(" y/o")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"{marker} was accompanied by a parent.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "age expression used without a patient noun" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_bare_not_an_identifier_disclaimer() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=2, batch_size=500, seed=20260820)
        if any(target["category"] == "gene_variant" for target in hard_negative_targets(row))
    )
    target = next(
        target for target in hard_negative_targets(case) if target["category"] == "gene_variant"
    )
    marked = _valid_marked_text(case).replace(
        target["value"], f"{target['value']}, not a patient identifier"
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "artificial test disclaimer" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_hard_negative_self_correction() -> None:
    case = build_batch_cases(batch_index=1, batch_size=2, seed=20260820)[0]
    hard_negative = hard_negative_targets(case)[0]
    marked = _valid_marked_text(case) + (
        f"\nSymptoms were present {hard_negative['value']}? No—the duration was entered again."
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "artificial self-correction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_redundant_measurement_context() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(
            target["category"] == "measurement" and target["value"].endswith(" kg")
            for target in hard_negative_targets(row)
        )
    )
    value = next(
        target["value"]
        for target in hard_negative_targets(case)
        if target["category"] == "measurement" and target["value"].endswith(" kg")
    )
    marked = _valid_marked_text(case).replace(value, f"measured weight was weight {value}")
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "redundant measurement construction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_redundant_hard_negative_context() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=2, batch_size=500, seed=20260820)
        if any(
            target["value"] == "may improve with hydration"
            for target in hard_negative_targets(row)
        )
    )
    marked = _valid_marked_text(case).replace(
        "may improve with hydration", "hydration may improve with hydration"
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "redundant hard-negative construction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


@pytest.mark.parametrize("literal_prefix", ["GMC-TEST ", "GMC-TEST-"])
def test_validation_rejects_duplicated_identifier_prefix(literal_prefix: str) -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=2, batch_size=500, seed=20260820)
        if any(target["slot"] == "caregiver.professional_id" for target in coverage_targets(row))
    )
    target = next(
        target
        for target in coverage_targets(case)
        if target["slot"] == "caregiver.professional_id"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Professional ID: {literal_prefix}{marker}",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "duplicated identifier prefix" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_redundant_age_group_and_stray_syntax() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(target["value"].startswith("aged ") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].startswith("aged ")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Clinical context: {marker} adult.\nGP\")]/\n//",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")
    issues = "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )

    assert "redundant age and age-group construction" in issues
    assert "stray generation syntax" in issues


def test_validation_rejects_subjectless_contextual_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(target["value"].startswith("age ") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].startswith("age ")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Review date: {marker} was reviewed during the shift.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "contextual age used without a patient noun" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_subjectless_aged_contextual_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(target["value"].startswith("aged ") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].startswith("aged ")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"On review, {marker} was alert and comfortable.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "contextual age used without a patient noun" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )

    marked_after_field = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Clinical context: {marker} was admitted for observation.",
    )
    doc_after_field = marked_text_to_english_doc(
        record=case, marked_text=marked_after_field, model="gpt-5.6-luna"
    )
    assert "contextual age used without a patient noun" in "\n".join(
        validate_english_llm_document(
            record=case, marked_text=marked_after_field, doc=doc_after_field
        )
    )

    marked_after_heading = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Reason for admission\n{marker} was admitted for observation.",
    )
    doc_after_heading = marked_text_to_english_doc(
        record=case, marked_text=marked_after_heading, model="gpt-5.6-luna"
    )
    assert "contextual age used without a patient noun" in "\n".join(
        validate_english_llm_document(
            record=case, marked_text=marked_after_heading, doc=doc_after_heading
        )
    )


def test_validation_rejects_annotation_schema_field_leakage() -> None:
    case = build_batch_cases(batch_index=1, batch_size=2, seed=20260820)[0]
    original = _valid_marked_text(case)
    marked = "Name:Patient: not recorded\n" + original
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "generation metadata leaked" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_age_under_date_field_and_generation_meta() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(target["value"].endswith("-year-old") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].endswith("-year-old")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Date: {marker} patient evaluated. The supplied clinical age form was recorded.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")
    issues = "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )

    assert "Age_Birthdate marker was used as the value of a Date field" in issues
    assert "generation metadata leaked into age context" in issues


def test_validation_rejects_missing_copula_after_hyphenated_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(target["value"].endswith("-year-old") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].endswith("-year-old")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"The {marker} patient awake and oriented.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "age construction is missing a copula" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )

    marked_admitted = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"This {marker} patient admitted to pulmonary medicine.",
    )
    doc_admitted = marked_text_to_english_doc(
        record=case, marked_text=marked_admitted, model="gpt-5.6-luna"
    )
    assert "age construction is missing a copula" in "\n".join(
        validate_english_llm_document(
            record=case, marked_text=marked_admitted, doc=doc_admitted
        )
    )


def test_validation_rejects_missing_copula_after_contextual_age() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=1, batch_size=500, seed=20260820)
        if any(target["value"].startswith("aged ") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].startswith("aged ")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"The patient, {marker}, reviewed on the morning round.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "age construction is missing a copula" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_corrupted_vital_sign_fragment() -> None:
    case = build_batch_cases(batch_index=1, batch_size=2, seed=20260820)[0]
    marked = _valid_marked_text(case) + "\nBP 104/ weten?/68 mmHg."
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "stray generation syntax" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_literal_generation_terminator() -> None:
    case = build_batch_cases(batch_index=1, batch_size=2, seed=20260820)[0]
    marked = _valid_marked_text(case) + "\n<|end|>"
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "stray generation syntax" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )

    marked_calc = _valid_marked_text(case) + "\nResult status: Authorised.calc"
    doc_calc = marked_text_to_english_doc(
        record=case, marked_text=marked_calc, model="gpt-5.6-luna"
    )
    assert "stray generation syntax" in "\n".join(
        validate_english_llm_document(
            record=case, marked_text=marked_calc, doc=doc_calc
        )
    )


def test_validation_rejects_article_after_age_field() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if any(target["value"].endswith("-year-old") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].endswith("-year-old")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Age: a {marker} patient.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "unnatural hyphenated age construction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_redundant_patient_age_patient_wording() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if any(target["value"].endswith("-year-old") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].endswith("-year-old")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"The patient is a {marker} patient.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "redundant age and age-group construction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_contextual_age_followed_by_patient() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=13, batch_size=500, seed=20260820)
        if any(target["value"].startswith("aged ") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].startswith("aged ")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Clinical context: {marker} patient.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "redundant age and age-group construction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_age_as_the_patient_field_value() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=11, batch_size=500, seed=20260820)
        if any(target["value"].endswith("-year-old") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].endswith("-year-old")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Patient: {marker} patient\nName: supplied separately",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "age was used as the Patient field value" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_artificial_role_and_phrase_hard_negative_contexts() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["category"] == "caregiver_role"
    )
    value = hard_negative_targets(case)[0]["value"]
    for replacement in (
        f"The relevant clinician role is {value}",
        f"The role of a {value} is not currently indicated",
        f"The phrase {value} was entered",
    ):
        marked = _valid_marked_text(case).replace(value, replacement)
        doc = marked_text_to_english_doc(
            record=case, marked_text=marked, model="gpt-5.6-luna"
        )
        assert "artificial test disclaimer" in "\n".join(
            validate_english_llm_document(record=case, marked_text=marked, doc=doc)
        )


def test_validation_rejects_invented_unmarked_profession_field() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if not any(target["label"] == "Profession" for target in coverage_targets(row))
    )
    marked = _valid_marked_text(case) + "\nProfession: secondary-school student"
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "potential invented or unmarked profession field" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_ungrammatical_duration_construction() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["value"] == "for 3 days"
    )
    marked = _valid_marked_text(case).replace("for 3 days", "Symptoms began for 3 days")
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "ungrammatical duration construction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_redundant_duration_field_construction() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["value"] == "for 3 days"
    )
    marked = _valid_marked_text(case).replace(
        "for 3 days", "Symptom duration: for 3 days"
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "redundant duration field construction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_redundant_duration_noted_as_construction() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["value"] == "for 3 days"
    )
    marked = _valid_marked_text(case).replace(
        "for 3 days", "Current symptom duration noted as for 3 days"
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "redundant duration field construction" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_malformed_credential_suffix() -> None:
    case = build_batch_cases(batch_index=3, batch_size=2, seed=20260820)[0]
    marked = _valid_marked_text(case) + "\nDischarging clinician: Dr Alex Smith, MBBS qc"
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "malformed clinician credential suffix" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_caregiver_role_after_signature() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["category"] == "caregiver_role"
    )
    value = hard_negative_targets(case)[0]["value"]
    marked = _valid_marked_text(case).replace(
        value,
        f"Sincerely,\nAlex Smith\nThe patient was not previously evaluated by an {value}",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "caregiver-role hard negative was placed after a signature" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_relative_time_that_modifies_continued_treatment() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["category"] == "relative_time"
    )
    value = hard_negative_targets(case)[0]["value"]
    marked = _valid_marked_text(case).replace(
        value,
        f"Continue oral rehydration and supportive care {value}",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "relative-time hard negative directly modifies continued treatment" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_patient_profession_attached_to_clinician_byline() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if any(target["label"] == "Profession" for target in coverage_targets(row))
    )
    original = _valid_marked_text(case)
    profession_line = next(
        line for line in original.splitlines() if "[[PII|Profession|" in line
    )
    profession_marker = profession_line.split(": ", 1)[1]
    marked = original.replace(
        profession_line,
        "Consulting clinician: Alex Smith\nOccupation: " + profession_marker,
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "patient profession is attached to a clinician byline" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_patient_profession_attached_to_seen_by_byline() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if any(target["label"] == "Profession" for target in coverage_targets(row))
    )
    original = _valid_marked_text(case)
    profession_line = next(
        line for line in original.splitlines() if "[[PII|Profession|" in line
    )
    profession_marker = profession_line.split(": ", 1)[1]
    marked = original.replace(
        profession_line,
        "Seen by: Alex Smith\nOccupation: " + profession_marker,
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "patient profession is attached to a clinician byline" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_unresolved_embedded_date_placeholder() -> None:
    case = build_batch_cases(batch_index=3, batch_size=2, seed=20260820)[0]
    marked = _valid_marked_text(case) + "\nReview after [clinic date]."
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "unresolved template placeholder" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_caregiver_identifier_used_as_name() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if any(target["slot"] == "caregiver.professional_id" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["slot"] == "caregiver.professional_id"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Discharging clinician: Dr {marker}.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "caregiver identifier used as a clinician name" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Relevant care coordination note: the role was identified in the referral pathway.",
        "Relevant comparison clinician role: attending neurologist.",
        "Relevant clinician is a consultant endocrinologist.",
        "Relevant clinician role noted in the chart: attending endocrinologist.",
        "Relevant clinician role recorded as attending endocrinologist.",
        "Relevant role for consultation: attending cardiologist.",
        "Relevant care team contact: attending endocrinologist.",
        "The relevant clinician’s role is consultant respiratory physician.",
        "The relevant attending pulmonologist may review the results.",
        "Relevant consultant role: attending cardiologist.",
    ],
)
def test_validation_rejects_additional_artificial_hard_negative_prose(phrase: str) -> None:
    case = build_batch_cases(batch_index=3, batch_size=2, seed=20260820)[0]
    value = hard_negative_targets(case)[0]["value"]
    marked = _valid_marked_text(case).replace(value, f"{value}. {phrase}")
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "artificial test disclaimer" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_relevant_clinician_is_without_an_article() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=5, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["category"] == "caregiver_role"
    )
    value = hard_negative_targets(case)[0]["value"]
    marked = _valid_marked_text(case).replace(value, f"Relevant clinician is {value}")
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "artificial test disclaimer" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("\nFollow-up in in 10 days.", "duplicated preposition"),
        ("\nFollow-up in in about 5 weeks.", "duplicated preposition"),
        ("\nOral rehydration may improve with hydration.", "redundant hard-negative"),
        ("\nHydration status may improve with hydration.", "redundant hard-negative"),
        ("\nThe patient reports that fluids may improve with hydration.", "redundant hard-negative"),
        ("\nOral rehydration solution may improve with hydration.", "redundant hard-negative"),
        ("\nEncourage fluids—may improve with hydration.", "redundant hard-negative"),
        ("\nOral rehydration solution; may improve with hydration.", "redundant hard-negative"),
        ("\nPlan reviewed; may improve with hydration.", "redundant hard-negative"),
        ("\nclinician salvaged?", "stray generation syntax"),
        ("\nPediatric Laboratory Medicine.calc", "stray generation syntax"),
        ("\nDocumented by AYAAD REEVES qc syndrom.", "stray generation syntax"),
        ("\nAvalaDr?", "stray generation syntax"),
        ("\nSigned: Alex Smith calibrator?", "stray generation syntax"),
        ("\nSigned: Alex Smith\nqdata?", "stray generation syntax"),
        ("\nNo lifting over deset?", "stray generation syntax"),
        ("\nJSImport", "stray generation syntax"),
        ("\nDischarging doctor: GMC-TEST-2916471", "caregiver identifier used as a clinician name"),
        ("\nSigned: Alex Smith\nJapgolly", "stray generation syntax"),
        ("\nValidated by: Alex Smith‑###", "stray generation syntax"),
        ("\nHome address: 882 I- 710, Los Angeles, CA 90017", "malformed US interstate display"),
        ("\nSigned: Alex Smith, pulmonary medicine、】【", "unexpected CJK punctuation"),
        ("\nVitals: HR heart rate 108 beats/minute.", "redundant vital-sign label"),
        ("\nAdmission date: [hospital record].", "unresolved template placeholder"),
        ("\nThe hard-negative instruction remains in about 11 weeks.", "generation metadata leaked"),
        ("\nRelevant clinician role: attending pulmonologist.", "artificial test disclaimer"),
        ("\nRelevant clinician: consultant neurologist.", "artificial test disclaimer"),
        ("\nRelevant specialist role: attending pulmonologist.", "artificial test disclaimer"),
        ("\nRelevant specialist: attending pulmonologist.", "artificial test disclaimer"),
        ("\nattending cardiologist-consulted case", "artificial test disclaimer"),
        ("\nGMC-TEST: GMC-TEST-1234567", "duplicated identifier prefix"),
        ("\nClinician: Alex Smith, GMC GMC-TEST-1234567", "unnatural GMC identifier label"),
        (
            "\nClinician: Alex Smith, GMC-TEST identifier GMC-TEST-1234567",
            "unnatural GMC identifier label",
        ),
        ("\nSigned by Alex Smith, MD-MD.", "duplicated clinician credential"),
        ("\nRespiratory rate thirty.", "spelled-out respiratory rate value"),
        ("\nAddress_Location:Patient: withheld", "generation metadata leaked"),
        ("\nAge_Birthdate: recorded", "generation metadata leaked"),
        ("\nDate: <Date>October 26, 2025", "generation metadata leaked"),
        ("\n10 y/o Resting in bed.", "age expression used without a patient noun"),
        ("\nડિસcharge teaching completed.", "unexpected non-Latin script"),
        (
            "\nDocumented by: Dr Alex Smith, MBBS, Registered Nurse",
            "physician credential conflicts with nurse role",
        ),
        (
            "\nDocumented by Nurse Dr Alex Smith, MBBS",
            "physician credential conflicts with nurse role",
        ),
        (
            "\nDischarging clinician: Dr, GMC GMC-TEST-1234567",
            "incomplete clinician name before professional identifier",
        ),
        (
            "\nDocumented by respiratory nursing team. GMC professional identifier: GMC-TEST-1234567",
            "GMC identifier was assigned to a nursing role",
        ),
        (
            "\nAlex Smith\nGeneral Practitioner\nconsultant respiratory physician",
            "clinician was assigned incompatible dual roles",
        ),
    ],
)
def test_validation_rejects_new_review_failure_classes(suffix: str, expected: str) -> None:
    case = build_batch_cases(batch_index=3, batch_size=2, seed=20260820)[0]
    marked = _valid_marked_text(case) + suffix
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert expected in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_british_clinical_spelling_in_us_prose() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=2, seed=20260820)
        if row["language"] == "en-US"
    )
    marked = _valid_marked_text(case) + "\nDiarrhoea is improving."
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "cross-locale British spelling" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_redundant_hyphenated_age_group_sentence() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if any(target["value"].endswith("-year-old") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["value"].endswith("-year-old")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    for replacement in (
        f"This {marker} patient is an adult.",
        f"Clinical context: a {marker} patient, adult.",
    ):
        original = _valid_marked_text(case)
        marked = original.replace(
            next(line for line in original.splitlines() if marker in line),
            replacement,
        )
        doc = marked_text_to_english_doc(
            record=case, marked_text=marked, model="gpt-5.6-luna"
        )

        assert "redundant age and age-group construction" in "\n".join(
            validate_english_llm_document(record=case, marked_text=marked, doc=doc)
        )


def test_validation_rejects_physician_credential_in_nurse_byline() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=3, batch_size=500, seed=20260820)
        if row["language"] == "en-US"
        and any(target["slot"] == "caregiver.credentialed_name" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["slot"] == "caregiver.credentialed_name"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Nurse: {marker}",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "physician credential was placed in a nurse byline" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_credentialed_name_prompt_forbids_an_external_credential() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=13, batch_size=500, seed=20260820)
        if any(
            target["slot"] == "caregiver.credentialed_name"
            for target in coverage_targets(row)
        )
    )

    prompt = build_english_llm_prompt(case)

    assert "already includes its MD/MBBS credential" in prompt
    assert "Do not write MD, MBBS, DO" in prompt


def test_gb_professional_id_prompt_forbids_added_identifier_prefixes() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=13, batch_size=500, seed=20260820)
        if row["language"] == "en-GB"
        and any(
            target["slot"] == "caregiver.professional_id"
            for target in coverage_targets(row)
        )
    )
    prompt = json.loads(build_english_llm_prompt(case))
    target = next(
        row
        for row in prompt["required_pii_targets"]
        if row["slot"] == "caregiver.professional_id"
    )

    assert "may be numeric-only" in target["usage_hint"]
    assert "Never prepend GMC, GMC-TEST, NPI, LIC" in target["usage_hint"]


def test_validation_rejects_age_span_wrapped_in_angle_brackets() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=4, batch_size=500, seed=20260820)
        if any(target["label"] == "Age_Birthdate" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["label"] == "Age_Birthdate"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(marker, f"<{marker}>", 1)
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "age PII span was wrapped in artificial angle brackets" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_healthcare_organization_with_dangling_punctuation() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=4, batch_size=500, seed=20260820)
        if any(
            target["slot"] == "healthcare.organization"
            for target in coverage_targets(row)
        )
    )
    case["hospital"] = "MALFORMED HOSPITAL -"
    marked = _valid_marked_text(case)
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "healthcare organization PII has dangling terminal punctuation" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_contextual_age_missing_copula_before_review_location() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=5, batch_size=500, seed=20260820)
        if any(target["slot"] == "patient.age_contextual" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["slot"] == "patient.age_contextual"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    for replacement in (
        f"The patient, {marker}, reviewed on the respiratory ward.",
        f"Patient, {marker}, reviewed with family at bedside.",
    ):
        marked = original.replace(
            next(line for line in original.splitlines() if marker in line),
            replacement,
        )
        doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

        assert "age construction is missing a copula" in "\n".join(
            validate_english_llm_document(record=case, marked_text=marked, doc=doc)
        )


def test_validation_rejects_name_span_concatenated_with_generation_text() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=5, batch_size=500, seed=20260820)
        if any(target["label"] == "Name:Caregiver" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["label"] == "Name:Caregiver"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    marked = _valid_marked_text(case).replace(marker, marker + "crescent", 1)
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "name PII span was concatenated with trailing alphabetic text" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_pediatric_civic_organization_as_school() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=5, batch_size=500, seed=20260820)
        if int(row["age_years"]) < 18
        and any(target["slot"] == "other.organization" for target in coverage_targets(row))
    )
    case["other_organisation"] = "Example Housing Authority"
    target = next(
        target for target in coverage_targets(case) if target["slot"] == "other.organization"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"School: {marker}.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "pediatric non-healthcare organization was assigned an age-inappropriate role" in (
        "\n".join(validate_english_llm_document(record=case, marked_text=marked, doc=doc))
    )


def test_validation_rejects_pediatric_generic_organization_as_school() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=5, batch_size=500, seed=20260820)
        if int(row["age_years"]) < 18
        and any(target["slot"] == "other.organization" for target in coverage_targets(row))
    )
    case["other_organisation"] = "Example Voluntary Service"
    target = next(
        target for target in coverage_targets(case) if target["slot"] == "other.organization"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"School: {marker}.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "pediatric non-healthcare organization was assigned an age-inappropriate role" in (
        "\n".join(validate_english_llm_document(record=case, marked_text=marked, doc=doc))
    )


def test_validation_rejects_impossible_adult_child_relationship() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=6, batch_size=500, seed=20260820)
        if 18 <= int(row["age_years"]) < 31
        and any(target["slot"].startswith("relative.") for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["slot"].startswith("relative.")
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Adult child {marker} was updated.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "patient age is incompatible with adult-child relationship" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_unnamed_impossible_adult_child_relationship() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=6, batch_size=500, seed=20260820)
        if 18 <= int(row["age_years"]) < 31
    )
    marked = _valid_marked_text(case) + "\nThe patient's adult child was updated."
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "patient age is incompatible with adult-child relationship" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_advanced_profession_for_young_adult() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=7, batch_size=500, seed=20260820)
        if 18 <= int(row["age_years"]) < 21
        and any(target["label"] == "Profession" for target in coverage_targets(row))
    )
    case["profession"] = "Geriatric Physical Therapist"
    target = next(
        target for target in coverage_targets(case) if target["label"] == "Profession"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    marked = _valid_marked_text(case)
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "profession is implausible for patient age" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_patient_age_attached_to_clinician_byline() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=7, batch_size=500, seed=20260820)
        if any(target["label"] == "Age_Birthdate" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["label"] == "Age_Birthdate"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Responsible clinician: Alex Smith, {marker}",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "patient age PII was assigned to a clinician" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_patient_age_after_clinician_byline_at_document_end() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=8, batch_size=500, seed=20260820)
        if any(target["label"] == "Age_Birthdate" for target in coverage_targets(row))
        and any(target["label"] == "Name:Caregiver" for target in coverage_targets(row))
    )
    age = next(target for target in coverage_targets(case) if target["label"] == "Age_Birthdate")
    caregiver = next(
        target for target in coverage_targets(case) if target["label"] == "Name:Caregiver"
    )
    age_marker = f"[[PII|{age['label']}|{age['slot']}]]"
    caregiver_marker = f"[[PII|{caregiver['label']}|{caregiver['slot']}]]"
    marked = _valid_marked_text(case)
    marked = "\n".join(
        line
        for line in marked.splitlines()
        if age_marker not in line and caregiver_marker not in line
    )
    marked += f"\nAttending: {caregiver_marker}\nAge: {age_marker}"
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "patient age PII was placed after a clinician byline" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_patient_contact_after_laboratory_validator() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=8, batch_size=500, seed=20260820)
        if row["document_type"] == "laboratory_report"
        and any(target["label"] == "Contactdetails" for target in coverage_targets(row))
        and any(target["label"] == "Name:Caregiver" for target in coverage_targets(row))
    )
    contact = next(target for target in coverage_targets(case) if target["label"] == "Contactdetails")
    caregiver = next(
        target for target in coverage_targets(case) if target["label"] == "Name:Caregiver"
    )
    contact_marker = f"[[PII|{contact['label']}|{contact['slot']}]]"
    caregiver_marker = f"[[PII|{caregiver['label']}|{caregiver['slot']}]]"
    original = _valid_marked_text(case)
    marked = "\n".join(
        line
        for line in original.splitlines()
        if contact_marker not in line and caregiver_marker not in line
    )
    marked += f"\nValidated by: {caregiver_marker}\nContact detail: {contact_marker}"
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "patient contact PII was placed after a laboratory validator" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_hard_negative_as_explicitly_inappropriate() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=8, batch_size=500, seed=20260820)
        if hard_negative_targets(row)[0]["category"] == "relative_time"
    )
    value = hard_negative_targets(case)[0]["value"]
    marked = _valid_marked_text(case).replace(value, f"{value} is not appropriate", 1)
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "hard-negative target was inserted as an artificial rejection" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_age_incompatible_college_as_child_school() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=9, batch_size=500, seed=20260820)
        if int(row["age_years"]) < 16
        and any(target["slot"] == "other.organization" for target in coverage_targets(row))
    )
    case["other_organisation"] = "Example Community College"
    target = next(
        target for target in coverage_targets(case) if target["slot"] == "other.organization"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"School: {marker}",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "education organization is incompatible with the pediatric patient's age" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_allows_college_as_explicit_aunt_employer() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=9, batch_size=500, seed=20260820)
        if int(row["age_years"]) < 16
        and any(target["slot"] == "other.organization" for target in coverage_targets(row))
    )
    case["other_organisation"] = "Example Community College"
    target = next(
        target for target in coverage_targets(case) if target["slot"] == "other.organization"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Aunt employer: {marker}",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "education organization is incompatible with the pediatric patient's age" not in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_synthetic_not_the_child_school_disclaimer() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=9, batch_size=500, seed=20260820)
        if int(row["age_years"]) < 16
        and any(target["slot"] == "other.organization" for target in coverage_targets(row))
    )
    case["other_organisation"] = "Example Community College"
    target = next(
        target for target in coverage_targets(case) if target["slot"] == "other.organization"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Aunt employer: {marker}, not the patient's school.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "education organization role was explained with a synthetic disclaimer" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_occupation_field_label_embedded_in_prose() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=9, batch_size=500, seed=20260820)
        if any(target["label"] == "Profession" for target in coverage_targets(row))
    )
    target = next(target for target in coverage_targets(case) if target["label"] == "Profession")
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"They work as Occupation: {marker}.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "occupation field label embedded in prose" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )


def test_validation_rejects_stray_token_after_caregiver_name() -> None:
    case = next(
        row
        for row in build_batch_cases(batch_index=9, batch_size=500, seed=20260820)
        if any(target["label"] == "Name:Caregiver" for target in coverage_targets(row))
    )
    target = next(
        target for target in coverage_targets(case) if target["label"] == "Name:Caregiver"
    )
    marker = f"[[PII|{target['label']}|{target['slot']}]]"
    original = _valid_marked_text(case)
    marked = original.replace(
        next(line for line in original.splitlines() if marker in line),
        f"Documented by: {marker} ose.",
    )
    doc = marked_text_to_english_doc(record=case, marked_text=marked, model="gpt-5.6-luna")

    assert "caregiver name PII has stray trailing syntax" in "\n".join(
        validate_english_llm_document(record=case, marked_text=marked, doc=doc)
    )
