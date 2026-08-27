import json
from datetime import date

import pytest
from meddeid_core import BERT_ENTITY_LABELS, validate_record

from meddeid_data import generation_profiles
from meddeid_data.generation_profile_en import (
    FIRST_NAMES as GB_FIRST_NAMES,
    HOSPITALS_BY_REGION,
    _looks_like_major_acute_hospital,
)
from meddeid_data.generation_profile_en_us import FIRST_NAMES as US_FIRST_NAMES
from meddeid_data.cli import _write_jsonl, main
from meddeid_data.generation_profiles import (
    GENERATION_PROFILE_CONTRACT,
    profile_from_case_records,
    resolve_generation_profile,
)


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_english_profiles_generate_all_reference_document_types(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    rows = profile.generate_documents(12, seed=42, require_synthea=False)
    reviewed, results, model_reviews = profile.review_documents(rows)

    assert reviewed == rows
    assert model_reviews == []
    assert all(result.passed for result in results)
    assert all(validate_record(row) == [] for row in rows)
    assert all(row["metadata"]["lang"] == profile_id for row in rows)
    assert all(row["metadata"]["generation_profile"] == profile_id for row in rows)
    assert {row["metadata"]["document_type"] for row in rows} == {
        "clinic_note",
        "discharge_summary",
        "emergency_note",
        "referral_letter",
        "laboratory_report",
        "nursing_note",
    }
    assert {span["label"] for row in rows for span in row["spans"]} == set(
        BERT_ENTITY_LABELS
    )
    assert all(span["label"] != "Anonymize_Other" for row in rows for span in row["spans"])
    assert all("patiënt" not in row["text"].lower() for row in rows)


def test_en_gb_generation_is_deterministic_and_two_stage_renderable() -> None:
    profile = resolve_generation_profile("en-GB")
    first = profile.generate_documents(6, seed=73, require_synthea=False)
    second = profile.generate_documents(6, seed=73, require_synthea=False)
    assert first == second

    cases = profile.build_case_records(6, seed=73)
    inferred = profile_from_case_records(cases)
    rendered = inferred.render_case_records(cases, seed=73)
    _, results, _ = inferred.review_documents(rendered)
    assert inferred.selection == "en-GB"
    assert all(result.passed for result in results)
    assert [row["document_id"] for row in rendered] == [
        f"synthetic-{index:05d}" for index in range(1, 7)
    ]


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_english_profiles_generate_varied_realistic_email_domains(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(48, seed=79)
    domains = {
        str(record["patient_email"]).rsplit("@", 1)[1]
        for record in cases
    }

    assert "example.test" not in domains
    assert len(domains) >= 5


def test_en_gb_profile_preserves_english_synthea_clinical_content() -> None:
    profile = resolve_generation_profile("en-GB")
    cases = profile.build_case_records(
        1,
        seed=8,
        synthea_seeds=[
            {
                "condition": "Viral sinusitis (disorder)",
                "medications": ["acetaminophen 325 mg oral tablet"],
                "observations": [("Body temperature", "38.1", "Cel")],
                "synthea_age_years": 41,
            }
        ],
    )
    rows = profile.render_case_records(cases, seed=8)
    assert "Viral sinusitis (disorder)" in rows[0]["text"]
    assert "acetaminophen 325 mg oral tablet" in rows[0]["text"]
    assert "patiënt" not in rows[0]["text"].lower()


def test_en_gb_reviewer_detects_a_removed_age_annotation() -> None:
    profile = resolve_generation_profile("en-GB")
    rows = profile.generate_documents(1, seed=42, require_synthea=False)
    rows[0]["spans"] = [
        span
        for span in rows[0]["spans"]
        if not (span["label"] == "Age_Birthdate" and span["text"].endswith("-year-old"))
    ]
    _, results, _ = profile.review_documents(rows)
    assert not results[0].passed
    assert any("Potential missed Age_Birthdate" in issue for issue in results[0].issues)


def test_generation_profile_manifest_pins_resources() -> None:
    profile = resolve_generation_profile("en_GB")
    manifest = profile.manifest()
    assert manifest["contract_version"] == GENERATION_PROFILE_CONTRACT
    assert manifest["profile_id"] == "en-GB"
    assert manifest["resources"]["maturity"] == "audited-language-pack-profile"
    assert manifest["allowed_labels"] == list(BERT_ENTITY_LABELS)
    assert manifest["resources"]["allowed_labels"] == list(BERT_ENTITY_LABELS)
    assert "Anonymize_Other" not in json.dumps(manifest)
    assert len(manifest["resources"]["resources"]["first_names"]["logical_sha256"]) == 64


def test_nl_nl_generation_profile_uses_netherlands_identity_resources() -> None:
    profile = resolve_generation_profile("nl-NL")
    record = profile.build_case_records(1, seed=7)[0]
    document = profile.generate_documents(1, seed=7, require_synthea=False)[0]
    bsn = record["identifiers"]["national_register"]

    assert profile.selection == "nl-NL"
    assert profile.default_preannotation_model == "stighellemans/meddeid-dutch-synth"
    assert record["language"] == "nl-NL"
    assert record["generation_profile"]["profile_id"] == "nl-NL"
    assert "+31 6 " in record["contact"]["patient_phone"]
    assert "@example." not in record["contact"]["patient_email"]
    assert len(bsn) == 9 and bsn.isdigit()
    assert (sum(int(value) * weight for value, weight in zip(bsn[:8], range(9, 1, -1))) - int(bsn[-1])) % 11 == 0
    assert document["metadata"]["lang"] == "nl-NL"
    assert document["metadata"]["generation_profile"] == "nl-NL"
    assert profile.manifest()["resources"]["language_resources"]["profile_id"] == "nl-NL"


def test_nl_nl_uses_dedicated_renderer_and_judge_for_all_document_types() -> None:
    profile = resolve_generation_profile("nl-NL")
    rows = profile.generate_documents(18, seed=41, require_synthea=False)
    reviewed, results, model_reviews = profile.review_documents(rows)
    combined_text = "\n".join(row["text"] for row in rows)

    assert reviewed == rows
    assert model_reviews == []
    assert all(result.passed for result in results)
    assert all(validate_record(row) == [] for row in rows)
    assert len({row["metadata"]["document_type"] for row in rows}) == 18
    assert all(
        row["metadata"]["generation_method"]
        == "netherlands-clinical-renderer-v1"
        for row in rows
    )
    assert {span["label"] for row in rows for span in row["spans"]} == set(
        BERT_ENTITY_LABELS
    )
    assert "SEH-verslag" in combined_text
    assert "Poliklinische consultnotitie" in combined_text
    assert "Wijkverpleegkundige rapportage" in combined_text
    assert "BIG-register" in combined_text


def test_nl_nl_case_and_rendered_text_have_no_belgian_context_leakage() -> None:
    profile = resolve_generation_profile("nl-NL")
    records = profile.build_case_records(36, seed=43)
    rows = profile.render_case_records(records, seed=43)
    serialized_records = json.dumps(records, ensure_ascii=False).lower()
    rendered_text = "\n".join(row["text"] for row in rows).lower()
    forbidden = (
        "riziv",
        "insz",
        "+32",
        "@example.be",
        "@example.nl",
        "universitair ziekenhuis antwerpen",
        "path-8405-be",
        "bhealth-",
        "bel-respira",
        "kinesist",
        "operatiekwartier",
        "basf antwerpen",
    )

    assert not any(value in serialized_records for value in forbidden)
    assert not any(value in rendered_text for value in forbidden)
    assert all("belgian_terms" not in record["style_profile"] for record in records)
    assert all("netherlands_terms" in record["style_profile"] for record in records)
    assert all(
        "catalog_medications" not in record["medical_details"] for record in records
    )


def test_nl_nl_judge_detects_missed_local_formats_and_belgian_leakage() -> None:
    profile = resolve_generation_profile("nl-NL")
    row = profile.generate_documents(1, seed=47, require_synthea=False)[0]
    row["spans"] = [
        span
        for span in row["spans"]
        if not (
            span["label"] == "Contactdetails"
            and span["text"].startswith("+31 6")
        )
    ]
    row["text"] += "\nAdministratieve bron: RIZIV."

    _, results, _ = profile.review_documents([row])

    assert not results[0].passed
    assert any("Potential missed Contactdetails" in issue for issue in results[0].issues)
    assert any("Belgian-context leakage" in issue for issue in results[0].issues)


def test_nl_nl_judge_checks_bsn_big_and_netherlands_postcode() -> None:
    profile = resolve_generation_profile("nl-NL")
    record = profile.build_case_records(1, seed=53)[0]
    row = profile.render_case_records([record], seed=53)[0]
    removed_values = {
        record["identifiers"]["national_register"],
        record["identifiers"]["caregiver_registry"],
        record["patient_address"]["text"],
    }
    row["spans"] = [
        span for span in row["spans"] if span["text"] not in removed_values
    ]

    _, results, _ = profile.review_documents([row])

    assert not results[0].passed
    issues = "\n".join(results[0].issues)
    assert record["identifiers"]["national_register"] in issues
    assert record["identifiers"]["caregiver_registry"] in issues
    assert record["patient_address"]["text"].split(", ", 1)[1].split(" ", 2)[0] in issues


def test_cli_generates_en_gb_dataset_and_profile_manifest(tmp_path) -> None:
    output = tmp_path / "english.jsonl"
    report = tmp_path / "english-report.md"
    assert (
        main(
            [
                "generate",
                "--language-profile",
                "en-GB",
                "--count",
                "6",
                "--seed",
                "91",
                "--output",
                str(output),
                "--judge-report",
                str(report),
            ]
        )
        == 0
    )

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 6
    assert manifest["contracts"]["language_profile"] == "en-GB"
    assert manifest["generation_profile"]["profile_id"] == "en-GB"
    assert manifest["language_profile"]["profile_id"] == "en-GB"
    assert manifest["language_profile"]["resources"]["manifest_version"] == "meddeid.language-resources.v2"
    assert manifest["generation_profile"]["allowed_labels"] == list(BERT_ENTITY_LABELS)
    assert "Anonymize_Other" not in json.dumps(manifest["generation_profile"])
    assert "Documents passed deterministic judge: 6/6" in report.read_text()


def test_cli_default_generation_remains_nl_be(tmp_path) -> None:
    output = tmp_path / "dutch.jsonl"
    assert main(["generate", "--count", "2", "--output", str(output)]) == 0
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["contracts"]["language_profile"] == "nl-BE"
    assert manifest["generation_profile"]["profile_id"] == "nl-BE"


def test_cli_round_trips_en_gb_case_records_without_repeat_selection(tmp_path) -> None:
    cases = tmp_path / "cases.jsonl"
    output = tmp_path / "rendered.jsonl"
    assert (
        main(
            [
                "build-cases",
                "--language-profile",
                "en-GB",
                "--count",
                "6",
                "--output",
                str(cases),
            ]
        )
        == 0
    )
    assert main(["render-cases", str(cases), "--output", str(output)]) == 0
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["metadata"]["renderer"] for row in rows} == {
        "clinic_note",
        "discharge_summary",
        "emergency_note",
        "referral_letter",
        "laboratory_report",
        "nursing_note",
    }


def test_unknown_generation_profile_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="built-ins: nl-BE, nl-NL, en-GB, en-US"):
        resolve_generation_profile("es-ES")


def test_bare_english_is_rejected_as_ambiguous() -> None:
    with pytest.raises(ValueError, match="en-GB, en-US"):
        resolve_generation_profile("en")


def test_gb_acute_hospital_sampler_excludes_screening_programmes() -> None:
    assert not _looks_like_major_acute_hospital(
        "LANCS BOWEL CANCER SCREENING - BURNLEY GENERAL HOSPITAL", "England"
    )
    assert not _looks_like_major_acute_hospital(
        "Regional Screening Programme at Example General Hospital", "England"
    )
    assert _looks_like_major_acute_hospital("Example District General Hospital", "England")


def test_gb_general_hospital_sampler_excludes_child_only_services() -> None:
    assert not any(
        any(term in hospital.lower() for term in (" child", "paediatric", "pediatric"))
        for hospitals in HOSPITALS_BY_REGION.values()
        for hospital in hospitals
    )


def test_person_name_pools_exclude_dangling_joiner_fragments() -> None:
    for value in (*GB_FIRST_NAMES, *US_FIRST_NAMES):
        assert value[-1].isalnum()
        assert not any(fragment in value for fragment in ("- ", " -", "' ", " '", "’ ", " ’"))


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_adult_cases_never_sample_pediatric_departments(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    adult_cases = [
        case for case in profile.build_case_records(500, seed=20260820)
        if case["age_years"] >= 18
    ]

    assert adult_cases
    assert not any(
        any(term in case["department"].lower() for term in ("paediatric", "pediatric", "child", "neonat"))
        for case in adult_cases
    )


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_anonymize_other_is_rejected_at_case_render_review_and_export_boundaries(
    profile_id: str, tmp_path
) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(1, seed=13)
    cases[0]["spans"] = [{"begin": 0, "end": 1, "label": "Anonymize_Other"}]
    with pytest.raises(ValueError, match="must not contain Anonymize_Other"):
        profile.render_case_records(cases, seed=13)

    docs = profile.generate_documents(1, seed=13, require_synthea=False)
    docs[0]["spans"].append(
        {"begin": 0, "end": 1, "text": docs[0]["text"][:1], "label": "Anonymize_Other"}
    )
    _, results, _ = profile.review_documents(docs)
    assert not results[0].passed
    assert any("Disallowed synthetic labels" in issue for issue in results[0].issues)
    with pytest.raises(ValueError, match="synthetic export must not contain Anonymize_Other"):
        _write_jsonl(docs, tmp_path / "forbidden.jsonl")


def test_us_profile_uses_us_dates_addresses_contacts_and_resource_provenance() -> None:
    profile = resolve_generation_profile("en_US")
    rows = profile.generate_documents(12, seed=27, require_synthea=False)
    text = "\n".join(row["text"] for row in rows)
    assert "Emergency department note" in text
    assert "A&E note" not in text
    assert "Organization:" in text
    assert "Organisation:" not in text
    assert "555-01" in text
    assert all(row["metadata"]["lookup_source"].endswith("en-US locked resources") for row in rows)
    assert any(", PR 00901" in row["text"] or ", DC 20001" in row["text"] or ", GU 96910" in row["text"] for row in rows)


def test_us_case_metadata_replaces_the_inherited_gb_address_region() -> None:
    profile = resolve_generation_profile("en-US")
    cases = profile.build_case_records(56, seed=29)
    gb_regions = {"England", "Scotland", "Wales", "Northern Ireland"}

    assert not {case["address_region"] for case in cases} & gb_regions
    assert {"Puerto Rico", "Guam", "District of Columbia"} <= {
        case["address_region"] for case in cases
    }


def test_us_interstate_display_closes_source_separator_space() -> None:
    from meddeid_data.generation_profile_en_us import _street_display

    assert _street_display("I- 710") == "I-710"


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_case_dependencies_keep_af_anticoagulation_and_age_plausible(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(500, seed=31)
    atrial_fibrillation = [
        case for case in cases if case["condition"]["name"].lower() == "atrial fibrillation"
    ]

    assert atrial_fibrillation
    assert all(case["age_years"] >= 65 for case in atrial_fibrillation)
    assert all("apixaban" in case["condition"]["medications"] for case in atrial_fibrillation)


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_case_dependencies_keep_copd_age_plausible(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(500, seed=31)
    copd = [
        case
        for case in cases
        if "chronic obstructive pulmonary disease" in case["condition"]["name"].lower()
    ]

    assert copd
    assert all(case["age_years"] >= 45 for case in copd)


def test_us_case_medications_use_us_systemic_steroid_name() -> None:
    profile = resolve_generation_profile("en-US")
    cases = profile.build_case_records(500, seed=33)
    medications = {
        medication
        for case in cases
        for medication in case["condition"].get("medications", [])
    }

    assert "prednisone" in medications
    assert "prednisolone" not in medications
    serialized_conditions = json.dumps(
        [case["condition"] for case in cases], ensure_ascii=False
    ).lower()
    assert not {
        "orthopnoea",
        "dyspnoea",
        "oedema",
        "haematuria",
        "ischaemic",
        "gastro-oesophageal",
        "generalised",
        "apnoea",
        "co-beneldopa",
        "ramipril",
    } & set(serialized_conditions.split('"'))


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_case_dates_cover_past_contemporary_and_future_shift_ranges(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(64, seed=35)
    years = {int(case["encounter_date"][:4]) for case in cases}

    assert any(year < 2000 for year in years)
    assert any(2000 <= year <= 2025 for year in years)
    assert any(year > 2025 for year in years)


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_case_birth_dates_equal_exact_completed_calendar_age(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(500, seed=37)

    for case in cases:
        encounter = date.fromisoformat(case["encounter_date"])
        birth = date.fromisoformat(case["birth_date"])
        completed_age = encounter.year - birth.year - int(
            (encounter.month, encounter.day) < (birth.month, birth.day)
        )
        assert completed_age == case["age_years"]


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_production_blocks_have_explicit_balanced_pediatric_strata(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(250, seed=20260820, start_index=250)
    pediatric = [case for case in cases if case["age_years"] < 18]

    assert len(pediatric) == 41
    assert {
        band: sum(case["age_group"] == band for case in pediatric)
        for band in ("infant", "early_childhood", "school_age", "adolescent")
    } == {
        "infant": 5,
        "early_childhood": 8,
        "school_age": 12,
        "adolescent": 16,
    }
    assert {case["age_unit"] for case in pediatric} == {
        "days", "weeks", "months", "years"
    }
    assert {case["document_type"] for case in pediatric} == {
        "clinic_note", "discharge_summary", "emergency_note",
        "referral_letter", "laboratory_report", "nursing_note",
    }
    neonates = [case for case in pediatric if case["clinical_age_group"] == "neonate"]
    assert len(neonates) == 2
    assert {case["condition"]["name"] for case in neonates} == {"neonatal jaundice"}


@pytest.mark.parametrize("profile_id", ["en-GB", "en-US"])
def test_non_year_pediatric_ages_are_consistent_with_birth_dates(profile_id: str) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(250, seed=20260820, start_index=250)

    for case in cases:
        if case["age_unit"] == "years":
            continue
        encounter = date.fromisoformat(case["encounter_date"])
        birth = date.fromisoformat(case["birth_date"])
        if case["age_unit"] == "days":
            assert (encounter - birth).days == case["age_value"]
        elif case["age_unit"] == "weeks":
            assert (encounter - birth).days == case["age_value"] * 7
        else:
            completed_months = (
                (encounter.year - birth.year) * 12
                + encounter.month
                - birth.month
                - int(encounter.day < birth.day)
            )
            assert completed_months == case["age_value"]


@pytest.mark.parametrize("profile_id,bad_phone", [("en-GB", "07123 456789"), ("en-US", "(202) 555-0200")])
def test_renderer_rejects_unapproved_synthetic_phones(profile_id: str, bad_phone: str) -> None:
    profile = resolve_generation_profile(profile_id)
    cases = profile.build_case_records(1, seed=19)
    cases[0]["patient_phone"] = bad_phone
    with pytest.raises(ValueError, match="unapproved telephone number"):
        profile.render_case_records(cases)


def test_plugin_cannot_return_an_incompatible_profile(monkeypatch) -> None:
    class IncompatibleEntryPoint:
        name = "incompatible-test-provider"

        @staticmethod
        def load():
            return lambda profile_id: resolve_generation_profile("en-GB")

    monkeypatch.setattr(
        generation_profiles,
        "_providers",
        lambda: [IncompatibleEntryPoint()],
    )

    with pytest.raises(ValueError, match="does not satisfy 'fr-FR'"):
        resolve_generation_profile("fr-FR")
