"""Deterministic English (United States and territories) generation profile."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from meddeid_core import BERT_ENTITY_LABELS
from meddeid_language_en import get_profile as get_language_profile
from meddeid_language_en import lookup_records, lookup_source, lookup_values

from .generation_profile_en import (
    DOCUMENT_TYPES,
    _age_display,
    _clean_person_values,
    _clean_values,
    _preferred_values,
    build_case_records as build_gb_case_records,
    render_case_records,
    review_documents,
)
from .generation_profiles import GENERATION_PROFILE_CONTRACT, GenerationProfile
from .email_domains import email_address
from .identifier_formats import (
    format_english_identifier,
    synthetic_national_identifier,
)
from .organization_formats import format_healthcare_organization
from .synthea_adapter import load_or_generate_synthea_csv_seeds

PROFILE_ID = "en-US"
ALLOWED_LABELS = tuple(BERT_ENTITY_LABELS)
FIRST_NAMES = _clean_person_values(lookup_values(PROFILE_ID, "first_names"))
FAMILY_NAMES = _clean_person_values(lookup_values(PROFILE_ID, "family_names"))
STREET_RECORDS = lookup_records(PROFILE_ID, "streets")
STREETS = _clean_values(tuple(record["value"] for record in STREET_RECORDS))
POSTAL_LOCALITIES = lookup_records(PROFILE_ID, "postal_localities")
POSTAL_CODE_LOCALITIES = lookup_records(PROFILE_ID, "postal_code_localities")


def _general_hospital_type(record: dict[str, Any]) -> bool:
    value = record.get("attributes", {}).get("hospital_type")
    if value is None:
        return True
    values = value if isinstance(value, list) else [value]
    return any(item in {"Acute Care Hospitals", "Critical Access Hospitals"} for item in values)


HOSPITAL_RECORDS = tuple(
    record
    for record in lookup_records(PROFILE_ID, "hospitals")
    if ",THE" not in record["value"].upper()
    and not record["value"].rstrip().endswith(("-", "/", "&"))
    and _general_hospital_type(record)
    if not any(
        term in record["value"].upper()
        for term in (
            "CHILDREN'S", "CHILDRENS", "PEDIATRIC", "PSYCHIATRIC", "BEHAVIORAL",
            "ORTHOPAEDIC", "ORTHOPEDIC", "EYE HOSPITAL", "REHABILITATION HOSPITAL",
            "MENTAL HEALTH",
        )
    )
)
HOSPITALS = tuple(record["value"] for record in HOSPITAL_RECORDS)
OTHER_ORGANISATIONS = lookup_values(PROFILE_ID, "other_organisations")
OTHER_ORGANISATION_SUFFIXES = (
    "Community College",
    "Community Center",
    "Housing Authority",
    "Public Library",
    "School District",
    "Senior Center",
    "Transit Authority",
    "Youth Services",
)
PROFESSIONS = _preferred_values(
    _clean_values(lookup_values(PROFILE_ID, "professions"), reject_leading_digit=True),
    (
        "accountant", "barista", "bread baker", "business office manager", "bus driver",
        "child care worker", "civil engineer", "clinical pharmacist", "commercial baker helper",
        "dental receptionist", "elementary school teacher", "front desk receptionist",
        "hairdresser", "high school teacher", "hospital pharmacist", "journalist",
        "legal administrative assistant", "paralegal", "pastry baker", "geriatric physical therapist",
        "plumber", "probate lawyer", "real estate administrative assistant",
        "software developer", "state highway police officer", "trial lawyer",
        "building carpenter", "children's librarian", "agricultural equipment mechanic",
        "certified personal chef", "architectural drafter", "auto electrician",
    ),
)
EARLY_CAREER_PROFESSIONS = tuple(
    value
    for value in PROFESSIONS
    if value.casefold()
    in {
        "barista",
        "commercial baker helper",
        "dental receptionist",
        "front desk receptionist",
        "hairdresser",
        "legal administrative assistant",
        "pastry baker",
        "real estate administrative assistant",
        "building carpenter",
        "auto electrician",
    }
)
if not EARLY_CAREER_PROFESSIONS:
    raise RuntimeError("English US profession resources have no early-career values")
DEPARTMENTS = lookup_values(PROFILE_ID, "departments")
ADULT_DEPARTMENTS = tuple(
    department
    for department in DEPARTMENTS
    if not any(
        term in department.lower()
        for term in ("paediatric", "pediatric", "child", "neonat")
    )
)

AREA_CODES = {
    "PR": ("787", "939"),
    "VI": ("340",),
    "GU": ("671",),
    "AS": ("684",),
    "MP": ("670",),
    "DC": ("202",),
}
REGION_BY_CODE = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
    "VI": "U.S. Virgin Islands", "GU": "Guam", "AS": "American Samoa",
    "MP": "Northern Mariana Islands",
}
ADDRESS_STATE_SEQUENCE = (
    "PR", "DC", "GU", "VI", "AS", "MP",
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY",
)


def _postal_options() -> dict[str, tuple[tuple[str, str], ...]]:
    grouped: dict[str, set[tuple[str, str]]] = {}
    composite = re.compile(r"^(\d{5})\s+(.+?)\s+([A-Z]{2})$")
    for record in POSTAL_CODE_LOCALITIES:
        attributes = record.get("attributes", {})
        state = attributes.get("state")
        locality = attributes.get("locality")
        value = str(record["value"])
        if isinstance(state, str) and isinstance(locality, str) and value.isdigit():
            grouped.setdefault(state, set()).add((locality, value))
            continue
        match = composite.fullmatch(value)
        if match:
            grouped.setdefault(match.group(3), set()).add((match.group(2), match.group(1)))
    return {state: tuple(sorted(options)) for state, options in grouped.items()}


POSTAL_OPTIONS_BY_STATE = _postal_options()
STREETS_BY_STATE = {
    state: tuple(
        record["value"]
        for record in STREET_RECORDS
        if region in record.get("regions", ())
        and 1 < len(record["value"]) <= 90
        and record["value"][0].isalnum()
        and not any(character in record["value"] for character in ('"', "{", "}"))
    )
    for state, region in REGION_BY_CODE.items()
}
HOSPITALS_BY_STATE = {
    state: tuple(
        record["value"]
        for record in HOSPITAL_RECORDS
        if region in record.get("regions", ())
    )
    for state, region in REGION_BY_CODE.items()
}


def _rng(seed: int, index: int) -> random.Random:
    return random.Random(f"{PROFILE_ID}|1|{seed}|{index}")


def _person(rng: random.Random) -> dict[str, str]:
    return {
        "given_name": rng.choice(FIRST_NAMES),
        "family_name": rng.choice(FAMILY_NAMES),
    }


def _localize(value: Any) -> Any:
    replacements = (
        ("paediatrics", "pediatrics"),
        ("paediatric", "pediatric"),
        ("anaemia", "anemia"),
        ("haemoglobin", "hemoglobin"),
        ("orthopnoea", "orthopnea"),
        ("dyspnoea", "dyspnea"),
        ("oedema", "edema"),
        ("haematuria", "hematuria"),
        ("ischaemic", "ischemic"),
        ("gastro-oesophageal", "gastroesophageal"),
        ("generalised", "generalized"),
        ("localised", "localized"),
        ("apnoea", "apnea"),
        ("paracetamol", "acetaminophen"),
        ("salbutamol", "albuterol"),
        ("beclometasone", "beclomethasone"),
        ("intravenous co-amoxiclav", "ampicillin-sulbactam"),
        ("co-amoxiclav", "amoxicillin-clavulanate"),
        ("flucloxacillin", "dicloxacillin"),
        ("cyclizine", "doxylamine-pyridoxine"),
        ("prednisolone", "prednisone"),
        ("aspirin 75 mg", "aspirin 81 mg"),
        ("ramipril", "lisinopril"),
        ("bisoprolol", "metoprolol succinate"),
        ("fondaparinux", "enoxaparin"),
        ("glyceryl trinitrate", "nitroglycerin"),
        ("co-beneldopa", "carbidopa-levodopa"),
        ("ferrous fumarate", "ferrous sulfate"),
        ("nitrofurantoin modified-release", "nitrofurantoin monohydrate/macrocrystals"),
        ("topical ibuprofen", "topical diclofenac"),
        ("general medicine", "internal medicine"),
    )
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target).replace(source.title(), target.title())
        return value
    if isinstance(value, list):
        return [_localize(item) for item in value]
    if isinstance(value, dict):
        return {key: _localize(item) for key, item in value.items()}
    return value


def _street_display(value: str) -> str:
    """Normalize source route separators without changing the proper name."""

    return re.sub(r"\bI-\s+(?=\d)", "I-", value)


def _address(rng: random.Random, index: int) -> tuple[str, str]:
    state = ADDRESS_STATE_SEQUENCE[index % len(ADDRESS_STATE_SEQUENCE)]
    options = POSTAL_OPTIONS_BY_STATE.get(state)
    if not options:
        raise RuntimeError(f"en-US resource pack has no postal geography for {state}")
    locality, postal_code = rng.choice(options)
    prefix = "URB San Patricio, " if state == "PR" else ""
    street = rng.choice(STREETS_BY_STATE.get(state) or STREETS)
    # TIGER route names occasionally encode an interstate as ``I- 710``.
    # Publication 28-style display closes that accidental separator space.
    street = _street_display(street)
    address = (
        f"{prefix}{rng.randrange(1, 900)} {street}, "
        f"{locality}, {state} {postal_code}"
    )
    return address, state


def _phone(rng: random.Random, state: str) -> str:
    area = rng.choice(AREA_CODES.get(state, ("202", "212", "312", "617")))
    return f"({area}) 555-{rng.randrange(100, 200):04d}"


def _email(person: dict[str, str], rng: random.Random) -> str:
    local = f"{person['given_name']}.{person['family_name']}".lower()
    numbered_local = f"{local}{rng.randrange(10, 99)}"
    return email_address(PROFILE_ID, numbered_local)


def _other_organisation(rng: random.Random, state: str) -> str:
    locality = rng.choice(POSTAL_OPTIONS_BY_STATE[state])[0]
    return f"{locality} {rng.choice(OTHER_ORGANISATION_SUFFIXES)}"


def build_case_records(
    count: int,
    *,
    seed: int = 20260508,
    synthea_seeds: list[dict[str, Any]] | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    # Use a locale-specific clinical seed so GB and US records at the same
    # profile index are not paired variants of one shared case backbone.
    us_clinical_seed = seed ^ 0x5EED5A17
    records = build_gb_case_records(
        count,
        seed=us_clinical_seed,
        synthea_seeds=synthea_seeds,
        start_index=start_index,
    )
    for offset, record in enumerate(records):
        index = start_index + offset
        rng = _rng(seed, index)
        patient = _person(rng)
        caregiver = _person(rng)
        relative = _person(rng)
        patient_address, state = _address(rng, index)
        caregiver_locality = rng.choice(POSTAL_OPTIONS_BY_STATE[state])[0]
        national_identifier = synthetic_national_identifier(PROFILE_ID, str(index))
        other_locality = rng.choice(POSTAL_OPTIONS_BY_STATE[state])[0]
        source_hospital = rng.choice(HOSPITALS_BY_STATE.get(state) or HOSPITALS)
        record.update(
            {
                "generation_profile": {
                    "contract_version": GENERATION_PROFILE_CONTRACT,
                    "profile_id": PROFILE_ID,
                },
                "language": PROFILE_ID,
                "patient": patient,
                "caregiver": caregiver,
                "relative": relative,
                "patient_address": patient_address,
                "address_region": REGION_BY_CODE[state],
                "caregiver_locality": caregiver_locality,
                "other_locality": other_locality,
                "hospital": format_healthcare_organization(
                    PROFILE_ID, source_hospital, str(index)
                ),
                "other_organisation": _other_organisation(rng, state),
                "profession": rng.choice(
                    EARLY_CAREER_PROFESSIONS
                    if int(record.get("age_years", 18)) < 21
                    else PROFESSIONS
                ),
                "department": (
                    _localize(record["department"])
                    if int(record.get("age_years", 18)) < 18
                    else ADULT_DEPARTMENTS[index % len(ADULT_DEPARTMENTS)]
                ),
                "condition": _localize(record["condition"]),
                "age_display": _age_display(
                    int(record["age_value"]), str(record["age_unit"]), PROFILE_ID
                ),
                "patient_phone": _phone(rng, state),
                "patient_email": _email(patient, rng),
                "patient_id": format_english_identifier(
                    PROFILE_ID, "patient.mrn", str(index)
                ),
                "caregiver_id": format_english_identifier(
                    PROFILE_ID, "caregiver.professional_id", str(index)
                ),
                "report_id": format_english_identifier(
                    PROFILE_ID, "patient.report_id", str(index)
                ),
                "national_id": national_identifier["value"],
                "national_id_type": national_identifier["kind"],
                "national_id_label": national_identifier["field_label"],
                "pii_policy": {
                    "source": "versioned meddeid-language-en en-US resources",
                    "note": (
                        "Person identities are independently recombined; public institutions "
                        "and geographic entities may be real; Synthea PII is not used."
                    ),
                },
            }
        )
    return records


def generate_documents(
    count: int,
    *,
    seed: int = 20260508,
    synthea_csv_dir: Path | None = None,
    auto_synthea: bool = False,
    synthea_repo_dir: Path = Path("external/synthea"),
    synthea_population: int | None = None,
    force_synthea: bool = False,
    require_synthea: bool = False,
) -> list[dict[str, Any]]:
    synthea_seeds, _ = load_or_generate_synthea_csv_seeds(
        synthea_csv_dir,
        limit=count,
        auto_generate=auto_synthea,
        synthea_repo_dir=synthea_repo_dir,
        population=synthea_population,
        seed=seed,
        force=force_synthea,
        require=require_synthea,
    )
    return render_case_records(
        build_case_records(count, seed=seed, synthea_seeds=synthea_seeds), seed=seed
    )


def _resource_manifest() -> dict[str, Any]:
    language_manifest = get_language_profile(PROFILE_ID).manifest()
    return {
        "manifest_version": "meddeid.generation-resources.v1",
        "package": "meddeid-data",
        "package_version": "0.4.0",
        "profile_id": PROFILE_ID,
        "maturity": "audited-language-pack-profile",
        "resources": language_manifest["resources"]["resources"],
        "language_resources": language_manifest["resources"],
        "allowed_labels": list(ALLOWED_LABELS),
        "provenance": {
            "description": (
                "Audited en-US language-pack resources covering the states, DC, "
                "Puerto Rico, USVI, Guam, American Samoa, and CNMI."
            ),
            "lookup_source": lookup_source(PROFILE_ID),
            "fictional_phone_source": "NANPA 555-0100 through 555-0199",
            "synthetic_identifier_formats": (
                "deterministic mix dominated by numeric-only values, with compact, "
                "grouped, slash, and short-prefix internal formats"
            ),
        },
        "document_types": list(DOCUMENT_TYPES),
    }


EN_US_GENERATION_PROFILE = GenerationProfile(
    profile_id=PROFILE_ID,
    language_tags=(PROFILE_ID,),
    description="English clinical notes in the United States and territories",
    generate_documents=generate_documents,
    build_case_records=build_case_records,
    render_case_records=render_case_records,
    review_documents=review_documents,
    resource_manifest_provider=_resource_manifest,
    language_manifest_provider=lambda: get_language_profile(PROFILE_ID).manifest(),
)
