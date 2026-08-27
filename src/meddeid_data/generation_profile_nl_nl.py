"""Deterministic Netherlands-Dutch synthetic generation profile.

This profile reuses only the language-neutral clinical case structure. Identity,
administrative conventions, document rendering, and judging are owned by the
Netherlands profile. It is executable support code; no corpus is generated as
part of the package.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from meddeid_language_nl import get_profile

from .clinical_cases import generate_case_records
from .generation_profiles import GENERATION_PROFILE_CONTRACT, GenerationProfile
from .email_domains import email_address
from .judge_nl_nl import judge_documents_nl_nl
from .lookups import LookupSampler, full_name
from .renderer_nl_nl import RENDERER_ID, render_documents_from_case_records_nl_nl


PROFILE_ID = "nl-NL"

NETHERLANDS_OTHER_ORGANIZATIONS = (
    "Noordzee Logistiek",
    "Stichting Groene Buurt",
    "Rijnstad Onderwijs",
    "Vereniging De Waterkant",
    "Polder Techniek",
    "Oranjehaven Administratie",
)

NETHERLANDS_TERMINOLOGY = {
    "spoedgevallen": "spoedeisende hulp",
    "spoedarts": "SEH-arts",
    "kinesitherapie": "fysiotherapie",
    "kinesist": "fysiotherapeut",
    "raadpleging": "polikliniekbezoek",
    "terugbrief": "specialistenbrief",
    "operatiekwartier": "operatiekamercomplex",
    "labo": "laboratorium",
    "mutualiteit": "zorgverzekeraar",
    "HAIO": "huisarts in opleiding",
    "ASO": "AIOS",
    "RIZIV": "BIG-register",
    "INSZ": "BSN",
}


def _bsn(rng: random.Random) -> str:
    """Generate a synthetic 9-digit value satisfying the Dutch 11-test."""

    while True:
        digits = [rng.randrange(10) for _ in range(8)]
        check = sum((9 - index) * value for index, value in enumerate(digits)) % 11
        if check < 10:
            return "".join(map(str, [*digits, check]))


def _person(sampler: LookupSampler) -> dict[str, Any]:
    name, sources = full_name(sampler)
    return {"name": name, "source_paths": sources}


def _localize_text(value: str) -> str:
    localized = value
    for source, target in NETHERLANDS_TERMINOLOGY.items():
        localized = re.sub(rf"\b{re.escape(source)}\b", target, localized, flags=re.I)
    return localized


def _localize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _localize_text(value)
    if isinstance(value, list):
        return [_localize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_localize_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _localize_value(item) for key, item in value.items()}
    return value


def _netherlands_identifiers(rng: random.Random) -> dict[str, str]:
    year = rng.randrange(2024, 2028)

    def token(length: int) -> str:
        return "".join(
            rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(length)
        )

    return {
        "patient_number": f"EPD-{rng.randrange(1000000, 9999999)}",
        "his_patient_id": f"ZIS-{rng.randrange(1000000, 9999999)}",
        "national_register": _bsn(rng),
        "lab_accession": f"LAB-NL-{rng.randrange(100000, 999999)}",
        "pathology_accession": f"PA-NL-{year}-{rng.randrange(10000, 99999)}",
        "imaging_key": f"PACS-NL-{token(4)}-{rng.randrange(100000, 999999)}",
        "operating_room_case": f"OK-NL-{year}-{rng.randrange(10000, 99999)}",
        "patient_file": f"DOSSIER-NL-{token(6)}",
        "study_name": f"STUDIE-NL-{token(5)}",
        "study_protocol_id": f"PROTOCOL-NL-{year}-{token(4)}",
        "study_protocol_name": f"ONDERZOEK-NL-{token(6)}",
        "study_reference": f"STUDIE-NL-{rng.randrange(10000, 99999)}",
        "cfdna_reference": f"CFDNA {rng.randrange(10000, 99999)}",
        "crisis_card": f"DOSSIER-NL-SEH-{rng.randrange(1000, 9999)}",
        "device_serial": f"DEVICE-NL-{token(4)}-{rng.randrange(100000, 999999)}",
        "caregiver_registry": f"BIG-{rng.randrange(10000000000, 99999999999)}",
        "material_lot": f"LOT-{rng.randrange(10000, 99999)}",
    }


def _localize_case(record: dict[str, Any], *, seed: int, index: int) -> dict[str, Any]:
    rng = random.Random(f"{PROFILE_ID}|{seed}|{index}")
    sampler = LookupSampler(seed=seed + index * 17, profile_id=PROFILE_ID)
    for field in ("patient", "caregiver", "secondary_caregiver", "relative"):
        record[field] = _person(sampler)

    street, street_source = sampler.street()
    locality, locality_source = sampler.locality()
    postcode = f"{rng.randrange(1000, 9999)} {rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}"
    record["patient_address"] = {
        "text": f"{street} {rng.randrange(1, 180)}, {postcode} {locality}",
        "source_paths": {"street": street_source, "locality": locality_source},
    }
    record["hospital"] = list(sampler.hospital())
    record["healthcare_institution"] = list(sampler.healthcare_institution())
    record["caregiver_locality"] = list(sampler.locality())
    record["other_location"] = list(sampler.locality())
    record["other_org"] = NETHERLANDS_OTHER_ORGANIZATIONS[
        index % len(NETHERLANDS_OTHER_ORGANIZATIONS)
    ]

    patient_name = record["patient"]["name"]
    relative_name = record["relative"]["name"]
    record["contact"] = {
        "patient_phone": f"+31 6 {rng.randrange(10000000, 99999999)}",
        "patient_email": email_address(
            PROFILE_ID,
            f"{patient_name.split()[0].lower()}.{patient_name.split()[-1].lower()}",
        ),
        "relative_phone": f"+31 6 {rng.randrange(10000000, 99999999)}",
        "relative_email": email_address(
            PROFILE_ID,
            f"{relative_name.split()[0].lower()}.{relative_name.split()[-1].lower()}",
        ),
        "caregiver_internal_phone": f"toestel {rng.randrange(1000, 9999)}",
    }
    record["identifiers"] = _netherlands_identifiers(rng)

    for field in (
        "department",
        "condition",
        "profession",
        "note_style",
        "annotation_policy",
        "style_profile",
        "medical_details",
    ):
        if field in record:
            record[field] = _localize_value(record[field])
    style_profile = record.get("style_profile")
    if isinstance(style_profile, dict):
        style_profile.pop("belgian_terms", None)
        style_profile["netherlands_terms"] = [
            "polikliniek",
            "spoedeisende hulp",
            "huisarts",
            "AIOS",
        ]
    medical_details = record.get("medical_details")
    if isinstance(medical_details, dict):
        # The shared case builder currently samples these from a Belgian CNK
        # catalogue. The dedicated Netherlands renderer never consumes them.
        medical_details.pop("catalog_medications", None)
    record["language"] = PROFILE_ID
    record["generation_profile"] = {
        "contract_version": GENERATION_PROFILE_CONTRACT,
        "profile_id": PROFILE_ID,
    }
    record["pii_policy"] = {
        "source": "Netherlands lookup lists and generated Netherlands identifiers",
        "note": "All person/contact/identifier combinations are synthetic; public institutions may be real.",
    }
    return record


def build_case_records(
    count: int,
    *,
    seed: int = 20260508,
    synthea_seeds: list[dict[str, Any]] | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    records = generate_case_records(
        count, seed=seed, synthea_seeds=synthea_seeds, start_index=start_index
    )
    return [
        _localize_case(record, seed=seed, index=start_index + index)
        for index, record in enumerate(records)
    ]


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
    from .synthea_adapter import load_or_generate_synthea_csv_seeds

    seeds, _ = load_or_generate_synthea_csv_seeds(
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
        build_case_records(count, seed=seed, synthea_seeds=seeds), seed=seed
    )


def render_case_records(
    records: list[dict[str, Any]], *, seed: int = 20260508
) -> list[dict[str, Any]]:
    return render_documents_from_case_records_nl_nl(records, seed=seed)


def _resource_manifest() -> dict[str, Any]:
    language_manifest = get_profile(PROFILE_ID).manifest()
    return {
        "manifest_version": "meddeid.generation-resources.v1",
        "package": "meddeid-data",
        "profile_id": PROFILE_ID,
        "language_resources": language_manifest["resources"],
        "generator": RENDERER_ID,
        "judge": "netherlands-clinical-judge-v1",
        "identifier_policy": {
            "national": "synthetic BSN satisfying the 11-test",
            "phone": "synthetic +31 mobile format",
            "email": "deterministic locale-aware consumer-domain mix",
        },
        "document_types": 18,
    }


NL_NL_GENERATION_PROFILE = GenerationProfile(
    profile_id=PROFILE_ID,
    language_tags=(PROFILE_ID,),
    description="Dutch clinical notes in the Netherlands setting",
    generate_documents=generate_documents,
    build_case_records=build_case_records,
    render_case_records=render_case_records,
    review_documents=judge_documents_nl_nl,
    resource_manifest_provider=_resource_manifest,
    language_manifest_provider=lambda: get_profile(PROFILE_ID).manifest(),
    default_preannotation_model="stighellemans/meddeid-dutch-synth",
)
