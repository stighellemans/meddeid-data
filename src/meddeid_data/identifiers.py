"""Identifier and lookalike code generation.

The annotation guideline distinguishes patient/caregiver identifiers from
medical codes that only look identifying. This module keeps that distinction
explicit for renderers and judge prompts.
"""

from __future__ import annotations

import random
from datetime import date


def patient_number(rng: random.Random) -> str:
    return rng.choice(
        [
            f"{rng.randrange(10000000, 99999999)}",
            f"P{rng.randrange(1000000, 9999999)}",
            f"MRN-{rng.randrange(100000, 999999)}",
            f"EPD-{_short_token(rng, 2)}-{rng.randrange(100000, 999999)}",
            f"UZ-{rng.randrange(2024, 2027)}-{rng.randrange(100000, 999999)}",
        ]
    )


def his_patient_id(rng: random.Random) -> str:
    return str(rng.randrange(100000000, 999999999))


def national_register(rng: random.Random, birth_date: date | None = None) -> str:
    if birth_date is None:
        yy = rng.randrange(45, 99)
        mm = rng.randrange(1, 13)
        dd = rng.randrange(1, 29)
    else:
        yy = birth_date.year % 100
        mm = birth_date.month
        dd = birth_date.day
    seq = rng.randrange(1, 999)
    chk = rng.randrange(1, 98)
    return f"{yy:02d}.{mm:02d}.{dd:02d}-{seq:03d}.{chk:02d}"


def phone(rng: random.Random) -> str:
    return f"+32 4{rng.randrange(70, 99)} {rng.randrange(10, 99)} {rng.randrange(10, 99)} {rng.randrange(10, 99)}"


def internal_phone(rng: random.Random) -> str:
    return rng.choice(
        [
            f"{rng.randrange(1000, 9999)}",
            f"toestel {rng.randrange(1000, 9999)}",
            f"lijn {rng.randrange(10, 99)} {rng.randrange(10, 99)}",
        ]
    )


def email(name: str, rng: random.Random) -> str:
    base = ".".join(part.lower().replace("'", "") for part in name.split()[:2])
    return f"{base}{rng.randrange(10, 99)}@example.be"


def caregiver_id(rng: random.Random) -> str:
    return rng.choice(
        [
            f"{rng.randrange(10000000000, 99999999999)}",
            f"DR-{rng.randrange(100000, 999999)}",
            f"VRN-{rng.randrange(100000, 999999)}",
            f"RIZIV Nr. {rng.randrange(10000000000, 99999999999)}",
        ]
    )


def material_lot_number(rng: random.Random) -> str:
    """Material lot numbers are ID:Caregiver per the guideline examples."""

    return f"LOT-{rng.randrange(10000, 99999)}"


def study_protocol_identifier(rng: random.Random) -> str:
    return rng.choice(
        [
            f"PROT-{rng.randrange(2024, 2028)}-{_short_token(rng, 5)}",
            f"UZA-PROT-{rng.randrange(1000, 9999)}",
            f"B{rng.randrange(10, 99)}-{rng.choice(['RESP', 'CARD', 'ONC', 'NEURO'])}-{rng.randrange(100, 999)}",
            f"EUDRACT-{rng.randrange(2024, 2028)}-{rng.randrange(100000, 999999)}",
        ]
    )


def study_protocol_name(rng: random.Random) -> str:
    return rng.choice(
        [
            f"BEL-{rng.choice(['RESPIRA', 'CARDIO', 'ONCO', 'NEURO'])}-{rng.randrange(100, 999)}",
            f"UZA {rng.choice(['LUNG', 'HEART', 'MDO', 'RECOVERY'])} cohort {rng.randrange(2024, 2028)}",
            f"Protocol {rng.choice(['AURORA', 'BRAVO', 'CORAL', 'DELTA'])}-{rng.randrange(10, 99)}",
            f"STUDIE-{rng.choice(['ASTMA', 'HF', 'CRC', 'MS'])}-{rng.randrange(2024, 2028)}",
        ]
    )


def _short_token(rng: random.Random, length: int = 6) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _dicom_uid(rng: random.Random) -> str:
    """Synthetic DICOM-style UID for study/series identifiers."""

    return ".".join(
        [
            "1",
            "2",
            "826",
            "0",
            "1",
            "3680043",
            "8",
            str(rng.randrange(1000, 9999)),
            str(rng.randrange(100000, 999999)),
            str(rng.randrange(1000000, 9999999)),
        ]
    )


def patient_identifier_bundle(rng: random.Random) -> list[str]:
    """Codes that should be annotated as ID:Patient."""

    study_uid = _dicom_uid(rng)
    series_uid = _dicom_uid(rng)
    return [
        rng.choice(
            [
                f"LAB-{rng.randrange(100000, 999999)}",
                f"LIS-{rng.randrange(2024, 2027)}-{rng.randrange(100000, 999999)}",
                f"GLIMS-{rng.randrange(10, 99)}-{rng.randrange(10000, 99999)}",
            ]
        ),
        rng.choice(
            [
                f"PATH-{rng.randrange(1000, 9999)}-BE",
                f"PA-{rng.randrange(2024, 2027)}-{rng.randrange(10000, 99999)}",
                f"CYTO-{rng.randrange(100000, 999999)}",
            ]
        ),
        rng.choice(
            [
                f"MI {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)}",
                f"PACSWEB-{rng.choice(['CT', 'MR', 'RX', 'US'])}{rng.randrange(10, 99)}-{rng.randrange(100000, 999999)}",
                f"ACSN-{rng.choice(['CT', 'MR', 'RX', 'US'])}-{rng.randrange(2024, 2027)}-{rng.randrange(100000, 999999)}",
                f"StudyUID {study_uid}",
            ]
        ),
        rng.choice(
            [
                f"OK nr. H{rng.randrange(1000, 9999)}L{rng.randrange(10, 99)}",
                f"OR-{rng.randrange(2024, 2027)}-{rng.randrange(10000, 99999)}",
            ]
        ),
        f"CFDNA {rng.randrange(10000, 99999)}",
        rng.choice(
            [
                f"BHEALTH-{rng.randrange(1000, 9999)}",
                f"TRIAL-{rng.randrange(2024, 2027)}-{_short_token(rng, 5)}",
            ]
        ),
        rng.choice(
            [
                f"K{rng.randrange(10000, 99999)}",
                f"STUDY-{_short_token(rng, 6)}",
                f"EUDRACT-{rng.randrange(2024, 2027)}-{rng.randrange(100000, 999999)}",
            ]
        ),
        f"CRISIS-2026-{rng.randrange(1000, 9999)}",
        rng.choice(
            [
                f"D{rng.randrange(100, 999)}-{rng.randrange(10000, 99999)}",
                f"DEV-{_short_token(rng, 4)}-{rng.randrange(100000, 999999)}",
                f"SN-{_short_token(rng, 3)}-{rng.randrange(1000000, 9999999)}",
            ]
        ),
        rng.choice(
            [
                f"PACSWEB-{rng.choice(['CT', 'MR', 'RX', 'US'])}{rng.randrange(10, 99)}-{rng.randrange(100000, 999999)}",
                f"DICOMweb study {study_uid}",
                f"WADO-RS studies/{study_uid}/series/{series_uid}",
                f"DICOM-SR-{rng.randrange(2024, 2027)}-{_short_token(rng)}.pdf",
                f"IMG-{rng.choice(['CT', 'MR', 'RX', 'US'])}-{rng.randrange(100000, 999999)}.dcm",
            ]
        ),
    ]


def hard_negative_codes(rng: random.Random) -> list[str]:
    """Code-like clinical strings that should normally stay unannotated."""

    pool = [
        "BRCA1",
        "BRCA2",
        "EGFR",
        "HER2",
        "ALK",
        "ROS1",
        "KRAS",
        "NRAS",
        "BRAF",
        "MTHFR C677T",
        "c.1799T>A",
        "c.35delG",
        "c.68_69delAG",
        "p.Val600Glu",
        "p.Gly12Asp",
        "p.Glu6Val",
        "NM_004333.6",
        "NM_007294.4",
        "LOINC 718-7",
        "LOINC 4548-4",
        "ICD-10 E11.9",
        "ICD-10 I10",
        "SNOMED 44054006",
        "SNOMED 38341003",
        "ISO 12452",
        "ISO 15189",
        "Medtronic Inceptiv",
        "Abbott Assurity MRI",
    ]
    selected = rng.sample(pool, k=8)
    selected.append(f"rs{rng.randrange(100000, 999999999)}")
    return selected
