"""Deterministic English (United Kingdom) synthetic generation profile."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from meddeid_core import BERT_ENTITY_LABELS, validate_record
from meddeid_language_en import (
    get_profile as get_language_profile,
    is_approved_synthetic_identifier,
    is_approved_synthetic_phone,
    lookup_records,
    lookup_source,
    lookup_values,
)

from .generation_profiles import GENERATION_PROFILE_CONTRACT, GenerationProfile
from .email_domains import email_address
from .identifier_formats import (
    format_english_identifier,
    synthetic_national_identifier,
)
from .organization_formats import format_healthcare_organization
from .span_builder import SpanBuilder
from .synthea_adapter import load_or_generate_synthea_csv_seeds

PROFILE_ID = "en-GB"
LOOKUP_SOURCE = lookup_source(PROFILE_ID)
ALLOWED_LABELS = tuple(BERT_ENTITY_LABELS)
ALLOWED_LABEL_SET = frozenset(ALLOWED_LABELS)

CONDITIONS = (
    {
        "name": "asthma",
        "symptoms": ["wheeze", "night-time cough", "shortness of breath"],
        "medications": ["salbutamol inhaler", "beclometasone inhaler"],
    },
    {
        "name": "atrial fibrillation",
        "symptoms": ["palpitations", "fatigue", "exertional breathlessness"],
        "medications": ["apixaban", "bisoprolol"],
    },
    {
        "name": "community-acquired pneumonia",
        "symptoms": ["productive cough", "fever", "pleuritic pain"],
        "medications": ["amoxicillin", "paracetamol"],
    },
    {
        "name": "migraine",
        "symptoms": ["unilateral headache", "nausea", "photophobia"],
        "medications": ["sumatriptan", "naproxen"],
    },
    {
        "name": "type 2 diabetes mellitus",
        "symptoms": ["thirst", "fatigue", "frequent urination"],
        "medications": ["metformin", "atorvastatin"],
    },
    {
        "name": "osteoarthritis of the knee",
        "symptoms": ["knee pain", "morning stiffness", "reduced mobility"],
        "medications": ["paracetamol", "topical ibuprofen"],
    },
    {
        "name": "iron-deficiency anaemia",
        "symptoms": ["tiredness", "dizziness", "reduced exercise tolerance"],
        "medications": ["ferrous fumarate"],
    },
    {
        "name": "essential hypertension",
        "symptoms": ["headache", "light-headedness"],
        "medications": ["amlodipine", "ramipril"],
    },
    {
        "name": "acute appendicitis",
        "symptoms": ["right lower-quadrant pain", "nausea", "loss of appetite"],
        "medications": ["paracetamol", "intravenous co-amoxiclav"],
    },
    {
        "name": "acute kidney injury",
        "symptoms": ["reduced urine output", "fatigue", "poor oral intake"],
        "medications": ["intravenous sodium chloride", "temporary medication hold"],
    },
    {
        "name": "acute pyelonephritis",
        "symptoms": ["flank pain", "dysuria", "rigors"],
        "medications": ["cefuroxime", "paracetamol"],
    },
    {
        "name": "allergic contact dermatitis",
        "symptoms": ["itching", "erythematous rash", "skin irritation"],
        "medications": ["hydrocortisone cream", "cetirizine"],
    },
    {
        "name": "benign paroxysmal positional vertigo",
        "symptoms": ["brief positional vertigo", "nausea", "unsteadiness"],
        "medications": ["prochlorperazine as required"],
    },
    {
        "name": "cellulitis of the lower leg",
        "symptoms": ["spreading erythema", "local warmth", "leg tenderness"],
        "medications": ["flucloxacillin", "paracetamol"],
    },
    {
        "name": "chronic obstructive pulmonary disease exacerbation",
        "symptoms": ["increased breathlessness", "wheeze", "purulent sputum"],
        "medications": ["prednisolone", "doxycycline", "salbutamol inhaler"],
    },
    {
        "name": "decompensated heart failure",
        "symptoms": ["orthopnoea", "ankle oedema", "reduced exercise tolerance"],
        "medications": ["furosemide", "bisoprolol"],
    },
    {
        "name": "diabetic foot ulcer",
        "symptoms": ["plantar ulceration", "reduced sensation", "local discharge"],
        "medications": ["wound care", "co-amoxiclav"],
    },
    {
        "name": "first-trimester hyperemesis",
        "symptoms": ["persistent vomiting", "dehydration", "weight loss"],
        "medications": ["cyclizine", "intravenous sodium chloride"],
    },
    {
        "name": "gastro-oesophageal reflux disease",
        "symptoms": ["retrosternal burning", "acid regurgitation", "night-time cough"],
        "medications": ["omeprazole", "alginate suspension"],
    },
    {
        "name": "generalised anxiety disorder",
        "symptoms": ["persistent worry", "poor sleep", "muscle tension"],
        "medications": ["sertraline"],
    },
    {
        "name": "gout flare",
        "symptoms": ["acute joint pain", "swelling", "erythema"],
        "medications": ["naproxen", "colchicine"],
    },
    {
        "name": "hypothyroidism",
        "symptoms": ["fatigue", "cold intolerance", "constipation"],
        "medications": ["levothyroxine"],
    },
    {
        "name": "infectious gastroenteritis",
        "symptoms": ["diarrhoea", "vomiting", "abdominal cramps"],
        "medications": ["oral rehydration solution", "ondansetron as required"],
    },
    {
        "name": "lumbar radiculopathy",
        "symptoms": ["low back pain", "shooting leg pain", "paraesthesia"],
        "medications": ["naproxen", "paracetamol"],
    },
    {
        "name": "major depressive episode",
        "symptoms": ["low mood", "anhedonia", "early-morning waking"],
        "medications": ["sertraline"],
    },
    {
        "name": "neonatal jaundice",
        "symptoms": ["jaundiced skin", "sleepy feeding", "reduced intake"],
        "medications": ["phototherapy", "feeding support"],
    },
    {
        "name": "non-ST-elevation myocardial infarction",
        "symptoms": ["central chest pressure", "diaphoresis", "nausea"],
        "medications": ["aspirin", "fondaparinux", "atorvastatin"],
    },
    {
        "name": "obstructive sleep apnoea",
        "symptoms": ["loud snoring", "witnessed apnoeas", "daytime somnolence"],
        "medications": ["continuous positive airway pressure"],
    },
    {
        "name": "panic attack",
        "symptoms": ["sudden palpitations", "chest tightness", "fear of dying"],
        "medications": ["breathing retraining"],
    },
    {
        "name": "pelvic inflammatory disease",
        "symptoms": ["lower abdominal pain", "abnormal discharge", "dyspareunia"],
        "medications": ["ceftriaxone", "doxycycline", "metronidazole"],
    },
    {
        "name": "postoperative wound infection",
        "symptoms": ["wound erythema", "purulent discharge", "local pain"],
        "medications": ["wound irrigation", "flucloxacillin"],
    },
    {
        "name": "pulmonary embolism",
        "symptoms": ["sudden breathlessness", "pleuritic pain", "tachycardia"],
        "medications": ["apixaban", "supplemental oxygen"],
    },
    {
        "name": "rheumatoid arthritis flare",
        "symptoms": ["morning stiffness", "swollen hand joints", "fatigue"],
        "medications": ["methotrexate", "folic acid", "naproxen"],
    },
    {
        "name": "sickle cell painful crisis",
        "symptoms": ["severe limb pain", "back pain", "dehydration"],
        "medications": ["morphine", "intravenous fluids"],
    },
    {
        "name": "transient ischaemic attack",
        "symptoms": ["transient arm weakness", "facial droop", "slurred speech"],
        "medications": ["aspirin", "atorvastatin"],
    },
    {
        "name": "uncomplicated urinary tract infection",
        "symptoms": ["dysuria", "urinary frequency", "suprapubic discomfort"],
        "medications": ["nitrofurantoin"],
    },
    {
        "name": "Parkinson disease",
        "symptoms": ["resting tremor", "bradykinesia", "muscular rigidity"],
        "medications": ["co-beneldopa"],
    },
    {
        "name": "Hodgkin lymphoma",
        "symptoms": ["painless lymphadenopathy", "night sweats", "unintentional weight loss"],
        "medications": ["supportive care pending tissue diagnosis"],
    },
    {
        "name": "hereditary breast and ovarian cancer risk assessment",
        "symptoms": ["significant family history", "request for predictive testing", "cancer-risk concern"],
        "medications": ["no current treatment"],
    },
    {
        "name": "bronchiolitis",
        "symptoms": ["cough", "increased work of breathing", "reduced feeding"],
        "medications": ["supportive care", "nasal saline as needed"],
    },
    {
        "name": "atopic eczema",
        "symptoms": ["itching", "dry inflamed skin", "sleep disturbance"],
        "medications": ["emollient", "topical hydrocortisone"],
    },
    {
        "name": "acute otitis media",
        "symptoms": ["ear pain", "fever", "irritability"],
        "medications": ["paracetamol", "supportive care"],
    },
    {
        "name": "viral croup",
        "symptoms": ["barking cough", "hoarse voice", "inspiratory stridor"],
        "medications": ["dexamethasone", "supportive care"],
    },
    {
        "name": "febrile seizure",
        "symptoms": ["generalised convulsion", "fever", "postictal drowsiness"],
        "medications": ["antipyretic comfort measures", "seizure first-aid advice"],
    },
    {
        "name": "gastroenteritis with dehydration",
        "symptoms": ["vomiting", "diarrhoea", "reduced urine output"],
        "medications": ["oral rehydration solution", "supportive care"],
    },
    {
        "name": "acute asthma exacerbation",
        "symptoms": ["wheeze", "increased work of breathing", "night-time cough"],
        "medications": ["salbutamol inhaler", "spacer teaching"],
    },
    {
        "name": "type 1 diabetes mellitus",
        "symptoms": ["thirst", "frequent urination", "weight loss"],
        "medications": ["insulin", "glucose monitoring"],
    },
)
DOCUMENT_TYPES = (
    "clinic_note",
    "discharge_summary",
    "emergency_note",
    "referral_letter",
    "laboratory_report",
    "nursing_note",
)
# Include deliberately shifted historical and future values so date detection
# remains stable after utility-preserving pseudonymisation. Most years remain
# contemporary; the tails reproduce the manuscript's date-value perturbations
# without creating paired duplicate documents.
ENCOUNTER_YEAR_SEQUENCE = (
    1988, 1996, 2004, 2012, 2018, 2019, 2020, 2021, 2022, 2023,
    2024, 2025, 2026, 2028, 2035, 2045,
)
CONDITION_NAMES_BY_DOCUMENT = {
    "clinic_note": {
        "asthma", "atrial fibrillation", "migraine", "type 2 diabetes mellitus",
        "osteoarthritis of the knee", "iron-deficiency anaemia", "essential hypertension",
        "allergic contact dermatitis", "benign paroxysmal positional vertigo",
        "chronic obstructive pulmonary disease exacerbation", "diabetic foot ulcer",
        "gastro-oesophageal reflux disease", "generalised anxiety disorder", "gout flare",
        "hypothyroidism", "lumbar radiculopathy", "major depressive episode",
        "obstructive sleep apnoea", "panic attack", "rheumatoid arthritis flare",
        "Parkinson disease", "Hodgkin lymphoma",
        "hereditary breast and ovarian cancer risk assessment",
    },
    "discharge_summary": {
        "community-acquired pneumonia", "acute appendicitis", "acute kidney injury",
        "acute pyelonephritis", "cellulitis of the lower leg",
        "chronic obstructive pulmonary disease exacerbation", "decompensated heart failure",
        "first-trimester hyperemesis", "infectious gastroenteritis",
        "non-ST-elevation myocardial infarction", "pelvic inflammatory disease",
        "postoperative wound infection", "pulmonary embolism", "sickle cell painful crisis",
        "transient ischaemic attack",
    },
    "emergency_note": {
        "community-acquired pneumonia", "acute appendicitis", "acute kidney injury",
        "acute pyelonephritis", "allergic contact dermatitis",
        "benign paroxysmal positional vertigo", "cellulitis of the lower leg",
        "decompensated heart failure", "first-trimester hyperemesis", "gout flare",
        "infectious gastroenteritis", "non-ST-elevation myocardial infarction",
        "panic attack", "pelvic inflammatory disease", "postoperative wound infection",
        "pulmonary embolism", "sickle cell painful crisis", "transient ischaemic attack",
        "uncomplicated urinary tract infection",
    },
    "referral_letter": {
        "asthma", "atrial fibrillation", "migraine", "type 2 diabetes mellitus",
        "osteoarthritis of the knee", "iron-deficiency anaemia", "essential hypertension",
        "diabetic foot ulcer", "gastro-oesophageal reflux disease", "generalised anxiety disorder",
        "hypothyroidism", "lumbar radiculopathy", "major depressive episode",
        "obstructive sleep apnoea", "rheumatoid arthritis flare", "Parkinson disease",
        "Hodgkin lymphoma", "hereditary breast and ovarian cancer risk assessment",
    },
    "laboratory_report": {
        "type 2 diabetes mellitus", "iron-deficiency anaemia", "essential hypertension",
        "acute kidney injury", "acute pyelonephritis", "diabetic foot ulcer", "gout flare",
        "hypothyroidism", "rheumatoid arthritis flare", "Hodgkin lymphoma",
        "hereditary breast and ovarian cancer risk assessment",
    },
    "nursing_note": {
        "community-acquired pneumonia", "acute kidney injury", "acute pyelonephritis",
        "cellulitis of the lower leg", "chronic obstructive pulmonary disease exacerbation",
        "decompensated heart failure", "diabetic foot ulcer", "first-trimester hyperemesis",
        "infectious gastroenteritis", "non-ST-elevation myocardial infarction",
        "postoperative wound infection", "pulmonary embolism", "rheumatoid arthritis flare",
        "sickle cell painful crisis", "transient ischaemic attack",
    },
}

# Production uses an explicit paediatric stratum instead of allowing the adult
# sampler to produce an occasional 16- or 17-year-old.  Each 250-document
# locale block contains exactly 41 paediatric cases: 5 infants, 8 children aged
# 1-4, 12 aged 5-11, and 16 adolescents aged 12-17.  Batch 1 was authored before
# this contract existed; the remaining thirteen blocks deliberately catch up so
# the final 7,000-document corpus still exceeds 15% paediatric coverage.
PRODUCTION_PROFILE_BLOCK_SIZE = 250
PEDIATRIC_PER_PROFILE_BLOCK = 41
PEDIATRIC_BAND_COUNTS = {
    "infant": 5,
    "early_childhood": 8,
    "school_age": 12,
    "adolescent": 16,
}
_PEDIATRIC_OFFSETS = tuple(
    (7 + 37 * position) % PRODUCTION_PROFILE_BLOCK_SIZE
    for position in range(PEDIATRIC_PER_PROFILE_BLOCK)
)
_PEDIATRIC_AGE_SPECS = (
    # Infant values exercise day-, week-, and month-based clinical age forms.
    (3, "days", "infant"),
    (2, "weeks", "infant"),
    (7, "weeks", "infant"),
    (4, "months", "infant"),
    (10, "months", "infant"),
    # Early childhood includes the especially important 18-month format.
    (18, "months", "early_childhood"),
    (2, "years", "early_childhood"),
    (3, "years", "early_childhood"),
    (4, "years", "early_childhood"),
    (2, "years", "early_childhood"),
    (3, "years", "early_childhood"),
    (4, "years", "early_childhood"),
    (18, "months", "early_childhood"),
    # School age.
    (5, "years", "school_age"),
    (6, "years", "school_age"),
    (7, "years", "school_age"),
    (8, "years", "school_age"),
    (9, "years", "school_age"),
    (10, "years", "school_age"),
    (11, "years", "school_age"),
    (5, "years", "school_age"),
    (6, "years", "school_age"),
    (8, "years", "school_age"),
    (10, "years", "school_age"),
    (11, "years", "school_age"),
    # Adolescence.
    (12, "years", "adolescent"),
    (13, "years", "adolescent"),
    (14, "years", "adolescent"),
    (15, "years", "adolescent"),
    (16, "years", "adolescent"),
    (17, "years", "adolescent"),
    (12, "years", "adolescent"),
    (13, "years", "adolescent"),
    (14, "years", "adolescent"),
    (15, "years", "adolescent"),
    (16, "years", "adolescent"),
    (17, "years", "adolescent"),
    (13, "years", "adolescent"),
    (14, "years", "adolescent"),
    (15, "years", "adolescent"),
    (16, "years", "adolescent"),
)
if len(_PEDIATRIC_OFFSETS) != len(_PEDIATRIC_AGE_SPECS):
    raise AssertionError("paediatric offset and age schedules must have equal length")
PEDIATRIC_AGE_BY_OFFSET = dict(zip(_PEDIATRIC_OFFSETS, _PEDIATRIC_AGE_SPECS))

PEDIATRIC_CONDITION_NAMES_BY_BAND = {
    "neonate": {"neonatal jaundice"},
    "infant": {
        "neonatal jaundice", "bronchiolitis", "atopic eczema",
        "gastroenteritis with dehydration", "uncomplicated urinary tract infection",
    },
    "early_childhood": {
        "atopic eczema", "acute otitis media", "viral croup", "febrile seizure",
        "gastroenteritis with dehydration", "uncomplicated urinary tract infection",
    },
    "school_age": {
        "acute asthma exacerbation", "acute appendicitis", "atopic eczema",
        "gastroenteritis with dehydration", "iron-deficiency anaemia",
        "sickle cell painful crisis", "type 1 diabetes mellitus",
    },
    "adolescent": {
        "acute asthma exacerbation", "acute appendicitis", "atopic eczema",
        "gastroenteritis with dehydration", "iron-deficiency anaemia", "migraine",
        "sickle cell painful crisis", "type 1 diabetes mellitus",
        "uncomplicated urinary tract infection",
    },
}
PEDIATRIC_CONDITION_NAMES_BY_DOCUMENT = {
    "clinic_note": {
        "neonatal jaundice", "atopic eczema", "acute asthma exacerbation", "iron-deficiency anaemia",
        "migraine", "type 1 diabetes mellitus",
    },
    "discharge_summary": {
        "neonatal jaundice", "bronchiolitis", "viral croup", "febrile seizure",
        "gastroenteritis with dehydration", "acute asthma exacerbation",
        "acute appendicitis", "sickle cell painful crisis",
    },
    "emergency_note": {
        "neonatal jaundice", "bronchiolitis", "acute otitis media", "viral croup", "febrile seizure",
        "gastroenteritis with dehydration", "acute asthma exacerbation",
        "acute appendicitis", "sickle cell painful crisis",
        "uncomplicated urinary tract infection",
    },
    "referral_letter": {
        "neonatal jaundice", "atopic eczema", "acute otitis media", "acute asthma exacerbation",
        "iron-deficiency anaemia", "migraine", "type 1 diabetes mellitus",
    },
    "laboratory_report": {
        "neonatal jaundice", "gastroenteritis with dehydration",
        "iron-deficiency anaemia", "type 1 diabetes mellitus",
        "uncomplicated urinary tract infection",
    },
    "nursing_note": {
        "neonatal jaundice", "bronchiolitis", "viral croup", "febrile seizure",
        "gastroenteritis with dehydration", "acute asthma exacerbation",
        "acute appendicitis", "sickle cell painful crisis",
    },
}

def _clean_values(values: tuple[str, ...], *, reject_leading_digit: bool = False) -> tuple[str, ...]:
    cleaned = tuple(
        value
        for value in values
        if 1 < len(value) <= 90
        and value == value.strip()
        and value[0].isalnum()
        and not (reject_leading_digit and value[0].isdigit())
        and not any(character in value for character in ('"', "{", "}"))
    )
    if not cleaned:
        raise RuntimeError("English language resource filtering removed every value")
    return cleaned


def _clean_person_values(values: tuple[str, ...]) -> tuple[str, ...]:
    """Exclude publisher fragments that cannot form a complete name token."""

    return tuple(
        value
        for value in _clean_values(values)
        if value[-1].isalnum()
        and not re.search(r"[-'’]\s|\s[-'’]", value)
    )


def _preferred_values(values: tuple[str, ...], preferred: tuple[str, ...]) -> tuple[str, ...]:
    by_normalized = {value.casefold(): value for value in values}
    selected = tuple(
        by_normalized[value.casefold()]
        for value in preferred
        if value.casefold() in by_normalized
    )
    if len(selected) < 20:
        raise RuntimeError("English profession resource is missing the curated common occupations")
    return selected


FIRST_NAMES = _clean_person_values(lookup_values(PROFILE_ID, "first_names"))
FAMILY_NAMES = _clean_person_values(lookup_values(PROFILE_ID, "family_names"))
STREET_RECORDS = lookup_records(PROFILE_ID, "streets")
STREET_NAMES = _clean_values(tuple(record["value"] for record in STREET_RECORDS))
POSTAL_CODE_LOCALITIES = tuple(
    (record.get("attributes", {}).get("locality"), record["value"])
    for record in lookup_records(PROFILE_ID, "postal_code_localities")
    if isinstance(record.get("attributes", {}).get("locality"), str)
    and 1 < len(record.get("attributes", {}).get("locality", "")) <= 70
    and record.get("attributes", {}).get("locality", "")[0].isalnum()
    and '"' not in record.get("attributes", {}).get("locality", "")
)
POSTAL_LOCALITY_RECORDS = lookup_records(PROFILE_ID, "postal_localities")
GB_REGION_SEQUENCE = ("England", "Scotland", "Wales", "Northern Ireland")
STREETS_BY_REGION = {
    region: tuple(
        record["value"]
        for record in STREET_RECORDS
        if region in record.get("regions", ())
        and 1 < len(record["value"]) <= 90
        and record["value"][0].isalnum()
        and not any(character in record["value"] for character in ('"', "{", "}"))
    )
    for region in GB_REGION_SEQUENCE
}
POSTAL_BY_REGION = {
    region: tuple(
        (record.get("attributes", {}).get("locality"), record["value"])
        for record in lookup_records(PROFILE_ID, "postal_code_localities")
        if region in record.get("regions", ())
        and isinstance(record.get("attributes", {}).get("locality"), str)
    )
    for region in GB_REGION_SEQUENCE[:-1]
}
LOCALITIES_BY_REGION = {
    region: tuple(
        record["value"]
        for record in POSTAL_LOCALITY_RECORDS
        if region in record.get("regions", ())
        and 1 < len(record["value"]) <= 70
        and record["value"][0].isalnum()
        and '"' not in record["value"]
    )
    for region in GB_REGION_SEQUENCE
}
POSTAL_LOCALITIES = POSTAL_CODE_LOCALITIES
HOSPITAL_RECORDS = lookup_records(PROFILE_ID, "hospitals")
HOSPITAL_EXCLUSIONS = (
    " child",
    " clinic",
    " dental",
    " department",
    " eye ",
    " hub",
    " liaison",
    " ophthalm",
    " outpatient",
    " out-patient",
    " urology",
    " vaccination",
    " hospice",
    " pharmacy",
    " mental health",
    " paediatric",
    " pediatric",
    " day hospital",
    " assessment service",
)
HOSPITALS_BY_REGION = {
    region: tuple(
        record["value"]
        for record in HOSPITAL_RECORDS
        if region in record.get("regions", ())
        and any(
            term in record["value"].lower()
            for term in ("hospital", "infirmary", "medical centre", "medical center")
        )
        and not any(term in f" {record['value'].lower()}" for term in HOSPITAL_EXCLUSIONS)
        and "(" not in record["value"]
        and ")" not in record["value"]
    )
    for region in GB_REGION_SEQUENCE
}


def _looks_like_major_acute_hospital(value: str, region: str) -> bool:
    lowered = value.lower()
    if any(
        term in lowered
        for term in (
            "community hospital", " trial", " control", " prescribing", " unit",
            " hospital site", " pts ", " at ", "screening", "programme",
        )
    ):
        return False
    if value in {"Ulster Hospital", "Aberdeen Royal Infirmary", "Royal Infirmary of Edinburgh"}:
        return True
    return bool(
        any(
            term in lowered
            for term in (
                "general hospital", "university hospital", "teaching hospital",
                "royal infirmary", "district general hospital",
            )
        )
        or re.search(r"\broyal .+ hospital\b", lowered)
        or ("queen" in lowered and "hospital" in lowered)
    )


ACUTE_HOSPITALS_BY_REGION = {
    region: tuple(
        value
        for value in HOSPITALS_BY_REGION[region]
        if _looks_like_major_acute_hospital(value, region)
    )
    for region in GB_REGION_SEQUENCE
}
HOSPITALS = tuple(
    value for region in GB_REGION_SEQUENCE for value in HOSPITALS_BY_REGION[region]
)
OTHER_ORGANISATION_RECORDS = lookup_records(PROFILE_ID, "other_organisations")
OTHER_ORGANISATIONS_BY_REGION = {
    region: tuple(
        record["value"]
        for record in OTHER_ORGANISATION_RECORDS
        if region in record.get("regions", ())
    )
    for region in GB_REGION_SEQUENCE
}
OTHER_ORGANISATIONS = tuple(
    record["value"] for record in OTHER_ORGANISATION_RECORDS
)
OTHER_ORGANISATION_SUFFIXES = (
    "Community Centre",
    "Further Education College",
    "Housing Association",
    "Public Library",
    "Secondary School",
    "Sports Club",
    "Town Council",
    "Voluntary Service",
)
PROFESSIONS = _preferred_values(
    _clean_values(lookup_values(PROFILE_ID, "professions"), reject_leading_digit=True),
    (
        "accountant", "administrative assistant", "architect", "baker", "barista",
        "builder", "bus driver", "care worker", "carpenter", "chef", "civil engineer",
        "cleaner", "electrician", "farmer", "graphic designer", "hairdresser",
        "journalist", "lawyer", "librarian", "mechanic", "office manager", "paralegal",
        "pharmacist", "physiotherapist", "plumber", "police officer", "postal worker",
        "receptionist", "retail assistant", "sales assistant", "shop manager",
        "software developer", "taxi driver", "university lecturer", "warehouse operative",
        "writer",
    ),
)
EARLY_CAREER_PROFESSIONS = tuple(
    value
    for value in PROFESSIONS
    if value.casefold()
    in {
        "baker",
        "barista",
        "builder",
        "care worker",
        "cleaner",
        "hairdresser",
        "retail assistant",
        "sales assistant",
        "warehouse operative",
    }
)
if not EARLY_CAREER_PROFESSIONS:
    raise RuntimeError("English GB profession resources have no early-career values")
DEPARTMENTS = lookup_values(PROFILE_ID, "departments")
ADULT_DEPARTMENTS = tuple(
    department
    for department in DEPARTMENTS
    if not any(
        term in department.lower()
        for term in ("paediatric", "pediatric", "child", "neonat")
    )
)
if not ADULT_DEPARTMENTS:
    raise RuntimeError("English department resources have no adult-compatible values")


def _stable_rng(seed: int, index: int) -> random.Random:
    return random.Random(f"{PROFILE_ID}|1|{seed}|{index}")


def _person(rng: random.Random) -> dict[str, str]:
    return {
        "given_name": rng.choice(FIRST_NAMES),
        "family_name": rng.choice(FAMILY_NAMES),
    }


def _full_name(person: dict[str, str]) -> str:
    return f"{person['given_name']} {person['family_name']}"


def _address(rng: random.Random, index: int) -> tuple[str, str]:
    region = GB_REGION_SEQUENCE[index % len(GB_REGION_SEQUENCE)]
    street = rng.choice(STREETS_BY_REGION[region])
    if region == "Northern Ireland":
        locality = rng.choice(LOCALITIES_BY_REGION[region])
        return f"{rng.randrange(1, 180)} {street}, {locality}", region
    locality, postcode = rng.choice(POSTAL_BY_REGION[region])
    return f"{rng.randrange(1, 180)} {street}, {locality}, {postcode}", region


def _phone(rng: random.Random) -> str:
    # Ofcom reserves 07700 900000-900999 for fictional drama use.
    return f"07700 900{rng.randrange(0, 1000):03d}"


def _other_organisation(rng: random.Random, region: str) -> str:
    locality = rng.choice(LOCALITIES_BY_REGION[region])
    return f"{locality} {rng.choice(OTHER_ORGANISATION_SUFFIXES)}"


def _email(person: dict[str, str], rng: random.Random) -> str:
    local = f"{person['given_name']}.{person['family_name']}".lower()
    numbered_local = f"{local}{rng.randrange(10, 99)}"
    return email_address(PROFILE_ID, numbered_local)


def _format_date(value: date, variant: int) -> str:
    months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    abbreviations = tuple(month[:3] for month in months)
    style = variant % 4
    if style == 0:
        return f"{value.day:02d}/{value.month:02d}/{value.year}"
    if style == 1:
        return f"{value.day} {months[value.month - 1]} {value.year}"
    if style == 2:
        return f"{value.day:02d} {abbreviations[value.month - 1]} {value.year}"
    return value.isoformat()


def _format_date_for_profile(value: date, variant: int, profile_id: str) -> str:
    if profile_id == "en-GB":
        return _format_date(value, variant)
    months = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )
    style = variant % 4
    if style == 0:
        return f"{value.month:02d}/{value.day:02d}/{value.year}"
    if style == 1:
        return f"{months[value.month - 1]} {value.day}, {value.year}"
    if style == 2:
        return f"{months[value.month - 1][:3]} {value.day}, {value.year}"
    return value.isoformat()


def _condition(
    seed: dict[str, Any] | None,
    rng: random.Random,
    document_type: str,
    *,
    pediatric_band: str | None = None,
) -> dict[str, Any]:
    if pediatric_band is None:
        allowed_names = CONDITION_NAMES_BY_DOCUMENT[document_type]
    else:
        allowed_names = (
            PEDIATRIC_CONDITION_NAMES_BY_BAND[pediatric_band]
            & PEDIATRIC_CONDITION_NAMES_BY_DOCUMENT[document_type]
        )
        if not allowed_names:
            raise RuntimeError(
                f"no paediatric condition for {pediatric_band}/{document_type}"
            )
    candidates = tuple(row for row in CONDITIONS if row["name"] in allowed_names)
    base = dict(rng.choice(candidates))
    base["symptoms"] = list(base["symptoms"])
    base["medications"] = list(base["medications"])
    if not seed:
        return base
    if seed.get("condition"):
        base["name"] = str(seed["condition"])
    if seed.get("medications"):
        base["medications"] = [str(value) for value in seed["medications"][:4]]
    observations = []
    for item in seed.get("observations", [])[:4]:
        if isinstance(item, (list, tuple)) and item:
            observations.append(
                {
                    "name": str(item[0]),
                    "value": str(item[1]) if len(item) > 1 else "",
                    "unit": str(item[2]) if len(item) > 2 else "",
                }
            )
    base["observations"] = observations
    return base


def _age_for_condition(condition_name: str, rng: random.Random) -> int:
    """Keep synthetic age and clinical scenario mutually plausible."""

    name = condition_name.lower()
    if "neonatal" in name:
        return 0
    if "first-trimester" in name or "pelvic inflammatory" in name:
        return rng.randrange(18, 46)
    if any(
        term in name
        for term in (
            "osteoarthritis",
            "heart failure",
            "sleep apnoea",
            "chronic obstructive pulmonary disease",
        )
    ):
        return rng.randrange(45, 91)
    # Every built-in AF scenario currently includes apixaban. Keep age >=65 so
    # anticoagulation has at least one defensible CHA2DS2-VASc risk factor.
    if "atrial fibrillation" in name:
        return rng.randrange(65, 91)
    if any(term in name for term in ("transient ischaemic", "myocardial")):
        return rng.randrange(40, 91)
    if "sickle cell" in name:
        return rng.randrange(18, 66)
    if any(term in name for term in ("type 2 diabetes", "hypertension", "gout")):
        return rng.randrange(30, 91)
    return rng.randrange(18, 91)


def _birth_date_for_age(encounter: date, age_years: int, rng: random.Random) -> date:
    """Construct a DOB whose completed calendar age is exactly ``age_years``."""

    month = rng.randrange(1, 13)
    day = rng.randrange(1, 29)
    birthday_still_to_come = (month, day) > (encounter.month, encounter.day)
    birth_year = encounter.year - age_years - int(birthday_still_to_come)
    return date(birth_year, month, day)


def _birth_date_for_precise_age(
    encounter: date, age_value: int, age_unit: str, rng: random.Random
) -> date:
    """Construct a DOB consistent with a day-, week-, month-, or year-age."""

    if age_value < 0:
        raise ValueError("age_value must not be negative")
    if age_unit == "years":
        return _birth_date_for_age(encounter, age_value, rng)
    if age_unit == "days":
        return encounter - timedelta(days=age_value)
    if age_unit == "weeks":
        return encounter - timedelta(weeks=age_value)
    if age_unit == "months":
        absolute_month = encounter.year * 12 + encounter.month - 1 - age_value
        year, zero_based_month = divmod(absolute_month, 12)
        # A day no later than the encounter day makes the completed-month value
        # exact, and limiting it to 28 is valid in every target month.
        return date(year, zero_based_month + 1, min(encounter.day, 28))
    raise ValueError(f"unsupported age unit {age_unit!r}")


def _completed_years(encounter: date, birth_date: date) -> int:
    return encounter.year - birth_date.year - int(
        (encounter.month, encounter.day) < (birth_date.month, birth_date.day)
    )


def _age_display(age_value: int, age_unit: str, profile_id: str) -> dict[str, str]:
    singular = {
        "days": "day",
        "weeks": "week",
        "months": "month",
        "years": "year",
    }[age_unit]
    plural = singular if age_value == 1 else age_unit
    contextual_prefix = "aged" if profile_id == "en-GB" else "age"
    if age_unit == "days":
        abbreviated = f"day {age_value} of life"
    elif age_unit == "weeks":
        abbreviated = f"{age_value} wk old"
    elif age_unit == "months":
        abbreviated = f"{age_value} m/o"
    else:
        abbreviated = f"{age_value} y/o"
    return {
        "hyphenated": f"{age_value}-{singular}-old",
        "contextual": f"{contextual_prefix} {age_value} {plural}",
        "abbreviated": abbreviated,
    }


def _pediatric_age_for_index(index: int) -> tuple[int, str, str] | None:
    return PEDIATRIC_AGE_BY_OFFSET.get(index % PRODUCTION_PROFILE_BLOCK_SIZE)


def _pediatric_department(document_type: str, clinical_age_group: str) -> str:
    if clinical_age_group == "neonate":
        return "neonatology"
    return {
        "clinic_note": "paediatric outpatient clinic",
        "discharge_summary": "paediatrics",
        "emergency_note": "paediatric emergency medicine",
        "referral_letter": "community paediatrics",
        "laboratory_report": "paediatric laboratory medicine",
        "nursing_note": "paediatric inpatient nursing",
    }[document_type]


def _case_record(
    index: int, seed: int, synthea_seed: dict[str, Any] | None
) -> dict[str, Any]:
    rng = _stable_rng(seed, index)
    encounter = date(ENCOUNTER_YEAR_SEQUENCE[index % len(ENCOUNTER_YEAR_SEQUENCE)], 1, 1) + timedelta(
        days=rng.randrange(0, 365)
    )
    document_type = DOCUMENT_TYPES[index % len(DOCUMENT_TYPES)]
    pediatric_age = _pediatric_age_for_index(index) if not synthea_seed else None
    pediatric_condition_band = pediatric_age[2] if pediatric_age else None
    if pediatric_age and (
        pediatric_age[1] == "days"
        or (pediatric_age[1] == "weeks" and pediatric_age[0] < 4)
    ):
        pediatric_condition_band = "neonate"
    condition = _condition(
        synthea_seed,
        rng,
        document_type,
        pediatric_band=pediatric_condition_band,
    )
    if synthea_seed and "synthea_age_years" in synthea_seed:
        age_value = int(synthea_seed["synthea_age_years"])
        age_unit = "years"
        birth_date = _birth_date_for_precise_age(encounter, age_value, age_unit, rng)
        age_years = age_value
        age_group = "pediatric_external" if age_years < 18 else "adult"
    elif pediatric_age:
        age_value, age_unit, age_group = pediatric_age
        birth_date = _birth_date_for_precise_age(encounter, age_value, age_unit, rng)
        age_years = _completed_years(encounter, birth_date)
        clinical_age_group = pediatric_condition_band
    else:
        age_value = _age_for_condition(str(condition["name"]), rng)
        age_unit = "years"
        birth_date = _birth_date_for_precise_age(encounter, age_value, age_unit, rng)
        age_years = age_value
        age_group = "adult"
        clinical_age_group = "adult"
    if synthea_seed and "synthea_age_years" in synthea_seed:
        clinical_age_group = age_group
    patient = _person(rng)
    caregiver = _person(rng)
    relative = _person(rng)
    if age_years < 18:
        relative_role = rng.choice(
            ("mother", "father", "parent", "legal guardian")
            if age_years < 12
            else ("mother", "father", "parent", "legal guardian", "aunt", "uncle")
        )
        department = _pediatric_department(document_type, clinical_age_group)
    elif age_years < 31:
        relative_role = rng.choice(
            ("spouse", "partner", "sibling", "friend", "emergency contact")
        )
        department = ADULT_DEPARTMENTS[index % len(ADULT_DEPARTMENTS)]
    else:
        relative_role = rng.choice(
            ("spouse", "partner", "adult child", "sibling", "friend", "emergency contact")
        )
        department = ADULT_DEPARTMENTS[index % len(ADULT_DEPARTMENTS)]
    patient_address, region = _address(rng, index)
    national_identifier = synthetic_national_identifier(PROFILE_ID, str(index))
    caregiver_locality = rng.choice(LOCALITIES_BY_REGION[region])
    other_locality = rng.choice(LOCALITIES_BY_REGION[region])
    source_hospital = rng.choice(
        (
            ACUTE_HOSPITALS_BY_REGION.get(region)
            if document_type in {"discharge_summary", "emergency_note", "nursing_note"}
            else HOSPITALS_BY_REGION.get(region)
        )
        or HOSPITALS
    )
    return {
        "case_id": f"case-{index + 1:05d}",
        "case_contract": "meddeid.synthetic-case.v1",
        "generation_profile": {
            "contract_version": GENERATION_PROFILE_CONTRACT,
            "profile_id": PROFILE_ID,
        },
        "language": PROFILE_ID,
        "document_type": document_type,
        "department": department,
        "condition": condition,
        "patient": patient,
        "caregiver": caregiver,
        "relative": relative,
        "relative_role": relative_role,
        "patient_address": patient_address,
        "address_region": region,
        "caregiver_locality": caregiver_locality,
        "other_locality": other_locality,
        "hospital": format_healthcare_organization(
            PROFILE_ID, source_hospital, str(index)
        ),
        "other_organisation": _other_organisation(rng, region),
        "profession": rng.choice(
            EARLY_CAREER_PROFESSIONS if age_years < 21 else PROFESSIONS
        ),
        "age_years": age_years,
        "age_value": age_value,
        "age_unit": age_unit,
        "age_group": age_group,
        "clinical_age_group": clinical_age_group,
        "age_display": _age_display(age_value, age_unit, PROFILE_ID),
        "birth_date": birth_date.isoformat(),
        "encounter_date": encounter.isoformat(),
        "followup_date": (encounter + timedelta(days=rng.randrange(7, 91))).isoformat(),
        "patient_phone": _phone(rng),
        "patient_email": _email(patient, rng),
        "patient_id": format_english_identifier(PROFILE_ID, "patient.mrn", str(index)),
        "caregiver_id": format_english_identifier(
            PROFILE_ID, "caregiver.professional_id", str(index)
        ),
        "report_id": format_english_identifier(
            PROFILE_ID, "patient.report_id", str(index)
        ),
        "national_id": national_identifier["value"],
        "national_id_type": national_identifier["kind"],
        "national_id_label": national_identifier["field_label"],
        "synthea_source": synthea_seed,
        "pii_policy": {
            "source": "versioned meddeid-language-en en-GB resources",
            "note": (
                "Person identities are independently recombined; public institutions "
                "and geographic entities may be real; Synthea PII is not used."
            ),
        },
        "pii_resource_categories": {
            "patient.given_name": "first_names",
            "patient.family_name": "family_names",
            "caregiver.given_name": "first_names",
            "caregiver.family_name": "family_names",
            "relative.given_name": "first_names",
            "relative.family_name": "family_names",
            "patient.address": "streets + postal_code_localities/localities",
            "healthcare.organization": "hospitals",
            "other.organization": "postal_localities + synthetic organisation suffix",
            "patient.profession": "professions",
        },
    }


def build_case_records(
    count: int,
    *,
    seed: int = 20260508,
    synthea_seeds: list[dict[str, Any]] | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must not be negative")
    seeds = synthea_seeds or []
    return [
        _case_record(
            index,
            seed,
            seeds[index % len(seeds)] if seeds else None,
        )
        for index in range(start_index, start_index + count)
    ]


def _metadata(case: dict[str, Any], renderer: str) -> dict[str, Any]:
    profile_id = str(case["language"])
    return {
        "generation_method": "generation-profile-renderer-v1",
        "generation_profile": profile_id,
        "renderer": renderer,
        "document_type": case["document_type"],
        "lang": profile_id,
        "synthetic": True,
        "lookup_source": lookup_source(profile_id),
        "patient": {
            **case["patient"],
            "birth_date": case["birth_date"],
        },
        "patient_age": {
            "value": case.get("age_value", case.get("age_years")),
            "unit": case.get("age_unit", "years"),
            "completed_years": case.get("age_years"),
            "group": case.get("age_group", "adult"),
            "clinical_group": case.get(
                "clinical_age_group", case.get("age_group", "adult")
            ),
        },
        "caregivers": [dict(case["caregiver"])],
        "document_creation_date": case["encounter_date"],
        "synthea_source": case.get("synthea_source"),
    }


def _add_header(b: SpanBuilder, case: dict[str, Any], variant: int) -> None:
    b.add("Patient: ")
    b.add(_full_name(case["patient"]), "Name:Patient")
    b.add(" | Date of birth: ")
    b.add(
        _format_date_for_profile(
            date.fromisoformat(case["birth_date"]), variant, case["language"]
        ),
        "Age_Birthdate",
    )
    b.add(" | MRN: ")
    b.add(case["patient_id"], "ID:Patient")
    b.add(f" | {case['national_id_label']}: ")
    b.add(case["national_id"], "ID:Patient")
    b.add("\nAddress: ")
    b.add(case["patient_address"], "Address_Location:Patient")
    b.add(" | Telephone: ")
    b.add(case["patient_phone"], "Contactdetails")
    b.add(" | Email: ")
    b.add(case["patient_email"], "Contactdetails")
    b.add("\n")


def _add_care_context(b: SpanBuilder, case: dict[str, Any]) -> None:
    b.add("Organisation: " if case["language"] == "en-GB" else "Organization: ")
    b.add(case["hospital"], "Organization:Healthcare")
    b.add(" | Clinician: ")
    b.add(f"Dr {_full_name(case['caregiver'])}", "Name:Caregiver")
    b.add(" | Professional ID: ")
    b.add(case["caregiver_id"], "ID:Caregiver")
    b.add("\n")


def _add_date(b: SpanBuilder, raw: str, variant: int, profile_id: str) -> None:
    b.add(_format_date_for_profile(date.fromisoformat(raw), variant, profile_id), "Date")


def _render_clinic_note(case: dict[str, Any], b: SpanBuilder, variant: int) -> None:
    b.add(f"Outpatient clinic note — {case['department']}\n")
    _add_header(b, case, variant)
    _add_care_context(b, case)
    b.add("Consultation date: ")
    _add_date(b, case["encounter_date"], variant + 1, case["language"])
    b.add("\n")
    b.add(_full_name(case["patient"]), "Name:Patient")
    b.add(" is a ")
    age_display = case.get("age_display") or {
        "hyphenated": f"{case['age_years']}-year-old"
    }
    b.add(str(age_display["hyphenated"]), "Age_Birthdate")
    if int(case["age_years"]) < 16:
        b.add(" patient accompanied by ")
        b.add(_full_name(case["relative"]), "Name:Other")
    else:
        b.add(" ")
        b.add(case["profession"], "Profession")
        b.add(" at ")
        b.add(case["other_organisation"], "Organization:Other")
    b.add(f" reporting {', '.join(case['condition']['symptoms'][:2])}. ")
    b.add(f"Assessment: {case['condition']['name']}. Plan: continue ")
    b.add(", ".join(case["condition"]["medications"][:2]))
    b.add(" and review on ")
    _add_date(b, case["followup_date"], variant + 2, case["language"])
    b.add(".")


def _render_discharge_summary(
    case: dict[str, Any], b: SpanBuilder, variant: int
) -> None:
    b.add("Discharge summary\n")
    _add_header(b, case, variant)
    b.add("Admitted to ")
    b.add(case["hospital"], "Organization:Healthcare")
    b.add(" on ")
    _add_date(b, case["encounter_date"], variant + 1, case["language"])
    b.add(f" with {case['condition']['name']}. Symptoms included ")
    b.add(", ".join(case["condition"]["symptoms"][:3]))
    b.add(". Discharge medication: ")
    b.add(", ".join(case["condition"]["medications"][:3]))
    b.add(". Summary authorised by ")
    b.add(f"Dr {_full_name(case['caregiver'])}", "Name:Caregiver")
    b.add(". Copy sent to the patient's relative, ")
    b.add(_full_name(case["relative"]), "Name:Other")
    b.add(", in ")
    b.add(case["caregiver_locality"], "Address_Location:Other")
    b.add(".")


def _render_emergency_note(case: dict[str, Any], b: SpanBuilder, variant: int) -> None:
    b.add("A&E note\n" if case["language"] == "en-GB" else "Emergency department note\n")
    _add_header(b, case, variant)
    b.add("Arrival: ")
    _add_date(b, case["encounter_date"], variant + 1, case["language"])
    b.add(" at ")
    b.add(case["hospital"], "Organization:Healthcare")
    b.add(". Emergency contact: ")
    b.add(_full_name(case["relative"]), "Name:Other")
    b.add(". Presenting complaint: ")
    b.add(", ".join(case["condition"]["symptoms"][:2]))
    b.add(f". Working diagnosis: {case['condition']['name']}.")


def _render_referral_letter(case: dict[str, Any], b: SpanBuilder, variant: int) -> None:
    b.add("Referral letter\n")
    b.add("From: ")
    b.add(f"Dr {_full_name(case['caregiver'])}", "Name:Caregiver")
    b.add(", ")
    b.add(case["hospital"], "Organization:Healthcare")
    b.add(", ")
    b.add(case["caregiver_locality"], "Address_Location:Caregiver")
    b.add("\nRe: ")
    b.add(_full_name(case["patient"]), "Name:Patient")
    b.add(", born ")
    b.add(
        _format_date_for_profile(
            date.fromisoformat(case["birth_date"]), variant, case["language"]
        ),
        "Age_Birthdate",
    )
    b.add(", identifier ")
    b.add(case["patient_id"], "ID:Patient")
    b.add("\nPlease assess this patient for ")
    b.add(case["condition"]["name"])
    b.add(". The planned appointment date is ")
    _add_date(b, case["followup_date"], variant + 1, case["language"])
    b.add(". The patient can be reached on ")
    b.add(case["patient_phone"], "Contactdetails")
    b.add(".")


def _render_laboratory_report(
    case: dict[str, Any], b: SpanBuilder, variant: int
) -> None:
    b.add("Laboratory report\n")
    b.add("Laboratory: ")
    b.add(case["hospital"], "Organization:Healthcare")
    b.add(" | Report: ")
    b.add(case["report_id"], "ID:Patient")
    b.add("\nPatient: ")
    b.add(_full_name(case["patient"]), "Name:Patient")
    b.add(" | MRN: ")
    b.add(case["patient_id"], "ID:Patient")
    b.add(" | Collected: ")
    _add_date(b, case["encounter_date"], variant, case["language"])
    observations = case["condition"].get("observations") or [
        {"name": "Haemoglobin", "value": "118", "unit": "g/L"},
        {"name": "C-reactive protein", "value": "14", "unit": "mg/L"},
    ]
    b.add("\nResults: ")
    b.add(
        "; ".join(
            " ".join(part for part in (row["name"], row["value"], row["unit"]) if part)
            for row in observations[:4]
        )
    )
    b.add(". Validated by ")
    b.add(f"Dr {_full_name(case['caregiver'])}", "Name:Caregiver")
    b.add(".")


def _render_nursing_note(case: dict[str, Any], b: SpanBuilder, variant: int) -> None:
    b.add("Nursing progress note\n")
    b.add("Patient ")
    b.add(_full_name(case["patient"]), "Name:Patient")
    b.add(" was reviewed on ")
    _add_date(b, case["encounter_date"], variant, case["language"])
    b.add(" at ")
    b.add(case["hospital"], "Organization:Healthcare")
    b.add(". Identity confirmed with ")
    b.add(case["patient_id"], "ID:Patient")
    b.add(". The patient reported ")
    b.add(", ".join(case["condition"]["symptoms"][:2]))
    b.add(". Family update given to ")
    b.add(_full_name(case["relative"]), "Name:Other")
    b.add(". Follow-up booked for ")
    _add_date(b, case["followup_date"], variant + 1, case["language"])
    b.add(".")


RENDERERS: dict[str, Callable[[dict[str, Any], SpanBuilder, int], None]] = {
    "clinic_note": _render_clinic_note,
    "discharge_summary": _render_discharge_summary,
    "emergency_note": _render_emergency_note,
    "referral_letter": _render_referral_letter,
    "laboratory_report": _render_laboratory_report,
    "nursing_note": _render_nursing_note,
}


def _render_one(case: dict[str, Any], index: int) -> dict[str, Any]:
    profile_id = str(case.get("language", "")).replace("_", "-")
    if profile_id not in {"en-GB", "en-US"}:
        raise ValueError(
            f"English generation profile cannot render case language {case.get('language')!r}"
        )
    if any(
        isinstance(span, dict) and span.get("label") == "Anonymize_Other"
        for span in case.get("spans", ())
    ):
        raise ValueError(
            f"{profile_id} synthetic case records must not contain Anonymize_Other"
        )
    if not is_approved_synthetic_phone(
        str(case.get("patient_phone", "")), profile_id=profile_id
    ):
        raise ValueError(
            f"{profile_id} synthetic case uses an unapproved telephone number"
        )
    for key in ("patient_id", "caregiver_id", "report_id", "national_id"):
        if not is_approved_synthetic_identifier(
            str(case.get(key, "")), profile_id=profile_id
        ):
            raise ValueError(
                f"{profile_id} synthetic case uses an unapproved identifier in {key}"
            )
    document_type = str(case.get("document_type", ""))
    try:
        renderer = RENDERERS[document_type]
    except KeyError as exc:
        supported = ", ".join(RENDERERS)
        raise ValueError(
            f"unsupported {profile_id} document type {document_type!r}; expected: {supported}"
        ) from exc
    b = SpanBuilder()
    renderer(case, b, index)
    case_id = str(case.get("case_id", f"case-{index + 1:05d}"))
    doc = b.doc(case_id.replace("case-", "synthetic-"), _metadata(case, document_type))
    labels = {span.get("label") for span in doc.get("spans", [])}
    disallowed = sorted(label for label in labels if label not in ALLOWED_LABEL_SET)
    if disallowed:
        raise ValueError(
            f"{profile_id} synthetic generation emitted disallowed labels: "
            f"{', '.join(disallowed)}"
        )
    problems = validate_record(doc)
    if problems:
        raise RuntimeError(f"{profile_id} renderer emitted an invalid document: {problems}")
    return doc


def render_case_records(
    records: list[dict[str, Any]], *, seed: int = 20260508
) -> list[dict[str, Any]]:
    del seed  # rendering variability is pinned to case order and case contents
    return [_render_one(case, index) for index, case in enumerate(records)]


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
        build_case_records(count, seed=seed, synthea_seeds=synthea_seeds),
        seed=seed,
    )


@dataclass
class ReviewResult:
    document_id: str
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)

    def fail(self, issue: str) -> None:
        self.passed = False
        self.issues.append(issue)


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b07700\s+900\d{3}\b")
PATIENT_ID_RE = re.compile(r"\b(?:MRN-GB|REPORT-GB)-\d{6}\b")
CAREGIVER_ID_RE = re.compile(r"\bGMC-TEST-\d{7}\b")
US_PHONE_RE = re.compile(r"\(\d{3}\)\s+555-01\d{2}\b")
US_PATIENT_ID_RE = re.compile(r"\b(?:MRN-US|REPORT-US)-\d{6}\b")
US_CAREGIVER_ID_RE = re.compile(r"\b1234567893\b")
AGE_RE = re.compile(
    r"\b\d{1,3}-(?:day|week|month|year)-old\b", re.IGNORECASE
)
DATE_RE = re.compile(
    r"\b(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
    re.IGNORECASE,
)
US_DATE_RE = re.compile(
    r"\b(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4})\b",
    re.IGNORECASE,
)
DUTCH_LEAK_RE = re.compile(
    r"\b(?:patiënt|geboren|huisarts|zorginstelling|behandelaar|ontslagbrief|spoednota|verslagdatum)\b",
    re.IGNORECASE,
)


def _covered(doc: dict[str, Any], begin: int, end: int, labels: set[str]) -> bool:
    return any(
        span["label"] in labels and span["begin"] <= begin and span["end"] >= end
        for span in doc.get("spans", [])
    )


def review_documents(
    docs: list[dict[str, Any]], **_: Any
) -> tuple[list[dict[str, Any]], list[ReviewResult], list[dict[str, Any]]]:
    results: list[ReviewResult] = []
    for doc in docs:
        result = ReviewResult(str(doc.get("document_id", "unknown")))
        for problem in validate_record(doc):
            result.fail(problem)
        metadata = doc.get("metadata", {})
        profile_id = str(metadata.get("lang", ""))
        if profile_id not in {"en-GB", "en-US"}:
            result.fail("metadata.lang must be 'en-GB' or 'en-US'")
        if metadata.get("generation_profile") != profile_id:
            result.fail("document does not pin its English generation profile")
        disallowed = sorted(
            {
                str(span.get("label"))
                for span in doc.get("spans", [])
                if span.get("label") not in ALLOWED_LABEL_SET
            }
        )
        if disallowed:
            result.fail(f"Disallowed synthetic labels: {', '.join(disallowed)}")
        for span in doc.get("spans", []):
            if span.get("label") == "Contactdetails" and "@" not in str(
                span.get("text", "")
            ) and re.search(
                r"\d", str(span.get("text", ""))
            ) and not is_approved_synthetic_phone(
                str(span.get("text", "")), profile_id=profile_id
            ):
                result.fail(
                    f"Unapproved synthetic telephone number: {span.get('text')!r}"
                )
            if span.get("label") in {"ID:Patient", "ID:Caregiver"} and not is_approved_synthetic_identifier(
                str(span.get("text", "")), profile_id=profile_id
            ):
                result.fail(f"Unapproved synthetic identifier: {span.get('text')!r}")
        if len(doc.get("spans", [])) < 3:
            result.fail("English reference document has insufficient PII coverage")
        leak = DUTCH_LEAK_RE.search(doc.get("text", ""))
        if leak:
            result.fail(
                f"Dutch renderer text leaked into English output: {leak.group()!r}"
            )
        locale_checks = (
            (PHONE_RE, PATIENT_ID_RE, CAREGIVER_ID_RE, DATE_RE)
            if profile_id == "en-GB"
            else (US_PHONE_RE, US_PATIENT_ID_RE, US_CAREGIVER_ID_RE, US_DATE_RE)
        )
        checks = (
            (EMAIL_RE, {"Contactdetails"}),
            (locale_checks[0], {"Contactdetails"}),
            (locale_checks[1], {"ID:Patient"}),
            (locale_checks[2], {"ID:Caregiver"}),
            (AGE_RE, {"Age_Birthdate"}),
            (locale_checks[3], {"Age_Birthdate", "Date"}),
        )
        for pattern, labels in checks:
            for match in pattern.finditer(doc.get("text", "")):
                if not _covered(doc, match.start(), match.end(), labels):
                    result.fail(
                        f"Potential missed {min(labels)}: {match.group()!r} "
                        f"at {match.start()}:{match.end()}"
                    )
        results.append(result)
    return docs, results, []


def _resource_manifest() -> dict[str, Any]:
    language_manifest = get_language_profile(PROFILE_ID).manifest()
    return {
        "manifest_version": "meddeid.generation-resources.v1",
        "package": "meddeid-data",
        "package_version": "0.4.1",
        "profile_id": PROFILE_ID,
        "maturity": "audited-language-pack-profile",
        "resources": language_manifest["resources"]["resources"],
        "language_resources": language_manifest["resources"],
        "allowed_labels": list(ALLOWED_LABELS),
        "provenance": {
            "description": (
                "Audited en-GB language-pack resources. Person names are independently "
                "recombined; public institutions and geographic entities may be real."
            ),
            "fictional_phone_range": "Ofcom 07700 900000-900999 drama range",
            "fictional_phone_source": (
                "https://www.ofcom.org.uk/phones-and-broadband/phone-numbers/"
                "numbers-for-drama"
            ),
            "synthetic_identifier_formats": (
                "deterministic mix dominated by numeric-only values, with compact, "
                "grouped, slash, and short-prefix internal formats"
            ),
        },
        "document_types": list(RENDERERS),
    }


EN_GB_GENERATION_PROFILE = GenerationProfile(
    profile_id=PROFILE_ID,
    language_tags=("en-GB",),
    description="English clinical notes in the United Kingdom setting",
    generate_documents=generate_documents,
    build_case_records=build_case_records,
    render_case_records=render_case_records,
    review_documents=review_documents,
    resource_manifest_provider=_resource_manifest,
    language_manifest_provider=lambda: get_language_profile("en-GB").manifest(),
)
