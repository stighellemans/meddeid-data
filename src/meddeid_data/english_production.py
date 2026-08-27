"""Batch-gated production pipeline for the combined English training corpus."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meddeid_core import BERT_ENTITY_LABELS, normalize_record, validate_record
from meddeid_language_en import lookup_values

from .corpus_quality import CorpusDiversityContract, audit_corpus_diversity
from .email_domains import choose_email_domain
from .identifier_formats import (
    SUPPORTED_SLOTS,
    format_english_identifier,
    synthetic_national_identifier,
)
from .generation_llm_en import (
    DEFAULT_ENGLISH_LLM_MODEL,
    EnglishLLMRenderError,
    hard_negative_targets,
    render_case_record_with_luna,
    review_english_document_with_luna,
    validate_english_llm_document,
)
from .generation_profiles import resolve_generation_profile
from .organization_formats import format_healthcare_organization
from .production_backends import LUNA_STANDARD_PRICING, estimate_usage_cost

PRODUCTION_TOTAL = 7_000
PRODUCTION_PER_PROFILE = 3_500
BATCH_SIZE = 500
BENCHMARK_PER_PROFILE = 150
PEDIATRIC_PER_PROFILE_AFTER_BATCH_1 = 41


@contextmanager
def _exclusive_generation_lock(output_dir: Path, batch_index: int):
    """Prevent overlapping writers for one production batch."""

    directory = _batch_paths(output_dir, batch_index)["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".generation.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"batch {batch_index + 1} already has an active generation process"
            ) from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
PEDIATRIC_FINAL_MINIMUM = 1_050
PEDIATRIC_FINAL_PER_PROFILE_MINIMUM = 525
PEDIATRIC_BENCHMARK_PER_PROFILE = 24
BENCHMARK_INFANT_UNIT_BY_DOCUMENT = {
    "clinic_note": "weeks",
    "discharge_summary": "days",
    "emergency_note": "months",
    "referral_letter": "days",
    "laboratory_report": "weeks",
    "nursing_note": "months",
}
PEDIATRIC_BAND_MINIMUMS_PER_PROFILE = {
    "infant": 5,
    "early_childhood": 8,
    "school_age": 12,
    "adolescent": 16,
}
WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)?", re.UNICODE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/+-]\d+)*\b")
SPACE_RE = re.compile(r"\s+")
HYPHENATED_AGE_OLD_SPAN_RE = re.compile(
    r"^(?P<age>\d{1,3}-(?:day|week|month|year))-old\b",
    re.IGNORECASE,
)
CONTEXTUAL_AGE_SPAN_RE = re.compile(
    r"^(?:aged|age)\s+(?P<age>\d{1,3}(?:\s+(?:day|week|month|year)s?)?)$",
    re.IGNORECASE,
)
SYNTHETIC_MRN_RE = re.compile(r"^MRN-(?:GB|US)-(?P<identifier>\d{6})$")
LEGACY_EMAIL_RE = re.compile(
    r"(?P<local>[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+)@example\.test\b"
)
DEMOGRAPHIC_ETHNICITIES = (
    "African American",
    "African-American",
    "Black",
    "White",
    "Caucasian",
    "Hispanic",
    "Latino",
    "Asian",
    "South Asian",
    "East Asian",
    "Middle Eastern",
    "Native American",
    "Pacific Islander",
    "multiracial",
)
DEMOGRAPHIC_PROFILES = tuple(
    (ethnicity, sex)
    for ethnicity in DEMOGRAPHIC_ETHNICITIES
    for sex in ("male", "female")
)
DEMOGRAPHIC_PREFIX_RE = re.compile(
    r"^(?:African American|African-American|Black|White|Caucasian|Hispanic|Latino|Asian|South Asian|"
    r"East Asian|Middle Eastern|Native American|Pacific Islander|multiracial)\b",
    re.IGNORECASE,
)
SEX_NOUN_RE = re.compile(r"^(?:male|female|man|woman|boy|girl)\b", re.IGNORECASE)
REQUIRED_500_HARD_NEGATIVES = frozenset(
    {
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
)
REQUIRED_500_FORMAT_SLOTS = frozenset(
    {
        "patient.name",
        "patient.name_upper",
        "patient.name_lower",
        "patient.given_name",
        "patient.initial_surname",
        "patient.given_family_initial",
        "caregiver.credentialed_name",
        "caregiver.initial_surname",
        "relative.name_upper",
        "relative.name_lower",
        "relative.initial_surname",
        "patient.birth_date",
        "patient.age_hyphenated",
        "patient.age_contextual",
        "patient.age_abbreviated",
    }
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_record(row), ensure_ascii=False) + "\n")


def normalize_english_age_birthdate_boundaries(record: dict[str, Any]) -> int:
    """Keep only the semantic age value inside ``Age_Birthdate`` spans.

    The clinical text is left unchanged.  For example, an ``Age_Birthdate`` span
    over ``50-year-old African American woman`` becomes ``50-year``.  Short
    clinical forms such as ``50 y/o`` are deliberately unaffected.  Contextual
    cues are excluded, so ``aged 43`` becomes ``43`` and ``age 43 years`` becomes
    ``43 years``.

    Returns the number of adjusted spans.
    """

    text = str(record.get("text", ""))
    changed = 0
    for span in record.get("spans", []):
        if span.get("label") != "Age_Birthdate":
            continue
        begin = span.get("begin")
        end = span.get("end")
        if not isinstance(begin, int) or not isinstance(end, int):
            continue
        if begin < 0 or end > len(text) or begin >= end:
            continue
        value = text[begin:end]
        hyphenated = HYPHENATED_AGE_OLD_SPAN_RE.match(value)
        contextual = CONTEXTUAL_AGE_SPAN_RE.fullmatch(value)
        if hyphenated is not None:
            new_begin = begin
            new_end = begin + hyphenated.end("age")
        elif contextual is not None:
            new_begin = begin + contextual.start("age")
            new_end = begin + contextual.end("age")
        else:
            continue
        if new_begin == begin and new_end == end:
            continue
        span["begin"] = new_begin
        span["end"] = new_end
        span["text"] = text[new_begin:new_end]
        changed += 1
    return changed


def _replace_record_text(
    record: dict[str, Any],
    begin: int,
    end: int,
    replacement: str,
    *,
    replaced_span: dict[str, Any] | None = None,
) -> None:
    """Replace one text range and shift all affected annotation offsets safely."""

    text = str(record.get("text", ""))
    if not (0 <= begin <= end <= len(text)):
        raise ValueError(f"invalid text replacement range {begin}:{end}")
    delta = len(replacement) - (end - begin)
    for span in record.get("spans", []):
        span_begin = int(span["begin"])
        span_end = int(span["end"])
        if span is replaced_span:
            if span_begin != begin or span_end != end:
                raise ValueError("replacement target does not match its annotation range")
            span["end"] = begin + len(replacement)
            span["text"] = replacement
            continue
        if span_end <= begin:
            continue
        if span_begin >= end:
            span["begin"] = span_begin + delta
            span["end"] = span_end + delta
            continue
        raise ValueError(
            "text replacement would partially overlap another annotation: "
            f"{span_begin}:{span_end} versus {begin}:{end}"
        )
    record["text"] = text[:begin] + replacement + text[end:]


def normalize_english_mrn_values(record: dict[str, Any]) -> int:
    """Convert synthetic ``MRN-GB/US-123456`` values to annotated ``123456``."""

    changed = 0
    for span in list(record.get("spans", [])):
        if span.get("label") != "ID:Patient" or span.get("source_slot") != "patient.mrn":
            continue
        match = SYNTHETIC_MRN_RE.fullmatch(str(span.get("text", "")))
        if match is None:
            continue
        _replace_record_text(
            record,
            int(span["begin"]),
            int(span["end"]),
            match.group("identifier"),
            replaced_span=span,
        )
        changed += 1
    return changed


def _email_domain(profile_id: str, local: str) -> str:
    if profile_id not in {"en-GB", "en-US"}:
        raise ValueError(f"unsupported English profile for email domain: {profile_id!r}")
    return choose_email_domain(profile_id, local)


def _replace_legacy_email(value: str, profile_id: str) -> str:
    return LEGACY_EMAIL_RE.sub(
        lambda match: f"{match.group('local')}@{_email_domain(profile_id, match.group('local'))}",
        value,
    )


def diversify_english_email_domains(record: dict[str, Any]) -> int:
    """Replace legacy ``example.test`` emails and preserve annotation offsets."""

    profile_id = str(record.get("metadata", {}).get("lang", ""))
    if profile_id not in {"en-GB", "en-US"}:
        return 0
    changed = 0
    for span in list(record.get("spans", [])):
        if span.get("label") != "Contactdetails":
            continue
        current = str(span.get("text", ""))
        replacement = _replace_legacy_email(current, profile_id)
        if replacement == current:
            continue
        _replace_record_text(
            record,
            int(span["begin"]),
            int(span["end"]),
            replacement,
            replaced_span=span,
        )
        changed += 1

    if not changed:
        return 0
    metadata = record.setdefault("metadata", {})
    for target in metadata.get("required_pii_targets", []):
        if isinstance(target.get("value"), str):
            target["value"] = _replace_legacy_email(target["value"], profile_id)
    metadata["email_domain_diversification"] = {
        "version": "english-email-domain-diversify-v1",
        "replacement_count": changed,
    }
    return changed


def diversify_english_identifier_formats(record: dict[str, Any]) -> int:
    """Diversify English ID surfaces while preserving all annotation offsets."""

    metadata = record.get("metadata", {})
    profile_id = str(metadata.get("lang", ""))
    if profile_id not in {"en-GB", "en-US"}:
        return 0
    production = metadata.get("production") or {}
    identity_key = str(
        production.get("profile_index", record.get("document_id", "unknown"))
    )
    replacements_by_slot: dict[str, str] = {}
    changed = 0
    for span in list(record.get("spans", [])):
        source_slot = str(span.get("source_slot", ""))
        if source_slot not in SUPPORTED_SLOTS:
            continue
        replacement = replacements_by_slot.setdefault(
            source_slot,
            format_english_identifier(profile_id, source_slot, identity_key),
        )
        current = str(span.get("text", ""))
        if replacement == current:
            continue
        _replace_record_text(
            record,
            int(span["begin"]),
            int(span["end"]),
            replacement,
            replaced_span=span,
        )
        changed += 1

    if not changed:
        return 0
    for target in metadata.get("required_pii_targets", []):
        source_slot = str(target.get("slot", ""))
        if source_slot in replacements_by_slot:
            target["value"] = replacements_by_slot[source_slot]
    metadata["identifier_format_diversification"] = {
        "version": "english-identifier-formats-v1",
        "replacement_count": changed,
    }
    return changed


def rebalance_healthcare_organization_case(record: dict[str, Any]) -> int:
    """Recase healthcare organizations without changing their annotations."""

    metadata = record.get("metadata", {})
    profile_id = str(metadata.get("lang", ""))
    if profile_id not in {"en-GB", "en-US"}:
        return 0
    production = metadata.get("production") or {}
    identity_key = str(
        production.get("profile_index", record.get("document_id", "unknown"))
    )
    replacements_by_value: dict[str, str] = {}
    changed = 0
    for span in list(record.get("spans", [])):
        if (
            span.get("label") != "Organization:Healthcare"
            or span.get("source_slot") != "healthcare.organization"
        ):
            continue
        current = str(span.get("text", ""))
        replacement = replacements_by_value.setdefault(
            current,
            format_healthcare_organization(profile_id, current, identity_key),
        )
        if replacement == current:
            continue
        _replace_record_text(
            record,
            int(span["begin"]),
            int(span["end"]),
            replacement,
            replaced_span=span,
        )
        changed += 1

    if not changed:
        return 0
    for target in metadata.get("required_pii_targets", []):
        if str(target.get("slot", "")) != "healthcare.organization":
            continue
        current = str(target.get("value", ""))
        target["value"] = replacements_by_value.get(
            current,
            format_healthcare_organization(profile_id, current, identity_key),
        )
    metadata["healthcare_organization_case_rebalancing"] = {
        "version": "healthcare-organization-case-v1",
        "replacement_count": changed,
        "uppercase_share_target_percent": 30,
    }
    return changed


def inject_national_patient_identifier(
    record: dict[str, Any], *, prevalence_percent: int = 40
) -> int:
    """Post-hoc add one public/national patient ID without regenerating prose.

    Selection and values are deterministic. The public benchmark label remains
    ``ID:Patient`` while ``source_slot`` records that this is not an MRN or a
    report/accession number.
    """

    if not 0 <= prevalence_percent <= 100:
        raise ValueError("prevalence_percent must be between 0 and 100")
    metadata = record.setdefault("metadata", {})
    profile_id = str(metadata.get("lang", ""))
    if profile_id not in {"en-GB", "en-US"}:
        return 0
    if any(
        str(span.get("source_slot", "")) == "patient.national_id"
        for span in record.get("spans", [])
    ):
        return 0

    document_id = str(record.get("document_id", ""))
    selector = int.from_bytes(
        hashlib.sha256(f"national-id-injection-v1|{document_id}".encode()).digest()[:4],
        "big",
    ) % 100
    if selector >= prevalence_percent:
        return 0

    production = metadata.get("production") or {}
    identity_key = str(production.get("profile_index", document_id))
    national_identifier = synthetic_national_identifier(profile_id, identity_key)
    value = national_identifier["value"]
    field_label = national_identifier["field_label"]
    text = str(record.get("text", ""))
    first_newline = text.find("\n")
    insertion_point = first_newline + 1 if first_newline >= 0 else 0
    prefix = f"{field_label}: "
    insertion = f"{prefix}{value}\n"
    _replace_record_text(record, insertion_point, insertion_point, insertion)

    begin = insertion_point + len(prefix)
    record.setdefault("spans", []).append(
        {
            "begin": begin,
            "end": begin + len(value),
            "text": value,
            "label": "ID:Patient",
            "category": "ID",
            "subtype": "Patient",
            "confirmed": True,
            "source_slot": "patient.national_id",
            "semantic_type": "patient_identifier.national",
        }
    )
    record["spans"].sort(key=lambda span: (int(span["begin"]), int(span["end"])))
    metadata.setdefault("required_pii_targets", []).append(
        {
            "slot": "patient.national_id",
            "value": value,
            "label": "ID:Patient",
            "semantic_type": "patient_identifier.national",
        }
    )
    metadata["national_identifier_injection"] = {
        "version": "national-id-injection-v1",
        "kind": national_identifier["kind"],
        "field_label": field_label,
    }
    return 1


def inject_demographic_hard_negative(
    record: dict[str, Any], profile_index: int
) -> int:
    """Add an unlabeled ethnicity/sex descriptor after one hyphenated age.

    Injection is restricted to the reviewed ``N-unit-old`` construction, where
    appending ``ethnicity sex`` before the following patient noun is grammatical.
    The demographic text is intentionally not added to ``spans``.
    """

    text = str(record.get("text", ""))
    candidate = next(
        (
            span
            for span in record.get("spans", [])
            if span.get("label") == "Age_Birthdate"
            and span.get("source_slot") == "patient.age_hyphenated"
            and text[int(span["end"]) : int(span["end"]) + 4].lower() == "-old"
        ),
        None,
    )
    if candidate is None:
        return 0
    insertion_point = int(candidate["end"]) + 4
    following = text[insertion_point:].lstrip()
    if DEMOGRAPHIC_PREFIX_RE.match(following):
        return 0

    ethnicity, sex = DEMOGRAPHIC_PROFILES[profile_index % len(DEMOGRAPHIC_PROFILES)]
    existing_sex = SEX_NOUN_RE.match(following)
    if existing_sex is not None:
        descriptor = f" {ethnicity}"
        rendered_sex = existing_sex.group(0).lower()
    else:
        descriptor = f" {ethnicity} {sex}"
        rendered_sex = sex
    _replace_record_text(record, insertion_point, insertion_point, descriptor)
    metadata = record.setdefault("metadata", {})
    metadata["demographic_hard_negative"] = {
        "ethnicity": ethnicity,
        "sex": rendered_sex,
        "annotated": False,
    }
    return 1


def _write_jsonl_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_sha256(document: dict[str, Any]) -> str:
    """Hash canonical document content for review/sign-off binding."""

    payload = json.dumps(
        normalize_record(document),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def _document_skeleton(doc: dict[str, Any]) -> str:
    text = str(doc.get("text", ""))
    parts: list[str] = []
    cursor = 0
    for span in sorted(doc.get("spans", []), key=lambda row: (row["begin"], row["end"])):
        parts.append(text[cursor : int(span["begin"])])
        parts.append(f" <{span['label']}> ")
        cursor = int(span["end"])
    parts.append(text[cursor:])
    return SPACE_RE.sub(" ", NUMBER_RE.sub("<N>", "".join(parts)).lower()).strip()


def _simhash(text: str) -> int:
    tokens = _tokens(text)
    shingles = [" ".join(tokens[index : index + 5]) for index in range(max(1, len(tokens) - 4))]
    vector = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _batch_paths(output_dir: Path, batch_index: int) -> dict[str, Path]:
    directory = output_dir / "batches" / f"batch-{batch_index + 1:02d}"
    return {
        "directory": directory,
        "cases": directory / "cases.jsonl",
        "docs": directory / "documents.jsonl",
        "marked": directory / "marked.jsonl",
        "failures": directory / "failures.jsonl",
        "usage_ledger": directory / "usage-attempts.jsonl",
        "invalidated": directory / "invalidated-documents.jsonl",
        "case_migrations": directory / "case-migrations.jsonl",
        "report_json": directory / "quality-report.json",
        "report_md": directory / "quality-report.md",
        "review": directory / "review-packet.md",
        "review_decisions": directory / "personal-review-decisions.jsonl",
        "signoff": directory / "personal-review-signoff.json",
        "manifest": directory / "manifest.json",
    }


def build_batch_cases(
    *, batch_index: int, batch_size: int = BATCH_SIZE, seed: int = 20260820
) -> list[dict[str, Any]]:
    if batch_size % 2:
        raise ValueError("English production batches must contain an even GB/US split")
    count_per_profile = batch_size // 2
    start_index = batch_index * count_per_profile
    rows: list[dict[str, Any]] = []
    for profile_id in ("en-GB", "en-US"):
        profile = resolve_generation_profile(profile_id)
        profile_rows = profile.build_case_records(
            count_per_profile,
            seed=seed,
            start_index=start_index,
        )
        for offset, row in enumerate(profile_rows):
            row["case_id"] = _canonical_case_id(row)
            row["production"] = {
                "corpus_contract": "meddeid.english-synthetic-corpus.v1",
                "batch_index": batch_index,
                "profile_index": start_index + offset,
                "independently_authored": True,
                "copied_source_text": False,
            }
        rows.extend(profile_rows)
    rows.sort(
        key=lambda row: (
            int(row["production"]["profile_index"]),
            0 if row["language"] == "en-GB" else 1,
        )
    )
    if batch_size >= BATCH_SIZE:
        _ensure_hard_negative_coverage(rows)
    return rows


def _ensure_hard_negative_coverage(rows: list[dict[str, Any]]) -> None:
    """Deterministically cover every required difficult-negative family."""

    selected = {
        str(row["case_id"]): hard_negative_targets(row)[0]["category"]
        for row in rows
    }
    counts = Counter(selected.values())
    for missing in sorted(REQUIRED_500_HARD_NEGATIVES - set(counts)):
        replacement: dict[str, Any] | None = None
        for row in rows:
            current = selected[str(row["case_id"])]
            if counts[current] <= 1:
                continue
            try:
                hard_negative_targets(row, category_override=missing)
            except ValueError:
                continue
            replacement = row
            break
        if replacement is None:
            raise RuntimeError(
                f"cannot construct required hard-negative category {missing!r}"
            )
        case_id = str(replacement["case_id"])
        displaced = selected[case_id]
        replacement.setdefault("production", {})[
            "hard_negative_category_override"
        ] = missing
        selected[case_id] = missing
        counts[displaced] -= 1
        counts[missing] += 1


def _canonical_case_id(case: dict[str, Any]) -> str:
    """Return a corpus-wide case identifier rather than a locale-local one."""
    language = str(case.get("language", "")).lower()
    match = re.search(r"(\d+)$", str(case.get("case_id", "")))
    if language not in {"en-gb", "en-us"} or match is None:
        raise ValueError(f"cannot canonicalize case identity: {case!r}")
    return f"{language}-case-{int(match.group(1)):05d}"


def _doc_id_for_case(case: dict[str, Any]) -> str:
    profile_id = str(case["language"]).lower()
    match = re.search(r"(\d+)$", str(case["case_id"]))
    if match is None:
        raise ValueError(f"case ID has no numeric suffix: {case['case_id']!r}")
    return f"{profile_id}-synthetic-{int(match.group(1)):05d}"


def _upgrade_case_traceability(
    *, paths: dict[str, Path], cases: list[dict[str, Any]], docs: list[dict[str, Any]]
) -> None:
    """Migrate pre-contract checkpoints whose case IDs collided across locales."""
    changed_cases = False
    for case in cases:
        canonical = _canonical_case_id(case)
        if case.get("case_id") != canonical:
            case["case_id"] = canonical
            changed_cases = True
    if changed_cases:
        _write_jsonl_raw(paths["cases"], cases)

    changed_docs = False
    for doc in docs:
        metadata = doc.get("metadata", {})
        source = str(metadata.get("source_case_id", ""))
        match = re.search(r"(\d+)$", source)
        language = str(metadata.get("lang", "")).lower()
        if match is not None and language in {"en-gb", "en-us"}:
            canonical = f"{language}-case-{int(match.group(1)):05d}"
            if source != canonical:
                metadata["source_case_id"] = canonical
                changed_docs = True
    if changed_docs:
        _write_jsonl(paths["docs"], docs)

    ledger = _read_jsonl(paths["usage_ledger"])
    changed_ledger = False
    for event in ledger:
        language = str(event.get("language", "")).lower()
        match = re.search(r"(\d+)$", str(event.get("case_id", "")))
        if match is not None and language in {"en-gb", "en-us"}:
            canonical = f"{language}-case-{int(match.group(1)):05d}"
            if event.get("case_id") != canonical:
                event["case_id"] = canonical
                changed_ledger = True
    if changed_ledger:
        _write_jsonl_raw(paths["usage_ledger"], ledger)


def _is_pediatric_case(case: dict[str, Any]) -> bool:
    return int(case.get("age_years", 18)) < 18


def _pediatric_band(case: dict[str, Any]) -> str:
    explicit = str(case.get("age_group", ""))
    if explicit in PEDIATRIC_BAND_MINIMUMS_PER_PROFILE:
        return explicit
    age = int(case.get("age_years", 18))
    if age < 1:
        return "infant"
    if age < 5:
        return "early_childhood"
    if age < 12:
        return "school_age"
    if age < 18:
        return "adolescent"
    return "adult"


def _transient_error(exc: Exception) -> bool:
    value = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in value
        for token in (
            "rate limit",
            "ratelimit",
            "timeout",
            "connection",
            "temporarily",
            "server error",
            "service unavailable",
            "gateway",
        )
    )


async def _render_one(
    *,
    case: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    validation_retries: int,
    api_retries: int,
    model_review: bool,
    usage_recorder: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    feedback_value = case.get("production", {}).get("review_feedback")
    feedback: str | None = str(feedback_value) if feedback_value else None
    last_error: Exception | None = None
    for validation_attempt in range(validation_retries + 1):
        for api_attempt in range(api_retries + 1):
            try:
                def record_usage(event: dict[str, Any]) -> None:
                    if usage_recorder is None:
                        return
                    usage_recorder(
                        {
                            "recorded_at": datetime.now(timezone.utc).isoformat(),
                            "document_id": _doc_id_for_case(case),
                            "case_id": case.get("case_id"),
                            "language": case.get("language"),
                            "validation_attempt": validation_attempt + 1,
                            "api_attempt": api_attempt + 1,
                            **event,
                        }
                    )

                async with semaphore:
                    result = await render_case_record_with_luna(
                        case,
                        client=client,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        reasoning_effort=reasoning_effort,
                        retry_feedback=feedback,
                        usage_recorder=record_usage,
                    )
                if model_review:
                    async with semaphore:
                        review = await review_english_document_with_luna(
                            record=case,
                            doc=result.doc,
                            client=client,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            usage_recorder=record_usage,
                        )
                    result.doc["metadata"]["clinical_model_review"] = review
                    if review.get("verdict") != "pass" or review.get("issues"):
                        compact = "; ".join(
                            f"{issue.get('category')}: {issue.get('explanation')} Fix: {issue.get('required_fix')}"
                            for issue in review.get("issues", [])[:8]
                        )
                        raise EnglishLLMRenderError(
                            "independent clinical review requires revision: " + compact
                        )
                result.doc["metadata"]["generation_attempts"] = validation_attempt + 1
                marked = {
                    "document_id": result.doc["document_id"],
                    "marked_text": result.marked_text,
                    "usage": result.usage,
                    "response_id": result.response_id,
                }
                return result.doc, marked
            except Exception as exc:
                last_error = exc
                if api_attempt < api_retries and _transient_error(exc):
                    await asyncio.sleep(min(30.0, 2.0**api_attempt))
                    continue
                break
        feedback = (
            "The prior document failed mandatory local review. Re-author the complete note "
            f"and fix these issues: {last_error}"
        )
    raise EnglishLLMRenderError(
        f"could not render {case.get('language')} {case.get('case_id')}: {last_error}"
    )


async def generate_batch(
    *,
    output_dir: Path,
    batch_index: int,
    batch_size: int,
    seed: int,
    model: str,
    concurrency: int,
    max_output_tokens: int,
    reasoning_effort: str,
    validation_retries: int,
    api_retries: int,
    resume: bool,
    model_review: bool,
) -> int:
    try:
        from dotenv import load_dotenv
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - exercised only without llm extra
        raise RuntimeError("Install meddeid-data[llm] before LLM generation") from exc

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    paths = _batch_paths(output_dir, batch_index)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    if resume and paths["cases"].is_file():
        cases = _read_jsonl(paths["cases"])
        if len(cases) != batch_size:
            raise RuntimeError(
                f"resume case manifest has {len(cases)} rows, expected {batch_size}"
            )
    else:
        cases = build_batch_cases(batch_index=batch_index, batch_size=batch_size, seed=seed)
        _write_jsonl_raw(paths["cases"], cases)

    existing_docs = _read_jsonl(paths["docs"]) if resume else []
    if resume:
        _upgrade_case_traceability(paths=paths, cases=cases, docs=existing_docs)
    existing_ids = {str(row.get("document_id")) for row in existing_docs}
    if not resume:
        for key in ("docs", "marked", "failures", "usage_ledger"):
            if paths[key].exists():
                paths[key].unlink()

    def record_usage(event: dict[str, Any]) -> None:
        recorded = dict(event)
        recorded["estimated_cost_usd"] = round(
            estimate_usage_cost(recorded.get("usage"), LUNA_STANDARD_PRICING),
            9,
        )
        recorded["pricing"] = {
            "model": "gpt-5.6-luna",
            "service_tier": "standard",
            "context": "short",
            "usd_per_million_tokens": dict(LUNA_STANDARD_PRICING),
            "pinned_at": "2026-08-20",
            "source": "https://platform.openai.com/pricing",
        }
        _append_jsonl(paths["usage_ledger"], recorded)
    pending = [case for case in cases if _doc_id_for_case(case) not in existing_ids]
    client = AsyncOpenAI(timeout=180.0, max_retries=0)
    # Bound complete author -> validate -> review chains. If every case queues
    # its author call against one semaphore, the later review calls sit behind
    # hundreds of authors and no completed document is checkpointed for a long
    # time. Separate pipeline/API semaphores preserve the same request ceiling
    # while allowing each case to finish end to end.
    pipeline_semaphore = asyncio.Semaphore(max(1, concurrency))
    api_semaphore = asyncio.Semaphore(max(1, concurrency))

    async def render_bounded(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        async with pipeline_semaphore:
            return await _render_one(
                case=case,
                client=client,
                semaphore=api_semaphore,
                model=model,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                validation_retries=validation_retries,
                        api_retries=api_retries,
                        model_review=model_review,
                        usage_recorder=record_usage,
                    )

    tasks = [
        asyncio.create_task(render_bounded(case))
        for case in pending
    ]
    started = time.monotonic()
    failures = 0
    completed = len(existing_docs)
    for future in asyncio.as_completed(tasks):
        try:
            doc, marked = await future
            _append_jsonl(paths["docs"], doc)
            _append_jsonl(paths["marked"], marked)
            completed += 1
            if completed % 10 == 0 or completed == batch_size:
                elapsed = time.monotonic() - started
                print(f"batch {batch_index + 1}: rendered {completed}/{batch_size} ({elapsed:.1f}s)", flush=True)
        except Exception as exc:
            failures += 1
            _append_jsonl(
                paths["failures"],
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            print(f"batch {batch_index + 1}: FAILED {exc}", flush=True)
    await client.close()

    docs = sorted(_read_jsonl(paths["docs"]), key=lambda row: str(row["document_id"]))
    _write_jsonl(paths["docs"], docs)
    report = audit_batch(output_dir=output_dir, batch_index=batch_index, expected=batch_size)
    write_batch_reports(paths=paths, docs=docs, report=report)
    return 0 if failures == 0 and report["gate_passed"] else 1


def _prior_documents(output_dir: Path, batch_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(batch_index):
        rows.extend(_read_jsonl(_batch_paths(output_dir, index)["docs"]))
    return rows


def _usage_totals(docs: list[dict[str, Any]]) -> dict[str, int | float]:
    input_tokens = output_tokens = cached_tokens = cache_write_tokens = 0
    for doc in docs:
        usage = doc.get("metadata", {}).get("openai_usage") or {}
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        details = usage.get("input_tokens_details") or {}
        cached_tokens += int(details.get("cached_tokens", 0) or 0)
        cache_write_tokens += int(details.get("cache_write_tokens", 0) or 0)
        review_usage = doc.get("metadata", {}).get("clinical_model_review", {}).get("usage") or {}
        input_tokens += int(review_usage.get("input_tokens", 0) or 0)
        output_tokens += int(review_usage.get("output_tokens", 0) or 0)
        review_details = review_usage.get("input_tokens_details") or {}
        cached_tokens += int(review_details.get("cached_tokens", 0) or 0)
        cache_write_tokens += int(review_details.get("cache_write_tokens", 0) or 0)
    uncached = max(0, input_tokens - cached_tokens - cache_write_tokens)
    estimated_usd = (
        uncached * LUNA_STANDARD_PRICING["input"]
        + cached_tokens * LUNA_STANDARD_PRICING["cached_input"]
        + cache_write_tokens * LUNA_STANDARD_PRICING["cache_write"]
        + output_tokens * LUNA_STANDARD_PRICING["output"]
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "pricing": {
            "model": "gpt-5.6-luna",
            "service_tier": "standard",
            "context": "short",
            "usd_per_million_tokens": dict(LUNA_STANDARD_PRICING),
            "pinned_at": "2026-08-20",
            "source": "https://platform.openai.com/pricing",
        },
        "estimated_usd_at_2026_08_20_list_price": round(estimated_usd, 6),
    }


def audit_batch(*, output_dir: Path, batch_index: int, expected: int) -> dict[str, Any]:
    paths = _batch_paths(output_dir, batch_index)
    docs = _read_jsonl(paths["docs"])
    cases = _read_jsonl(paths["cases"])
    cases_by_doc = {_doc_id_for_case(case): case for case in cases}
    marked_by_doc = {
        str(row.get("document_id")): row for row in _read_jsonl(paths["marked"])
    }
    prior = _prior_documents(output_dir, batch_index)
    all_reference = prior + docs
    validation_failures: list[dict[str, Any]] = []
    model_review_failures: list[dict[str, Any]] = []
    for doc in docs:
        case = cases_by_doc.get(str(doc.get("document_id")))
        marked_row = marked_by_doc.get(str(doc.get("document_id")))
        issues = list(validate_record(doc, strict_taxonomy=True))
        if case is None:
            issues.append("document has no matching structured case")
        elif marked_row is None:
            issues.append("document has no retained marked source")
        else:
            issues.extend(
                validate_english_llm_document(
                    record=case,
                    marked_text=str(marked_row.get("marked_text", "")),
                    doc=doc,
                )
            )
        if issues:
            validation_failures.append(
                {"document_id": doc.get("document_id"), "issues": list(dict.fromkeys(issues))}
            )
        model_review = doc.get("metadata", {}).get("clinical_model_review")
        review_issues: list[str] = []
        if isinstance(model_review, dict):
            if model_review.get("verdict") != "pass":
                review_issues.append(f"review verdict is {model_review.get('verdict')!r}")
            if model_review.get("issues"):
                review_issues.append("review retained one or more unresolved issues")
            if not model_review.get("response_id"):
                review_issues.append("review response ID is missing")
        if review_issues:
            model_review_failures.append(
                {"document_id": doc.get("document_id"), "issues": review_issues}
            )

    exact_groups: dict[str, list[str]] = defaultdict(list)
    skeleton_groups: dict[str, list[str]] = defaultdict(list)
    for doc in all_reference:
        doc_id = str(doc.get("document_id"))
        exact = hashlib.sha256(SPACE_RE.sub(" ", str(doc.get("text", "")).lower()).strip().encode()).hexdigest()
        skeleton = hashlib.sha256(_document_skeleton(doc).encode()).hexdigest()
        exact_groups[exact].append(doc_id)
        skeleton_groups[skeleton].append(doc_id)
    current_ids = {str(doc.get("document_id")) for doc in docs}
    exact_duplicates = [ids for ids in exact_groups.values() if len(ids) > 1 and current_ids.intersection(ids)]
    skeleton_duplicates = [ids for ids in skeleton_groups.values() if len(ids) > 1 and current_ids.intersection(ids)]

    prior_hashes = [(str(doc.get("document_id")), _simhash(_document_skeleton(doc))) for doc in prior]
    current_hashes = [(str(doc.get("document_id")), _simhash(_document_skeleton(doc))) for doc in docs]
    near_duplicates: list[dict[str, Any]] = []
    for index, (doc_id, fingerprint) in enumerate(current_hashes):
        candidates = prior_hashes + current_hashes[:index]
        for other_id, other_fingerprint in candidates:
            distance = _hamming(fingerprint, other_fingerprint)
            if distance <= 3:
                near_duplicates.append(
                    {"document_id": doc_id, "similar_to": other_id, "simhash_distance": distance}
                )

    repeated_phrases: Counter[str] = Counter()
    for doc in docs:
        tokens = _tokens(_document_skeleton(doc))
        phrases = {" ".join(tokens[index : index + 12]) for index in range(max(0, len(tokens) - 11))}
        repeated_phrases.update(phrases)
    phrase_threshold = max(8, int(len(docs) * 0.08))
    high_repetition = [
        {"phrase": phrase, "documents": count}
        for phrase, count in repeated_phrases.most_common()
        if count >= phrase_threshold
    ][:50]

    words = [len(_tokens(str(doc.get("text", "")))) for doc in docs]
    labels = Counter(span.get("label") for doc in docs for span in doc.get("spans", []))
    locales = Counter(doc.get("metadata", {}).get("lang") for doc in docs)
    document_types = Counter(
        f"{doc.get('metadata', {}).get('lang')}:{doc.get('metadata', {}).get('document_type')}"
        for doc in docs
    )
    styles = Counter(
        doc.get("metadata", {}).get("style_profile", {}).get("name") for doc in docs
    )
    hard_negatives = Counter(
        target.get("category")
        for doc in docs
        for target in doc.get("metadata", {}).get("hard_negative_targets", [])
    )
    source_slots = Counter(
        str(span.get("source_slot"))
        for doc in docs
        for span in doc.get("spans", [])
        if span.get("source_slot")
    )
    date_formats: Counter[str] = Counter()
    date_year_buckets: Counter[str] = Counter()
    for doc in docs:
        for span in doc.get("spans", []):
            if span.get("label") != "Date":
                continue
            value = str(span.get("text", ""))
            if re.match(r"^\d{4}-", value):
                date_formats["iso"] += 1
            elif "/" in value:
                date_formats["numeric_slash"] += 1
            elif re.match(r"^[A-Za-z]", value):
                date_formats["month_first_words"] += 1
            else:
                date_formats["day_first_words"] += 1
            year_match = re.search(r"\b((?:19|20)\d{2})\b", value)
            if year_match:
                year = int(year_match.group(1))
                bucket = "pre_2000" if year < 2000 else "post_2025" if year > 2025 else "2000_2025"
                date_year_buckets[bucket] += 1
    conditions = Counter(
        f"{case.get('language')}:{case.get('condition', {}).get('name')}" for case in cases
    )
    conditions_by_locale: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        conditions_by_locale[str(case.get("language"))][
            str(case.get("condition", {}).get("name"))
        ] += 1
    pediatric_cases = [case for case in cases if _is_pediatric_case(case)]
    pediatric_by_locale = Counter(str(case.get("language")) for case in pediatric_cases)
    pediatric_bands_by_locale: dict[str, Counter[str]] = defaultdict(Counter)
    pediatric_documents_by_locale: dict[str, Counter[str]] = defaultdict(Counter)
    pediatric_conditions_by_locale: dict[str, Counter[str]] = defaultdict(Counter)
    pediatric_age_units_by_locale: dict[str, Counter[str]] = defaultdict(Counter)
    pediatric_docs_with_age_signal: Counter[str] = Counter()
    pediatric_age_source_slots: Counter[str] = Counter()
    for case in pediatric_cases:
        locale = str(case.get("language"))
        pediatric_bands_by_locale[locale][_pediatric_band(case)] += 1
        pediatric_documents_by_locale[locale][str(case.get("document_type"))] += 1
        pediatric_conditions_by_locale[locale][
            str(case.get("condition", {}).get("name"))
        ] += 1
        pediatric_age_units_by_locale[locale][str(case.get("age_unit", "years"))] += 1
        doc = next(
            (
                candidate
                for candidate in docs
                if str(candidate.get("document_id")) == _doc_id_for_case(case)
            ),
            None,
        )
        if doc is not None:
            age_spans = [
                span for span in doc.get("spans", []) if span.get("label") == "Age_Birthdate"
            ]
            if age_spans:
                pediatric_docs_with_age_signal[locale] += 1
            pediatric_age_source_slots.update(
                str(span.get("source_slot"))
                for span in age_spans
                if span.get("source_slot")
            )
    styles_by_cell: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        metadata = doc.get("metadata", {})
        cell = f"{metadata.get('lang')}:{metadata.get('document_type')}"
        styles_by_cell[cell].add(str(metadata.get("style_profile", {}).get("name")))
    patient_names = {
        f"{case.get('patient', {}).get('given_name')} {case.get('patient', {}).get('family_name')}"
        for case in cases
    }
    patient_addresses = {str(case.get("patient_address")) for case in cases}
    healthcare_organizations = {str(case.get("hospital")) for case in cases}
    all_fourgrams: list[str] = []
    for doc in docs:
        tokens = _tokens(_document_skeleton(doc))
        all_fourgrams.extend(
            " ".join(tokens[index : index + 4])
            for index in range(max(0, len(tokens) - 3))
        )
    lexical_fourgram_ratio = (
        len(set(all_fourgrams)) / len(all_fourgrams) if all_fourgrams else 0.0
    )
    allowed = list(BERT_ENTITY_LABELS)
    gate_failures: list[str] = []
    if len(docs) != expected:
        gate_failures.append(f"expected {expected} documents, found {len(docs)}")
    if locales != Counter({"en-GB": expected // 2, "en-US": expected // 2}):
        gate_failures.append(f"locale split is not exact: {dict(locales)}")
    if validation_failures:
        gate_failures.append(f"{len(validation_failures)} documents failed exhaustive validation")
    if model_review_failures:
        gate_failures.append(
            f"{len(model_review_failures)} documents lack a clean independent clinical review"
        )
    if exact_duplicates:
        gate_failures.append(f"{len(exact_duplicates)} exact duplicate groups")
    if skeleton_duplicates:
        gate_failures.append(f"{len(skeleton_duplicates)} PII-normalized duplicate groups")
    if near_duplicates:
        gate_failures.append(f"{len(near_duplicates)} near-duplicate pairs at SimHash distance <= 3")
    missing_labels = [label for label in allowed if labels.get(label, 0) == 0]
    if missing_labels:
        gate_failures.append(f"missing allowed labels: {', '.join(missing_labels)}")
    if high_repetition:
        gate_failures.append(
            f"{len(high_repetition)} twelve-token phrases occur in >= {phrase_threshold} documents"
        )
    if any(span.get("label") == "Anonymize_Other" for doc in docs for span in doc.get("spans", [])):
        gate_failures.append("Anonymize_Other was emitted")
    if expected >= BATCH_SIZE:
        per_cell_floor = expected // 12
        per_cell_ceiling = (expected + 11) // 12
        if len(document_types) != 12 or any(
            count < per_cell_floor or count > per_cell_ceiling
            for count in document_types.values()
        ):
            gate_failures.append(
                f"locale/document-family cells are not balanced within {per_cell_floor}-{per_cell_ceiling}"
            )
        if len(styles) < 8 or any(len(values) < 3 for values in styles_by_cell.values()):
            gate_failures.append("format styles do not cover all profiles and at least three styles per locale/document cell")
        missing_hard_negatives = sorted(REQUIRED_500_HARD_NEGATIVES - set(hard_negatives))
        if missing_hard_negatives:
            gate_failures.append(
                "missing 500-document hard-negative categories: "
                + ", ".join(missing_hard_negatives)
            )
        missing_format_slots = sorted(REQUIRED_500_FORMAT_SLOTS - set(source_slots))
        if missing_format_slots:
            gate_failures.append(
                "missing formatting-robustness slots: " + ", ".join(missing_format_slots)
            )
        if set(date_formats) != {
            "iso", "numeric_slash", "month_first_words", "day_first_words"
        }:
            gate_failures.append("date formatting does not cover ISO, numeric, month-first, and day-first forms")
        if any(date_year_buckets.get(bucket, 0) < 5 for bucket in ("pre_2000", "2000_2025", "post_2025")):
            gate_failures.append("date values do not cover past, contemporary, and future robustness ranges")
        if any(len(counter) < 25 for counter in conditions_by_locale.values()):
            gate_failures.append("fewer than 25 distinct clinical conditions in one locale")
        if any(
            count > (expected / 2) * 0.10
            for counter in conditions_by_locale.values()
            for count in counter.values()
        ):
            gate_failures.append("one clinical condition exceeds 10% of the batch")
        if len(patient_names) / max(1, len(cases)) < 0.95:
            gate_failures.append("fewer than 95% unique recombined patient names")
        if len(patient_addresses) / max(1, len(cases)) < 0.95:
            gate_failures.append("fewer than 95% unique patient addresses")
        if lexical_fourgram_ratio < 0.55:
            gate_failures.append(
                f"lexical four-gram diversity is too low: {lexical_fourgram_ratio:.3f}"
            )
        # Batch 1 had already been authored and partly reviewed when the
        # paediatric contract was introduced.  Report its gap, but enforce the
        # exact catch-up allocation on every subsequent 500-document batch.
        if batch_index >= 1:
            for locale in ("en-GB", "en-US"):
                if pediatric_by_locale.get(locale, 0) < PEDIATRIC_PER_PROFILE_AFTER_BATCH_1:
                    gate_failures.append(
                        f"{locale} has fewer than {PEDIATRIC_PER_PROFILE_AFTER_BATCH_1} paediatric cases"
                    )
                for band, minimum in PEDIATRIC_BAND_MINIMUMS_PER_PROFILE.items():
                    if pediatric_bands_by_locale[locale].get(band, 0) < minimum:
                        gate_failures.append(
                            f"{locale} paediatric band {band} has fewer than {minimum} cases"
                        )
                if set(pediatric_documents_by_locale[locale]) != set(
                    ("clinic_note", "discharge_summary", "emergency_note", "referral_letter", "laboratory_report", "nursing_note")
                ):
                    gate_failures.append(
                        f"{locale} paediatric cases do not cover all six document families"
                    )
                if pediatric_docs_with_age_signal.get(locale, 0) != pediatric_by_locale.get(locale, 0):
                    gate_failures.append(
                        f"{locale} has paediatric documents without an Age_Birthdate signal"
                    )
                if set(pediatric_age_units_by_locale[locale]) != {
                    "days", "weeks", "months", "years"
                }:
                    gate_failures.append(
                        f"{locale} paediatric ages do not cover days, weeks, months, and years"
                    )

    # Emit the shared, language-neutral report as part of every batch audit.
    # English keeps the additional domain-specific checks above (for example
    # exact paediatric bands and source-slot coverage), while future language
    # packs can reuse these common dimensions and thresholds directly.
    shared_contract = CorpusDiversityContract(
        expected_documents=expected,
        profiles=("en-GB", "en-US"),
        document_families=(
            "clinic_note",
            "discharge_summary",
            "emergency_note",
            "referral_letter",
            "laboratory_report",
            "nursing_note",
        ),
        allowed_labels=tuple(BERT_ENTITY_LABELS),
        forbidden_labels=("Anonymize_Other",),
        required_hard_negative_categories=(
            tuple(sorted(REQUIRED_500_HARD_NEGATIVES)) if expected >= BATCH_SIZE else ()
        ),
        require_balanced_profile_family_cells=expected >= BATCH_SIZE,
        min_styles_per_profile_family=3 if expected >= BATCH_SIZE else 0,
        min_fourgram_distinct_ratio=0.55 if expected >= BATCH_SIZE else 0.0,
        # The English-specific calculation above already compares SimHash
        # values with all prior batches. Avoid repeating that quadratic pass.
        near_duplicate_distance=None,
        reference_year=2025,
        min_date_values_per_period=5 if expected >= BATCH_SIZE else 0,
        pediatric_minimum_by_profile=(
            {
                "en-GB": PEDIATRIC_PER_PROFILE_AFTER_BATCH_1,
                "en-US": PEDIATRIC_PER_PROFILE_AFTER_BATCH_1,
            }
            if expected >= BATCH_SIZE and batch_index >= 1
            else {}
        ),
        required_pediatric_age_units=("days", "weeks", "months", "years"),
    )
    shared_diversity_report = audit_corpus_diversity(
        docs,
        contract=shared_contract,
        prior_records=prior,
        case_records=cases,
    )
    for failure in shared_diversity_report["failures"]:
        if failure not in gate_failures:
            gate_failures.append(f"shared diversity contract: {failure}")
    return {
        "contract": "meddeid.english-batch-quality.v1",
        "batch_index": batch_index,
        "expected_documents": expected,
        "documents": len(docs),
        "gate_passed": not gate_failures,
        "gate_failures": gate_failures,
        "validation_failures": validation_failures,
        "model_review_failures": model_review_failures,
        "exact_duplicate_groups": exact_duplicates,
        "skeleton_duplicate_groups": skeleton_duplicates,
        "near_duplicate_pairs": near_duplicates,
        "high_repetition_phrases": high_repetition,
        "counts": {
            "locales": dict(sorted(locales.items())),
            "document_types": dict(sorted(document_types.items())),
            "styles": dict(sorted(styles.items())),
            "labels": {label: labels.get(label, 0) for label in allowed},
            "hard_negative_categories": dict(sorted(hard_negatives.items())),
            "conditions": dict(sorted(conditions.items())),
            "source_slots": dict(sorted(source_slots.items())),
            "date_formats": dict(sorted(date_formats.items())),
            "date_year_buckets": dict(sorted(date_year_buckets.items())),
            "pediatric": {
                "total": len(pediatric_cases),
                "by_locale": dict(sorted(pediatric_by_locale.items())),
                "bands_by_locale": {
                    locale: dict(sorted(counter.items()))
                    for locale, counter in sorted(pediatric_bands_by_locale.items())
                },
                "document_types_by_locale": {
                    locale: dict(sorted(counter.items()))
                    for locale, counter in sorted(pediatric_documents_by_locale.items())
                },
                "conditions_by_locale": {
                    locale: dict(sorted(counter.items()))
                    for locale, counter in sorted(pediatric_conditions_by_locale.items())
                },
                "age_units_by_locale": {
                    locale: dict(sorted(counter.items()))
                    for locale, counter in sorted(pediatric_age_units_by_locale.items())
                },
                "documents_with_age_signal_by_locale": dict(
                    sorted(pediatric_docs_with_age_signal.items())
                ),
                "age_source_slots": dict(sorted(pediatric_age_source_slots.items())),
            },
        },
        "diversity": {
            "distinct_conditions_by_locale": {
                locale: len(counter) for locale, counter in sorted(conditions_by_locale.items())
            },
            "max_condition_count_by_locale": {
                locale: max(counter.values(), default=0)
                for locale, counter in sorted(conditions_by_locale.items())
            },
            "styles_by_locale_document_cell": {
                cell: sorted(values) for cell, values in sorted(styles_by_cell.items())
            },
            "unique_patient_names": len(patient_names),
            "unique_patient_addresses": len(patient_addresses),
            "unique_healthcare_organizations": len(healthcare_organizations),
            "lexical_fourgram_distinct_ratio": round(lexical_fourgram_ratio, 4),
        },
        "shared_diversity_contract": shared_diversity_report,
        "word_counts": {
            "min": min(words) if words else 0,
            "median": statistics.median(words) if words else 0,
            "mean": round(statistics.mean(words), 2) if words else 0,
            "max": max(words) if words else 0,
        },
        "usage": _usage_totals(docs),
        "clinical_model_review": {
            "completed": sum(
                isinstance(doc.get("metadata", {}).get("clinical_model_review"), dict)
                for doc in docs
            ),
            "policy": "optional targeted second opinion; direct personal review remains mandatory",
        },
    }


def _review_selection(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The production requirement is direct review of every authored document,
    # not only a statistically representative sample.  Keep the function so a
    # future UI can page the packet without changing the batch contract.
    return sorted(docs, key=lambda row: str(row["document_id"]))


def _review_decision_for_document(
    document: dict[str, Any], existing: dict[str, Any] | None
) -> dict[str, Any]:
    """Preserve a legacy decision once, then bind it to immutable content."""

    document_id = str(document["document_id"])
    digest = _document_sha256(document)
    if existing is not None and existing.get("document_sha256") in {None, digest}:
        migrated = dict(existing)
        migrated["contract_version"] = "meddeid.review-decision.v1"
        migrated["document_sha256"] = digest
        return migrated
    return {
        "contract_version": "meddeid.review-decision.v1",
        "document_id": document_id,
        "document_sha256": digest,
        "decision": "pending",
        "notes": "",
        "reviewed_at": None,
    }


def write_batch_reports(
    *, paths: dict[str, Path], docs: list[dict[str, Any]], report: dict[str, Any]
) -> None:
    paths["report_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# English production batch {report['batch_index'] + 1}",
        "",
        f"Quality gate: {'PASS' if report['gate_passed'] else 'FAIL'}",
        f"Documents: {report['documents']}/{report['expected_documents']}",
        f"Word counts: {report['word_counts']}",
        f"Estimated generation cost: USD {report['usage']['estimated_usd_at_2026_08_20_list_price']:.4f}",
        "",
        "## Gate failures",
        "",
    ]
    lines.extend(f"- {failure}" for failure in report["gate_failures"])
    if not report["gate_failures"]:
        lines.append("- None")
    lines.extend(["", "## Coverage", "", "```json", json.dumps(report["counts"], indent=2), "```", ""])
    lines.extend(["## Diversity", "", "```json", json.dumps(report["diversity"], indent=2), "```", ""])
    paths["report_md"].write_text("\n".join(lines), encoding="utf-8")

    review_lines = [
        f"# Personal review packet — batch {report['batch_index'] + 1}",
        "",
        "This packet contains every document in the batch. Personal sign-off requires document-by-document review of clinical coherence, locale fidelity, hard-negative naturalness, PII semantics, annotation boundaries, and language quality.",
        "",
    ]
    selected = _review_selection(docs)
    selected_by_id = {str(row["document_id"]): row for row in selected}
    for failure in report.get("validation_failures", []):
        doc_id = str(failure.get("document_id"))
        match = next((row for row in docs if str(row.get("document_id")) == doc_id), None)
        if match is not None:
            selected_by_id[doc_id] = match
    for doc_id in sorted(selected_by_id):
        doc = selected_by_id[doc_id]
        metadata = doc.get("metadata", {})
        review_lines.extend(
            [
                f"## {doc_id}",
                "",
                f"Locale/type/style: {metadata.get('lang')} / {metadata.get('document_type')} / {metadata.get('style_profile', {}).get('name')}",
                f"Words/spans: {len(_tokens(str(doc.get('text', ''))))} / {len(doc.get('spans', []))}",
                "",
                "```text",
                str(doc.get("text", "")),
                "```",
                "",
                "Review decision: PENDING",
                "",
            ]
        )
    paths["review"].write_text("\n".join(review_lines), encoding="utf-8")
    existing_decisions = {
        str(row.get("document_id")): row
        for row in _read_jsonl(paths["review_decisions"])
    }
    _write_jsonl_raw(
        paths["review_decisions"],
        [
            _review_decision_for_document(
                doc, existing_decisions.get(str(doc["document_id"]))
            )
            for doc in selected
        ],
    )
    manifest = {
        "contract": "meddeid.english-production-batch.v1",
        "batch_index": report["batch_index"],
        "documents": report["documents"],
        "allowed_labels": list(BERT_ENTITY_LABELS),
        "forbidden_generated_label": "Anonymize_Other",
        "independently_authored": True,
        "quality_gate_passed": report["gate_passed"],
        "files": {
            key: {"path": path.name, "sha256": _sha256_file(path)}
            for key, path in paths.items()
            if key not in {"directory", "manifest"} and path.is_file()
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def record_document_review(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    decision: str,
    notes: str,
) -> dict[str, Any]:
    paths = _batch_paths(output_dir, batch_index)
    rows = _read_jsonl(paths["review_decisions"])
    match = next((row for row in rows if row.get("document_id") == document_id), None)
    if match is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    document = next(
        (
            row
            for row in _read_jsonl(paths["docs"])
            if row.get("document_id") == document_id
        ),
        None,
    )
    if document is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    match.update(
        {
            "contract_version": "meddeid.review-decision.v1",
            "document_sha256": _document_sha256(document),
            "decision": decision,
            "notes": notes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_jsonl_raw(paths["review_decisions"], rows)
    return match


def invalidate_document(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    reason: str,
) -> dict[str, Any]:
    """Archive an accepted document and make its structured case resumable."""

    paths = _batch_paths(output_dir, batch_index)
    docs = _read_jsonl(paths["docs"])
    marked_rows = _read_jsonl(paths["marked"])
    cases = _read_jsonl(paths["cases"])
    doc = next((row for row in docs if row.get("document_id") == document_id), None)
    marked = next(
        (row for row in marked_rows if row.get("document_id") == document_id), None
    )
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if doc is None or marked is None or case is None:
        raise KeyError(f"cannot invalidate incomplete or unknown document {document_id!r}")
    invalidated_at = datetime.now(timezone.utc).isoformat()
    archive_row = {
        "document_id": document_id,
        "reason": reason,
        "invalidated_at": invalidated_at,
        "document": doc,
        "marked": marked,
    }
    _append_jsonl(paths["invalidated"], archive_row)
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    _write_jsonl(paths["docs"], [row for row in docs if row is not doc])
    _write_jsonl_raw(
        paths["marked"], [row for row in marked_rows if row is not marked]
    )
    decisions = [
        row
        for row in _read_jsonl(paths["review_decisions"])
        if row.get("document_id") != document_id
    ]
    _write_jsonl_raw(paths["review_decisions"], decisions)
    return {
        "document_id": document_id,
        "reason": reason,
        "invalidated_at": invalidated_at,
    }


def migrate_case_age(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    age_years: int,
    reason: str,
) -> dict[str, Any]:
    """Change age/DOB together while preserving exact calendar-age consistency."""

    if age_years < 0 or age_years > 120:
        raise ValueError("age_years must be between 0 and 120")
    paths = _batch_paths(output_dir, batch_index)
    cases = _read_jsonl(paths["cases"])
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if case is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    if any(row.get("document_id") == document_id for row in _read_jsonl(paths["docs"])):
        raise RuntimeError("invalidate the accepted document before migrating its case")
    encounter = datetime.strptime(str(case["encounter_date"]), "%Y-%m-%d").date()
    old_birth = datetime.strptime(str(case["birth_date"]), "%Y-%m-%d").date()
    birth_year = encounter.year - age_years - int(
        (old_birth.month, old_birth.day) > (encounter.month, encounter.day)
    )
    new_birth = old_birth.replace(year=birth_year)
    before = {
        "age_years": int(case["age_years"]),
        "birth_date": str(case["birth_date"]),
    }
    after = {"age_years": age_years, "birth_date": new_birth.isoformat()}
    case.update(after)
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    migration = {
        "document_id": document_id,
        "field": "age_years,birth_date",
        "from": before,
        "to": after,
        "reason": reason,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(paths["case_migrations"], migration)
    return migration


def migrate_case_hospital(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    hospital: str,
    reason: str,
) -> dict[str, Any]:
    """Replace an incompatible facility with another locked profile resource."""

    paths = _batch_paths(output_dir, batch_index)
    cases = _read_jsonl(paths["cases"])
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if case is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    if any(row.get("document_id") == document_id for row in _read_jsonl(paths["docs"])):
        raise RuntimeError("invalidate the accepted document before migrating its case")
    profile_id = str(case["language"])
    if hospital not in set(lookup_values(profile_id, "hospitals")):
        raise ValueError(
            f"hospital {hospital!r} is not a locked {profile_id} healthcare resource"
        )
    before = str(case["hospital"])
    case["hospital"] = hospital
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    migration = {
        "document_id": document_id,
        "field": "hospital",
        "from": before,
        "to": hospital,
        "reason": reason,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(paths["case_migrations"], migration)
    return migration


def migrate_case_address_display(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    patient_address: str,
    reason: str,
) -> dict[str, Any]:
    """Correct display-only address defects while preserving the sampled geography."""

    paths = _batch_paths(output_dir, batch_index)
    cases = _read_jsonl(paths["cases"])
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if case is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    if any(row.get("document_id") == document_id for row in _read_jsonl(paths["docs"])):
        raise RuntimeError("invalidate the accepted document before migrating its case")
    before = str(case["patient_address"])
    normalized_before = re.sub(r"\bI-\s+(?=\d)", "I-", before)
    if patient_address != normalized_before:
        raise ValueError(
            "address display migration may only close a TIGER interstate separator space"
        )
    case["patient_address"] = patient_address
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    migration = {
        "document_id": document_id,
        "field": "patient_address",
        "from": before,
        "to": patient_address,
        "reason": reason,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(paths["case_migrations"], migration)
    return migration


def migrate_case_person_name(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    role: str,
    given_name: str,
    family_name: str,
    reason: str,
) -> dict[str, Any]:
    """Replace a malformed sampled person token with locked resource values."""

    if role not in {"patient", "caregiver", "relative"}:
        raise ValueError("role must be patient, caregiver, or relative")
    paths = _batch_paths(output_dir, batch_index)
    cases = _read_jsonl(paths["cases"])
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if case is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    if any(row.get("document_id") == document_id for row in _read_jsonl(paths["docs"])):
        raise RuntimeError("invalidate the accepted document before migrating its case")
    profile_id = str(case["language"])
    if given_name not in set(lookup_values(profile_id, "first_names")):
        raise ValueError(f"given name {given_name!r} is not a locked {profile_id} resource")
    if family_name not in set(lookup_values(profile_id, "family_names")):
        raise ValueError(f"family name {family_name!r} is not a locked {profile_id} resource")
    before = dict(case[role])
    after = {"given_name": given_name, "family_name": family_name}
    case[role] = after
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    migration = {
        "document_id": document_id,
        "field": role,
        "from": before,
        "to": after,
        "reason": reason,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(paths["case_migrations"], migration)
    return migration


def migrate_case_department(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    department: str,
    reason: str,
) -> dict[str, Any]:
    """Replace an age-incompatible department with a locked profile value."""

    paths = _batch_paths(output_dir, batch_index)
    cases = _read_jsonl(paths["cases"])
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if case is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    if any(row.get("document_id") == document_id for row in _read_jsonl(paths["docs"])):
        raise RuntimeError("invalidate the accepted document before migrating its case")
    profile_id = str(case["language"])
    if department not in set(lookup_values(profile_id, "departments")):
        raise ValueError(
            f"department {department!r} is not a locked {profile_id} department resource"
        )
    before = str(case["department"])
    case["department"] = department
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    migration = {
        "document_id": document_id,
        "field": "department",
        "from": before,
        "to": department,
        "reason": reason,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(paths["case_migrations"], migration)
    return migration


def migrate_case_years(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    years: int,
    reason: str,
) -> dict[str, Any]:
    """Shift all case calendar dates together while preserving clinical age."""

    if years == 0:
        raise ValueError("date migration must use a non-zero year shift")
    paths = _batch_paths(output_dir, batch_index)
    cases = _read_jsonl(paths["cases"])
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if case is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    if any(row.get("document_id") == document_id for row in _read_jsonl(paths["docs"])):
        raise RuntimeError("invalidate the accepted document before migrating its case")

    def shifted(value: str) -> str:
        parsed = datetime.fromisoformat(value)
        try:
            return parsed.replace(year=parsed.year + years).date().isoformat()
        except ValueError:  # 29 February into a non-leap year
            return parsed.replace(year=parsed.year + years, day=28).date().isoformat()

    fields = ("birth_date", "encounter_date", "followup_date")
    before = {field: str(case[field]) for field in fields}
    after = {field: shifted(before[field]) for field in fields}
    case.update(after)
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    migration = {
        "document_id": document_id,
        "field": ",".join(fields),
        "from": before,
        "to": after,
        "reason": reason,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(paths["case_migrations"], migration)
    return migration


def migrate_case_profession(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    profession: str,
    reason: str,
) -> dict[str, Any]:
    """Replace an implausible profession/organisation pairing with a locked value."""

    paths = _batch_paths(output_dir, batch_index)
    cases = _read_jsonl(paths["cases"])
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if case is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    if any(row.get("document_id") == document_id for row in _read_jsonl(paths["docs"])):
        raise RuntimeError("invalidate the accepted document before migrating its case")
    profile_id = str(case["language"])
    if profession not in set(lookup_values(profile_id, "professions")):
        raise ValueError(
            f"profession {profession!r} is not a locked {profile_id} profession resource"
        )
    before = str(case["profession"])
    case["profession"] = profession
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    migration = {
        "document_id": document_id,
        "field": "profession",
        "from": before,
        "to": profession,
        "reason": reason,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(paths["case_migrations"], migration)
    return migration


def migrate_case_medication(
    *,
    output_dir: Path,
    batch_index: int,
    document_id: str,
    old: str,
    new: str,
    reason: str,
) -> dict[str, Any]:
    """Apply a reviewed locale-specific medication terminology correction."""

    paths = _batch_paths(output_dir, batch_index)
    cases = _read_jsonl(paths["cases"])
    case = next((row for row in cases if _doc_id_for_case(row) == document_id), None)
    if case is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    if any(row.get("document_id") == document_id for row in _read_jsonl(paths["docs"])):
        raise RuntimeError("invalidate the accepted document before migrating its case")
    medications = list(case.get("condition", {}).get("medications", []))
    if old not in medications:
        raise ValueError(f"case does not contain medication {old!r}")
    medications[medications.index(old)] = new
    case["condition"]["medications"] = medications
    case.setdefault("production", {})["review_feedback"] = reason
    _write_jsonl_raw(paths["cases"], cases)
    migration = {
        "document_id": document_id,
        "field": "condition.medications",
        "from": old,
        "to": new,
        "reason": reason,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(paths["case_migrations"], migration)
    return migration


def sign_off_batch(
    *,
    output_dir: Path,
    batch_index: int,
    reviewer: str,
    decision: str,
    notes: str,
) -> dict[str, Any]:
    paths = _batch_paths(output_dir, batch_index)
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    if decision == "pass" and not report.get("gate_passed"):
        raise RuntimeError("cannot pass personal review while the automated quality gate fails")
    decisions = _read_jsonl(paths["review_decisions"])
    if decision == "pass":
        documents = _read_jsonl(paths["docs"])
        expected_ids = {str(doc.get("document_id")) for doc in documents}
        document_hashes = {
            str(doc.get("document_id")): _document_sha256(doc) for doc in documents
        }
        decision_ids = {str(row.get("document_id")) for row in decisions}
        if decision_ids != expected_ids:
            raise RuntimeError("personal review decisions do not match the batch documents")
        incomplete = [
            str(row.get("document_id"))
            for row in decisions
            if row.get("decision") != "pass"
        ]
        if incomplete:
            raise RuntimeError(
                f"cannot pass batch with {len(incomplete)} non-passing document reviews"
            )
        stale = [
            str(row.get("document_id"))
            for row in decisions
            if row.get("document_sha256")
            != document_hashes.get(str(row.get("document_id")))
        ]
        if stale:
            raise RuntimeError(
                f"cannot pass batch with {len(stale)} stale document reviews"
            )
    signoff = {
        "contract": "meddeid.english-personal-review.v1",
        "batch_index": batch_index,
        "reviewer": reviewer,
        "decision": decision,
        "reviewed_documents": int(report["documents"]),
        "review_scope": "all_documents",
        "review_packet_sha256": _sha256_file(paths["review"]),
        "review_decisions_sha256": (
            _sha256_file(paths["review_decisions"])
            if paths["review_decisions"].is_file()
            else None
        ),
        "quality_report_sha256": _sha256_file(paths["report_json"]),
        "notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["signoff"].write_text(
        json.dumps(signoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return signoff


def validate_unversioned_english_generation_profiles(
    docs: list[dict[str, Any]],
) -> Counter[str]:
    """Keep corpus provenance stable across compatible profile resource updates."""

    counts: Counter[str] = Counter()
    supported = {"en-GB", "en-US"}
    for index, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata") or {}
        language = str(metadata.get("lang") or "").strip().replace("_", "-")
        profile = str(metadata.get("generation_profile") or "").strip().replace("_", "-")
        if "@" in profile:
            raise ValueError(
                f"document {index} metadata.generation_profile must be unversioned: {profile!r}"
            )
        if profile not in supported:
            raise ValueError(
                f"document {index} has unsupported metadata.generation_profile={profile!r}"
            )
        if language != profile:
            raise ValueError(
                f"document {index} metadata.lang={language!r} conflicts with "
                f"metadata.generation_profile={profile!r}"
            )
        counts[profile] += 1
    return counts


def finalize_corpus(output_dir: Path, *, expected_total: int = PRODUCTION_TOTAL) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    cases_by_doc: dict[str, dict[str, Any]] = {}
    batch_index = 0
    while True:
        paths = _batch_paths(output_dir, batch_index)
        if not paths["docs"].is_file():
            break
        report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
        if not report.get("gate_passed"):
            raise RuntimeError(f"batch {batch_index + 1} has not passed its quality gate")
        if not paths["signoff"].is_file():
            raise RuntimeError(f"batch {batch_index + 1} has no personal-review sign-off")
        signoff = json.loads(paths["signoff"].read_text(encoding="utf-8"))
        if signoff.get("decision") != "pass":
            raise RuntimeError(f"batch {batch_index + 1} did not pass personal review")
        if signoff.get("quality_report_sha256") != _sha256_file(paths["report_json"]):
            raise RuntimeError(
                f"batch {batch_index + 1} quality report changed after sign-off"
            )
        if signoff.get("review_decisions_sha256") != _sha256_file(
            paths["review_decisions"]
        ):
            raise RuntimeError(
                f"batch {batch_index + 1} review decisions changed after sign-off"
            )
        batch_docs = _read_jsonl(paths["docs"])
        for doc in batch_docs:
            normalize_english_age_birthdate_boundaries(doc)
            normalize_english_mrn_values(doc)
            diversify_english_email_domains(doc)
            diversify_english_identifier_formats(doc)
            rebalance_healthcare_organization_case(doc)
            inject_national_patient_identifier(doc)
        docs.extend(batch_docs)
        cases_by_doc.update(
            {
                _doc_id_for_case(case): case
                for case in _read_jsonl(paths["cases"])
            }
        )
        batch_index += 1
    if len(docs) != expected_total:
        raise RuntimeError(f"expected {expected_total} gated documents, found {len(docs)}")
    generation_profile_counts = validate_unversioned_english_generation_profiles(docs)
    if generation_profile_counts != Counter({"en-GB": 3_500, "en-US": 3_500}):
        raise RuntimeError(
            f"English generation profiles are unbalanced: {dict(generation_profile_counts)}"
        )
    pediatric_final_by_locale = Counter(
        str(cases_by_doc[str(doc["document_id"])]["language"])
        for doc in docs
        if _is_pediatric_case(cases_by_doc[str(doc["document_id"])])
    )
    pediatric_total = sum(pediatric_final_by_locale.values())
    if pediatric_total < PEDIATRIC_FINAL_MINIMUM:
        raise RuntimeError(
            f"final corpus has {pediatric_total} paediatric documents; "
            f"minimum is {PEDIATRIC_FINAL_MINIMUM}"
        )
    for profile_id in ("en-GB", "en-US"):
        if (
            pediatric_final_by_locale.get(profile_id, 0)
            < PEDIATRIC_FINAL_PER_PROFILE_MINIMUM
        ):
            raise RuntimeError(
                f"final corpus has {pediatric_final_by_locale.get(profile_id, 0)} "
                f"{profile_id} paediatric documents; minimum is "
                f"{PEDIATRIC_FINAL_PER_PROFILE_MINIMUM}"
            )

    benchmark_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        metadata = doc.get("metadata", {})
        grouped[(str(metadata.get("lang")), str(metadata.get("document_type")))].append(doc)
    for profile_id in ("en-GB", "en-US"):
        profile_selected = 0
        for document_type in sorted({key[1] for key in grouped if key[0] == profile_id}):
            cell = grouped[(profile_id, document_type)]
            pediatric_rows = [
                row
                for row in cell
                if _is_pediatric_case(cases_by_doc[str(row["document_id"])])
            ]
            adult_rows = sorted(
                (
                    row
                    for row in cell
                    if not _is_pediatric_case(cases_by_doc[str(row["document_id"])])
                ),
                key=lambda row: hashlib.sha256(
                    f"sealed-benchmark|{row['document_id']}".encode()
                ).hexdigest(),
            )
            pediatric_take = PEDIATRIC_BENCHMARK_PER_PROFILE // 6
            total_take = BENCHMARK_PER_PROFILE // 6
            selected_pediatric: list[dict[str, Any]] = []
            for band in ("infant", "early_childhood", "school_age", "adolescent"):
                candidates = [
                    row
                    for row in pediatric_rows
                    if _pediatric_band(cases_by_doc[str(row["document_id"])]) == band
                ]
                if band == "infant":
                    preferred_unit = BENCHMARK_INFANT_UNIT_BY_DOCUMENT[document_type]
                    candidates = [
                        row
                        for row in candidates
                        if str(cases_by_doc[str(row["document_id"])].get("age_unit"))
                        == preferred_unit
                    ]
                candidates.sort(
                    key=lambda row: hashlib.sha256(
                        f"sealed-benchmark|{row['document_id']}".encode()
                    ).hexdigest()
                )
                if not candidates:
                    raise RuntimeError(
                        f"not enough {band} cases for {profile_id}/{document_type} benchmark cell"
                    )
                selected_pediatric.append(candidates[0])
            if len(selected_pediatric) != pediatric_take:
                raise AssertionError("paediatric benchmark stratum must select four cases")
            selected = selected_pediatric + adult_rows[
                : total_take - pediatric_take
            ]
            benchmark_ids.update(str(row["document_id"]) for row in selected)
            profile_selected += len(selected)
        if profile_selected != BENCHMARK_PER_PROFILE:
            raise RuntimeError(f"could not construct exact benchmark for {profile_id}")
    benchmark = sorted(
        (doc for doc in docs if str(doc["document_id"]) in benchmark_ids),
        key=lambda row: str(row["document_id"]),
    )
    benchmark_pediatric_bands: dict[str, Counter[str]] = defaultdict(Counter)
    benchmark_pediatric_units: dict[str, Counter[str]] = defaultdict(Counter)
    for doc in benchmark:
        case = cases_by_doc[str(doc["document_id"])]
        if not _is_pediatric_case(case):
            continue
        locale = str(case["language"])
        benchmark_pediatric_bands[locale][_pediatric_band(case)] += 1
        benchmark_pediatric_units[locale][str(case.get("age_unit", "years"))] += 1
    for profile_id in ("en-GB", "en-US"):
        if benchmark_pediatric_bands[profile_id] != Counter(
            {
                "infant": 6,
                "early_childhood": 6,
                "school_age": 6,
                "adolescent": 6,
            }
        ):
            raise RuntimeError(f"paediatric benchmark age bands are unbalanced for {profile_id}")
        if not {"days", "weeks", "months", "years"} <= set(
            benchmark_pediatric_units[profile_id]
        ):
            raise RuntimeError(f"paediatric benchmark age formats are incomplete for {profile_id}")
    development = sorted(
        (doc for doc in docs if str(doc["document_id"]) not in benchmark_ids),
        key=lambda row: str(row["document_id"]),
    )
    for rows in (development, benchmark):
        demographic_index = 0
        for doc in rows:
            if inject_demographic_hard_negative(doc, demographic_index):
                demographic_index += 1
    final_dir = output_dir / "final"
    development_path = final_dir / "development.jsonl"
    benchmark_path = final_dir / "benchmark.jsonl"
    _write_jsonl(development_path, development)
    _write_jsonl(benchmark_path, benchmark)
    manifest = {
        "contract": "meddeid.english-synthetic-corpus.v1",
        "total_documents": len(docs),
        "development_documents": len(development),
        "benchmark_documents": len(benchmark),
        "profiles": {"en-GB": 3_500, "en-US": 3_500},
        "benchmark_profiles": {"en-GB": 150, "en-US": 150},
        "pediatric_documents_minimum": PEDIATRIC_FINAL_MINIMUM,
        "pediatric_documents_actual": pediatric_total,
        "pediatric_profiles_actual": dict(sorted(pediatric_final_by_locale.items())),
        "benchmark_pediatric_profiles": {
            "en-GB": PEDIATRIC_BENCHMARK_PER_PROFILE,
            "en-US": PEDIATRIC_BENCHMARK_PER_PROFILE,
        },
        "benchmark_pediatric_bands": {
            locale: dict(sorted(counter.items()))
            for locale, counter in sorted(benchmark_pediatric_bands.items())
        },
        "benchmark_pediatric_age_units": {
            locale: dict(sorted(counter.items()))
            for locale, counter in sorted(benchmark_pediatric_units.items())
        },
        "allowed_labels": list(BERT_ENTITY_LABELS),
        "forbidden_generated_label": "Anonymize_Other",
        "annotation_policy": {
            "hyphenated_age": (
                "Annotate the numeric age and unit (for example 50-year); "
                "exclude -old and following demographic descriptors."
            ),
            "contextual_age": (
                "Exclude cue words such as age and aged; annotate only the numeric "
                "age and an explicit unit when present."
            ),
            "abbreviated_age": (
                "Keep short clinical age forms such as 50 y/o inside Age_Birthdate."
            ),
            "mrn": "Render MRN: 538265 and annotate only the six-digit identifier.",
            "email_domains": (
                "Use a deterministic, locale-appropriate mix of common consumer "
                "email domains; do not emit the legacy example.test domain."
            ),
            "identifier_formats": (
                "Use deterministic locale- and slot-aware identifier formats. "
                "Most values are numeric-only, with compact, grouped, slash, "
                "and short-prefix variants for structural diversity."
            ),
            "national_identifiers": (
                "Include a deterministic mix of national/public patient identifiers "
                "under ID:Patient, with patient.national_id provenance. Values use "
                "realistic SSN, NHS-number, or National-Insurance-number surfaces "
                "but deliberately invalid/reserved components."
            ),
            "healthcare_organization_case": (
                "Render a deterministic approximately 30% uppercase and 70% "
                "natural-display-case mix per English locale."
            ),
            "demographic_hard_negatives": (
                "Synthetic ethnicity and sex descriptors may follow hyphenated ages; "
                "they are clinical context and remain unannotated."
            ),
        },
        "base_model": "FacebookAI/roberta-base",
        "files": {
            "development": {"path": development_path.name, "sha256": _sha256_file(development_path)},
            "benchmark": {"path": benchmark_path.name, "sha256": _sha256_file(benchmark_path)},
        },
    }
    (final_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="meddeid-english-production")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-batch")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--batch-index", type=int, required=True)
    generate.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    generate.add_argument("--seed", type=int, default=20260820)
    generate.add_argument("--model", default=DEFAULT_ENGLISH_LLM_MODEL)
    generate.add_argument("--concurrency", type=int, default=8)
    generate.add_argument("--max-output-tokens", type=int, default=1800)
    generate.add_argument("--reasoning-effort", default="low")
    generate.add_argument("--validation-retries", type=int, default=4)
    generate.add_argument("--api-retries", type=int, default=5)
    generate.add_argument(
        "--model-review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an optional targeted second Luna clinical/editorial review (off by default to preserve the corpus budget).",
    )
    generate.add_argument("--resume", action="store_true")
    audit = sub.add_parser("audit-batch")
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--batch-index", type=int, required=True)
    audit.add_argument("--expected", type=int, default=BATCH_SIZE)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--expected-total", type=int, default=PRODUCTION_TOTAL)
    signoff = sub.add_parser("sign-off-batch")
    signoff.add_argument("--output-dir", type=Path, required=True)
    signoff.add_argument("--batch-index", type=int, required=True)
    signoff.add_argument("--reviewer", required=True)
    signoff.add_argument("--decision", choices=("pass", "fail"), required=True)
    signoff.add_argument("--notes", required=True)
    review = sub.add_parser("review-document")
    review.add_argument("--output-dir", type=Path, required=True)
    review.add_argument("--batch-index", type=int, required=True)
    review.add_argument("--document-id", required=True)
    review.add_argument("--decision", choices=("pass", "fail"), required=True)
    review.add_argument("--notes", default="")
    invalidate = sub.add_parser("invalidate-document")
    invalidate.add_argument("--output-dir", type=Path, required=True)
    invalidate.add_argument("--batch-index", type=int, required=True)
    invalidate.add_argument("--document-id", required=True)
    invalidate.add_argument("--reason", required=True)
    migrate_age = sub.add_parser("migrate-case-age")
    migrate_age.add_argument("--output-dir", type=Path, required=True)
    migrate_age.add_argument("--batch-index", type=int, required=True)
    migrate_age.add_argument("--document-id", required=True)
    migrate_age.add_argument("--age-years", type=int, required=True)
    migrate_age.add_argument("--reason", required=True)
    migrate_hospital = sub.add_parser("migrate-case-hospital")
    migrate_hospital.add_argument("--output-dir", type=Path, required=True)
    migrate_hospital.add_argument("--batch-index", type=int, required=True)
    migrate_hospital.add_argument("--document-id", required=True)
    migrate_hospital.add_argument("--hospital", required=True)
    migrate_hospital.add_argument("--reason", required=True)
    migrate_department = sub.add_parser("migrate-case-department")
    migrate_department.add_argument("--output-dir", type=Path, required=True)
    migrate_department.add_argument("--batch-index", type=int, required=True)
    migrate_department.add_argument("--document-id", required=True)
    migrate_department.add_argument("--department", required=True)
    migrate_department.add_argument("--reason", required=True)
    migrate_years = sub.add_parser("migrate-case-years")
    migrate_years.add_argument("--output-dir", type=Path, required=True)
    migrate_years.add_argument("--batch-index", type=int, required=True)
    migrate_years.add_argument("--document-id", required=True)
    migrate_years.add_argument("--years", type=int, required=True)
    migrate_years.add_argument("--reason", required=True)
    migrate_profession = sub.add_parser("migrate-case-profession")
    migrate_profession.add_argument("--output-dir", type=Path, required=True)
    migrate_profession.add_argument("--batch-index", type=int, required=True)
    migrate_profession.add_argument("--document-id", required=True)
    migrate_profession.add_argument("--profession", required=True)
    migrate_profession.add_argument("--reason", required=True)
    migrate_medication = sub.add_parser("migrate-case-medication")
    migrate_medication.add_argument("--output-dir", type=Path, required=True)
    migrate_medication.add_argument("--batch-index", type=int, required=True)
    migrate_medication.add_argument("--document-id", required=True)
    migrate_medication.add_argument("--old", required=True)
    migrate_medication.add_argument("--new", required=True)
    migrate_medication.add_argument("--reason", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate-batch":
        with _exclusive_generation_lock(args.output_dir, args.batch_index):
            return asyncio.run(
                generate_batch(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    model=args.model,
                    concurrency=args.concurrency,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    validation_retries=args.validation_retries,
                    api_retries=args.api_retries,
                    resume=args.resume,
                    model_review=args.model_review,
                )
            )
    if args.command == "audit-batch":
        paths = _batch_paths(args.output_dir, args.batch_index)
        docs = _read_jsonl(paths["docs"])
        report = audit_batch(
            output_dir=args.output_dir,
            batch_index=args.batch_index,
            expected=args.expected,
        )
        write_batch_reports(paths=paths, docs=docs, report=report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["gate_passed"] else 1
    if args.command == "review-document":
        print(
            json.dumps(
                record_document_review(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    document_id=args.document_id,
                    decision=args.decision,
                    notes=args.notes,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "invalidate-document":
        print(
            json.dumps(
                invalidate_document(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    document_id=args.document_id,
                    reason=args.reason,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "migrate-case-age":
        print(
            json.dumps(
                migrate_case_age(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    document_id=args.document_id,
                    age_years=args.age_years,
                    reason=args.reason,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "migrate-case-hospital":
        print(
            json.dumps(
                migrate_case_hospital(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    document_id=args.document_id,
                    hospital=args.hospital,
                    reason=args.reason,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "migrate-case-department":
        print(
            json.dumps(
                migrate_case_department(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    document_id=args.document_id,
                    department=args.department,
                    reason=args.reason,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "migrate-case-years":
        print(
            json.dumps(
                migrate_case_years(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    document_id=args.document_id,
                    years=args.years,
                    reason=args.reason,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "migrate-case-profession":
        print(
            json.dumps(
                migrate_case_profession(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    document_id=args.document_id,
                    profession=args.profession,
                    reason=args.reason,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "migrate-case-medication":
        print(
            json.dumps(
                migrate_case_medication(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    document_id=args.document_id,
                    old=args.old,
                    new=args.new,
                    reason=args.reason,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "sign-off-batch":
        print(
            json.dumps(
                sign_off_batch(
                    output_dir=args.output_dir,
                    batch_index=args.batch_index,
                    reviewer=args.reviewer,
                    decision=args.decision,
                    notes=args.notes,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    manifest = finalize_corpus(args.output_dir, expected_total=args.expected_total)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
