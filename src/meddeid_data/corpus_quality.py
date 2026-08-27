"""Reusable corpus diversity and stability gates for synthetic datasets."""

from __future__ import annotations

import hashlib
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from .semantic_types import semantic_type_for_target


WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)?", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/+-]\d+)*\b")
YEAR_RE = re.compile(r"\b((?:18|19|20|21)\d{2})\b")


@dataclass(frozen=True)
class CorpusDiversityContract:
    """Thresholds that turn corpus diversity into a blocking contract.

    Locale packages can add stricter gates, but the dimensions and report shape
    remain stable across languages.
    """

    expected_documents: int | None = None
    profiles: tuple[str, ...] = ()
    document_families: tuple[str, ...] = ()
    allowed_labels: tuple[str, ...] = ()
    forbidden_labels: tuple[str, ...] = ()
    required_hard_negative_categories: tuple[str, ...] = ()
    require_balanced_profile_family_cells: bool = False
    min_styles_per_profile_family: int = 0
    min_fourgram_distinct_ratio: float = 0.0
    near_duplicate_distance: int | None = 3
    reference_year: int = 2025
    min_date_values_per_period: int = 0
    pediatric_minimum_by_profile: Mapping[str, int] = field(default_factory=dict)
    required_pediatric_age_units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_documents is not None and self.expected_documents < 0:
            raise ValueError("expected_documents must be >= 0")
        if not 0.0 <= self.min_fourgram_distinct_ratio <= 1.0:
            raise ValueError("min_fourgram_distinct_ratio must be between 0 and 1")
        if self.near_duplicate_distance is not None and not 0 <= self.near_duplicate_distance <= 64:
            raise ValueError("near_duplicate_distance must be between 0 and 64")


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _profile(record: Mapping[str, Any]) -> str:
    metadata = _metadata(record)
    value = metadata.get("lang") or metadata.get("generation_profile") or record.get("language")
    return str(value or "unknown").split("@", 1)[0].replace("_", "-")


def _document_family(record: Mapping[str, Any]) -> str:
    metadata = _metadata(record)
    return str(metadata.get("document_type") or record.get("document_type") or "unknown")


def _style(record: Mapping[str, Any]) -> str:
    style = _metadata(record).get("style_profile")
    if isinstance(style, Mapping):
        return str(style.get("name") or "unknown")
    return str(style or "unknown")


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def _normalized_text(record: Mapping[str, Any]) -> str:
    text = unicodedata.normalize("NFKC", str(record.get("text") or ""))
    return SPACE_RE.sub(" ", text).strip().casefold()


def document_skeleton(record: Mapping[str, Any]) -> str:
    """Replace PII spans and incidental numbers before duplicate comparison."""

    text = str(record.get("text") or "")
    parts: list[str] = []
    cursor = 0
    spans = sorted(
        (span for span in record.get("spans", []) if isinstance(span, Mapping)),
        key=lambda span: (int(span.get("begin", 0)), int(span.get("end", 0))),
    )
    for span in spans:
        begin, end = int(span.get("begin", 0)), int(span.get("end", 0))
        if begin < cursor or end < begin or end > len(text):
            continue
        parts.append(text[cursor:begin])
        parts.append(f" <{span.get('label', 'PII')}> ")
        cursor = end
    parts.append(text[cursor:])
    value = unicodedata.normalize("NFKC", "".join(parts)).casefold()
    return SPACE_RE.sub(" ", NUMBER_RE.sub("<n>", value)).strip()


def _simhash(text: str) -> int:
    tokens = _tokens(text)
    shingles = [
        " ".join(tokens[index : index + 5])
        for index in range(max(1, len(tokens) - 4))
    ]
    vector = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _document_id(record: Mapping[str, Any], index: int) -> str:
    return str(record.get("document_id") or record.get("id") or f"row-{index}")


def _date_period(value: str, reference_year: int) -> str | None:
    match = YEAR_RE.search(value)
    if match is None:
        return None
    year = int(match.group(1))
    if year < 2000:
        return "historical"
    if year <= reference_year:
        return "contemporary"
    return "future_shifted"


def _age_surface(value: str) -> str:
    folded = value.casefold()
    if re.search(r"\b(?:day|days|d)\b", folded):
        return "days"
    if re.search(r"\b(?:week|weeks|wk|wks)\b", folded):
        return "weeks"
    if re.search(r"\b(?:month|months|mo|mos)\b", folded):
        return "months"
    if re.search(r"\b(?:year|years|yr|yrs|y/o|yo)\b", folded):
        return "years"
    if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}", value):
        return "birthdate"
    return "other"


def _surface_format(label: str, value: str) -> str:
    if label.startswith("Name:"):
        tokens = value.split()
        if any(re.fullmatch(r"[A-Z]\.?", token) for token in tokens):
            return "contains_initial"
        letters = "".join(character for character in value if character.isalpha())
        if letters and letters.isupper():
            return "uppercase"
        if letters and letters.islower():
            return "lowercase"
        return "single_token" if len(tokens) == 1 else "mixed_or_title"
    if label.startswith("Address_Location:"):
        folded = value.casefold()
        if "\n" in value:
            return "multiline"
        if re.search(r"\b(?:po|p\.o\.)\s+box\b", folded):
            return "po_box"
        if "urbaniz" in folded:
            return "urbanizacion"
        return "single_line"
    if label == "Date":
        if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", value):
            return "iso"
        if "/" in value:
            return "numeric_slash"
        return "worded_month" if re.search(r"[A-Za-z]", value) else "other"
    if label.startswith("ID:"):
        if value.isdigit():
            return "digits_only"
        if "-" in value:
            return "hyphenated"
        if " " in value:
            return "spaced"
        return "alphanumeric"
    return "default"


def _documentation_shapes(text: str) -> set[str]:
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    shapes = {"single_paragraph" if len(nonempty) <= 1 else "multiline"}
    if any(re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line) for line in lines):
        shapes.add("list_like")
    if any(re.match(r"^\s*[^:\n]{1,40}:\s*\S", line) for line in lines):
        shapes.add("field_rows")
    if any(re.match(r"^\s*[A-Za-z][^:\n]{0,40}:\s*$", line) for line in lines):
        shapes.add("section_headings")
    if "\n\n" in text:
        shapes.add("paragraph_breaks")
    return shapes


def _parse_iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _is_pediatric_document(record: Mapping[str, Any]) -> bool:
    metadata = _metadata(record)
    patient = metadata.get("patient")
    if not isinstance(patient, Mapping):
        return False
    born = _parse_iso_date(patient.get("birth_date"))
    encounter = _parse_iso_date(metadata.get("document_creation_date"))
    if born is None or encounter is None:
        return False
    years = encounter.year - born.year - ((encounter.month, encounter.day) < (born.month, born.day))
    return years < 18


def _pediatric_case(case: Mapping[str, Any]) -> bool:
    try:
        return float(case.get("age_years", 18)) < 18
    except (TypeError, ValueError):
        return False


def _duplicate_groups(
    current: Sequence[Mapping[str, Any]],
    prior: Sequence[Mapping[str, Any]],
    *,
    skeleton: bool,
) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    all_records = [*prior, *current]
    for index, record in enumerate(all_records):
        value = document_skeleton(record) if skeleton else _normalized_text(record)
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        groups[digest].append(_document_id(record, index))
    current_ids = {
        _document_id(record, index + len(prior)) for index, record in enumerate(current)
    }
    return [ids for ids in groups.values() if len(ids) > 1 and current_ids.intersection(ids)]


def audit_corpus_diversity(
    records: Sequence[Mapping[str, Any]],
    *,
    contract: CorpusDiversityContract,
    prior_records: Sequence[Mapping[str, Any]] = (),
    case_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Audit the dimensions that every language onboarding must make explicit."""

    profile_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    semantic_types: Counter[str] = Counter()
    hard_negatives: Counter[str] = Counter()
    styles_by_cell: dict[str, set[str]] = defaultdict(set)
    documentation_shapes: Counter[str] = Counter()
    surface_formats: dict[str, Counter[str]] = defaultdict(Counter)
    date_periods: Counter[str] = Counter()
    pediatric_documents: Counter[str] = Counter()
    pediatric_age_surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    fourgrams: list[str] = []
    word_counts: list[int] = []

    for record in records:
        profile = _profile(record)
        family = _document_family(record)
        cell = f"{profile}:{family}"
        profile_counts[profile] += 1
        cell_counts[cell] += 1
        styles_by_cell[cell].add(_style(record))
        text = str(record.get("text") or "")
        tokens = _tokens(document_skeleton(record))
        word_counts.append(len(_tokens(text)))
        fourgrams.extend(
            " ".join(tokens[index : index + 4])
            for index in range(max(0, len(tokens) - 3))
        )
        documentation_shapes.update(_documentation_shapes(text))
        pediatric = _is_pediatric_document(record)
        if pediatric:
            pediatric_documents[profile] += 1

        for target in _metadata(record).get("hard_negative_targets", []):
            if isinstance(target, Mapping):
                hard_negatives[str(target.get("category") or "unknown")] += 1

        for span in record.get("spans", []):
            if not isinstance(span, Mapping):
                continue
            label = str(span.get("label") or "")
            value = str(span.get("text") or "")
            labels[label] += 1
            semantic_types[
                semantic_type_for_target(dict(span), document_family=family)
            ] += 1
            surface_formats[label][_surface_format(label, value)] += 1
            if label == "Date":
                period = _date_period(value, contract.reference_year)
                if period:
                    date_periods[period] += 1
            if pediatric and label == "Age_Birthdate":
                pediatric_age_surfaces[profile][_age_surface(value)] += 1

    pediatric_cases: Counter[str] = Counter()
    pediatric_case_units: dict[str, Counter[str]] = defaultdict(Counter)
    for case in case_records:
        if not _pediatric_case(case):
            continue
        profile = _profile(case)
        pediatric_cases[profile] += 1
        pediatric_case_units[profile][str(case.get("age_unit") or "years")] += 1

    exact_duplicates = _duplicate_groups(records, prior_records, skeleton=False)
    skeleton_duplicates = _duplicate_groups(records, prior_records, skeleton=True)
    near_duplicates: list[dict[str, Any]] = []
    if contract.near_duplicate_distance is not None:
        prior_hashes = [
            (_document_id(record, index), _simhash(document_skeleton(record)))
            for index, record in enumerate(prior_records)
        ]
        current_hashes = [
            (
                _document_id(record, index + len(prior_records)),
                _simhash(document_skeleton(record)),
            )
            for index, record in enumerate(records)
        ]
        for index, (document_id, fingerprint) in enumerate(current_hashes):
            for other_id, other_fingerprint in [*prior_hashes, *current_hashes[:index]]:
                distance = _hamming(fingerprint, other_fingerprint)
                if distance <= contract.near_duplicate_distance:
                    near_duplicates.append(
                        {
                            "document_id": document_id,
                            "similar_to": other_id,
                            "simhash_distance": distance,
                        }
                    )

    fourgram_ratio = len(set(fourgrams)) / len(fourgrams) if fourgrams else 0.0
    failures: list[str] = []
    if contract.expected_documents is not None and len(records) != contract.expected_documents:
        failures.append(
            f"expected {contract.expected_documents} documents, found {len(records)}"
        )
    missing_profiles = sorted(set(contract.profiles) - set(profile_counts))
    if missing_profiles:
        failures.append("missing profiles: " + ", ".join(missing_profiles))
    missing_labels = sorted(set(contract.allowed_labels) - set(labels))
    if missing_labels:
        failures.append("missing allowed labels: " + ", ".join(missing_labels))
    emitted_forbidden = sorted(set(contract.forbidden_labels).intersection(labels))
    if emitted_forbidden:
        failures.append("forbidden generated labels: " + ", ".join(emitted_forbidden))
    unexpected_labels = sorted(
        set(labels) - set(contract.allowed_labels) - set(contract.forbidden_labels)
    )
    if contract.allowed_labels and unexpected_labels:
        failures.append("unexpected generated labels: " + ", ".join(unexpected_labels))
    if exact_duplicates:
        failures.append(f"{len(exact_duplicates)} exact duplicate groups")
    if skeleton_duplicates:
        failures.append(f"{len(skeleton_duplicates)} PII-normalized duplicate groups")
    if near_duplicates:
        failures.append(
            f"{len(near_duplicates)} near-duplicate pairs at SimHash distance <= "
            f"{contract.near_duplicate_distance}"
        )
    if fourgram_ratio < contract.min_fourgram_distinct_ratio:
        failures.append(
            f"lexical four-gram diversity {fourgram_ratio:.4f} is below "
            f"{contract.min_fourgram_distinct_ratio:.4f}"
        )

    expected_cells = {
        f"{profile}:{family}"
        for profile in contract.profiles
        for family in contract.document_families
    }
    if expected_cells and set(cell_counts) != expected_cells:
        missing = sorted(expected_cells - set(cell_counts))
        extra = sorted(set(cell_counts) - expected_cells)
        if missing:
            failures.append("missing profile/document-family cells: " + ", ".join(missing))
        if extra:
            failures.append("unexpected profile/document-family cells: " + ", ".join(extra))
    if contract.require_balanced_profile_family_cells and cell_counts:
        if max(cell_counts.values()) - min(cell_counts.values()) > 1:
            failures.append("profile/document-family cells differ by more than one document")
    sparse_style_cells = sorted(
        cell
        for cell in expected_cells
        if len(styles_by_cell.get(cell, set())) < contract.min_styles_per_profile_family
    )
    if sparse_style_cells:
        failures.append(
            "too few formatting styles in cells: " + ", ".join(sparse_style_cells)
        )
    missing_negatives = sorted(
        set(contract.required_hard_negative_categories) - set(hard_negatives)
    )
    if missing_negatives:
        failures.append("missing hard-negative categories: " + ", ".join(missing_negatives))
    if contract.min_date_values_per_period:
        sparse_periods = [
            period
            for period in ("historical", "contemporary", "future_shifted")
            if date_periods.get(period, 0) < contract.min_date_values_per_period
        ]
        if sparse_periods:
            failures.append("insufficient date periods: " + ", ".join(sparse_periods))
    pediatric_source = pediatric_cases or pediatric_documents
    for profile, minimum in contract.pediatric_minimum_by_profile.items():
        if pediatric_source.get(profile, 0) < minimum:
            failures.append(f"{profile} has fewer than {minimum} pediatric documents")
    for profile in contract.pediatric_minimum_by_profile:
        missing_units = sorted(
            set(contract.required_pediatric_age_units)
            - set(pediatric_case_units.get(profile, pediatric_age_surfaces.get(profile, Counter())))
        )
        if missing_units:
            failures.append(
                f"{profile} is missing pediatric age units: {', '.join(missing_units)}"
            )

    return {
        "contract": "meddeid.corpus-diversity.v1",
        "configuration": asdict(contract),
        "passed": not failures,
        "failures": failures,
        "counts": {
            "documents": len(records),
            "profiles": dict(sorted(profile_counts.items())),
            "profile_document_families": dict(sorted(cell_counts.items())),
            "labels": dict(sorted(labels.items())),
            "semantic_types": dict(sorted(semantic_types.items())),
            "hard_negative_categories": dict(sorted(hard_negatives.items())),
            "date_periods": dict(sorted(date_periods.items())),
            "documentation_shapes": dict(sorted(documentation_shapes.items())),
            "surface_formats_by_label": {
                label: dict(sorted(counter.items()))
                for label, counter in sorted(surface_formats.items())
            },
            "pediatric_documents": dict(sorted(pediatric_documents.items())),
            "pediatric_cases": dict(sorted(pediatric_cases.items())),
            "pediatric_age_units": {
                profile: dict(sorted(counter.items()))
                for profile, counter in sorted(pediatric_case_units.items())
            },
            "pediatric_age_surfaces": {
                profile: dict(sorted(counter.items()))
                for profile, counter in sorted(pediatric_age_surfaces.items())
            },
        },
        "formatting": {
            "styles_by_profile_document_family": {
                cell: sorted(values) for cell, values in sorted(styles_by_cell.items())
            }
        },
        "diversity": {
            "fourgram_distinct_ratio": round(fourgram_ratio, 6),
            "exact_duplicate_groups": exact_duplicates,
            "skeleton_duplicate_groups": skeleton_duplicates,
            "near_duplicate_pairs": near_duplicates,
        },
        "word_counts": {
            "min": min(word_counts) if word_counts else 0,
            "median": statistics.median(word_counts) if word_counts else 0,
            "mean": round(statistics.mean(word_counts), 2) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
        },
    }


__all__ = [
    "CorpusDiversityContract",
    "audit_corpus_diversity",
    "document_skeleton",
]
