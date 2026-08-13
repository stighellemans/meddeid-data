"""Production post-processing for generated annotation spans."""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Any

from meddeid_language_nl.date_age_variants import parse_date_text
from meddeid_language_nl.postprocess import post_process_spans

from .labels import split_label

VERSION = "production-postprocess-v10"
AGE_BIRTHDATE_CUE_RE = re.compile(r"^\s*(?:geb\.?|geboren)\b\.?\s*[:;-]?\s*", re.I)
INTERNAL_CONTACT_CUE_RE = re.compile(r"^\s*(?:intern|lijn)\b\s*[:;-]?\s*", re.I)
CFDNA_PATIENT_ID_CUE_RE = re.compile(r"^\s*CFDNA\b\s*[:#/-]?\s*", re.I)
CAREGIVER_NAME_CUE_RE = re.compile(
    r"""
    ^\s*(?:
        door
        |
        uitvoerder
        |
        uitgevoerd\s+door
        |
        gevalideerd\s+door
        |
        opgesteld\s+door
        |
        ondertekend\s+door
        |
        verslag(?:geving)?\s+door
        |
        verslag\s+(?:opgesteld|opgemaakt|voorbereid)\s+door
        |
        verslag\s+gezien\s+en\s+geadviseerd\s+door
        |
        mede-?beoordeling\s+door
        |
        beoordeling\s+door
        |
        en\s+geadviseerd\s+door
        |
        (?:secundair\s+)?overleg\s+met
        |
        i\.?\s*o\.?\s*m\.?
        |
        in\s+overleg\s+met
        |
        assistent[-\s]+diensthoofd
        |
        ass\.\s*diensthoofd
        |
        zo\s+nodig\s+met
        |
        met\s+collegiale\s+groet(?:en)?
    )\b\s*[:=/-]?\s*
    """,
    re.I | re.X,
)
CAREGIVER_EMBEDDED_NAME_CUE_RE = re.compile(
    r"""
    (?:
        verslag\s+door
        |
        verslag(?:geving)?\s+door
        |
        verslag\s+(?:opgesteld|opgemaakt|voorbereid)\s+door
        |
        verslag\s+gezien\s+en\s+geadviseerd\s+door
        |
        door
        |
        uitgevoerd\s+door
        |
        gevalideerd\s+door
        |
        opgesteld\s+door
        |
        ondertekend\s+door
        |
        mede-?beoordeling\s+door
        |
        beoordeling\s+door
        |
        en\s+geadviseerd\s+door
        |
        i\.?\s*o\.?\s*m\.?
        |
        in\s+overleg\s+met
        |
        assistent[-\s]+diensthoofd
        |
        ass\.\s*diensthoofd
    )\s+
    (?=[A-ZÀ-Ý])
    """,
    re.I | re.X,
)
LEADING_LIST_CLUTTER_RE = re.compile(r"^[\s.,;:()\[\]\-–—]+")
NAME_TOKEN_RE = re.compile(r"[A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]+")
CAREGIVER_SIGNATURE_SUFFIX_RE = re.compile(
    r"\s*,\s*(?:arts|dr\.?|dokter|prof(?:essor)?\.?\s*(?:dr\.?|dokter)|verpleegkundige|specialist)\b",
    re.I,
)
MONTH_TOKEN_PATTERN = (
    r"jan(?:uari)?|feb(?:r(?:uari)?)?|mrt|maart|apr(?:il)?|mei|jun(?:i)?|"
    r"jul(?:i)?|aug(?:ustus)?|sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|mars"
)
NUMERIC_MONTH_YEAR_PATTERN = r"(?:0?[1-9]|1[0-2])[/-](?:19\d{2}|20\d{2})"
WEEKDAY_TOKEN_PATTERN = (
    r"(?:maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag|"
    r"ma|di|wo|do|vr|za|zo)\.?"
)
WEEKDAY_TOKEN_RE = re.compile(rf"\b{WEEKDAY_TOKEN_PATTERN}\b", re.I)
WEEKDAY_CLUTTER_SUFFIX_RE = re.compile(
    rf"((?:\b{WEEKDAY_TOKEN_PATTERN}\b[\s,;:/.-]*)+)$",
    re.I,
)
DATE_CORE_RE = re.compile(
    r"""
    \b(?:
        \d{1,2}[./-]\d{1,2}[./-]\d{2,4}
        |
        \d{4}[./-]\d{1,2}[./-]\d{1,2}
        |
        \d{1,2}\s*[./-]\s*[A-Za-zÀ-ÿ]{3,}\.?\s*[./-]\s*\d{2,4}
        |
        \d{1,2}\.?\s+
        (?:jan(?:uari)?|feb(?:ruari)?|mrt|maart|apr(?:il)?|mei|jun(?:i)?|
           jul(?:i)?|aug(?:ustus)?|sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|
           dec(?:ember)?|mars)\.?\s+'?\d{2,4}
    )\b
    """,
    re.I | re.X,
)
CONTEXTUAL_DATE_CANDIDATE_RE = re.compile(
    rf"""
    (?<![\w+/-])
    (?:
        {NUMERIC_MONTH_YEAR_PATTERN}
        |
        (?:{MONTH_TOKEN_PATTERN})\.?\s+(?:19\d{{2}}|20\d{{2}})
        |
        (?:19\d{{2}}|20\d{{2}})
    )
    (?![\w/-])
    """,
    re.I | re.X,
)
NUMERIC_MONTH_YEAR_VALUE_RE = re.compile(rf"^{NUMERIC_MONTH_YEAR_PATTERN}$")
# A valid numeric month/year is a date form by itself. Year-only and textual
# month/year forms stay context-gated to reduce false positives.
DATE_CONTEXT_RE = re.compile(
    r"""
    \b(?:op|d\.d\.|dd\.|datum|date|consult|consultatie|contact|controle|
        follow-up|fu|opvolg|afspraak|voorzien|gezien|gepland|opname|
        verslag|verslagdatum|herbeoordeling|vanaf|tot|tussen|van|week|
        nota|overdracht|terugbrief|sinds|rookstop|planning|planjaar|plan|
        jaarcontrole|mdo|historiek|tijdslijn|antecedenten|voorgeschiedenis|
        review|start|gevolgd|herbekeken|afgewerkt|richting|lange\s+termijn|
        reeds\s+uitgevoerd)\b
        |
        \[\s*$
    """,
    re.I | re.X,
)
BIRTHDATE_CONTEXT_RE = re.compile(
    r"(?:°\s*$|\b(?:geb\.?|geboren|geboortedatum|geb\.datum|leeftijd)\W{0,24}$)",
    re.I,
)
FULL_MONTH_AT_END_RE = re.compile(rf"\b(?:{MONTH_TOKEN_PATTERN})\.$", re.I)
PATIENT_ID_CODE_RE = re.compile(
    r"""
    \b(?:
        UZ-\d{4}-\d{6}
        |
        MRN-\d{6}
        |
        EPD-[A-Z2-9]{2}-\d{6}
    )\b
    """,
    re.X,
)
CONTEXTUAL_PATIENT_ID_RE = re.compile(
    r"""
    \b(?:
        HIS(?:\s+(?:Patient\s+ID|patient-id|id))?
        |
        Dossier(?:\s*-\s*ID)?
        |
        patient(?:en)?(?:nr|nummer|nummer)?
        |
        MRN
    )\b
    \s*[:#/-]?\s*
    (?P<id>
        UZ-\d{4}-\d{6}
        |
        MRN-\d{6}
        |
        EPD-[A-Z2-9]{2}-\d{6}
        |
        P\d{7}
        |
        \d{8,9}
    )\b
    """,
    re.I | re.X,
)
def _annotation_signature(ann: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return ann.get("begin"), ann.get("end"), ann.get("label"), ann.get("text")


def _external_post_process_source_info() -> dict[str, Any]:
    return {"package": "meddeid-core"}


def _load_external_post_process():
    return post_process_spans


def _sync_label_fields(ann: dict[str, Any]) -> dict[str, Any]:
    label = ann.get("label")
    if not isinstance(label, str):
        return ann
    category, subtype = split_label(label)
    ann["Category"] = category
    ann["Subtype"] = subtype
    ann["confirmed"] = True
    return ann


def _covered_or_overlapped(
    begin: int,
    end: int,
    annotations: list[dict[str, Any]],
) -> bool:
    for ann in annotations:
        ann_begin = ann.get("begin")
        ann_end = ann.get("end")
        if not isinstance(ann_begin, int) or not isinstance(ann_end, int):
            continue
        if ann_begin <= begin and ann_end >= end:
            return True
        if begin < ann_end and ann_begin < end:
            return True
    return False


def _make_annotation(text: str, begin: int, end: int, label: str) -> dict[str, Any]:
    return _sync_label_fields(
        {
            "begin": begin,
            "end": end,
            "label": label,
            "text": text[begin:end],
        }
    )


def _set_annotation_span(ann: dict[str, Any], text: str, begin: int, end: int) -> None:
    while begin < end and text[begin] in " \t,;:/-":
        begin += 1
    while end > begin and text[end - 1] in " \t,;:/-":
        end -= 1
    ann["begin"] = begin
    ann["end"] = end
    ann["text"] = text[begin:end]
    _sync_label_fields(ann)


def _name_begin_after_cue(text: str, begin: int, end: int) -> int:
    segment = text[begin:end]
    cue_match: re.Match[str] | None = None
    for pattern in (
        r"(?:^|\b)(?:in samenwerking met|worden met|contact opgenomen worden met|"
        r"opgemaakt door|medebeoordeling door|uitgevoerd op [^\n]{0,40}? door|"
        r"door|met)\s+",
        r"\n",
    ):
        for match in re.finditer(pattern, segment, re.I):
            cue_match = match
    if cue_match is None:
        return begin
    return begin + cue_match.end()


def _trim_caregiver_name_before(text: str, ann: dict[str, Any], limit: int) -> bool:
    begin = ann.get("begin")
    end = ann.get("end")
    if not isinstance(begin, int) or not isinstance(end, int) or limit <= begin:
        return False

    new_end = limit
    comma = text.rfind(",", begin, limit)
    if comma > begin:
        new_end = comma
    new_begin = _name_begin_after_cue(text, begin, new_end)
    if new_begin >= new_end:
        return False
    _set_annotation_span(ann, text, new_begin, new_end)
    return True


def normalize_overlapping_annotations(
    text: str,
    annotations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Normalize known post-process overlap shapes without changing text."""

    processed = [deepcopy(ann) for ann in annotations if isinstance(ann, dict)]
    changed = 0
    again = True
    while again:
        again = False
        processed.sort(
            key=lambda item: (item.get("begin", -1), item.get("end", -1), item.get("label", ""))
        )
        for index in range(len(processed) - 1):
            left = processed[index]
            right = processed[index + 1]
            left_begin = left.get("begin")
            left_end = left.get("end")
            right_begin = right.get("begin")
            right_end = right.get("end")
            if not all(isinstance(value, int) for value in (left_begin, left_end, right_begin, right_end)):
                continue
            if left_end <= right_begin:
                continue

            left_label = left.get("label")
            right_label = right.get("label")

            if left_label == right_label and (
                (left_begin <= right_begin and left_end >= right_end)
                or (right_begin <= left_begin and right_end >= left_end)
            ):
                left_width = left_end - left_begin
                right_width = right_end - right_begin
                del processed[index + 1 if left_width >= right_width else index]
                changed += 1
                again = True
                break

            if (
                left_begin == right_begin
                and left_end == right_end
                and {left_label, right_label} == {"Age_Birthdate", "Name:Patient"}
            ):
                del processed[index if left_label == "Name:Patient" else index + 1]
                changed += 1
                again = True
                break

            if (
                left_begin == right_begin
                and left_end == right_end
                and {left_label, right_label} == {"Age_Birthdate", "Date"}
            ):
                match = re.search(r"\s+-\s+", text[left_begin:left_end])
                if match:
                    age_ann = left if left_label == "Age_Birthdate" else right
                    date_ann = left if left_label == "Date" else right
                    _set_annotation_span(age_ann, text, left_begin, left_begin + match.start())
                    _set_annotation_span(date_ann, text, left_begin + match.end(), left_end)
                    changed += 1
                    again = True
                    break

            if left_label == "Date" and right_label == "Date":
                begin = min(left_begin, right_begin)
                end = max(left_end, right_end)
                if "\n" not in text[begin:end]:
                    _set_annotation_span(left, text, begin, end)
                    del processed[index + 1]
                    changed += 1
                    again = True
                    break

            if right_label == "Name:Caregiver" and left_label in {"Date", "Organization:Healthcare"}:
                new_begin = _name_begin_after_cue(text, right_begin, right_end)
                if new_begin < right_end and new_begin != right_begin:
                    _set_annotation_span(right, text, new_begin, right_end)
                    changed += 1
                    again = True
                    break

            if left_label == "Name:Caregiver" and right_label in {"Date", "Organization:Healthcare"}:
                if _trim_caregiver_name_before(text, left, right_begin):
                    changed += 1
                    again = True
                    break

            if left_label == "Name:Caregiver" and right_label in {"ID:Caregiver", "Address_Location:Caregiver"}:
                if _trim_caregiver_name_before(text, left, right_begin):
                    changed += 1
                    again = True
                    break

            if right_label == "Name:Caregiver" and left_label in {"ID:Caregiver", "Address_Location:Caregiver"}:
                new_begin = max(_name_begin_after_cue(text, right_begin, right_end), left_end)
                if new_begin < right_end:
                    _set_annotation_span(right, text, new_begin, right_end)
                    changed += 1
                    again = True
                    break

            if left_label == "Name:Caregiver" and right_label == "Name:Caregiver":
                union_begin = min(left_begin, right_begin)
                union_end = max(left_end, right_end)
                newline = text[union_begin:union_end].find("\n")
                if newline != -1:
                    _set_annotation_span(left, text, union_begin, union_begin + newline)
                    _set_annotation_span(right, text, union_begin + newline + 1, union_end)
                    changed += 1
                    again = True
                    break
                new_right_begin = _name_begin_after_cue(text, right_begin, right_end)
                if new_right_begin != right_begin and _trim_caregiver_name_before(text, left, right_begin):
                    _set_annotation_span(right, text, new_right_begin, right_end)
                    changed += 1
                    again = True
                    break

    processed.sort(
        key=lambda item: (item.get("begin", -1), item.get("end", -1), item.get("label", ""))
    )
    return processed, changed


def _trim_contextual_date_candidate(text: str, begin: int, end: int) -> tuple[int, int]:
    while begin < end and text[begin].isspace():
        begin += 1
    while end > begin and text[end - 1].isspace():
        end -= 1
    value = text[begin:end]
    if value.endswith(".") and FULL_MONTH_AT_END_RE.search(value):
        end -= 1
    return begin, end


def missing_contextual_annotations(
    text: str,
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return high-confidence missing annotations for generated admin/date text."""

    additions: list[dict[str, Any]] = []
    occupied = list(annotations)

    def add_if_free(begin: int, end: int, label: str) -> None:
        if begin >= end or _covered_or_overlapped(begin, end, occupied):
            return
        ann = _make_annotation(text, begin, end, label)
        additions.append(ann)
        occupied.append(ann)

    for match in PATIENT_ID_CODE_RE.finditer(text):
        add_if_free(match.start(), match.end(), "ID:Patient")

    for match in CONTEXTUAL_PATIENT_ID_RE.finditer(text):
        add_if_free(match.start("id"), match.end("id"), "ID:Patient")

    for match in CONTEXTUAL_DATE_CANDIDATE_RE.finditer(text):
        begin, end = _trim_contextual_date_candidate(text, match.start(), match.end())
        if begin >= end or _covered_or_overlapped(begin, end, occupied):
            continue

        before = text[max(0, begin - 80) : begin]
        label = None
        if BIRTHDATE_CONTEXT_RE.search(before):
            label = "Age_Birthdate"
        elif DATE_CONTEXT_RE.search(before) or NUMERIC_MONTH_YEAR_VALUE_RE.fullmatch(text[begin:end]):
            label = "Date"
        if label is None:
            continue

        add_if_free(begin, end, label)

    return additions


def _spans_are_external_ready(spans: list[dict[str, Any]], text: str) -> bool:
    for ann in spans:
        begin = ann.get("begin")
        end = ann.get("end")
        if not isinstance(begin, int) or not isinstance(end, int):
            return False
        if begin < 0 or end > len(text) or begin >= end:
            return False
        if ann.get("text") != text[begin:end]:
            return False
    return True


def remove_age_birthdate_cue(value: str) -> tuple[str, bool]:
    match = AGE_BIRTHDATE_CUE_RE.match(value)
    if not match:
        return value, False
    return value[match.end() :], True


def remove_internal_contact_cue(value: str) -> tuple[str, bool]:
    match = INTERNAL_CONTACT_CUE_RE.match(value)
    if not match:
        return value, False
    return value[match.end() :], True


def remove_cfdna_patient_id_cue(value: str) -> tuple[str, bool]:
    match = CFDNA_PATIENT_ID_CUE_RE.match(value)
    if not match:
        return value, False
    return value[match.end() :], True


def remove_caregiver_name_cue(value: str) -> tuple[str, bool]:
    match = CAREGIVER_NAME_CUE_RE.match(value)
    embedded_matches = list(CAREGIVER_EMBEDDED_NAME_CUE_RE.finditer(value))
    if embedded_matches:
        embedded_match = embedded_matches[-1]
        if match is None or embedded_match.end() > match.end():
            match = embedded_match
    if match is None:
        return value, False
    candidate = value[match.end() :]
    if not candidate.strip():
        return value, False
    return candidate, True


def _metadata_caregivers(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []

    names: list[str] = []
    source_pii = metadata.get("source_pii")
    if isinstance(source_pii, dict):
        for value in source_pii.get("caregivers") or []:
            if isinstance(value, str) and value.strip():
                names.append(value.strip())

    for key in ("caregiver", "secondary_caregiver"):
        value = metadata.get(key)
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            names.append(value["name"].strip())

    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def expand_caregiver_name_subspan(
    text: str,
    begin: int,
    end: int,
    *,
    metadata: Any = None,
) -> tuple[int, int, bool]:
    current = text[begin:end]
    if not current.strip():
        return begin, end, False

    for full_name in _metadata_caregivers(metadata):
        if current not in full_name or current == full_name:
            continue
        search_start = max(0, begin - len(full_name))
        search_end = min(len(text), end + len(full_name))
        window = text[search_start:search_end]
        offset = 0
        while True:
            relative = window.find(full_name, offset)
            if relative == -1:
                break
            full_begin = search_start + relative
            full_end = full_begin + len(full_name)
            if full_begin <= begin and end <= full_end:
                return full_begin, full_end, True
            offset = relative + 1

    if not NAME_TOKEN_RE.fullmatch(current.strip()):
        return begin, end, False
    if begin + len(current.lstrip()) != end:
        return begin, end, False

    following = text[end : min(len(text), end + 80)]
    surname_match = re.match(r"\s+([A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]+)", following)
    if not surname_match:
        return begin, end, False
    suffix = following[surname_match.end() :]
    if not CAREGIVER_SIGNATURE_SUFFIX_RE.match(suffix):
        return begin, end, False
    return begin, end + surname_match.end(), True


def normalize_weekday_date_clutter(text: str, begin: int, end: int) -> tuple[int, bool]:
    current = text[begin:end]
    date_match = DATE_CORE_RE.search(current)
    if date_match:
        date_begin = begin + date_match.start()
    else:
        date_match = DATE_CORE_RE.match(text[begin : min(len(text), end + 40)])
        if not date_match:
            return begin, False
        date_begin = begin

    prefix_start = max(0, date_begin - 80)
    prefix = text[prefix_start:date_begin]
    clutter_match = WEEKDAY_CLUTTER_SUFFIX_RE.search(prefix)
    if not clutter_match:
        return begin, False

    weekday_matches = list(WEEKDAY_TOKEN_RE.finditer(clutter_match.group(1)))
    if not weekday_matches:
        return begin, False

    last_weekday = weekday_matches[-1]
    new_begin = prefix_start + clutter_match.start(1) + last_weekday.start()
    return new_begin, new_begin != begin


def trim_leading_date_list_clutter(
    text: str,
    begin: int,
    end: int,
    *,
    label: str,
) -> tuple[int, bool]:
    current = text[begin:end]
    match = LEADING_LIST_CLUTTER_RE.match(current)
    if not match:
        return begin, False

    new_begin = begin + match.end()
    candidate = text[new_begin:end]
    if not candidate.strip():
        return begin, False
    if parse_date_text(candidate, label=label) is None:
        return begin, False
    return new_begin, new_begin != begin


def is_relative_date_hard_negative(value: str) -> bool:
    parsed = parse_date_text(value, label="Date")
    return parsed is not None and parsed.precision == "relative"


def _delete_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    parts: list[str] = []
    cursor = 0
    for begin, end in ranges:
        parts.append(text[cursor:begin])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _map_position(position: int, deletion_ranges: list[tuple[int, int]]) -> int:
    shift = 0
    for begin, end in deletion_ranges:
        if position <= begin:
            break
        if position < end:
            return begin - shift
        shift += end - begin
    return position - shift


def _contact_cue_deletion_ranges(text: str, annotations: list[dict[str, Any]]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for ann in annotations:
        if not isinstance(ann, dict) or ann.get("label") != "Contactdetails":
            continue
        begin = ann.get("begin")
        end = ann.get("end")
        if not isinstance(begin, int) or not isinstance(end, int):
            continue
        if begin < 0 or end > len(text) or begin >= end:
            continue
        match = INTERNAL_CONTACT_CUE_RE.match(text[begin:end])
        if match:
            ranges.append((begin, begin + match.end()))

    merged: list[tuple[int, int]] = []
    for begin, end in sorted(set(ranges)):
        if not merged or begin > merged[-1][1]:
            merged.append((begin, end))
            continue
        previous_begin, previous_end = merged[-1]
        merged[-1] = (previous_begin, max(previous_end, end))
    return merged


def apply_production_post_process(
    doc: dict[str, Any],
    *,
    use_external_post_process: bool = True,
) -> dict[str, Any]:
    """Return a copy of *doc* with production span corrections applied.

    Current rules:
    - remove leading ``geb.``/``geboren`` cues from ``Age_Birthdate`` spans
    - remove leading ``CFDNA`` cues from ``ID:Patient`` spans
    - remove leading cue phrases such as ``door``/``overleg met`` from ``Name:Caregiver`` spans
    - expand ``Name:Caregiver`` subspans when the contiguous written full name is known
    - collapse repeated weekday clutter before ``Date`` spans to the final weekday token
    - trim previous-line punctuation/list markers from date-like spans
    - remove leading ``intern``/``lijn`` cues from ``Contactdetails`` spans and text
    - run the canonical :mod:`meddeid_core` span pipeline
    - remove spans that are empty after post-processing
    """

    new_doc = deepcopy(doc)
    text = new_doc.get("text", "")
    annotations = new_doc.get("spans", [])
    if not isinstance(text, str) or not isinstance(annotations, list):
        return new_doc

    processed: list[dict[str, Any]] = []
    removed_empty = 0
    adjusted_birthdate_cues = 0
    adjusted_cfdna_patient_id_cues = 0
    adjusted_caregiver_name_cues = 0
    adjusted_caregiver_name_expansions = 0
    adjusted_weekday_date_clutter = 0
    adjusted_leading_date_list_clutter = 0
    adjusted_overlapping_spans = 0
    removed_relative_date_spans = 0
    removed_contact_cues = 0
    added_contextual_annotations: Counter[str] = Counter()
    original_text_length = len(text)

    contact_deletions = _contact_cue_deletion_ranges(text, annotations)
    if contact_deletions:
        text = _delete_ranges(text, contact_deletions)
        new_doc["text"] = text
        removed_contact_cues = len(contact_deletions)

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        begin = ann.get("begin")
        end = ann.get("end")
        if not isinstance(begin, int) or not isinstance(end, int):
            processed.append(ann)
            continue
        if begin < 0 or end > original_text_length or begin > end:
            processed.append(ann)
            continue

        new_ann = deepcopy(ann)
        begin = _map_position(begin, contact_deletions)
        end = _map_position(end, contact_deletions)

        if new_ann.get("label") == "Age_Birthdate":
            current = text[begin:end]
            _, adjusted = remove_age_birthdate_cue(current)
            if adjusted:
                match = AGE_BIRTHDATE_CUE_RE.match(current)
                begin += match.end() if match else 0
                adjusted_birthdate_cues += 1

        if new_ann.get("label") == "ID:Patient":
            current = text[begin:end]
            _, adjusted = remove_cfdna_patient_id_cue(current)
            if adjusted:
                match = CFDNA_PATIENT_ID_CUE_RE.match(current)
                begin += match.end() if match else 0
                adjusted_cfdna_patient_id_cues += 1

        if new_ann.get("label") == "Name:Caregiver":
            current = text[begin:end]
            without_cue, adjusted = remove_caregiver_name_cue(current)
            if adjusted:
                begin += len(current) - len(without_cue)
                adjusted_caregiver_name_cues += 1
            begin, end, expanded = expand_caregiver_name_subspan(
                text,
                begin,
                end,
                metadata=new_doc.get("metadata"),
            )
            if expanded:
                adjusted_caregiver_name_expansions += 1

        if new_ann.get("label") == "Date":
            begin, adjusted = normalize_weekday_date_clutter(text, begin, end)
            if adjusted:
                adjusted_weekday_date_clutter += 1

        if new_ann.get("label") in {"Date", "Age_Birthdate"}:
            begin, adjusted = trim_leading_date_list_clutter(
                text,
                begin,
                end,
                label=str(new_ann.get("label")),
            )
            if adjusted:
                adjusted_leading_date_list_clutter += 1

        if new_ann.get("label") == "Date" and is_relative_date_hard_negative(text[begin:end]):
            removed_relative_date_spans += 1
            continue

        if begin >= end or not text[begin:end].strip():
            removed_empty += 1
            continue

        new_ann["begin"] = begin
        new_ann["end"] = end
        new_ann["text"] = text[begin:end]
        processed.append(_sync_label_fields(new_ann))

    processed.sort(
        key=lambda item: (item.get("begin", -1), item.get("end", -1), item.get("label", ""))
    )

    for ann in missing_contextual_annotations(text, processed):
        processed.append(ann)
        added_contextual_annotations[str(ann.get("label", "unknown"))] += 1

    if added_contextual_annotations:
        processed.sort(
            key=lambda item: (item.get("begin", -1), item.get("end", -1), item.get("label", ""))
        )

    processed, overlap_adjusted = normalize_overlapping_annotations(text, processed)
    adjusted_overlapping_spans += overlap_adjusted

    new_doc["spans"] = processed

    external_applied = False
    external_available = False
    external_skip_reason = None
    if not use_external_post_process:
        external_skip_reason = "disabled"
    elif not _spans_are_external_ready(processed, text):
        external_skip_reason = "spans_not_ready"
    else:
        post_process_spans = _load_external_post_process()
        external_available = callable(post_process_spans)
        if callable(post_process_spans):
            external_processed = post_process_spans(
                processed,
                text,
                metadata=new_doc.get("metadata"),
            )
            if isinstance(external_processed, list):
                new_doc["spans"] = [
                    _sync_label_fields(deepcopy(ann))
                    for ann in external_processed
                    if isinstance(ann, dict)
                ]
                external_applied = True
            else:
                external_skip_reason = "unexpected_return_type"
        else:
            external_skip_reason = "unavailable"

    if external_applied:
        post_external_processed: list[dict[str, Any]] = []
        for ann in new_doc.get("spans", []):
            if not isinstance(ann, dict):
                continue
            begin = ann.get("begin")
            end = ann.get("end")
            if not isinstance(begin, int) or not isinstance(end, int):
                post_external_processed.append(ann)
                continue
            if begin < 0 or end > len(text) or begin > end:
                post_external_processed.append(ann)
                continue

            new_ann = deepcopy(ann)
            if new_ann.get("label") == "Age_Birthdate":
                current = text[begin:end]
                _, adjusted = remove_age_birthdate_cue(current)
                if adjusted:
                    match = AGE_BIRTHDATE_CUE_RE.match(current)
                    begin += match.end() if match else 0
                    adjusted_birthdate_cues += 1

            if new_ann.get("label") == "ID:Patient":
                current = text[begin:end]
                _, adjusted = remove_cfdna_patient_id_cue(current)
                if adjusted:
                    match = CFDNA_PATIENT_ID_CUE_RE.match(current)
                    begin += match.end() if match else 0
                    adjusted_cfdna_patient_id_cues += 1

            if new_ann.get("label") == "Name:Caregiver":
                current = text[begin:end]
                without_cue, adjusted = remove_caregiver_name_cue(current)
                if adjusted:
                    begin += len(current) - len(without_cue)
                    adjusted_caregiver_name_cues += 1
                begin, end, expanded = expand_caregiver_name_subspan(
                    text,
                    begin,
                    end,
                    metadata=new_doc.get("metadata"),
                )
                if expanded:
                    adjusted_caregiver_name_expansions += 1

            if new_ann.get("label") == "Date":
                begin, adjusted = normalize_weekday_date_clutter(text, begin, end)
                if adjusted:
                    adjusted_weekday_date_clutter += 1

            if new_ann.get("label") in {"Date", "Age_Birthdate"}:
                begin, adjusted = trim_leading_date_list_clutter(
                    text,
                    begin,
                    end,
                    label=str(new_ann.get("label")),
                )
                if adjusted:
                    adjusted_leading_date_list_clutter += 1

            if new_ann.get("label") == "Date" and is_relative_date_hard_negative(text[begin:end]):
                removed_relative_date_spans += 1
                continue

            if begin >= end or not text[begin:end].strip():
                removed_empty += 1
                continue

            new_ann["begin"] = begin
            new_ann["end"] = end
            new_ann["text"] = text[begin:end]
            post_external_processed.append(_sync_label_fields(new_ann))

        post_external_processed.sort(
            key=lambda item: (item.get("begin", -1), item.get("end", -1), item.get("label", ""))
        )

        for ann in missing_contextual_annotations(text, post_external_processed):
            post_external_processed.append(ann)
            added_contextual_annotations[str(ann.get("label", "unknown"))] += 1

        if added_contextual_annotations:
            post_external_processed.sort(
                key=lambda item: (item.get("begin", -1), item.get("end", -1), item.get("label", ""))
            )

        post_external_processed, overlap_adjusted = normalize_overlapping_annotations(
            text,
            post_external_processed,
        )
        adjusted_overlapping_spans += overlap_adjusted

        new_doc["spans"] = post_external_processed

    before = [_annotation_signature(ann) for ann in annotations if isinstance(ann, dict)]
    after = [
        _annotation_signature(ann)
        for ann in new_doc.get("spans", [])
        if isinstance(ann, dict)
    ]
    metadata = new_doc.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        new_doc["metadata"] = metadata
    production_metadata = {
        "version": VERSION,
        "applied": True,
        "changed": before != after,
        "adjusted_age_birthdate_cues": adjusted_birthdate_cues,
        "adjusted_cfdna_patient_id_cues": adjusted_cfdna_patient_id_cues,
        "adjusted_caregiver_name_cues": adjusted_caregiver_name_cues,
        "adjusted_caregiver_name_expansions": adjusted_caregiver_name_expansions,
        "adjusted_weekday_date_clutter": adjusted_weekday_date_clutter,
        "adjusted_leading_date_list_clutter": adjusted_leading_date_list_clutter,
        "adjusted_overlapping_spans": adjusted_overlapping_spans,
        "removed_relative_date_spans": removed_relative_date_spans,
        "removed_contact_cues": removed_contact_cues,
        "added_contextual_annotations": dict(sorted(added_contextual_annotations.items())),
        "removed_empty_spans": removed_empty,
        "external_post_process_requested": use_external_post_process,
        "external_post_process_available": external_available,
        "external_post_process_applied": external_applied,
    }
    if external_skip_reason:
        production_metadata["external_post_process_skip_reason"] = external_skip_reason
    if external_available:
        production_metadata["external_post_process_source"] = _external_post_process_source_info()
    metadata["production_postprocess"] = production_metadata

    return new_doc


def apply_production_post_process_to_documents(
    docs: list[dict[str, Any]],
    *,
    use_external_post_process: bool = True,
) -> list[dict[str, Any]]:
    return [
        apply_production_post_process(
            doc,
            use_external_post_process=use_external_post_process,
        )
        for doc in docs
    ]
