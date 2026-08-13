"""Quality checks and optional correction layer for synthetic annotations."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from meddeid_language_nl.date_age_variants import parse_date_text
from meddeid_language_nl.postprocess import post_process_spans

from .labels import LABELS, split_label
from .production_postprocess import (
    apply_production_post_process,
    missing_contextual_annotations,
)

PATIENT_ID_CODE_RE = re.compile(
    r"""
    \b(?:
        \d{2}[./-]\d{2}[./-]\d{2}[- ]\d{3}[. -]\d{2}
        | LAB-\d{6}
        | LIS-\d{4}-\d{6}
        | GLIMS-\d{2}-\d{5}
        | PATH-\d{4}-BE
        | PA-\d{4}-\d{5}
        | CYTO-\d{6}
        | MI\ \d{4}\ \d{4}
        | PACSWEB-(?:CT|MR|RX|US)\d{2}-\d{6}
        | ACSN-(?:CT|MR|RX|US)-\d{4}-\d{6}
        | StudyUID\ \d+(?:\.\d+){2,}
        | OK\ nr\.\ H\d{4}L\d{2}
        | OR-\d{4}-\d{5}
        | (?<=CFDNA\ )\d{5}
        | BHEALTH-\d{4}
        | TRIAL-\d{4}-[A-Z2-9]{5}
        | K\d{5}
        | STUDY-[A-Z2-9]{6}
        | EUDRACT-\d{4}-\d{6}
        | CRISIS-2026-\d{4}
        | D\d{3}-\d{5}
        | DEV-[A-Z2-9]{4}-\d{6}
        | SN-[A-Z2-9]{3}-\d{7}
        | DICOMweb\ study\ \d+(?:\.\d+){2,}
        | WADO-RS\ studies/\d+(?:\.\d+){2,}/series/\d+(?:\.\d+){2,}
        | DICOM-SR-\d{4}-[A-Z2-9]{6}\.pdf
        | IMG-(?:CT|MR|RX|US)-\d{6}\.dcm
        | Eindrapport\ \d{6}-UZA-MDO\.pdf
    )\b
    """,
    re.X,
)


REGEX_CHECKS = [
    ("Contactdetails", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    (
        "Contactdetails",
        re.compile(r"https?://[^\s.,;)]*[^\s.,;)]|www\.[^\s.,;)]*[^\s.,;)]"),
    ),
    (
        "Contactdetails",
        re.compile(r"(?<!\d)(?:\+32|0032|0)\s?4\d{2}(?:[\s./-]?\d{2}){3}(?!\d)"),
    ),
    (
        "Contactdetails",
        re.compile(r"\b(?:toestel|intern|ext\.|lijn)\s+\d{2,4}(?:\s\d{2})?\b", re.I),
    ),
    ("ID:Patient", PATIENT_ID_CODE_RE),
    ("ID:Caregiver", re.compile(r"\b(?:LOT|DR|VRN)-\d{5,11}\b|\bRIZIV\s+Nr\.\s+\d{11}\b")),
    ("Organization:Healthcare", re.compile(r"\b[Rr]oute\s+\d+[A-Za-z]?\b")),
    ("Organization:Healthcare", re.compile(r"\bOK\s+\d+[A-Za-z]?\b")),
    ("Organization:Healthcare", re.compile(r"\b[A-Z]\d{1,2}[A-Z]?:BOX\d+[A-Za-z]?\b")),
    ("Date", re.compile(r"\b\d{1,2}([/-])\d{1,2}\1\d{2,4}\b")),
    (
        "Date",
        re.compile(
            r"(?<!\d\.)\b\d{1,2}\.\d{1,2}\.\d{2,4}\b(?![- ]?\d{3}[. -]\d{2}|\.\d)"
        ),
    ),
    ("Date", re.compile(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b")),
    ("Date", re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:19\d{2}|20\d{2})\b")),
    (
        "Date",
        re.compile(
            r"\b\d{1,2}\.?\s+(?:jan(?:uari)?|feb(?:ruari)?|mrt|maart|apr(?:il)?|mei|jun(?:i)?|jul(?:i)?|aug(?:ustus)?|"
            r"sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|dec(?:ember)|mars)\.?\s+'?\d{2,4}\b",
            re.I,
        ),
    ),
    (
        "Date",
        re.compile(
            r"\b(?:jan(?:uari)?|feb(?:ruari)?|mrt|maart|apr(?:il)?|mei|jun(?:i)?|jul(?:i)?|aug(?:ustus)?|"
            r"sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|dec(?:ember))\.?\s+'?\d{2,4}\b",
            re.I,
        ),
    ),
    (
        "Age_Birthdate",
        re.compile(
            r"(?<!\d)(?:(?:[1-9]|1[0-7])[- ]jarige|"
            r"(?:1[89]|[2-9]\d|1[01]\d|120)\s*(?:[- ](?:jarige|jaar|jaren)|j|jr)|"
            r"(?:[1-9]|1[01])[- ]maanden)(?!\w)",
            re.I,
        ),
    ),
]


REGEX_GROUP_CHECKS = [
    (
        "Organization:Healthcare",
        re.compile(
            r"\b(?:kamer|afdeling|gebouw|blok)\s+"
            r"(?P<code>[A-Z]\d{1,2}[A-Z]?(?::BOX\d+[A-Za-z]?)?)\b",
            re.I,
        ),
        "code",
    ),
]


DATE_TIME_FALSE_YEAR_VALUE_RE = re.compile(
    r"""
    \b\d{1,2}\.?\s+
    (?:jan(?:uari)?|feb(?:ruari)?|mrt|maart|apr(?:il)?|mei|jun(?:i)?|jul(?:i)?|
       aug(?:ustus)?|sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|dec(?:ember)?|mars)\.?
    \s+\d{1,2}
    """,
    re.I | re.X,
)
MONTH_TIME_FALSE_YEAR_VALUE_RE = re.compile(
    r"""
    \b(?:jan(?:uari)?|feb(?:ruari)?|mrt|maart|apr(?:il)?|mei|jun(?:i)?|jul(?:i)?|
       aug(?:ustus)?|sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|dec(?:ember)?|mars)\.?
    \s+\d{1,2}
    """,
    re.I | re.X,
)
AGE_DURATION_CONTEXT_RE = re.compile(
    r"""
    (?:
        \b(?:binnen|na|over|elke|iedere|gedurende|sinds|voor|vanaf|tegen)\s+
        |
        \bom\s+de\s+
        |
        \b(?:controle|hercontrole|follow-up|opvolging|vervolg|evaluatie|afspraak|planmatig\s+vervolg)\W{0,24}
    )$
    """,
    re.I | re.X,
)
AGE_POSITIVE_CONTEXT_BEFORE_RE = re.compile(
    r"""
    (?:
        \b(?:leeftijd|geb\.?|geboren|geboortedatum|oud|pati[eë]nt|pt\.?|man|vrouw|
            jongen|meisje|kind|baby|zuigeling|neonaat|peuter|kleuter)\W{0,36}
        |
        ^[^\n]{0,80}?\(\s*
    )$
    """,
    re.I | re.X,
)
AGE_POSITIVE_CONTEXT_AFTER_RE = re.compile(
    r"^\W{0,16}(?:oud|bij geboorte|op consult|opname|met)\b",
    re.I,
)


def _is_false_positive_date_match(text: str, begin: int, end: int) -> bool:
    value = text[begin:end]
    if "\n" in value:
        return True
    if parse_date_text(value, label="Date") is None:
        return True
    if DATE_TIME_FALSE_YEAR_VALUE_RE.fullmatch(value) and re.match(
        r"[.:uUhH]\d{2}\b",
        text[end : min(len(text), end + 4)],
    ):
        return True
    if (
        MONTH_TIME_FALSE_YEAR_VALUE_RE.fullmatch(value)
        and re.search(r"\b\d{1,2}\.?\s+$", text[max(0, begin - 6) : begin])
        and re.match(r"[.:uUhH]\d{2}\b", text[end : min(len(text), end + 4)])
    ):
        return True

    before = text[max(0, begin - 40) : begin].lower()
    after = text[end : min(len(text), end + 24)].lower()
    if re.search(r"\b(?:duratears|seneuval|caps?\.?|ocul\.?|zalf|cnk|mg|ml|g)\W*$", before):
        return True
    if re.search(r"^\W*(?:mg|ml|g|caps?\.?|ocul\.?|zalf|harde|siroop|cnk)\b", after):
        return True
    if re.search(r"/10\b", value):
        return True
    return False


def _is_false_positive_age_match(text: str, begin: int, end: int) -> bool:
    value = text[begin:end]
    if begin > 0 and (text[begin - 1].isalnum() or text[begin - 1] in "-_"):
        return True
    if end < len(text) and (text[end].isalnum() or text[end] in "-_"):
        return True

    before = text[max(0, begin - 80) : begin]
    after = text[end : min(len(text), end + 50)]

    if AGE_DURATION_CONTEXT_RE.search(before):
        return True

    if re.search(r"\bmaanden?\b", value, re.I):
        return not (
            AGE_POSITIVE_CONTEXT_BEFORE_RE.search(before)
            or AGE_POSITIVE_CONTEXT_AFTER_RE.search(after)
        )

    return False


@dataclass
class JudgeResult:
    document_id: str
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.passed = False
        if message not in self.issues:
            self.issues.append(message)

    def correction(self, message: str) -> None:
        self.corrections.append(message)


def _is_covered(
    begin: int, end: int, annotations: list[dict], labels: set[str] | None = None
) -> bool:
    for ann in annotations:
        if labels is not None and ann["label"] not in labels:
            continue
        if ann["begin"] <= begin and ann["end"] >= end:
            return True
    return False


def normalize_annotations(doc: dict[str, Any], result: JudgeResult) -> None:
    text = doc["text"]
    normalized: list[dict[str, Any]] = []
    for ann in doc.get("spans", []):
        begin = ann.get("begin")
        end = ann.get("end")
        if (
            not isinstance(begin, int)
            or not isinstance(end, int)
            or begin < 0
            or end > len(text)
            or begin >= end
        ):
            if (
                isinstance(begin, int)
                and isinstance(end, int)
                and 0 <= begin <= len(text)
                and begin == end
                and not str(ann.get("text", "")).strip()
            ):
                result.correction(
                    f"Removed empty zero-width annotation at {begin}:{end}."
                )
                continue
            result.fail(f"Invalid span bounds: {ann}")
            continue

        actual = text[begin:end]
        if ann.get("text") != actual:
            ann["text"] = actual
            result.correction(f"Repaired annotation text at {begin}:{end}.")

        label = ann.get("label")
        if label not in LABELS:
            result.fail(f"Unknown label {label!r} for span {actual!r}.")
            continue

        category, subtype = split_label(label)
        if ann.get("Category") != category:
            ann["Category"] = category
            result.correction(f"Repaired Category for {actual!r}.")
        if ann.get("Subtype") != subtype:
            ann["Subtype"] = subtype
            result.correction(f"Repaired Subtype for {actual!r}.")
        normalized.append(ann)
    doc["spans"] = normalized


def deterministic_judge(doc: dict[str, Any]) -> JudgeResult:
    result = JudgeResult(document_id=doc.get("document_id", ""))
    text = doc.get("text", "")
    annotations = sorted(
        doc.get("spans", []), key=lambda x: (x.get("begin", -1), x.get("end", -1))
    )
    doc["spans"] = annotations

    normalize_annotations(doc, result)
    annotations = sorted(
        doc.get("spans", []), key=lambda x: (x.get("begin", -1), x.get("end", -1))
    )
    doc["spans"] = annotations

    for left, right in zip(annotations, annotations[1:]):
        if left["end"] > right["begin"]:
            result.fail(
                f"Overlapping spans: {left['text']!r} ({left['label']}) and {right['text']!r} ({right['label']})."
            )

    seen = set()
    for ann in annotations:
        key = (ann["begin"], ann["end"], ann["label"])
        if key in seen:
            result.fail(f"Duplicate span {key}.")
        seen.add(key)

    for expected_label, pattern in REGEX_CHECKS:
        labels = {expected_label}
        if expected_label == "Contactdetails":
            labels = {"Contactdetails"}
        if expected_label == "Date":
            labels = {"Date", "Age_Birthdate"}
        if expected_label == "Age_Birthdate":
            labels = {"Age_Birthdate", "Date"}
        for match in pattern.finditer(text):
            if expected_label == "Date" and _is_false_positive_date_match(
                text,
                match.start(),
                match.end(),
            ):
                continue
            if expected_label == "Age_Birthdate" and _is_false_positive_age_match(
                text,
                match.start(),
                match.end(),
            ):
                continue
            if not _is_covered(match.start(), match.end(), annotations, labels):
                result.fail(
                    f"Potential missed {expected_label}: {match.group(0)!r} at {match.start()}:{match.end()}."
                )

    for expected_label, pattern, group_name in REGEX_GROUP_CHECKS:
        labels = {expected_label}
        for match in pattern.finditer(text):
            begin, end = match.span(group_name)
            if not _is_covered(begin, end, annotations, labels):
                result.fail(
                    f"Potential missed {expected_label}: {match.group(group_name)!r} at {begin}:{end}."
                )

    for ann in missing_contextual_annotations(text, annotations):
        result.fail(
            f"Potential missed {ann['label']}: {ann['text']!r} at {ann['begin']}:{ann['end']}."
        )

    return result


def apply_post_process_if_available(doc: dict[str, Any], result: JudgeResult) -> None:
    before = [(a["begin"], a["end"], a["label"], a["text"]) for a in doc["spans"]]
    try:
        processed = post_process_spans(
            doc["spans"], doc["text"], metadata=doc.get("metadata")
        )
    except Exception as exc:  # pragma: no cover - optional local integration
        result.fail(f"post_process_spans failed: {exc}")
        return
    after = [(a["begin"], a["end"], a["label"], a["text"]) for a in processed]
    if before != after:
        doc["spans"] = processed
        result.correction("Applied post_process_spans corrections.")


def openai_judge_if_configured(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Ask an OpenAI model for correction suggestions when OPENAI_API_KEY is set.

    The deterministic judge is the source of truth for offsets. The model is used only
    as a second reviewer for missed or wrongly labelled PII.
    """

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return {
            "status": "skipped",
            "reason": "openai package is not installed; run pip install -r requirements.txt",
        }

    client = OpenAI()
    model = os.getenv("OPENAI_JUDGE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    prompt = {
        "task": "Review this synthetic Belgian clinical de-identification example. Return JSON only.",
        "labels": sorted(LABELS),
        "instructions": [
            "Check whether every directly identifying span is annotated.",
            "Check whether category/subtype matches the guideline labels.",
            "Family, friends, other patients, and external non-caregiver people are Name:Other, not Name:Patient.",
            "Relationship words such as moeder, vader, dochter, partner, broer, frere are not annotated unless they are part of a name.",
            "Profession is annotated for patients and relatives only; professions, specialties, and functions of treating caregivers are not annotated as Profession.",
            "School year or education level for the patient is Profession.",
            "Medication names, brand names, formulations, doses, routes, and schedules are clinical content, not PII.",
            "Administrative gender/sex markers such as M, V, X, man, vrouw, onbekend are clinical/admin context and not PII.",
            "Named schools, employers, clubs, and non-healthcare organizations are Organization:Other even when they are linked to the patient.",
            "Healthcare room, building, ward, operating-room, route, and box codes such as D4:BOX5, OK 1, and Route 28 are Organization:Healthcare according to the guideline.",
            "For caregiver names, include adjacent titles such as dr. or prof. dr.; do not require specialties such as psycholoog or pneumologie to be annotated.",
            "Address/location annotations should cover the identifying address or place text, not generic descriptor words such as praktijkregio or woonadres.",
            "Identifier annotations should include stable prefixes when the prefix is part of the identifier string, such as LAB-313143 or CRISIS-2026-1376.",
            "Patient-linked dossier numbers, report numbers, pathology numbers, lab accession IDs, imaging/PACS/DICOMweb keys, study names/references, study protocol IDs/names, and patient-linked files are ID:Patient.",
            "Codes for genes or biomarkers, HGVS variants, rsIDs, transcript IDs, LOINC/SNOMED/ICD terminology codes, ISO standards, and generic device model names are not ID unless the context explicitly makes them patient- or organization-linked.",
            "Material LOT numbers such as LOT-23532 are ID:Caregiver according to the guideline.",
            "Do not include generic descriptor words before identifiers, such as rijksregisternummer, dossier, patientnummer, INSZ, RIZIV, or identifiant medecin, unless they are physically part of the code itself.",
            "Do not invent offsets; if a correction is needed, quote the exact text and proposed label.",
        ],
        "document": doc,
        "response_schema": {
            "issues": [
                {
                    "exact_text": "text needing attention",
                    "proposed_label": "one label from labels",
                    "reason": "short reason",
                }
            ],
            "verdict": "pass or needs_changes",
        },
    }
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            }
        ],
    )
    try:
        review = json.loads(response.output_text)
    except Exception:
        return {"status": "unparsed", "raw": response.output_text}
    if isinstance(review, dict) and str(review.get("verdict", "")).lower() == "pass":
        review["issues"] = [
            issue
            for issue in review.get("issues", [])
            if str(issue.get("proposed_label", "")).lower()
            not in {"no annotation", "none", "n/a"}
        ]
    return review


def _openai_review_has_actionable_issues(review: dict[str, Any] | None) -> bool:
    if not isinstance(review, dict):
        return False
    if review.get("status") in {"skipped", "unparsed"}:
        return False
    issues = review.get("issues")
    if isinstance(issues, list) and len(issues) > 0:
        return True
    return str(review.get("verdict", "")).lower() not in {"", "pass"}


def judge_documents(
    docs: list[dict[str, Any]],
    *,
    use_openai: bool = False,
    openai_advisory: bool = False,
    use_post_process: bool = False,
) -> tuple[list[dict[str, Any]], list[JudgeResult], list[dict[str, Any]]]:
    load_dotenv()
    results: list[JudgeResult] = []
    model_reviews: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        doc = apply_production_post_process(doc)
        docs[index] = doc
        pre_result = deterministic_judge(doc)
        if use_post_process:
            apply_post_process_if_available(doc, pre_result)
        result = deterministic_judge(doc)
        result.corrections = pre_result.corrections + result.corrections
        if pre_result.issues:
            result.issues = pre_result.issues + result.issues
            result.passed = False
        result.issues = list(dict.fromkeys(result.issues))
        result.corrections = list(dict.fromkeys(result.corrections))
        if use_openai:
            review = openai_judge_if_configured(doc)
            if review is not None:
                model_reviews.append(
                    {"document_id": doc["document_id"], "review": review}
                )
                if not openai_advisory and _openai_review_has_actionable_issues(review):
                    result.fail(
                        f"OpenAI judge flagged actionable issues: {json.dumps(review, ensure_ascii=False)}"
                    )
        results.append(result)
    return docs, results, model_reviews


def write_report(
    results: list[JudgeResult], model_reviews: list[dict[str, Any]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Synthetic DEID Judge Report", ""]
    passed = sum(1 for result in results if result.passed)
    lines.append(f"Documents passed deterministic judge: {passed}/{len(results)}")
    lines.append("")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"## {result.document_id}: {status}")
        if result.corrections:
            lines.append("Corrections:")
            lines.extend(f"- {item}" for item in result.corrections)
        if result.issues:
            lines.append("Issues:")
            lines.extend(f"- {item}" for item in result.issues)
        if not result.corrections and not result.issues:
            lines.append("No issues.")
        lines.append("")
    if model_reviews:
        lines.append("## Optional Judge Reviews")
        lines.append("```json")
        lines.append(json.dumps(model_reviews, ensure_ascii=False, indent=2))
        lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
