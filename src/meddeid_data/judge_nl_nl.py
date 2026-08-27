"""Netherlands-specific quality judge for synthetic clinical documents."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from .judge import (
    JudgeResult,
    _is_covered,
    _openai_review_has_actionable_issues,
    apply_post_process_if_available,
    deterministic_judge,
)
from .labels import LABELS
from .production_postprocess import apply_production_post_process


PROFILE_ID = "nl-NL"

NETHERLANDS_REGEX_CHECKS = (
    (
        "Contactdetails",
        re.compile(
            r"(?<!\d)(?:\+31\s?6|0031\s?6|06)(?:[\s./-]?\d){8}(?!\d)"
        ),
        {"Contactdetails"},
        None,
    ),
    (
        "ID:Patient",
        re.compile(r"\bBSN\s*:?[ \t]*(?P<value>\d{9})\b", re.I),
        {"ID:Patient"},
        "value",
    ),
    (
        "ID:Caregiver",
        re.compile(
            r"\bBIG(?:-register)?(?:nummer)?\s*:?[ \t]*(?P<value>BIG-\d{11}|\d{11})\b",
            re.I,
        ),
        {"ID:Caregiver"},
        "value",
    ),
    (
        "ID:Patient",
        re.compile(
            r"\b(?:EPD|ZIS|LAB-NL|PA-NL|PACS-NL|OK-NL|DOSSIER-NL|"
            r"STUDIE-NL|PROTOCOL-NL|ONDERZOEK-NL|DEVICE-NL)-[A-Z0-9-]+\b"
        ),
        {"ID:Patient"},
        None,
    ),
    (
        "Address_Location:Patient",
        re.compile(r"\b(?P<value>[1-9]\d{3}\s?[A-Z]{2})\b"),
        {
            "Address_Location:Patient",
            "Address_Location:Caregiver",
            "Address_Location:Other",
        },
        "value",
    ),
)

BELGIAN_CONTEXT_PATTERNS = (
    re.compile(r"(?<!\d)(?:\+32|0032)(?!\d)"),
    re.compile(r"@example\.be\b", re.I),
    re.compile(r"\b(?:RIZIV|INSZ|rijksregisternummer|kinesist|operatiekwartier)\b", re.I),
    re.compile(r"\bUZA\b|Universitair Ziekenhuis Antwerpen", re.I),
    re.compile(r"\bPATH-\d+-BE\b|\bBHEALTH-\d+\b|\bBEL-[A-Z0-9-]+\b"),
)


def _add_netherlands_checks(doc: dict[str, Any], result: JudgeResult) -> None:
    text = str(doc.get("text", ""))
    annotations = doc.get("spans", [])
    metadata = doc.get("metadata", {})
    if not isinstance(metadata, dict):
        result.fail("nl-NL document metadata must be an object.")
        metadata = {}
    if metadata.get("lang") != PROFILE_ID:
        result.fail("nl-NL judge requires metadata.lang='nl-NL'.")
    if metadata.get("generation_profile") != PROFILE_ID:
        result.fail("nl-NL judge requires metadata.generation_profile='nl-NL'.")

    for expected_label, pattern, labels, group_name in NETHERLANDS_REGEX_CHECKS:
        for match in pattern.finditer(text):
            begin, end = match.span(group_name) if group_name else match.span()
            value = match.group(group_name) if group_name else match.group(0)
            if not _is_covered(begin, end, annotations, labels):
                result.fail(
                    f"Potential missed {expected_label}: {value!r} at {begin}:{end}."
                )

    for pattern in BELGIAN_CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            result.fail(
                f"Belgian-context leakage in nl-NL document: {match.group(0)!r} "
                f"at {match.start()}:{match.end()}."
            )


def deterministic_judge_nl_nl(doc: dict[str, Any]) -> JudgeResult:
    """Run shared annotation checks plus Netherlands locale checks."""

    result = deterministic_judge(doc)
    _add_netherlands_checks(doc, result)
    result.issues = list(dict.fromkeys(result.issues))
    result.corrections = list(dict.fromkeys(result.corrections))
    return result


def openai_judge_nl_nl_if_configured(
    doc: dict[str, Any],
) -> dict[str, Any] | None:
    """Request an advisory Netherlands-context review when explicitly enabled."""

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return {
            "status": "skipped",
            "reason": "openai package is not installed; install the judge extra",
        }

    prompt = {
        "task": (
            "Review this synthetic Netherlands clinical de-identification example. "
            "Return JSON only."
        ),
        "locale": PROFILE_ID,
        "labels": sorted(LABELS),
        "instructions": [
            "Check whether every directly identifying span is annotated with the supplied label set.",
            "Use Netherlands healthcare language and administrative conventions.",
            "BSN, EPD/ZIS numbers and patient-linked accessions are ID:Patient.",
            "BIG-register numbers are ID:Caregiver.",
            "Netherlands postcodes, streets and residences belong to the applicable address/location span.",
            "Telephone forms beginning +31, 0031 or 06 and email addresses are Contactdetails.",
            "SEH, polikliniek, huisarts, AIOS, ANIOS, fysiotherapeut and wijkverpleegkundige are valid Netherlands terms.",
            "Flag Belgian leakage such as RIZIV, INSZ, rijksregisternummer, +32, example.be, UZA, kinesist or operatiekwartier.",
            "Caregiver specialties and functions are not Profession; a patient or relative profession is Profession.",
            "Medication names, doses, routes, diagnosis codes and generic device models are clinical content, not PII.",
            "Do not invent offsets; quote the exact source text for every proposed correction.",
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
    client = OpenAI()
    model = os.getenv("OPENAI_JUDGE_MODEL") or os.getenv(
        "OPENAI_MODEL", "gpt-5.4-mini"
    )
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


def judge_documents_nl_nl(
    docs: list[dict[str, Any]],
    *,
    use_openai: bool = False,
    openai_advisory: bool = False,
    use_post_process: bool = False,
) -> tuple[list[dict[str, Any]], list[JudgeResult], list[dict[str, Any]]]:
    """Review documents using Netherlands rules without a Belgian judge prompt."""

    results: list[JudgeResult] = []
    model_reviews: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        doc = apply_production_post_process(doc)
        docs[index] = doc
        pre_result = deterministic_judge_nl_nl(doc)
        if use_post_process:
            apply_post_process_if_available(doc, pre_result)
        result = deterministic_judge_nl_nl(doc)
        result.corrections = list(
            dict.fromkeys(pre_result.corrections + result.corrections)
        )
        result.issues = list(dict.fromkeys(pre_result.issues + result.issues))
        result.passed = not result.issues
        if use_openai:
            review = openai_judge_nl_nl_if_configured(doc)
            if review is not None:
                model_reviews.append(
                    {"document_id": doc["document_id"], "review": review}
                )
                if not openai_advisory and _openai_review_has_actionable_issues(
                    review
                ):
                    result.fail(
                        "OpenAI Netherlands judge flagged actionable issues: "
                        f"{json.dumps(review, ensure_ascii=False)}"
                    )
        results.append(result)
    return docs, results, model_reviews


__all__ = [
    "BELGIAN_CONTEXT_PATTERNS",
    "deterministic_judge_nl_nl",
    "judge_documents_nl_nl",
    "openai_judge_nl_nl_if_configured",
]
