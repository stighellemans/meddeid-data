"""Training-grade English clinical-note rendering with exact PII markers.

The structured case owns every value that may be identifying.  The language
model authors the clinical prose, but must wrap any selected PII value in an
exact marker.  Markers are removed locally and converted to canonical offsets.
This is deliberately separate from the small deterministic reference renderer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from meddeid_core import BERT_ENTITY_LABELS, split_label, validate_record
from meddeid_language_en import (
    is_approved_synthetic_identifier,
    is_approved_synthetic_phone,
    lookup_source,
)

from .semantic_types import semantic_type_for_target

DEFAULT_ENGLISH_LLM_MODEL = "gpt-5.6-luna"
MARKER_START = "[[PII|"
MARKER_RE = re.compile(r"\[\[PII\|([^|\]]+)\|([^\]]+)\]\]")
SHORT_MARKER_START = "[[PII_"
SHORT_MARKER_RE = re.compile(r"\[\[PII_(\d{2})\]\]")
ALLOWED_LABELS = tuple(BERT_ENTITY_LABELS)
ALLOWED_LABEL_SET = frozenset(ALLOWED_LABELS)
UsageRecorder = Callable[[dict[str, Any]], None]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
GB_PHONE_RE = re.compile(r"(?<!\d)07700\s+900\d{3}(?!\d)")
US_PHONE_RE = re.compile(r"(?<!\d)\(\d{3}\)\s+555-01\d{2}(?!\d)")
GB_ID_RE = re.compile(
    r"\b(?:(?:MRN|LAB)-0\d{6,8}|S2\d-0\d{6}|GMC(?:-TEST-?|\s)0\d{6}|"
    r"(?:MRN-GB|REPORT-GB)-\d{6})\b"
)
US_ID_RE = re.compile(
    r"\b(?:(?:MRN|LAB)-0\d{6,8}|S2\d-0\d{6}|LIC-?\d{8}|"
    r"(?:MRN-US|REPORT-US)-\d{6}|1234567893)\b"
)
LABELED_REPORT_ID_RE = re.compile(
    r"\b(?:report(?:\s*(?:and|/)\s*accession)?|accession)"
    r"(?:\s+(?:ID|identifier|number))?\s*:\s*"
    r"(?=[A-Z0-9._/-]*\d)([A-Z][A-Z0-9._/-]*[A-Z0-9])",
    re.I,
)
TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)?", re.UNICODE)
FULL_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
    re.I,
)
SPELLED_NUMBER_RANGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*[–—-]\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|"
    r"five|six|seven|eight|nine))?\b",
    re.I,
)
UNNATURAL_HARD_NEGATIVE_RE = re.compile(
    r"\b(?:unrelated to|not relevant to|for completeness|does not represent (?:a |an )?"
    r"(?:patient )?identifier|is not (?:a |an )?(?:patient )?(?:identifier|pii)|"
    r"not (?:a |an )?(?:patient )?(?:identifier|pii)|non-pii|test example|"
    r"the relevant (?:clinician(?:[’']s)? )?role is|"
    r"the relevant (?:attending|consultant)\s+[a-z-]+|the phrase|the role of (?:a |an )|"
    r"relevant care coordination note|relevant comparison clinician role|"
    r"role identified in (?:the )?referral pathway|relevant clinician is (?:(?:a|an) )?|"
    r"relevant clinician role (?:noted|recorded)|relevant clinician role\s*:\s*[a-z-]+|"
    r"relevant role for consultation|(?:attending|consultant)\s+[a-z-]+-consulted case)\b",
    re.I,
)
ARTIFICIAL_ROLE_FIELD_RE = re.compile(
    r"\b(?:relevant (?:clinician(?: role)?|specialist(?: role)?|consultant role)\s*:\s*[a-z-]+|"
    r"the relevant clinician[’']s role is\s+[a-z-]+|"
    r"the relevant (?:attending|consultant)\s+[a-z-]+|"
    r"relevant care team contact\s*:\s*[a-z-]+|"
    r"(?:attending|consultant)\s+[a-z-]+-consulted case)\b",
    re.I,
)
EMBEDDED_OCCUPATION_FIELD_RE = re.compile(
    r"\b(?:works?|employed)\s+(?:as\s+)?(?:an?\s+)?Occupation\s*:", re.I
)
HARD_NEGATIVE_SELF_CORRECTION_RE = re.compile(r"^\?\s*No(?:\b|[—–-])", re.I)
UNNATURAL_AGE_RE = re.compile(
    r"\b(?:Age\s*:?\s*(?:(?:a|an)\s+)?|aged\s+)\d+-year-old\b", re.I
)
REDUNDANT_AGE_PREFIX_RE = re.compile(r"\bAge\s*:?\s*(?:aged|age)\b", re.I)
WRONG_AGE_ARTICLE_RE = re.compile(
    r"(?:\ba\s+(?:8\d*|11|18)-|\ban\s+(?!(?:8\d*|11|18)-)\d+-)"
    r"(?:day|week|month|year)-old\b",
    re.I,
)
MISSING_AGE_ARTICLE_RE = re.compile(
    r"\b(?:(?:in|of|with|for)\s+\d{1,3}-(?:day|week|month|year)-old\s+"
    r"(?:child|patient|infant|adolescent|adult)|patient\s+is\s+\d{1,3}-"
    r"(?:day|week|month|year)-old)\b",
    re.I,
)
SUBJECTLESS_AGE_RE = re.compile(
    r"(?m)(?:^|[.!?]\s+|\n)\d{1,3}\s+y/o\s+"
    r"(?:was\b|Resting\b|Alert\b|Awake\b|Oriented\b|Stable\b|Comfortable\b)",
    re.I,
)
SUBJECTLESS_CONTEXTUAL_AGE_RE = re.compile(
    r"(?:\bage\s+\d{1,3}(?:\s+(?:days?|weeks?|months?|years?))?\s+was\s+reviewed\b|"
    r"(?:^|[.!?]\s+|[,:;]\s+)aged\s+\d{1,3}\s+(?:days?|weeks?|months?|years?)\s+was\b)",
    re.I | re.M,
)
REDUNDANT_AGE_GROUP_RE = re.compile(
    r"\b(?:aged|age)\s+\d{1,3}\s+(?:days?|weeks?|months?|years?)\s+"
    r"(?:patient|neonate|infant|child|adolescent|adult)\b|"
    r"\b(?:the\s+)?patient\s+is\s+(?:a|an)\s+\d{1,3}-"
    r"(?:day|week|month|year)-old\s+patient\b|"
    r"\b(?:This|The)\s+\d{1,3}-(?:day|week|month|year)-old\s+patient\s+is\s+"
    r"(?:a|an)\s+(?:neonate|infant|child|adolescent|adult)\b|"
    r"\b(?:a|an)\s+\d{1,3}-(?:day|week|month|year)-old\s+patient\s*,\s*"
    r"(?:neonate|infant|child|adolescent|adult)\b",
    re.I,
)
REDUNDANT_MEASUREMENT_RE = re.compile(
    r"\b(?:(?:measured|recorded|current)\s+)?weight\s+(?:was|is)\s+weight\b", re.I
)
REDUNDANT_HARD_NEGATIVE_RE = re.compile(
    r"\b(?:(?:oral rehydration|hydration(?: status)?)\s+may improve with hydration|"
    r"oral rehydration solution(?:\s*;\s*|\s+)may improve with hydration|"
    r"fluids?\s+may improve with hydration)\b|"
    r"(?:^|[.;—–]\s*)may improve with hydration\b",
    re.I | re.M,
)
DUPLICATED_ID_PREFIX_RE = re.compile(
    r"\b(?:GMC-TEST|MRN-(?:GB|US)|REPORT-(?:GB|US))(?:\s*:?\s+|-)"
    r"(?:GMC-TEST|MRN-(?:GB|US)|REPORT-(?:GB|US))-",
    re.I,
)
UNNATURAL_GMC_ID_LABEL_RE = re.compile(
    r"\b(?:GMC\s+GMC-TEST|GMC-TEST(?:\s+(?:professional\s+)?identifier)?\s+GMC-TEST)-\d{7}\b",
    re.I,
)
HARD_NEGATIVE_REJECTION_RE = re.compile(
    r"^\s+is\s+not\s+(?:appropriate|indicated|needed|relevant)\b", re.I
)
DUPLICATED_CREDENTIAL_RE = re.compile(r"\b(?:MD|DO|MBBS)-(?:MD|DO|MBBS)\b", re.I)
SPELLED_RESPIRATORY_RATE_RE = re.compile(
    r"\brespiratory rate\s+(?:twenty|thirty|forty|fifty|sixty)\b",
    re.I,
)
ADVANCED_PROFESSION_RE = re.compile(
    r"\b(?:architect|attorney|civil engineer|clinical pharmacist|doctor|engineer|"
    r"geriatric physical therapist|hospital pharmacist|lawyer|lecturer|pharmacist|"
    r"physical therapist|physiotherapist|police officer|teacher)\b",
    re.I,
)
MISSING_AGE_COPULA_RE = re.compile(
    r"(?:\b(?:The|This)\s+\d{1,3}-year-old\s+patient\s+"
    r"(?:awake|alert|oriented|stable|comfortable|admitted|reviewed)\b|"
    r"\b(?:The\s+)?patient,\s+(?:aged|age)\s+\d{1,3}(?:\s+(?:days?|weeks?|months?|years?))?,?\s+"
    r"reviewed\s+(?:on|in|by|during|at|with)\b)",
    re.I,
)
GENERATION_META_RE = re.compile(
    r"(?:\b(?:supplied|provided) clinical age form\b|"
    r"\b(?:Name:(?:Patient|Caregiver|Other)|Address_Location:(?:Patient|Caregiver|Other)|"
    r"Age(?:/Birthdate|_Birthdate)|Contactdetails|ID:(?:Patient|Caregiver)|"
    r"Organization:(?:Healthcare|Other))\s*:|\bhard-negative\b|"
    r"<(?:Date|Name:(?:Patient|Caregiver|Other)|Age_Birthdate|Contactdetails|"
    r"Address_Location:(?:Patient|Caregiver|Other)|ID:(?:Patient|Caregiver)|"
    r"Organization:(?:Healthcare|Other)|Profession)>)",
    re.I,
)
BAD_PREPOSITION_RE = re.compile(
    r"(?:\b(?:planned|scheduled|discussed|arranged)\s+for\s+in\b|"
    r"\bin\s+in\s+(?=(?:about\s+)?\d))",
    re.I,
)
EXPLICIT_AGE_RE = re.compile(
    r"\b(?:aged\s+|age\s*:?\s*)\d{1,3}(?:\s+(?:days?|weeks?|months?|years?))?\b|"
    r"\b\d{1,3}\s*(?:d/o|wk\s+old|m/o|y/o|(?:days?|weeks?|months?|years?)\s+old|"
    r"-(?:day|week|month|year)-old)\b|"
    r"\bday\s+\d{1,3}\s+of\s+life\b|"
    r",\s*\d{1,3}\s+(?:days?|weeks?|months?|years?)\b",
    re.I,
)
EXPLICIT_BIRTH_YEAR_RE = re.compile(
    r"\b(?:born\s+in|birth\s+year\s*:?|year\s+of\s+birth\s*:?)\s*((?:19|20)\d{2})\b",
    re.I,
)
UNGRAMMATICAL_DISCHARGE_RE = re.compile(
    r"\bstable for discharge home with (?:tolerating|ambulating|eating|drinking)\b",
    re.I,
)
UNGRAMMATICAL_DURATION_RE = re.compile(
    r"\b(?:symptoms?|breathlessness|pain|headache|cough|fever)\s+"
    r"(?:began|commenced|(?:has|had)\s+developed)\s+for\s+\d+\b",
    re.I,
)
REDUNDANT_DURATION_FIELD_RE = re.compile(
    r"(?i)(?:\b(?:symptom\s+)?duration\s*:\s*|"
    r"\b(?:current\s+)?symptom\s+duration\s+(?:was\s+)?(?:noted|recorded)\s+as\s+)"
    r"for\s+\d+\s+days?\b"
)
MALFORMED_CREDENTIAL_SUFFIX_RE = re.compile(
    r"\b(?:MBBS|MBChB|MD|DO)\s+qc\b", re.I
)
UNRESOLVED_TEMPLATE_RE = re.compile(
    r"\[[^\]\n]{0,30}\b(?:day|date|time|name|address|phone|identifier|hospital\s+record|insert|tbd)"
    r"\b[^\]\n]{0,30}\]",
    re.I,
)
STRAY_GENERATION_SYNTAX_RE = re.compile(
    r'(?:\bGP"\)\]|(?m:^//\s*$)|\b[A-Za-z]{2,}\?/\d|<\|end\|>|'
    r'\bAuthorised\.calc\b|\b(?:Laboratory|Medicine)\.calc\b|\bclinician salvaged\?|\bdeset\?|\bJSImport\b|'
    r'\bqc syndrom\b|\bAvalaDr\?|\bcalibrator\?|\bqdata\?|\bJapgolly\b|[-‑]#{3})'
)
REDUNDANT_VITAL_LABEL_RE = re.compile(r"\bHR\s+heart rate\b", re.I)
UNEXPECTED_CJK_PUNCTUATION_RE = re.compile(r"[、\u3008-\u301f]")
MALFORMED_US_INTERSTATE_RE = re.compile(r"\bI-\s+\d+\b")
UNEXPECTED_SCRIPT_RE = re.compile(
    "[\u0370-\u052f\u0590-\u08ff\u0900-\u109f\u1780-\u18af\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]"
)
US_BRITISH_SPELLING_RE = re.compile(
    r"\b(?:diarrhoea|paediatric|haemorrhage|haematology|haemoglobin|oedema|anaemia|"
    r"foetus|favour|colour|organisation|speciality)\b",
    re.I,
)
US_JURISDICTION_RE = re.compile(
    r"\b(?:United States|Puerto Rico|U\.S\. Virgin Islands|United States Virgin Islands|"
    r"American Samoa|Northern Mariana Islands|District of Columbia|Alabama|Alaska|Arizona|"
    r"Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|"
    r"Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
    r"Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|"
    r"North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|"
    r"South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|"
    r"Wyoming|Guam)\b",
    re.I,
)
GB_JURISDICTION_RE = re.compile(
    r"\b(?:United Kingdom|England|Scotland|Wales|Northern Ireland)\b",
    re.I,
)
ADDED_CREDENTIAL_RE = re.compile(
    r"^\s*(?:,\s*)?(?:RN|RPh|PharmD|PA-C|NP|APRN|DO|MD|MBBS|MBChB|FRCP|FRCGP)\b",
    re.I,
)
ADDED_HONORIFIC_RE = re.compile(r"\b(?:Dr|Professor|Prof)\s+$", re.I)
LABELED_PROFESSION_FIELD_RE = re.compile(
    r"(?mi)^(?:Occupation|Profession)\s*:\s*([^\n]+)$"
)
CAREGIVER_BYLINE_PROFESSION_RE = re.compile(
    r"(?mi)^(?:(?:(?:consulting|treating|referring|discharging)\s+)?"
    r"clinician|seen\s+by)\s*:[^\n]+\n\s*(?:Occupation|Profession)\s*:"
)
NULL_FIELD_VALUE_RE = re.compile(
    r"^(?:not\s+(?:recorded|stated|known|applicable)|unknown|none|n/?a|retired)\.?$",
    re.I,
)
CAREGIVER_ID_AS_NAME_RE = re.compile(
    r"\bDr\s+(?:GMC-TEST-\d{7}|1234567893)\b|"
    r"^(?:(?:responsible|discharging|treating|reviewing|covering)\s+)?"
    r"(?:doctor|clinician|consultant)\s*:\s*(?:GMC-TEST-\d{7}|1234567893)\s*$",
    re.I | re.M,
)
UNNATURAL_PATIENT_AGE_FIELD_RE = re.compile(
    r"(?mi)^Patient\s*:\s*\d{1,3}-(?:day|week|month|year)-old\s+patient\s*$"
)
DETACHED_AGE_RE = re.compile(r"(?m)\n\d{1,3}\s+y/o\s*$", re.I)
PHYSICIAN_NURSE_ROLE_RE = re.compile(
    r"(?:\bDr\s+[^\n,]{1,80},\s*(?:MBBS|MBChB|MD|DO)\b[^\n]{0,24}\b"
    r"(?:registered\s+nurse|staff\s+nurse|RN)\b|"
    r"\bNurse\s+Dr\s+[^\n,]{1,80},\s*(?:MBBS|MBChB|MD|DO)\b)",
    re.I,
)
INCOMPLETE_CLINICIAN_RE = re.compile(
    r"(?mi)^(?:Discharging\s+)?(?:clinician|consultant)\s*:\s*Dr\s*,\s*GMC\b"
)
GMC_NURSING_ROLE_RE = re.compile(
    r"(?mi)^(?:Documented\s+by\s+)?(?:the\s+)?(?:respiratory\s+)?"
    r"(?:nursing\s+team|nurse)\b[^\n]{0,80}\bGMC\b"
)
INCOMPATIBLE_DUAL_CLINICIAN_ROLE_RE = re.compile(
    r"(?mi)\b(?:General Practitioner|GP)\s*\n\s*"
    r"(?:consultant|attending)\s+(?:[a-z-]+\s+){0,3}(?:physician|surgeon|neurologist|"
    r"cardiologist|endocrinologist|pulmonologist|oncologist)\b"
)


class EnglishLLMRenderError(RuntimeError):
    """An LLM output could not satisfy the English generation contract."""


@dataclass(frozen=True)
class EnglishLLMRenderResult:
    doc: dict[str, Any]
    marked_text: str
    usage: dict[str, Any] | None
    response_id: str | None


STYLE_PROFILES = (
    {
        "name": "compact-clinician",
        "shape": "terse clinician-entered headings, field rows, and concise fragments",
        "length": "130-220 words",
        "quirks": "standard abbreviations and clipped syntax mixed with short sentences; no polished essay opening or long explanatory paragraphs",
        "structural_contract": "Use multiple terse field, result, problem, or action lines; at least half of the clinical entries should be fragments or single-clause chart statements.",
    },
    {
        "name": "narrative-scribe",
        "shape": "natural narrative with short assessment and plan sections",
        "length": "180-290 words",
        "quirks": "complete sentences and varied paragraph lengths",
    },
    {
        "name": "structured-record",
        "shape": "EHR field block, record-style headings, short paragraphs, and compact lists",
        "length": "150-250 words",
        "quirks": "dense factual presentation with uneven section lengths and selective fragments",
        "structural_contract": "Include a genuine field block plus compact problem/result/action lines; do not place polished prose paragraphs under every heading.",
    },
    {
        "name": "handover-note",
        "shape": "problem-oriented handover with priorities and safety-netting",
        "length": "120-210 words",
        "quirks": "telegraphic fragments, abbreviations, and action-led lines mixed with complete sentences",
        "structural_contract": "Use a recognisable handover flow such as situation/background/concern/actions or equivalent terse priority lines; it must not read like a referral essay.",
    },
    {
        "name": "specialist-report",
        "shape": "formal specialist report with findings and interpretation",
        "length": "180-300 words",
        "quirks": "technical but readable; values and units retained exactly",
    },
    {
        "name": "longitudinal-summary",
        "shape": "chronological clinical summary with a compact problem list",
        "length": "210-330 words",
        "quirks": "varied chronology; avoid repetitive timeline labels",
    },
    {
        "name": "minimal-ehr",
        "shape": "raw EHR fragments, tight field labels, and short problem-oriented sections",
        "length": "100-180 words",
        "quirks": "selective abbreviations, omitted subjects, clipped syntax, and exact numeric results",
        "structural_contract": "Prefer standalone fields and fragment lines; no prose paragraph may contain more than two complete sentences.",
    },
    {
        "name": "patient-facing-letter",
        "shape": "clinician letter mixing explanation with a technical plan",
        "length": "180-290 words",
        "quirks": "plain language where possible; no generic greeting template",
    },
)


# From production batch 4 onward, documentation-native formats intentionally
# dominate. Repeated names are deterministic weights, not duplicated content.
FUTURE_STYLE_WEIGHTS = {
    "clinic_note": (
        "compact-clinician", "compact-clinician", "compact-clinician",
        "structured-record", "structured-record", "structured-record",
        "minimal-ehr", "minimal-ehr", "minimal-ehr",
        "specialist-report", "specialist-report",
        "narrative-scribe", "patient-facing-letter",
    ),
    "discharge_summary": (
        "compact-clinician", "compact-clinician", "compact-clinician",
        "structured-record", "structured-record", "structured-record", "structured-record",
        "longitudinal-summary", "longitudinal-summary",
        "narrative-scribe", "patient-facing-letter",
    ),
    "emergency_note": (
        "compact-clinician", "compact-clinician", "compact-clinician",
        "structured-record", "structured-record", "structured-record",
        "handover-note", "handover-note", "handover-note",
        "minimal-ehr", "minimal-ehr", "minimal-ehr",
        "narrative-scribe",
    ),
    "referral_letter": (
        "compact-clinician", "compact-clinician", "compact-clinician",
        "specialist-report", "specialist-report", "specialist-report",
        "narrative-scribe", "narrative-scribe", "patient-facing-letter",
    ),
    "laboratory_report": (
        "compact-clinician", "compact-clinician", "compact-clinician",
        "structured-record", "structured-record", "structured-record",
        "specialist-report", "specialist-report",
        "minimal-ehr", "minimal-ehr", "minimal-ehr",
    ),
    "nursing_note": (
        "compact-clinician", "compact-clinician", "compact-clinician",
        "structured-record", "structured-record", "structured-record",
        "handover-note", "handover-note", "handover-note",
        "minimal-ehr", "minimal-ehr", "minimal-ehr",
        "narrative-scribe",
    ),
}

DOCUMENT_GUIDANCE = {
    "clinic_note": (
        "A genuine outpatient or primary-care encounter note. Vary among history/exam/"
        "assessment-plan, problem-oriented, scribe narrative, and terse EHR layouts."
    ),
    "discharge_summary": (
        "A discharge summary. Select a plausible subset of: reason for admission, key "
        "results, hospital course, procedures, medication changes, condition at discharge, "
        "follow-up, and patient instructions. Do not reproduce one fixed heading sequence."
    ),
    "emergency_note": (
        "An emergency assessment with a credible presenting complaint, focused examination, "
        "investigations, decision-making, treatment, and disposition."
    ),
    "referral_letter": (
        "A referral or consultation letter with a specific clinical question, relevant history, "
        "current findings, prior management, and the requested action."
    ),
    "laboratory_report": (
        "A laboratory report or laboratory-facing interpretive note. Include plausible analytes, "
        "units, reference interpretation, specimen context, and a concise clinical comment."
    ),
    "nursing_note": (
        "A nursing progress or handover note with observations, symptoms, mobility or intake, "
        "interventions, response, risks, escalation, and the next shift plan."
    ),
}


COMMON_HARD_NEGATIVES = (
    {
        "category": "time",
        "value": "08:15",
        "instruction": "Use as a clock time. It is not Date.",
    },
    {
        "category": "vital_sign",
        "value": "120/80 mmHg",
        "instruction": "Use as blood pressure. It is not Date or an identifier.",
    },
    {
        "category": "pain_score",
        "value": "5/10",
        "instruction": "Use as a clinical score. It is not Date, Age, or ID.",
    },
    {
        "category": "duration",
        "value": "for 3 days",
        "instruction": (
            "Integrate this naturally into a symptom sentence, for example 'cough for 3 days'. "
            "Do not put it after a Duration field because 'Duration: for 3 days' is unnatural. "
            "It is not Date or Age_Birthdate."
        ),
    },
    {
        "category": "relative_time",
        "value": "in 2 weeks",
        "instruction": (
            "Use as a relative follow-up, review, return, or reassessment interval. "
            "Do not attach it directly to continuing a medication or treatment. It is not Date."
        ),
    },
    {
        "category": "gestational_age",
        "value": "32+5 weeks' gestation",
        "instruction": "Use as gestational duration. It is not patient age or Date.",
    },
    {
        "category": "medication",
        "value": "metoprolol 25 mg twice daily",
        "instruction": "Use as medication and dose. It is not PII.",
    },
    {
        "category": "terminology_code",
        "value": "SNOMED CT 233604007",
        "instruction": "Use as a clinical terminology code. It is not a patient ID.",
    },
    {
        "category": "laboratory_code",
        "value": "LOINC 718-7",
        "instruction": "Use as a laboratory terminology code. It is not a patient ID.",
    },
    {
        "category": "gene_variant",
        "value": "BRCA1 c.68_69delAG",
        "instruction": "Use as a biomedical variant. It is not a name or patient ID.",
    },
    {
        "category": "eponym",
        "value": "Parkinson disease",
        "instruction": "Use as a medical eponym. Parkinson is not a person-name span here.",
    },
    {
        "category": "eponym",
        "value": "Hodgkin lymphoma",
        "instruction": "Use as a diagnosis. Hodgkin is not a person-name span here.",
    },
    {
        "category": "clinical_scale",
        "value": "Glasgow Coma Scale 15",
        "instruction": "Use as a clinical scale. Glasgow is not an address span here.",
    },
    {
        "category": "measurement",
        "value": "sodium 138 mmol/L",
        "instruction": "Use as a result. It is not PII.",
    },
    {
        "category": "impossible_date",
        "value_gb": "31/04/2025",
        "value_us": "04/31/2025",
        "instruction": "Mention as a rejected/invalid entered value. It is not Date.",
    },
    {
        "category": "device_model",
        "value": "Model X3 cardiac monitor",
        "instruction": "Use as a generic device model. It is not an identifier.",
    },
)


LOCALE_HARD_NEGATIVES = {
    "en-GB": (
        {
            "category": "caregiver_role",
            "value": "consultant cardiologist",
            "instruction": "Use as a caregiver role. It is not Profession.",
        },
        {
            "category": "ambiguous_month_word",
            "value": "may improve with hydration",
            "instruction": (
                "Place this exact phrase after a symptom or finding that could improve. "
                "Never make hydration, fluids, or rehydration its grammatical subject. "
                "The word may is not Date."
            ),
        },
    ),
    "en-US": (
        {
            "category": "caregiver_role",
            "value": "attending physician",
            "instruction": "Use as a caregiver role. It is not Profession.",
        },
        {
            "category": "ambiguous_month_word",
            "value": "may improve with hydration",
            "instruction": (
                "Place this exact phrase after a symptom or finding that could improve. "
                "Never make hydration, fluids, or rehydration its grammatical subject. "
                "The word may is not Date."
            ),
        },
    ),
}


MEDICATION_HARD_NEGATIVES = {
    "acetaminophen": "acetaminophen 650 mg every 6 hours as needed",
    "albuterol": "albuterol two puffs every 4 to 6 hours as needed",
    "amlodipine": "amlodipine 5 mg once daily",
    "amoxicillin": "amoxicillin 500 mg three times daily",
    "amoxicillin-clavulanate": "amoxicillin-clavulanate 875/125 mg twice daily",
    "apixaban": "apixaban 5 mg twice daily",
    "aspirin": "aspirin 75 mg once daily",
    "atorvastatin": "atorvastatin 40 mg at night",
    "beclometasone": "beclometasone two puffs twice daily",
    "beclomethasone": "beclomethasone two puffs twice daily",
    "bisoprolol": "bisoprolol 2.5 mg once daily",
    "cefuroxime": "cefuroxime 500 mg twice daily",
    "cetirizine": "cetirizine 10 mg once daily",
    "co-amoxiclav": "co-amoxiclav 625 mg three times daily",
    "colchicine": "colchicine 500 micrograms twice daily",
    "dicloxacillin": "dicloxacillin 500 mg four times daily",
    "doxycycline": "doxycycline 100 mg twice daily",
    "doxylamine-pyridoxine": "doxylamine-pyridoxine two tablets at bedtime",
    "ferrous fumarate": "ferrous fumarate 210 mg once daily",
    "ferrous sulfate": "ferrous sulfate 325 mg once daily",
    "flucloxacillin": "flucloxacillin 500 mg four times daily",
    "folic acid": "folic acid 5 mg once weekly",
    "furosemide": "furosemide 40 mg once daily",
    "hydrocortisone": "hydrocortisone 1% cream twice daily",
    "levothyroxine": "levothyroxine 75 micrograms once daily",
    "metformin": "metformin 500 mg twice daily with food",
    "methotrexate": "methotrexate 15 mg once weekly",
    "naproxen": "naproxen 500 mg twice daily with food",
    "nitrofurantoin": "nitrofurantoin 100 mg modified-release twice daily",
    "nitrofurantoin monohydrate/macrocrystals": "nitrofurantoin monohydrate/macrocrystals 100 mg twice daily",
    "omeprazole": "omeprazole 20 mg once daily",
    "ondansetron": "ondansetron 4 mg every 8 hours as needed",
    "paracetamol": "paracetamol 1 g up to four times daily",
    "prednisolone": "prednisolone 40 mg once daily for 5 days",
    "prednisone": "prednisone 40 mg once daily for 5 days",
    "prochlorperazine": "prochlorperazine 5 mg up to three times daily as needed",
    "ramipril": "ramipril 2.5 mg once daily",
    "sertraline": "sertraline 50 mg once daily",
    "sumatriptan": "sumatriptan 50 mg at onset of migraine",
}


CONDITION_MEASUREMENTS = {
    "acute kidney injury": "creatinine 186 µmol/L",
    "acute pyelonephritis": "C-reactive protein 86 mg/L",
    "atrial fibrillation": "ventricular rate 118 beats/minute",
    "community-acquired pneumonia": "oxygen saturation 91% on room air",
    "decompensated heart failure": "BNP 2,840 pg/mL",
    "diabetic foot ulcer": "HbA1c 74 mmol/mol",
    "first-trimester hyperemesis": "potassium 3.2 mmol/L",
    "gout flare": "serum urate 510 µmol/L",
    "hereditary breast and ovarian cancer risk assessment": "estimated lifetime breast-cancer risk 38%",
    "hodgkin lymphoma": "LDH 340 U/L",
    "hypothyroidism": "TSH 8.2 mIU/L",
    "infectious gastroenteritis": "creatinine 1.1 mg/dL",
    "iron-deficiency anaemia": "haemoglobin 94 g/L",
    "iron-deficiency anemia": "hemoglobin 9.4 g/dL",
    "non-st-elevation myocardial infarction": "high-sensitivity troponin 32 ng/L",
    "type 2 diabetes mellitus": "HbA1c 8.6%",
}


def _specialist_role(condition_name: str, profile_id: str) -> str | None:
    prefix = "consultant" if profile_id == "en-GB" else "attending"
    specialties = (
        (("atrial fibrillation", "heart failure", "myocardial", "hypertension"), "cardiologist"),
        (("pneumonia", "asthma", "pulmonary embolism", "sleep apnoea", "obstructive pulmonary"), "respiratory physician" if profile_id == "en-GB" else "pulmonologist"),
        (("kidney", "pyelonephritis"), "nephrologist"),
        (("hodgkin",), "haematologist" if profile_id == "en-GB" else "hematologist"),
        (("parkinson", "transient ischaemic", "migraine", "radiculopathy"), "neurologist"),
        (("hyperemesis", "pelvic inflammatory"), "obstetrician-gynaecologist" if profile_id == "en-GB" else "obstetrician-gynecologist"),
        (("diabetic", "diabetes", "hypothyroidism"), "endocrinologist"),
    )
    for terms, specialty in specialties:
        if any(term in condition_name for term in terms):
            return f"{prefix} {specialty}"
    return None


def _stable_int(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _case_number(record: dict[str, Any]) -> int:
    match = re.search(r"(\d+)$", str(record.get("case_id", "")))
    return int(match.group(1)) if match else _stable_int(record.get("case_id")) % 1_000_000


def _full_name(person: dict[str, Any]) -> str:
    return " ".join(
        value.strip()
        for value in (str(person.get("given_name", "")), str(person.get("family_name", "")))
        if value.strip()
    )


def _format_date(raw: str, *, profile_id: str, variant: int) -> str:
    value = date.fromisoformat(raw)
    months = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )
    month = months[value.month - 1]
    style = variant % 5
    if style == 0:
        return value.isoformat()
    if profile_id == "en-GB":
        if style == 1:
            return f"{value.day:02d}/{value.month:02d}/{value.year}"
        if style == 2:
            return f"{value.day} {month} {value.year}"
        if style == 3:
            return f"{value.day} {month[:3]} {value.year}"
        return f"{value.day}{_ordinal_suffix(value.day)} {month} {value.year}"
    if style == 1:
        return f"{value.month:02d}/{value.day:02d}/{value.year}"
    if style == 2:
        return f"{month} {value.day}, {value.year}"
    if style == 3:
        return f"{month[:3]} {value.day}, {value.year}"
    return f"{month} {value.day}{_ordinal_suffix(value.day)}, {value.year}"


def _ordinal_suffix(day: int) -> str:
    if 10 < day % 100 < 14:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def pii_slots(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return every exact synthetic value the model is allowed to mark."""

    profile_id = str(record["language"])
    if profile_id not in {"en-GB", "en-US"}:
        raise ValueError(f"English LLM rendering requires en-GB or en-US, got {profile_id!r}")
    index = _case_number(record)
    patient = _full_name(record["patient"])
    caregiver = _full_name(record["caregiver"])
    relative = _full_name(record["relative"])
    patient_initial = f"{record['patient']['given_name'][0]}. {record['patient']['family_name']}"
    patient_given_family_initial = (
        f"{record['patient']['given_name']} {record['patient']['family_name'][0]}."
    )
    caregiver_initial = f"{record['caregiver']['given_name'][0]}. {record['caregiver']['family_name']}"
    relative_initial = f"{record['relative']['given_name'][0]}. {record['relative']['family_name']}"
    supplied_age_display = record.get("age_display")
    if isinstance(supplied_age_display, dict) and {
        "hyphenated", "contextual", "abbreviated"
    } <= set(supplied_age_display):
        age_variants = (
            str(supplied_age_display["hyphenated"]),
            str(supplied_age_display["contextual"]),
            str(supplied_age_display["abbreviated"]),
        )
    elif profile_id == "en-GB":
        age_variants = (
            f"{record['age_years']}-year-old",
            f"aged {record['age_years']}",
            f"{record['age_years']} y/o",
        )
    else:
        age_variants = (
            f"{record['age_years']}-year-old",
            f"age {record['age_years']}",
            f"{record['age_years']} y/o",
        )
    if profile_id == "en-GB":
        credentialed = f"Dr {caregiver}, MBBS"
    else:
        credentialed = f"{caregiver}, MD"
    rows = [
        ("patient.name", patient, "Name:Patient"),
        ("patient.name_upper", patient.upper(), "Name:Patient"),
        ("patient.name_lower", patient.lower(), "Name:Patient"),
        ("patient.given_name", str(record["patient"]["given_name"]), "Name:Patient"),
        ("patient.initial_surname", patient_initial, "Name:Patient"),
        ("patient.given_family_initial", patient_given_family_initial, "Name:Patient"),
        ("caregiver.name", caregiver, "Name:Caregiver"),
        ("caregiver.name_upper", caregiver.upper(), "Name:Caregiver"),
        ("caregiver.name_lower", caregiver.lower(), "Name:Caregiver"),
        ("caregiver.initial_surname", caregiver_initial, "Name:Caregiver"),
        ("caregiver.credentialed_name", credentialed, "Name:Caregiver"),
        ("relative.name", relative, "Name:Other"),
        ("relative.name_upper", relative.upper(), "Name:Other"),
        ("relative.name_lower", relative.lower(), "Name:Other"),
        ("relative.given_name", str(record["relative"]["given_name"]), "Name:Other"),
        ("relative.initial_surname", relative_initial, "Name:Other"),
        ("patient.address", str(record["patient_address"]), "Address_Location:Patient"),
        ("caregiver.locality", str(record["caregiver_locality"]), "Address_Location:Caregiver"),
        ("other.locality", str(record.get("other_locality") or record["caregiver_locality"]), "Address_Location:Other"),
        ("healthcare.organization", str(record["hospital"]), "Organization:Healthcare"),
        ("other.organization", str(record["other_organisation"]), "Organization:Other"),
        ("patient.profession", str(record["profession"]), "Profession"),
        ("patient.birth_date", _format_date(str(record["birth_date"]), profile_id=profile_id, variant=index), "Age_Birthdate"),
        ("patient.age_hyphenated", age_variants[0], "Age_Birthdate"),
        ("patient.age_contextual", age_variants[1], "Age_Birthdate"),
        ("patient.age_abbreviated", age_variants[2], "Age_Birthdate"),
        ("encounter.date", _format_date(str(record["encounter_date"]), profile_id=profile_id, variant=index + 1), "Date"),
        ("followup.date", _format_date(str(record["followup_date"]), profile_id=profile_id, variant=index + 2), "Date"),
        ("patient.phone", str(record["patient_phone"]), "Contactdetails"),
        ("patient.email", str(record["patient_email"]), "Contactdetails"),
        ("patient.mrn", str(record["patient_id"]), "ID:Patient"),
        ("patient.report_id", str(record["report_id"]), "ID:Patient"),
        ("patient.national_id", str(record["national_id"]), "ID:Patient"),
        ("caregiver.professional_id", str(record["caregiver_id"]), "ID:Caregiver"),
    ]
    result: list[dict[str, str]] = []
    for slot, value, label in rows:
        if not value:
            continue
        target = {"slot": slot, "value": value, "label": label}
        target["semantic_type"] = (
            "healthcare_organization.acute_hospital"
            if slot == "healthcare.organization"
            else semantic_type_for_target(
                target,
                document_family=str(record.get("document_type") or ""),
            )
        )
        result.append(target)
    if any(row["label"] == "Anonymize_Other" for row in result):
        raise AssertionError("English LLM slot construction must not expose Anonymize_Other")
    return result


LABEL_SLOT_PREFERENCES = {
    "Address_Location:Caregiver": ("caregiver.locality",),
    "Address_Location:Other": ("other.locality",),
    "Address_Location:Patient": ("patient.address",),
    "Age_Birthdate": ("patient.birth_date", "patient.age_hyphenated", "patient.age_contextual", "patient.age_abbreviated"),
    "Contactdetails": ("patient.phone", "patient.email"),
    "Date": ("encounter.date", "followup.date"),
    "ID:Caregiver": ("caregiver.professional_id",),
    "ID:Patient": ("patient.mrn", "patient.report_id", "patient.national_id"),
    "Name:Caregiver": (
        "caregiver.name", "caregiver.name_upper", "caregiver.name_lower",
        "caregiver.initial_surname", "caregiver.credentialed_name",
    ),
    "Name:Other": (
        "relative.name", "relative.name_upper", "relative.name_lower",
        "relative.given_name", "relative.initial_surname",
    ),
    "Name:Patient": (
        "patient.name", "patient.name_upper", "patient.name_lower", "patient.given_name",
        "patient.initial_surname", "patient.given_family_initial",
    ),
    "Organization:Healthcare": ("healthcare.organization",),
    "Organization:Other": ("other.organization",),
    "Profession": ("patient.profession",),
}


# Label coverage must respect the semantics of the document family.  The order
# provides a deterministic first-cycle coverage design; later cycles rotate
# within a larger set so the same family does not always receive the same PII.
DOCUMENT_LABEL_PREFERENCES = {
    "clinic_note": (
        "Profession", "Organization:Other", "Address_Location:Patient", "Contactdetails",
        "Age_Birthdate", "Date", "ID:Patient", "Name:Caregiver",
        "Organization:Healthcare",
    ),
    "discharge_summary": (
        "Date", "Name:Other", "Address_Location:Other", "Organization:Healthcare",
        "ID:Patient", "Name:Caregiver", "ID:Caregiver", "Address_Location:Patient",
        "Contactdetails", "Age_Birthdate",
    ),
    "emergency_note": (
        "Age_Birthdate", "Address_Location:Patient", "Contactdetails", "ID:Patient",
        "Name:Other", "Organization:Healthcare", "Date", "Name:Caregiver",
    ),
    "referral_letter": (
        "Address_Location:Caregiver", "ID:Caregiver", "Name:Caregiver",
        "Organization:Healthcare", "ID:Patient", "Date", "Address_Location:Patient",
        "Profession", "Organization:Other", "Name:Other",
    ),
    "laboratory_report": (
        "ID:Patient", "Date", "Organization:Healthcare", "Name:Caregiver",
        "ID:Caregiver", "Age_Birthdate", "Address_Location:Patient", "Contactdetails",
    ),
    "nursing_note": (
        "Age_Birthdate", "Date", "Name:Caregiver", "Name:Other",
        "Organization:Healthcare", "Contactdetails", "Address_Location:Patient",
        "ID:Patient", "ID:Caregiver",
    ),
}


def coverage_targets(record: dict[str, Any]) -> list[dict[str, str]]:
    """Select a document-appropriate subset with aggregate 14-label coverage."""

    slots = {row["slot"]: row for row in pii_slots(record)}
    index = _case_number(record)
    selected_labels = ["Name:Patient"]
    is_pediatric = int(record.get("age_years", 18)) < 18
    if is_pediatric:
        # Every paediatric document must contribute a direct age/DOB training
        # signal, rather than merely having a young age hidden in its case.
        selected_labels.append("Age_Birthdate")
        if str(record["document_type"]) != "laboratory_report":
            selected_labels.append("Name:Other")
    if str(record["document_type"]) in {"referral_letter", "laboratory_report"}:
        selected_labels.append("Name:Caregiver")
    if str(record["document_type"]) == "laboratory_report":
        # A realistic laboratory report needs both an explicit specimen/report
        # date and a patient-associated report/MRN identifier. Supplying these
        # slots prevents the author from inventing unannotated header values.
        selected_labels.extend(("Date", "ID:Patient"))
    labels = DOCUMENT_LABEL_PREFERENCES[str(record["document_type"])]
    cycle = max(0, (index - 1) // len(DOCUMENT_GUIDANCE))
    start = (cycle * 3) % len(labels)
    for offset in range(len(labels) * 2):
        if len(selected_labels) >= 5:
            break
        label = labels[(start + offset) % len(labels)]
        if label == "Profession" and int(record.get("age_years", 18)) < 18:
            continue
        if label not in selected_labels:
            selected_labels.append(label)
    targets: list[dict[str, str]] = []
    for label in selected_labels:
        preferences = LABEL_SLOT_PREFERENCES[label]
        slot_name = preferences[_stable_int(record.get("case_id"), label) % len(preferences)]
        targets.append(slots[slot_name])
    return targets


def hard_negative_targets(
    record: dict[str, Any], *, category_override: str | None = None
) -> list[dict[str, str]]:
    """Choose a small, clinically endogenous false-positive challenge set.

    A hard negative is useful only when it belongs in the document.  The
    production corpus therefore uses one condition- and document-aware target
    per ordinary case, with a small number of semantically forced challenges
    for eponyms, terminology codes, gene variants, and invalid data-entry dates.
    """

    profile_id = str(record["language"])
    document_type = str(record["document_type"])
    index = _case_number(record)
    condition_name = str(record.get("condition", {}).get("name", "")).lower()
    condition = record.get("condition", {})
    is_pediatric = int(record.get("age_years", 18)) < 18
    clinical_age_group = str(
        record.get("clinical_age_group", record.get("age_group", "adult"))
    )
    symptoms = " ".join(str(value).lower() for value in condition.get("symptoms", []))
    medications = [str(value).lower() for value in condition.get("medications", [])]
    followup_days = (
        date.fromisoformat(str(record["followup_date"]))
        - date.fromisoformat(str(record["encounter_date"]))
    ).days
    relative_followup = (
        f"in {followup_days} days"
        if followup_days < 21
        else f"in about {round(followup_days / 7)} weeks"
    )
    effective_override = category_override or record.get("production", {}).get(
        "hard_negative_category_override"
    )

    def target(category: str, value: str, instruction: str) -> dict[str, str]:
        return {"category": category, "value": value, "instruction": instruction}

    forced: list[dict[str, str]] = []
    if "parkinson" in condition_name:
        forced.append(target("eponym", "Parkinson disease", "Use as the actual diagnosis."))
    if "hodgkin" in condition_name:
        forced.append(target("eponym", "Hodgkin lymphoma", "Use as the actual diagnosis."))
    if "hereditary breast and ovarian" in condition_name:
        forced.append(
            target(
                "gene_variant",
                "BRCA1 c.68_69delAG",
                "Use as the variant under genetics review, not as a person or patient ID.",
            )
        )
    if "community-acquired pneumonia" in condition_name:
        forced.append(
            target(
                "terminology_code",
                "SNOMED CT 233604007",
                "Use as the code for community-acquired pneumonia in a realistic diagnosis/coding context.",
            )
        )
    if "first-trimester hyperemesis" in condition_name:
        forced.append(
            target(
                "gestational_age",
                "9+4 weeks' gestation",
                "Use as the current gestational duration; it is not the patient's age.",
            )
        )
    if document_type == "laboratory_report" and index % 60 == 5:
        impossible = "31/04/2025" if profile_id == "en-GB" else "04/31/2025"
        forced.append(
            target(
                "impossible_date",
                impossible,
                "Use only as an invalid specimen-date entry rejected during data-quality checking.",
            )
        )
    elif document_type == "laboratory_report" and "iron-deficiency" in condition_name:
        forced.append(
            target(
                "laboratory_code",
                "LOINC 718-7",
                "Use as the haemoglobin/hemoglobin test code in realistic result metadata.",
            )
        )
    if forced:
        matching_forced = [
            candidate
            for candidate in forced
            if candidate["category"] == effective_override
        ]
        if matching_forced:
            return matching_forced[:1]
        if effective_override is None:
            return forced[:1]

    candidates: list[dict[str, str]] = [
        target(
            "duration",
            "for 3 days",
            "Integrate this naturally into a symptom sentence, such as 'cough for 3 days'; "
            "never write the redundant field form 'Duration: for 3 days'.",
        ),
        target(
            "relative_time",
            relative_followup,
            "Use as a relative follow-up, review, return, or reassessment interval consistent "
            "with the structured follow-up date; never attach it directly to continuing a "
            "medication or treatment.",
        ),
    ]
    if document_type in {"emergency_note", "nursing_note", "discharge_summary"}:
        vital_value = (
            {
                "neonate": "heart rate 142 beats/minute",
                "infant": "heart rate 128 beats/minute",
                "early_childhood": "heart rate 108 beats/minute",
                "school_age": "heart rate 92 beats/minute",
                "adolescent": "blood pressure 108/66 mmHg",
            }.get(clinical_age_group, "heart rate 100 beats/minute")
            if is_pediatric
            else "120/80 mmHg"
        )
        candidates.extend(
            [
                target("time", "08:15", "Use as a clock time for an event or observation."),
                target("vital_sign", vital_value, "Use as a current vital-sign reading."),
            ]
        )
    if document_type == "emergency_note":
        candidates.append(
            target(
                "clinical_scale",
                "AVPU response: alert"
                if clinical_age_group in {"neonate", "infant", "early_childhood"}
                else "Glasgow Coma Scale 15",
                "Use as a neurological assessment.",
            )
        )
    if (
        clinical_age_group not in {"neonate", "infant", "early_childhood"}
        and any(
            term in f"{condition_name} {symptoms}"
            for term in ("pain", "headache", "tender", "cramp")
        )
    ):
        candidates.append(target("pain_score", "5/10", "Use as a current pain score."))

    if is_pediatric:
        age_group = clinical_age_group
        weight = {
            "neonate": "3.4 kg",
            "infant": "6.4 kg",
            "early_childhood": "14.2 kg",
            "school_age": "28.6 kg",
            "adolescent": "54.0 kg",
        }.get(age_group, "42.0 kg")
        candidates.append(
            target(
                "measurement",
                weight,
                "Use as a current measured weight; it is not an age or identifier.",
            )
        )
        if age_group in {"neonate", "infant"}:
            candidates.append(
                target(
                    "gestational_age",
                    "born at 39+2 weeks' gestation",
                    "Use as birth history; gestational duration is not the infant's current age.",
                )
            )

    # Avoid adult fixed-dose hard negatives in paediatric cases.  The clinical
    # author has no supplied weight on which to ground a weight-based dose.
    for medication in (() if is_pediatric else medications):
        # Match the most specific medicine name first.  Without this ordering,
        # ``amoxicillin-clavulanate`` is incorrectly reduced to standalone
        # amoxicillin and can create unsafe-looking duplicate therapy.
        for key in sorted(MEDICATION_HARD_NEGATIVES, key=len, reverse=True):
            dosed_value = MEDICATION_HARD_NEGATIVES[key]
            if profile_id == "en-US" and key == "aspirin":
                dosed_value = "aspirin 81 mg once daily"
            if key in medication:
                candidates.append(
                    target(
                        "medication",
                        dosed_value,
                        "Copy this complete exact value contiguously on a natural Medication, "
                        "Treatment, Administered, or Discharge medicines line. It is already "
                        "present in the case and is mandatory even in a compact or minimal note; "
                        "do not abbreviate or omit its dose or frequency.",
                    )
                )
                break
        else:
            continue
        break

    measurement = CONDITION_MEASUREMENTS.get(condition_name)
    if profile_id == "en-US":
        measurement = {
            "acute kidney injury": "creatinine 2.1 mg/dL",
            "diabetic foot ulcer": "HbA1c 8.7%",
            "gout flare": "serum urate 8.6 mg/dL",
        }.get(condition_name, measurement)
    elif condition_name == "infectious gastroenteritis":
        measurement = "creatinine 97 µmol/L"
    if document_type == "laboratory_report":
        measurement = {
            "acute kidney injury": "2.1 mg/dL" if profile_id == "en-US" else "186 µmol/L",
            "diabetic foot ulcer": "8.7%" if profile_id == "en-US" else "74 mmol/mol",
            "gout flare": "8.6 mg/dL" if profile_id == "en-US" else "510 µmol/L",
            "hypothyroidism": "8.2 mIU/L",
            "iron-deficiency anaemia": "94 g/L",
            "iron-deficiency anemia": "9.4 g/dL",
            "type 2 diabetes mellitus": "8.6%" if profile_id == "en-US" else "70 mmol/mol",
        }.get(condition_name, measurement)
    if measurement and document_type in {
        "laboratory_report", "emergency_note", "discharge_summary", "referral_letter"
    }:
        candidates.append(
            target("measurement", measurement, "Use as a clinically relevant result for this condition.")
        )
    role = _specialist_role(condition_name, profile_id)
    if role and document_type in {"referral_letter", "discharge_summary", "clinic_note"}:
        candidates.append(
            target(
                "caregiver_role",
                role,
                "Integrate as a naturally involved or requested specialist role. In a referral, "
                "do not append it as a second, incompatible title to the GP/referring signatory; "
                "mention the distinct specialist in the body or address instead.",
            )
        )

    cardiac_context = any(
        term in condition_name
        for term in ("atrial fibrillation", "heart failure", "myocardial infarction")
    )
    if cardiac_context and document_type in {"emergency_note", "nursing_note", "discharge_summary"}:
        candidates.append(
            target("device_model", "Model X3 cardiac monitor", "Use as the monitor used during this encounter.")
        )
    if any(term in condition_name for term in ("dehydration", "kidney injury", "gastroenteritis")):
        candidates.append(
            target(
                "ambiguous_month_word",
                "may improve with hydration",
                "Use this exact phrase after a symptom or finding that could improve, for example "
                "'nausea may improve with hydration'. Never make hydration, fluids, oral "
                "rehydration, or rehydration solution the grammatical subject, because that "
                "would produce a redundant '<hydration> may improve with hydration' sentence. "
                "The word 'may' is ordinary prose.",
            )
        )

    if effective_override is not None:
        matching = [
            candidate
            for candidate in candidates
            if candidate["category"] == effective_override
        ]
        if not matching:
            raise ValueError(
                f"hard-negative category {effective_override!r} is not natural for "
                f"{record.get('case_id')}"
            )
        index = _stable_int(
            record.get("case_id"), profile_id, effective_override, "hard-negative-override"
        )
        return [matching[index % len(matching)]]
    return [candidates[_stable_int(record.get("case_id"), profile_id, "hard-negative") % len(candidates)]]


def style_profile(record: dict[str, Any]) -> dict[str, str]:
    allowed_names = {
        "clinic_note": (
            "compact-clinician", "narrative-scribe", "structured-record",
            "specialist-report", "minimal-ehr", "patient-facing-letter",
        ),
        "discharge_summary": (
            "compact-clinician", "narrative-scribe", "structured-record",
            "longitudinal-summary", "patient-facing-letter",
        ),
        "emergency_note": (
            "compact-clinician", "narrative-scribe", "structured-record",
            "handover-note", "minimal-ehr",
        ),
        "referral_letter": (
            "compact-clinician", "narrative-scribe", "specialist-report",
            "patient-facing-letter",
        ),
        "laboratory_report": (
            "compact-clinician", "structured-record", "specialist-report", "minimal-ehr",
        ),
        "nursing_note": (
            "compact-clinician", "narrative-scribe", "structured-record",
            "handover-note", "minimal-ehr",
        ),
    }[str(record["document_type"])]
    production_batch = int(record.get("production", {}).get("batch_index", -1))
    if production_batch >= 3:
        allowed_names = FUTURE_STYLE_WEIGHTS[str(record["document_type"])]
    by_name = {profile["name"]: profile for profile in STYLE_PROFILES}
    index = _stable_int(record.get("case_id"), record.get("language"), record.get("document_type"))
    return dict(by_name[allowed_names[index % len(allowed_names)]])


def _case_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: record[key]
        for key in (
            "case_id",
            "language",
            "document_type",
            "department",
            "condition",
            "relative_role",
        )
        if key in record
    }
    if int(record.get("age_years", 18)) < 18:
        payload["clinical_age_group"] = str(
            record.get("clinical_age_group", record.get("age_group", "pediatric"))
        )
        payload["paediatric_case"] = True
    elif (
        int(record.get("age_years", 18)) < 31
        and payload.get("relative_role") == "adult child"
    ):
        # Older saved production cases can predate the age/relationship rule.
        # Keep their fixed PII values but give the author a plausible relationship.
        payload["relative_role"] = "sibling"
    return payload


def _prompt_pii_targets(record: dict[str, Any]) -> list[dict[str, str]]:
    """Expose typed placeholders, never literal PII, to the language model."""

    usage_hints = {
        "Address_Location:Caregiver": "Use as the clinician, practice, service, or correspondence locality.",
        "Address_Location:Other": "Use as a relative's or emergency contact's locality.",
        "Address_Location:Patient": "Use as the patient's home or mailing address.",
        "Age_Birthdate": "Use naturally as the patient's age or date of birth, according to the slot.",
        "Contactdetails": (
            "Use explicitly as the patient's contact detail. Label it 'Patient contact', "
            "'Patient phone', or 'Patient email' where role ambiguity is possible; never "
            "place it after a clinician signature or validation line where it could read "
            "as the clinician's contact detail."
        ),
        "Date": "Use as the encounter, collection, discharge, or follow-up date, according to the slot.",
        "ID:Caregiver": "Use as the named clinician's professional identifier.",
        "ID:Patient": "Use as the patient's MRN or report/accession identifier, according to the slot.",
        "Name:Caregiver": "Use as a treating, referring, validating, or signing clinician.",
        "Name:Other": "Use as a named relative, parent, carer, or emergency contact.",
        "Name:Patient": "Use as the patient's name.",
        "Organization:Healthcare": "Use as a treating, referring, or reporting healthcare organisation.",
        "Organization:Other": "Use as the patient's employer, school, council, community service, or other non-healthcare affiliation.",
        "Profession": "Use only as the patient's occupation. Prefer a field such as 'Occupation: <placeholder>' and do not place an article such as a/an immediately before the placeholder because source capitalisation must be preserved.",
    }
    targets: list[dict[str, str]] = []
    for target_index, row in enumerate(coverage_targets(record), start=1):
        hint = usage_hints[row["label"]]
        if row["slot"] == "other.organization" and int(record.get("age_years", 18)) < 18:
            organisation = str(record.get("other_organisation", ""))
            organisation_lower = organisation.casefold()
            patient_age = int(record.get("age_years", 18))
            age_incompatible_school = (
                patient_age < 16
                and any(
                    token in organisation_lower
                    for token in ("college", "university", "further education")
                )
            ) or (
                patient_age < 11
                and any(
                    token in organisation_lower
                    for token in ("secondary school", "high school")
                )
            )
            if age_incompatible_school:
                hint = (
                    "This education organisation is not age-appropriate as the patient's own school. "
                    "Use it only as the supplied parent, guardian, aunt, uncle, or other adult relative's "
                    "workplace or affiliation. Render this concisely as a natural field or phrase such as "
                    "'Mother's employer: <placeholder>' or 'the aunt works for <placeholder>'. Never label "
                    "it School, say the patient attends it, share the patient's school plan with it, or "
                    "present it as a family support service. Do not add an explanatory disclaimer saying "
                    "that it is not the patient's school or is unrelated to the patient's care."
                )
            elif any(
                token in organisation_lower
                for token in ("school", "college", "academy", "nursery")
            ):
                hint = (
                    "Use as the child's genuine school or education affiliation. "
                    "Do not call it an employer or a healthcare organisation."
                )
            elif any(
                token in organisation_lower
                for token in ("housing authority", "transit authority", "town council", "senior center")
            ):
                hint = (
                    "Use only as a family support service, housing/transport/community contact, "
                    "correspondence recipient, or the parent/guardian's affiliation. Never call it "
                    "the child's school, daycare, education provider, employer, or a place the child attends."
                )
            elif "sports club" in organisation_lower and int(record.get("age_years", 18)) < 5:
                hint = (
                    "Use only as a family or community support affiliation. The patient is too young "
                    "for this to be presented as their workplace, school/daycare, or an independently attended club."
                )
            else:
                hint = (
                    "Use only as a family support, community programme, correspondence recipient, "
                    "or parent/guardian affiliation. Because the organisation name has no school marker, "
                    "never call it the child's school, daycare, or education provider, and never present "
                    "the child as employed."
                )
        if row["slot"] == "patient.birth_date":
            hint = "Use as the patient's date of birth with a DOB/date-of-birth descriptor."
        elif row["slot"] == "patient.age_hyphenated":
            hint = (
                "Use adjectivally in a natural phrase such as '<placeholder> patient' or "
                "after 'is a'. Never write 'Age: <placeholder>', 'Age <placeholder>', "
                "'aged <placeholder>', or 'Age recorded as <placeholder>'. When a singular "
                "noun follows, include the grammatically correct determiner: for example, "
                "'an <18-month-old> child' or 'a <3-year-old> patient'; never write "
                "'in <placeholder> child'. In a structured field block, keep the actual name in "
                "the Patient field and use a separate natural line such as "
                "'Clinical context: a <placeholder> patient'. Never use the age phrase itself as "
                "the value of a Patient field."
            )
        elif row["slot"] == "patient.age_contextual":
            example = (
                "the patient, <placeholder>, was reviewed"
                if record["language"] == "en-GB"
                else "the patient, <placeholder>, was reviewed"
            )
            hint = (
                f"Use the supplied contextual age directly in a clause such as '{example}'. "
                "Do not put 'Age:', 'aged', or another age word immediately before it because the placeholder already contains that wording."
            )
        elif row["slot"] == "patient.age_abbreviated":
            hint = (
                "Use the supplied short clinical age form naturally. It may be an abbreviation "
                "or a neonatal form such as 'day N of life'. "
                "Do not put 'Age:', 'age', or 'aged' immediately before it and do not label it DOB."
            )
        elif row["slot"] == "encounter.date":
            hint = "Use as the encounter, admission, collection, or report date; it may be the document date."
        elif row["slot"] == "followup.date":
            hint = "Use explicitly as a future follow-up appointment or review date; never use it as the current document, admission, collection, or discharge date."
        elif row["slot"] == "patient.mrn":
            hint = "Use only as the patient's MRN or patient identifier; never describe it as a report or accession identifier."
        elif row["slot"] == "patient.report_id":
            hint = (
                "Use only as a report, result, case, or accession identifier; never describe it as an MRN or patient identifier. "
                "Prefer a normal field such as 'Report/accession ID:' or an ordinary clinical reference. Never explain that it is synthetic, annotated, non-PII, or included for dataset purposes."
            )
        elif row["slot"] == "patient.national_id":
            field_label = str(record.get("national_id_label") or "National identifier")
            hint = (
                f"Use only as the patient's {field_label}; never describe it as an MRN, "
                "hospital patient number, report ID, or accession number. Copy the supplied "
                "value exactly and use it in a natural administrative field. Never explain "
                "that it is synthetic or included for dataset purposes."
            )
        elif row["slot"] == "caregiver.professional_id" and record["language"] == "en-GB":
            hint = (
                "Use as the GMC-style professional identifier of the supplied named doctor/physician. "
                "Do not describe that clinician as a nurse, midwife, pharmacist, or other non-doctor profession. "
                "Use a natural field such as 'GMC number:' or 'Professional identifier:'. "
                "Copy the supplied value exactly: it may be numeric-only or may already contain a designator. "
                "Never prepend GMC, GMC-TEST, NPI, LIC, or another identifier prefix to the placeholder."
            )
        elif row["slot"] in {
            "patient.name_upper",
            "patient.name_lower",
            "caregiver.name_upper",
            "caregiver.name_lower",
            "relative.name_upper",
            "relative.name_lower",
        }:
            hint += " Preserve the supplied capitalization exactly; this is deliberate formatting diversity."
        elif row["slot"] in {
            "patient.given_name",
            "relative.given_name",
        }:
            hint += " Use this first-name-only form naturally after the person has been introduced or in a suitable field."
        if row["slot"].startswith("relative."):
            patient_age = int(record.get("age_years", 18))
            if patient_age >= 70:
                hint += " Use as an adult child, sibling, spouse/partner, friend, or emergency contact; do not make this person the patient's parent."
            elif patient_age < 18:
                role = str(record.get("relative_role", "parent or guardian"))
                hint += (
                    f" Use as the patient's {role}; keep that relationship consistent "
                    "throughout the document."
                )
            elif patient_age < 31:
                hint += (
                    " Use as a sibling, spouse/partner, friend, or emergency contact. "
                    "Do not describe this person as the patient's adult child."
                )
        elif row["slot"] in {
            "patient.initial_surname",
            "patient.given_family_initial",
            "caregiver.initial_surname",
            "relative.initial_surname",
        }:
            hint += " Preserve the supplied initial punctuation and name format exactly."
        if record["document_type"] == "laboratory_report" and row["slot"] in {
            "caregiver.name",
            "caregiver.name_upper",
            "caregiver.name_lower",
            "caregiver.initial_surname",
            "caregiver.credentialed_name",
        }:
            hint = (
                "Use as the laboratory validating, reporting, or signing clinician. "
                "Do not also use this person as the ordering clinician; keep any ordering clinician unnamed. "
                "Preserve the supplied name spelling, capitalization, initials, and credentials exactly."
            )
        if (
            record["document_type"] == "nursing_note"
            and row["slot"] == "caregiver.credentialed_name"
        ):
            hint = (
                "Use as the named physician/doctor who medically reviewed the patient or was contacted by nursing staff. "
                "The credentials belong to a physician: never call this person a nurse, RN, staff nurse, documenting nurse, or nursing author. "
                "Preserve the supplied name and physician credentials exactly."
            )
        if record["document_type"] == "referral_letter":
            if row["slot"] in {
                "caregiver.name",
                "caregiver.name_upper",
                "caregiver.name_lower",
                "caregiver.initial_surname",
                "caregiver.credentialed_name",
            }:
                hint = (
                    "Use exactly once as the referring or signing clinician who authored the letter, never as its recipient. "
                    "Preserve the supplied name spelling, capitalization, initials, and credentials exactly."
                )
            elif row["slot"] == "caregiver.professional_id":
                hint = "Use as the professional identifier of the referring/signing clinician, never the recipient."
            elif row["slot"] == "caregiver.locality":
                hint = "Use as the referring clinician or service locality, never the recipient's locality."
            elif row["slot"] == "healthcare.organization":
                hint = "Use as the referring healthcare organisation; address the letter to an unnamed appropriate specialist or service."
        if row["slot"] == "caregiver.credentialed_name":
            hint += (
                " The placeholder is the complete clinician-name span and already includes its "
                "MD/MBBS credential. Do not write MD, MBBS, DO, or any other credential before "
                "or after the placeholder."
            )
        targets.append({
            "slot": row["slot"],
            "label": row["label"],
            "semantic_type": row["semantic_type"],
            # Keep author-facing tokens deliberately short. The label and slot
            # remain typed in this target object and are expanded locally;
            # long delimiter-rich tokens proved unnecessarily error-prone.
            "placeholder": f"[[PII_{target_index:02d}]]",
            "usage_hint": hint,
        })
    return targets


def build_english_llm_prompt(
    record: dict[str, Any], *, retry_feedback: str | None = None
) -> str:
    profile_id = str(record["language"])
    locale_name = "British English" if profile_id == "en-GB" else "American English"
    setting = (
        "the NHS and the four UK nations; do not use Crown Dependency conventions"
        if profile_id == "en-GB"
        else "the United States, including state and territorial clinical settings"
    )
    prompt: dict[str, Any] = {
        "task": (
            f"Independently author one realistic {locale_name} clinical document for {setting}. "
            "This must be a fresh composition, not a filled template or a paraphrase of another note."
        ),
        "document_guidance": DOCUMENT_GUIDANCE[str(record["document_type"])],
        "style_profile": style_profile(record),
        "authorship_nonce": hashlib.sha256(
            f"english-independent-authorship|{profile_id}|{record.get('case_id')}".encode()
        ).hexdigest()[:20],
        "output_contract": [
            "Return only the clinical document. Do not return JSON, Markdown fences, commentary, or a preface.",
            "Return plain clinical text. Do not use Markdown emphasis markers such as **bold** or Markdown heading markers such as # Heading.",
            "Realistic EHR-style field labels such as Patient:, Occupation:, Employer:, MRN:, Date:, and Contact: are welcome formatting diversity. Integrate them cleanly, but do not treat a completed field label as an unresolved template artifact.",
            "Write in the native shape of the requested document, as a clinician, nurse, or laboratory professional would enter it during routine workflow. Most notes should be terse records rather than polished educational prose: fragments, selective abbreviations, compact field blocks, uneven section lengths, and mixed line layouts are welcome when the style profile calls for them.",
            "Do not make every thought a complete sentence, do not explain routine clinical conventions to another clinician, and do not force every possible section into the document. Referral and patient-facing letters may remain natural prose when their selected style calls for it.",
            "Treat style_profile.structural_contract as mandatory. A heading followed by polished multi-sentence paragraphs does not satisfy a compact, handover, structured-record, or minimal-EHR style. Preserve natural chart shorthand and line structure rather than turning the note into an essay.",
            "Keep only clinical detail needed to make the PII and hard-negative contexts natural. Avoid exhaustive negative review-of-systems lists, generic educational explanation, and repeated safety-net boilerplate; this corpus is for PII detection, not a showcase clinical vignette.",
            "Use the structured case as clinical grounding, but write a genuinely new note with varied organisation and phrasing.",
            "You may create additional plausible synthetic clinical findings, normal/abnormal results, reasoning, and treatment detail. Do not contradict the supplied condition, symptoms, medications, age, or dates.",
            "Never invent a person, place, address, organisation, phone number, email address, URL, date, age, or identifier. The local pipeline will insert PII after authorship.",
            "Do not state or infer a numeric patient age anywhere unless required_pii_targets contains an age slot and you insert that slot's placeholder. Age-looking numbers outside a supplied placeholder are forbidden gold false negatives.",
            "A hyphenated age placeholder such as '10-year-old' is an adjective, not a standalone Age-field value. Write '<placeholder> patient/child' or 'is a <placeholder> patient'; never write 'Age: <placeholder>'.",
            "When case.paediatric_case is true, the stated clinical_age_group is a hard clinical constraint. Use an age-appropriate history, examination, differential, setting, safeguarding tone, consent/assent, parent or guardian involvement, and treatment plan. Never turn an adult note into a child note merely by changing the age. Do not invent an exact weight (apart from a supplied hard-negative weight), developmental milestone, weight-based dose, school year, or another numeric age.",
            "For an infant or young child, distinguish the patient's history from the parent or guardian's account. For an adolescent, preserve appropriate confidentiality and assent while involving a guardian only where clinically natural.",
            "Insert each required_pii_targets.placeholder token verbatim at least once in a clinically plausible place. A placeholder is the complete PII field; do not put a value inside it and do not add a closing marker.",
            "A valid placeholder looks exactly like [[PII_01]]. Preserve the two-digit number, both opening brackets, and both closing brackets. The target object's label and slot supply its type; never rewrite them into the placeholder.",
            "Use every item in required_pii_targets. A placeholder may recur when the same PII naturally belongs in a header and signature/footer, but every recurrence must keep the same semantic role. Do not repeat PII merely to increase label density. No other PII values or placeholders are available or permitted.",
            "The only permitted labels are the 14 labels in allowed_labels. Anonymize_Other is forbidden and must never appear.",
            "Use every exact hard_negative_targets.value as ordinary unmarked clinical text. These are deliberately difficult non-PII examples; never wrap any part of them in a PII marker.",
            "Hard-negative values are byte-exact required strings, including medication dose/frequency wording such as 'once daily' or 'as needed'. Copy each full value contiguously at least once; do not abbreviate, paraphrase, reorder, or split it across lines.",
            "A medication hard negative remains mandatory in compact and minimal-EHR styles. Put its complete exact value on a natural Medication, Treatment, Administered, or Discharge medicines line; never shorten the dose/frequency to save words.",
            "Integrate each hard negative naturally into the history, examination, results, medication list, device use, data-quality note, or plan. Never dump them into a separate 'non-PII', 'unrelated terms', 'for completeness', or test-example paragraph, and never explain that a term is not an identifier.",
            "Clock times, vital signs, measurements, scores, doses, durations, gestational durations, relative intervals, medical eponyms, clinical roles, biomedical codes, gene variants, generic device models, and impossible dates are not PII unless explicitly supplied as a PII slot.",
            "Do not add a derived score or calculated value unless it is useful to the note and exactly agrees with the displayed inputs and scoring convention. This includes NEWS2, GCS components, anion gap, corrected values, percentages, and renal indices.",
            "If a Parkinson disease case already includes co-beneldopa, describe an established diagnosis previously assessed by neurology or another appropriate specialist, with prescribing continued in primary care, and seek medication/multidisciplinary review. Do not say the diagnosis and dopaminergic initiation occurred solely in primary care, and do not call it an untreated suspected diagnosis awaiting first specialist confirmation.",
            "Only the patient or a relative's occupation from patient.profession is Profession. Caregiver roles, specialties, and credentials are not Profession. If a clinician byline is nearby, explicitly write 'Patient occupation:' so the profession cannot appear to belong to the clinician.",
            "If other.organization is required, present it as a genuine non-healthcare affiliation such as the patient's employer, school, council, community service, or correspondence recipient. Never describe it as a treating, referring, requesting, or discharging healthcare organisation.",
            "In a referral letter, the supplied caregiver is the referring/signing author, not the addressee. Repetition in the signature is allowed, but do not make one person both sender and recipient.",
            "In a laboratory report, the supplied caregiver is the validating/reporting/signing laboratory clinician, not also the ordering clinician. Keep any ordering clinician unnamed.",
            "In a nursing note, the supplied caregiver may be the documenting nurse or another treating clinician. If used as the author/nurse, do not also say that this same person was notified, contacted, or escalated to; name the author only once or refer to an unnamed covering provider.",
            "More generally, never write that a named author discussed, escalated, handed over, or notified the case to themself. Use the supplied caregiver either as author or as a consulted clinician in that interaction, not both.",
            "For an NSTEMI discharge summary, include the cardiology-determined antiplatelet plan. Unless a contraindication or explicit cardiology rationale is stated, medically managed NSTEMI should not be discharged on aspirin alone without a P2Y12 inhibitor plan.",
            "A report/accession identifier must be described in the clinical prose only as a report, result, case, or accession identifier, never as an MRN or patient ID. The suite intentionally annotates both patient-record MRNs and patient-associated report/accession identifiers with the interoperable ID:Patient label. An MRN must be described as an MRN or patient identifier, never as a report/accession identifier.",
            "In en-GB, a GMC-TEST professional identifier belongs to a doctor/physician and must not be assigned to a nurse, midwife, pharmacist, or other profession with a different regulator.",
            "A terminology, laboratory, gene, or device code is not a patient identifier merely because it contains capitals, digits, punctuation, or a person's historical surname.",
            "Keep descriptors such as DOB, MRN, telephone, address, from, to, clinician, and signed by outside the placeholder.",
            "Do not reproduce or imitate text from MIMIC or any source note. Abstract clinical-document conventions are allowed, but the prose must be newly authored.",
            "Never mention dataset construction, annotation, placeholders, synthetic generation, quality assurance, benchmark selection, or a sealed review inside the clinical document.",
        ],
        "locale_rules": (
            [
                "Use British spelling and natural NHS language such as GP, surgery, A&E or emergency department, ward, consultant, discharge medicines, and postcode where appropriate.",
                "Ambiguous numeric dates are day/month/year. Do not use a US ZIP or SSN format.",
                "Do not imply that the Crown Dependencies are part of this profile.",
            ]
            if profile_id == "en-GB"
            else [
                "Use American spelling and natural US language such as primary care, office, ED, attending, resident, discharge medications, and ZIP code where appropriate.",
                "Ambiguous numeric dates are month/day/year. Do not use NHS, NINO, CHI, H&C, or UK postcode conventions.",
                "Territorial formats are valid when the supplied address uses them; do not rewrite them as mainland formats.",
            ]
        ),
        "allowed_labels": list(ALLOWED_LABELS),
        "required_pii_targets": _prompt_pii_targets(record),
        "hard_negative_targets": hard_negative_targets(record),
        "case": _case_payload(record),
    }
    if retry_feedback:
        prompt["previous_output_problem"] = retry_feedback
        prompt["retry_instruction"] = (
            "Write the complete document again from scratch while correcting every listed problem."
        )
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def english_system_prompt(profile_id: str) -> str:
    locale = "British" if profile_id == "en-GB" else "American"
    return (
        f"You create high-fidelity synthetic clinical documents in {locale} English for a "
        "medical de-identification training corpus. Documentation authenticity, exact marker syntax, "
        "locale fidelity, independent authorship, and clinically coherent context are mandatory. The supplied people and "
        "PII are synthetic or approved public/non-assignable values. Never invent additional PII. "
        "Return only the marked clinical document."
    )


def _strip_wrapping(text: str) -> str:
    result = text.strip()
    if result.startswith("```"):
        lines = result.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        result = "\n".join(lines).strip()
    return result


def _expand_short_markers(
    record: dict[str, Any],
    source: str,
    *,
    targets: list[dict[str, str]] | None = None,
) -> str:
    """Expand compact author-facing markers to canonical typed markers."""

    effective_targets = targets if targets is not None else coverage_targets(record)

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(effective_targets):
            raise EnglishLLMRenderError(f"Unknown compact PII placeholder {match.group()!r}")
        target = effective_targets[index]
        return f"[[PII|{target['label']}|{target['slot']}]]"

    expanded = SHORT_MARKER_RE.sub(replace, source)
    if SHORT_MARKER_START in expanded:
        raise EnglishLLMRenderError("Unbalanced or malformed compact PII marker")
    return expanded


def marked_text_to_english_doc(
    *,
    record: dict[str, Any],
    marked_text: str,
    model: str,
    usage: dict[str, Any] | None = None,
    response_id: str | None = None,
) -> dict[str, Any]:
    source = _expand_short_markers(record, _strip_wrapping(marked_text))
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    position = 0
    allowed_slots = {row["slot"]: row for row in pii_slots(record)}
    for match in MARKER_RE.finditer(source):
        prefix = source[cursor : match.start()]
        parts.append(prefix)
        position += len(prefix)
        label, slot = match.group(1), match.group(2)
        if label == "Anonymize_Other":
            raise EnglishLLMRenderError("Anonymize_Other is forbidden in English synthetic data")
        if label not in ALLOWED_LABEL_SET:
            raise EnglishLLMRenderError(f"Unknown or disallowed marker label {label!r}")
        expected = allowed_slots.get(slot)
        if expected is None:
            raise EnglishLLMRenderError(f"Unknown PII placeholder slot {slot!r}")
        if expected["label"] != label:
            raise EnglishLLMRenderError(
                f"Placeholder slot {slot!r} used {label!r}, expected {expected['label']!r}"
            )
        value = expected["value"]
        begin = position
        parts.append(value)
        position += len(value)
        category, subtype = split_label(label)
        spans.append(
            {
                "begin": begin,
                "end": position,
                "label": label,
                "text": value,
                "Category": category,
                "Subtype": subtype,
                "confirmed": True,
                "source_slot": slot,
                "semantic_type": expected["semantic_type"],
            }
        )
        cursor = match.end()
    parts.append(source[cursor:])
    clean_text = "".join(parts)
    if MARKER_START in clean_text or "[[/PII]]" in clean_text:
        raise EnglishLLMRenderError("Unbalanced or malformed PII marker")
    if not spans:
        raise EnglishLLMRenderError("Model output contained no PII markers")
    profile_id = str(record["language"])
    case_match = re.search(r"(\d+)$", str(record.get("case_id", "")))
    if case_match is None:
        raise EnglishLLMRenderError("Case ID must end in a numeric corpus index")
    doc_id = f"{profile_id.lower()}-synthetic-{int(case_match.group(1)):05d}"
    metadata: dict[str, Any] = {
        "generation_method": "english-case-luna-marked-renderer-v1",
        "generation_profile": profile_id,
        "renderer": "luna_marked_clinical_document",
        "model": model,
        "document_type": record["document_type"],
        "lang": profile_id,
        "synthetic": True,
        "independently_authored": True,
        "lookup_source": lookup_source(profile_id),
        "style_profile": style_profile(record),
        "hard_negative_targets": hard_negative_targets(record),
        "required_pii_targets": coverage_targets(record),
        "patient": {**record["patient"], "birth_date": record["birth_date"]},
        "patient_age": {
            "value": record.get("age_value", record.get("age_years")),
            "unit": record.get("age_unit", "years"),
            "completed_years": record.get("age_years"),
            "group": record.get("age_group", "adult"),
            "clinical_group": record.get(
                "clinical_age_group", record.get("age_group", "adult")
            ),
        },
        "caregivers": [dict(record["caregiver"])],
        "document_creation_date": record["encounter_date"],
        "source_case_id": record.get("case_id"),
        "response_id": response_id,
        "pii_policy": record.get("pii_policy"),
        "pii_resource_categories": record.get("pii_resource_categories"),
        "production": record.get("production"),
    }
    if usage:
        metadata["openai_usage"] = usage
    return {
        "document_id": doc_id,
        "text": clean_text,
        "spans": sorted(spans, key=lambda row: (row["begin"], row["end"])),
        "metadata": metadata,
        "annotated": True,
    }


def _covered(
    doc: dict[str, Any], begin: int, end: int, labels: set[str] | None = None
) -> bool:
    return any(
        (labels is None or span.get("label") in labels)
        and int(span.get("begin", -1)) <= begin
        and int(span.get("end", -1)) >= end
        for span in doc.get("spans", [])
    )


def _literal_ranges(text: str, value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    while value:
        begin = text.find(value, cursor)
        if begin < 0:
            break
        end = begin + len(value)
        result.append((begin, end))
        cursor = end
    return result


def validate_english_llm_document(
    *, record: dict[str, Any], marked_text: str, doc: dict[str, Any]
) -> list[str]:
    issues = list(validate_record(doc, strict_taxonomy=True))
    profile_id = str(record["language"])
    stored_targets = doc.get("metadata", {}).get("required_pii_targets")
    expected_targets = (
        [dict(row) for row in stored_targets]
        if isinstance(stored_targets, list) and stored_targets
        else coverage_targets(record)
    )
    allowed = {str(row["slot"]): row for row in expected_targets}
    stripped = _expand_short_markers(
        record,
        _strip_wrapping(marked_text),
        targets=expected_targets,
    )
    marker_rows = list(MARKER_RE.finditer(stripped))
    if len(marker_rows) != len(doc.get("spans", [])):
        issues.append(
            f"marker count {len(marker_rows)} does not match span count {len(doc.get('spans', []))}"
        )
    seen_slots: set[str] = set()
    for marker in marker_rows:
        label, slot = marker.group(1), marker.group(2)
        seen_slots.add(slot)
        expected = allowed.get(slot)
        if expected is None:
            issues.append(f"unknown PII marker slot {slot!r}")
            continue
        if label != expected["label"]:
            issues.append(f"slot {slot!r} used label {label!r}, expected {expected['label']!r}")
    for target in expected_targets:
        if target["slot"] not in seen_slots:
            issues.append(f"missing required PII target {target['slot']!r}")
    for marker, span in zip(marker_rows, doc.get("spans", [])):
        label, slot = marker.group(1), marker.group(2)
        expected = allowed.get(slot)
        if str(span.get("label")) != label or str(span.get("source_slot")) != slot:
            issues.append(
                f"marker {slot!r}/{label!r} does not match serialized span "
                f"{span.get('source_slot')!r}/{span.get('label')!r}"
            )
        if expected is not None and str(span.get("text")) != str(expected.get("value")):
            issues.append(f"serialized span text does not match stored target {slot!r}")
        if (
            expected is not None
            and expected.get("semantic_type")
            and str(span.get("semantic_type")) != str(expected.get("semantic_type"))
        ):
            issues.append(f"serialized span semantic type does not match stored target {slot!r}")
    for marker in marker_rows:
        label, slot = marker.group(1), marker.group(2)
        line_prefix = stripped[stripped.rfind("\n", 0, marker.start()) + 1 : marker.start()]
        if label != "Date" and re.search(
            r"\bDate(?:\s+of\s+[A-Za-z ]+)?\s*:\s*$", line_prefix, re.I
        ):
            issues.append(f"{label} marker was used as the value of a Date field")
        if slot == "patient.report_id" and re.search(
            r"\b(?:MRN|patient\s+(?:ID|identifier))\b", line_prefix, re.I
        ):
            issues.append("report/accession identifier was described as an MRN or patient identifier")
        if slot == "caregiver.professional_id" and re.search(
            r"(?:\bDr\.?|\bDoctor|\bDischarging\s+clinician\s*:|"
            r"\bDischarging\s+doctor\s*:)\s*$",
            line_prefix,
            re.I,
        ):
            issues.append("caregiver identifier used as a clinician name")
        if slot == "caregiver.professional_id" and re.search(
            r"\b(?:GMC-TEST|NPI-TEST|LIC)\s*-?\s*$", line_prefix, re.I
        ):
            issues.append("duplicated identifier prefix")
        if (
            str(record["document_type"]) == "laboratory_report"
            and slot in {
                "caregiver.name",
                "caregiver.name_upper",
                "caregiver.name_lower",
                "caregiver.initial_surname",
                "caregiver.credentialed_name",
            }
            and re.search(r"\bordering\s+(?:clinician|physician|provider|doctor)\b", line_prefix, re.I)
        ):
            issues.append(
                "laboratory validating clinician was also presented as the ordering clinician"
            )
    text = str(doc.get("text", ""))
    stored_patient_age = doc.get("metadata", {}).get("patient_age", {})
    validation_age = (
        stored_patient_age.get("completed_years")
        if isinstance(stored_patient_age, dict)
        else None
    )
    if validation_age is None:
        validation_age = record.get("age_years", 18)
    if int(validation_age) < 18 and any(
        span.get("label") == "Profession" for span in doc.get("spans", [])
    ):
        issues.append("Profession target was assigned to a pediatric patient")
    if int(validation_age) < 21 and any(
        span.get("label") == "Profession"
        and ADVANCED_PROFESSION_RE.search(str(span.get("text", "")))
        for span in doc.get("spans", [])
    ):
        issues.append("profession is implausible for patient age")
    if int(validation_age) < 31 and re.search(r"\badult child\b", text, re.I):
        issues.append("patient age is incompatible with adult-child relationship")
    for span in doc.get("spans", []):
        begin = int(span.get("begin", -1))
        end = int(span.get("end", -1))
        line_start = text.rfind("\n", 0, begin) + 1 if begin >= 0 else 0
        previous_line_end = max(0, line_start - 1)
        previous_line_start = text.rfind("\n", 0, previous_line_end) + 1
        previous_line = text[previous_line_start:previous_line_end]
        if (
            span.get("label") == "Age_Birthdate"
            and begin > 0
            and end < len(text)
            and text[begin - 1] == "<"
            and text[end] == ">"
        ):
            issues.append("age PII span was wrapped in artificial angle brackets")
        if span.get("label") == "Age_Birthdate" and begin >= 0:
            line_prefix = text[text.rfind("\n", 0, begin) + 1 : begin]
            if re.search(
                r"\b(?:(?:responsible|treating|reviewing|referring|discharging)\s+)?"
                r"(?:clinician|doctor|physician|nurse|consultant)\s*:",
                line_prefix,
                re.I,
            ):
                issues.append("patient age PII was assigned to a clinician")
            if (
                not text[end:].strip()
                and re.search(
                    r"^\s*(?:attending|clinician|signed(?:\s+by)?|validated(?:\s+by)?|"
                    r"documented(?:\s+by)?)\s*:",
                    previous_line,
                    re.I,
                )
            ):
                issues.append("patient age PII was placed after a clinician byline")
        if (
            span.get("label") == "Contactdetails"
            and str(record.get("document_type")) == "laboratory_report"
            and re.search(r"\b(?:validated|authorised|authorized|signed)\s+by\s*:", previous_line, re.I)
        ):
            issues.append("patient contact PII was placed after a laboratory validator")
        if span.get("label") == "Organization:Healthcare" and str(
            span.get("text", "")
        ).rstrip().endswith(("-", "/", "&")):
            issues.append("healthcare organization PII has dangling terminal punctuation")
        if (
            str(span.get("label", "")).startswith("Name:")
            and end >= 0
            and end < len(text)
            and text[end].isalpha()
        ):
            issues.append("name PII span was concatenated with trailing alphabetic text")
        if (
            span.get("label") == "Name:Caregiver"
            and end >= 0
            and re.match(r"^\s+ose\.(?:\s|$)", text[end : end + 12], re.I)
        ):
            issues.append("caregiver name PII has stray trailing syntax")
        if (
            span.get("label") == "Organization:Other"
            and int(validation_age) < 18
        ):
            organisation = str(span.get("text", "")).casefold()
            context = text[max(0, begin - 55) : min(len(text), end + 55)].casefold()
            civic_or_adult_service = any(
                token in organisation
                for token in (
                    "housing authority",
                    "transit authority",
                    "town council",
                    "senior center",
                )
            )
            if civic_or_adult_service and re.search(
                r"\b(?:school|daycare|education|attends?|affiliation)\b", context
            ):
                issues.append(
                    "pediatric non-healthcare organization was assigned an age-inappropriate role"
                )
            has_school_marker = any(
                token in organisation
                for token in ("school", "college", "academy", "nursery")
            )
            if not has_school_marker and re.search(
                r"\b(?:school|education)(?:\s+(?:attended|provider|setting))?\s*:",
                context,
            ):
                issues.append(
                    "pediatric non-healthcare organization was assigned an age-inappropriate role"
                )
            if (
                "sports club" in organisation
                and int(validation_age) < 5
                and re.search(r"\battends?\b", context)
            ):
                issues.append(
                    "pediatric non-healthcare organization was assigned an age-inappropriate role"
                )
            age_incompatible_school = (
                int(validation_age) < 16
                and any(
                    token in organisation
                    for token in ("college", "university", "further education")
                )
            ) or (
                int(validation_age) < 11
                and any(
                    token in organisation
                    for token in ("secondary school", "high school")
                )
            )
            explicit_adult_affiliation = re.search(
                r"(?:parent|guardian|mother|father|aunt|uncle|adult relative).{0,45}"
                r"(?:employer|workplace|works?\s+(?:at|for)|affiliation)",
                context,
                re.I,
            )
            if age_incompatible_school and not explicit_adult_affiliation:
                issues.append(
                    "education organization is incompatible with the pediatric patient's age"
                )
            education_context = text[
                max(0, begin - 80) : min(len(text), end + 140)
            ].casefold()
            if age_incompatible_school and re.search(
                r"\b(?:not\s+(?:the\s+)?(?:child|patient)[’']?s?\s+school|"
                r"not\s+involved\s+in\s+(?:the\s+patient[’']?s?|your)\s+"
                r"(?:education|medical care))\b",
                education_context,
                re.I,
            ):
                issues.append(
                    "education organization role was explained with a synthetic disclaimer"
                )
    for span in doc.get("spans", []):
        if span.get("label") != "Name:Caregiver":
            continue
        begin = int(span.get("begin", -1))
        end = int(span.get("end", -1))
        if begin >= 0 and ADDED_HONORIFIC_RE.search(text[max(0, begin - 16) : begin]):
            issues.append(
                "caregiver honorific was added outside the annotated name span"
            )
        if end >= 0 and ADDED_CREDENTIAL_RE.match(text[end : end + 24]):
            issues.append(
                "caregiver credential was added outside the annotated name span"
            )
        line_prefix = text[text.rfind("\n", 0, begin) + 1 : begin]
        span_text = text[begin:end]
        if re.search(r"\bNurse\s*:\s*$", line_prefix, re.I) and re.search(
            r",\s*(?:MD|DO)\s*$", span_text, re.I
        ):
            issues.append(
                "physician credential was placed in a nurse byline"
            )
    # Literal PII values are not sent to the model.  Exact placeholder accounting
    # above therefore provides the authoritative leakage check without producing
    # false alarms when two typed slots intentionally resolve to the same locality.
    stored_hard_negatives = doc.get("metadata", {}).get("hard_negative_targets")
    expected_hard_negatives = (
        [dict(row) for row in stored_hard_negatives]
        if isinstance(stored_hard_negatives, list) and stored_hard_negatives
        else hard_negative_targets(record)
    )
    for target in expected_hard_negatives:
        value = target["value"]
        ranges = _literal_ranges(text, value)
        if not ranges:
            issues.append(f"missing hard-negative target {target['category']}: {value!r}")
        for begin, end in ranges:
            if _covered(doc, begin, end):
                issues.append(f"hard-negative target was annotated as PII: {value!r}")
            context = text[max(0, begin - 100) : min(len(text), end + 100)]
            if UNNATURAL_HARD_NEGATIVE_RE.search(context):
                issues.append(
                    f"hard-negative target was inserted as an artificial test disclaimer: {value!r}"
                )
            if HARD_NEGATIVE_SELF_CORRECTION_RE.search(text[end : end + 20]):
                issues.append(
                    f"hard-negative target was inserted as an artificial self-correction: {value!r}"
                )
            if HARD_NEGATIVE_REJECTION_RE.search(text[end : end + 60]):
                issues.append(
                    f"hard-negative target was inserted as an artificial rejection: {value!r}"
                )
            if target["category"] == "caregiver_role":
                signature_matches = list(
                    re.finditer(
                        r"(?mi)^(?:sincerely|yours\s+(?:sincerely|faithfully)|"
                        r"signed(?:\s+by)?\s*:?)\b",
                        text[:begin],
                    )
                )
                if signature_matches:
                    after_signature = text[signature_matches[-1].end() : begin]
                    # A specialty on the line beneath a signatory is a natural title.
                    # New patient-directed prose after the closing/signature is not.
                    if re.search(r"\b(?:the|this)\s+patient\b", after_signature, re.I):
                        issues.append(
                            f"caregiver-role hard negative was placed after a signature: {value!r}"
                        )
            if target["category"] == "relative_time":
                treatment_prefix = re.search(
                    r"\bcontinue\b(?P<body>[^.;\n]{0,120})$",
                    text[max(0, begin - 140) : begin],
                    re.I,
                )
                if treatment_prefix and not re.search(
                    r"\b(?:follow(?:-|\s)?up|review|return|reassess|appointment)\b",
                    treatment_prefix.group("body"),
                    re.I,
                ):
                    issues.append(
                        f"relative-time hard negative directly modifies continued treatment: {value!r}"
                    )
    hard_negative_date_ranges = {
        pair
        for target in expected_hard_negatives
        if target["category"] == "impossible_date"
        for pair in _literal_ranges(text, target["value"])
    }
    for match in FULL_DATE_RE.finditer(text):
        if (match.start(), match.end()) in hard_negative_date_ranges:
            continue
        if not _covered(doc, match.start(), match.end(), {"Date", "Age_Birthdate"}):
            issues.append(f"potential invented or unmarked full date: {match.group()!r}")
    for match in EXPLICIT_AGE_RE.finditer(text):
        if "gestational" in text[max(0, match.start() - 16) : match.end()].lower():
            continue
        overlaps_age_span = any(
            span.get("label") == "Age_Birthdate"
            and int(span.get("begin", -1)) < match.end()
            and int(span.get("end", -1)) > match.start()
            for span in doc.get("spans", [])
        )
        if not overlaps_age_span:
            issues.append(f"potential invented or unmarked explicit age: {match.group()!r}")
    for match in EXPLICIT_BIRTH_YEAR_RE.finditer(text):
        if not _covered(doc, match.start(1), match.end(1), {"Age_Birthdate"}):
            issues.append(
                f"potential invented or unmarked birth year: {match.group(1)!r}"
            )
    jurisdiction_re = (
        US_JURISDICTION_RE if profile_id == "en-US" else GB_JURISDICTION_RE
    )
    for match in jurisdiction_re.finditer(text):
        if not _covered(doc, match.start(), match.end()):
            issues.append(
                f"potential invented or unmarked jurisdiction: {match.group()!r}"
            )
    for match in SPELLED_NUMBER_RANGE_RE.finditer(text):
        issues.append(f"malformed mixed numeric/word range: {match.group()!r}")
    for match in LABELED_PROFESSION_FIELD_RE.finditer(text):
        field_value = match.group(1).strip()
        if NULL_FIELD_VALUE_RE.fullmatch(field_value):
            continue
        if not any(
            span.get("label") == "Profession"
            and int(span.get("begin", -1)) < match.end(1)
            and int(span.get("end", -1)) > match.start(1)
            for span in doc.get("spans", [])
        ):
            issues.append(
                f"potential invented or unmarked profession field: {field_value!r}"
            )
    if "**" in text or re.search(r"(?m)^#{1,6}\s+", text):
        issues.append("document contains Markdown emphasis or heading syntax")
    for pattern, description in (
        (ARTIFICIAL_ROLE_FIELD_RE, "artificial test disclaimer"),
        (EMBEDDED_OCCUPATION_FIELD_RE, "occupation field label embedded in prose"),
        (UNNATURAL_AGE_RE, "unnatural hyphenated age construction"),
        (REDUNDANT_AGE_PREFIX_RE, "redundant age-field construction"),
        (WRONG_AGE_ARTICLE_RE, "incorrect indefinite article before age"),
        (MISSING_AGE_ARTICLE_RE, "missing indefinite article before age"),
        (SUBJECTLESS_AGE_RE, "age expression used without a patient noun"),
        (SUBJECTLESS_CONTEXTUAL_AGE_RE, "contextual age used without a patient noun"),
        (REDUNDANT_AGE_GROUP_RE, "redundant age and age-group construction"),
        (REDUNDANT_MEASUREMENT_RE, "redundant measurement construction"),
        (REDUNDANT_HARD_NEGATIVE_RE, "redundant hard-negative construction"),
        (DUPLICATED_ID_PREFIX_RE, "duplicated identifier prefix"),
        (UNNATURAL_GMC_ID_LABEL_RE, "unnatural GMC identifier label"),
        (DUPLICATED_CREDENTIAL_RE, "duplicated clinician credential"),
        (SPELLED_RESPIRATORY_RATE_RE, "spelled-out respiratory rate value"),
        (MISSING_AGE_COPULA_RE, "age construction is missing a copula"),
        (GENERATION_META_RE, "generation metadata leaked into age context"),
        (BAD_PREPOSITION_RE, "duplicated preposition in interval context"),
        (UNGRAMMATICAL_DISCHARGE_RE, "ungrammatical discharge construction"),
        (UNGRAMMATICAL_DURATION_RE, "ungrammatical duration construction"),
        (REDUNDANT_DURATION_FIELD_RE, "redundant duration field construction"),
        (MALFORMED_CREDENTIAL_SUFFIX_RE, "malformed clinician credential suffix"),
        (CAREGIVER_BYLINE_PROFESSION_RE, "patient profession is attached to a clinician byline"),
        (REDUNDANT_VITAL_LABEL_RE, "redundant vital-sign label"),
        (UNEXPECTED_CJK_PUNCTUATION_RE, "unexpected CJK punctuation in English document"),
        (MALFORMED_US_INTERSTATE_RE, "malformed US interstate display"),
        (CAREGIVER_ID_AS_NAME_RE, "caregiver identifier used as a clinician name"),
        (UNNATURAL_PATIENT_AGE_FIELD_RE, "age was used as the Patient field value"),
        (DETACHED_AGE_RE, "detached age expression at document end"),
        (PHYSICIAN_NURSE_ROLE_RE, "physician credential conflicts with nurse role"),
        (INCOMPLETE_CLINICIAN_RE, "incomplete clinician name before professional identifier"),
        (GMC_NURSING_ROLE_RE, "GMC identifier was assigned to a nursing role"),
        (INCOMPATIBLE_DUAL_CLINICIAN_ROLE_RE, "clinician was assigned incompatible dual roles"),
    ):
        match = pattern.search(text)
        if match:
            issues.append(f"{description}: {match.group()!r}")
    unresolved = UNRESOLVED_TEMPLATE_RE.search(text)
    if unresolved:
        issues.append(f"unresolved template placeholder: {unresolved.group()!r}")
    stray_syntax = STRAY_GENERATION_SYNTAX_RE.search(text)
    if stray_syntax:
        issues.append(f"stray generation syntax: {stray_syntax.group()!r}")
    unexpected_script = UNEXPECTED_SCRIPT_RE.search(text)
    if unexpected_script:
        issues.append(
            f"unexpected non-Latin script in English document: {unexpected_script.group()!r}"
        )
    if profile_id == "en-US":
        for match in US_BRITISH_SPELLING_RE.finditer(text):
            # Proper names can legitimately retain any regional spelling.
            if not _covered(doc, match.start(), match.end()):
                issues.append(
                    f"cross-locale British spelling in en-US prose: {match.group()!r}"
                )
    fena_patterns = {
        "serum_creatinine": r"Serum creatinine:\s*(\d+(?:\.\d+)?)\s*mg/dL",
        "urine_sodium": r"Urine sodium:\s*(\d+(?:\.\d+)?)\s*mmol/L",
        "serum_sodium": r"Serum sodium:\s*(\d+(?:\.\d+)?)\s*mmol/L",
        "urine_creatinine": r"Urine creatinine:\s*(\d+(?:\.\d+)?)\s*mg/dL",
        "reported_fena": r"(?:FeNa|Fractional excretion of sodium[^:\n]*):\s*(?:approximately\s*)?(\d+(?:\.\d+)?)%",
    }
    fena_values: dict[str, float] = {}
    for key, pattern in fena_patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            fena_values[key] = float(match.group(1))
    if len(fena_values) == len(fena_patterns):
        expected_fena = (
            fena_values["urine_sodium"]
            * fena_values["serum_creatinine"]
            / (fena_values["serum_sodium"] * fena_values["urine_creatinine"])
            * 100
        )
        if abs(fena_values["reported_fena"] - expected_fena) > max(
            0.05, expected_fena * 0.15
        ):
            issues.append(
                "incorrect FeNa calculation: "
                f"reported {fena_values['reported_fena']:.2f}%, expected approximately {expected_fena:.2f}%"
            )
    patterns = (
        (EMAIL_RE, {"Contactdetails"}),
        (GB_PHONE_RE if profile_id == "en-GB" else US_PHONE_RE, {"Contactdetails"}),
        (GB_ID_RE if profile_id == "en-GB" else US_ID_RE, {"ID:Patient", "ID:Caregiver"}),
    )
    for pattern, labels in patterns:
        for match in pattern.finditer(text):
            if not _covered(doc, match.start(), match.end(), labels):
                issues.append(f"potential unmarked {min(labels)}: {match.group()!r}")
    for match in LABELED_REPORT_ID_RE.finditer(text):
        begin, end = match.span(1)
        if not _covered(doc, begin, end, {"ID:Patient"}):
            issues.append(f"potential invented or unmarked report identifier: {match.group(1)!r}")
    for span in doc.get("spans", []):
        label = str(span.get("label", ""))
        value = str(span.get("text", ""))
        if label == "Anonymize_Other" or label not in ALLOWED_LABEL_SET:
            issues.append(f"disallowed synthetic label {label!r}")
        if label == "Contactdetails" and any(character.isdigit() for character in value) and "@" not in value:
            if not is_approved_synthetic_phone(value, profile_id=profile_id):
                issues.append(f"unapproved synthetic phone {value!r}")
        if label in {"ID:Patient", "ID:Caregiver"}:
            if not is_approved_synthetic_identifier(value, profile_id=profile_id):
                issues.append(f"unapproved synthetic identifier {value!r}")
        if label == "Profession":
            prefix = text[max(0, int(span["begin"]) - 8) : int(span["begin"])]
            article_match = re.search(r"\b(a|an)\s+$", prefix, re.I)
            first_letter = next((character for character in value if character.isalpha()), "")
            if article_match and first_letter:
                expected_article = "an" if first_letter.lower() in "aeiou" else "a"
                if article_match.group(1).lower() != expected_article:
                    issues.append(
                        f"incorrect article before profession: {article_match.group(1)!r} before {value!r}"
                    )
    word_count = len(TOKEN_RE.findall(text))
    style_name = str(style_profile(record)["name"])
    minimum_words = {
        "minimal-ehr": 100,
        "handover-note": 110,
        "compact-clinician": 120,
        "structured-record": 130,
        "narrative-scribe": 160,
        "specialist-report": 160,
        "patient-facing-letter": 170,
        "longitudinal-summary": 190,
    }[style_name]
    if word_count < minimum_words:
        issues.append(
            f"document is too short for {style_name}: {word_count} words; "
            f"minimum {minimum_words}"
        )
    if word_count > 900:
        issues.append(f"document is too long for the production corpus: {word_count} words")
    lowered = text.lower()
    locale_leaks = (
        ("zip code", "social security number", "primary care physician")
        if profile_id == "en-GB"
        else ("nhs number", "postcode", "a&e", "gp surgery")
    )
    for phrase in locale_leaks:
        if phrase in lowered:
            issues.append(f"cross-locale phrase in {profile_id}: {phrase!r}")
    return list(dict.fromkeys(issues))


def _usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if hasattr(usage, key)
    }


async def render_case_record_with_luna(
    record: dict[str, Any],
    *,
    client: Any,
    model: str = DEFAULT_ENGLISH_LLM_MODEL,
    max_output_tokens: int = 1800,
    reasoning_effort: str = "low",
    retry_feedback: str | None = None,
    usage_recorder: UsageRecorder | None = None,
) -> EnglishLLMRenderResult:
    """Render one structured case through the Responses API and validate locally."""

    profile_id = str(record["language"])
    response = await client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": english_system_prompt(profile_id)},
            {
                "role": "user",
                "content": build_english_llm_prompt(record, retry_feedback=retry_feedback),
            },
        ],
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
    )
    marked_text = str(response.output_text)
    usage = _usage_dict(response)
    response_id = getattr(response, "id", None)
    if usage_recorder is not None:
        usage_recorder(
            {
                "stage": "author",
                "model": model,
                "response_id": response_id,
                "usage": usage,
            }
        )
    doc = marked_text_to_english_doc(
        record=record,
        marked_text=marked_text,
        model=model,
        usage=usage,
        response_id=response_id,
    )
    issues = validate_english_llm_document(
        record=record, marked_text=marked_text, doc=doc
    )
    if issues:
        raise EnglishLLMRenderError("; ".join(issues[:12]))
    doc["metadata"]["local_validation_passed"] = True
    return EnglishLLMRenderResult(
        doc=doc,
        marked_text=marked_text,
        usage=usage,
        response_id=response_id,
    )


async def review_english_document_with_luna(
    *,
    record: dict[str, Any],
    doc: dict[str, Any],
    client: Any,
    model: str = DEFAULT_ENGLISH_LLM_MODEL,
    reasoning_effort: str = "low",
    usage_recorder: UsageRecorder | None = None,
) -> dict[str, Any]:
    """Run a second, structured clinical/editorial review over one rendered note."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "severity": {"type": "string", "enum": ["major", "minor"]},
                        "category": {
                            "type": "string",
                            "enum": [
                                "clinical_accuracy",
                                "internal_consistency",
                                "demographic_plausibility",
                                "numeric_or_unit_error",
                                "locale_fidelity",
                                "document_realism",
                                "hard_negative_naturalness",
                                "pii_semantics",
                                "language_quality",
                            ],
                        },
                        "exact_quote": {"type": "string"},
                        "explanation": {"type": "string"},
                        "required_fix": {"type": "string"},
                    },
                    "required": [
                        "severity",
                        "category",
                        "exact_quote",
                        "explanation",
                        "required_fix",
                    ],
                },
            },
        },
        "required": ["verdict", "issues"],
    }
    payload = {
        "task": "Act as a strict clinical editor and synthetic DEID gold-data reviewer.",
        "instructions": [
            "Return pass only when the document is clinically coherent, numerically well formed, internally consistent, locale-correct, and realistic for its document type.",
            "Flag wrong or contradictory clinical statements, implausible age/condition combinations, malformed numbers or units, nonsensical department/organisation roles, and unsafe or unsupported medication advice.",
            "Recalculate stated derived values from the numbers in the document, including FeNa, corrected values, ratios, anion gaps, and percentages; flag material arithmetic disagreement.",
            "Flag unresolved template artifacts such as [date], [day of admission], [name], TBD, or insert-value instructions.",
            "When a SNOMED CT, LOINC, ICD, or other terminology code is present, flag it if its described clinical meaning does not match the code or if it is inserted without a realistic coding/result context.",
            "For en-US, flag distinctly UK-only prescribing language or drugs not ordinarily used in that US scenario, including flucloxacillin or co-amoxiclav as active treatment and adult asthma/COPD bursts written as prednisolone rather than the usual US prednisone. For en-GB, flag distinctly US-only administrative or prescribing language.",
            "Flag hard-negative phrases that were dumped as unrelated examples, identifier explanations, or otherwise unnatural test content instead of being integrated into the clinical document.",
            "Flag a non-healthcare organisation used as a treating, referring, requesting, reporting, or discharging healthcare provider.",
            "Perform a sentence-level grammar review. Flag malformed age constructions, dangling modifiers, broken discharge-condition phrases, duplicated words, and Markdown presentation that would not belong in extracted plain clinical text.",
            "In referral letters, verify that a named referring clinician is the author/signatory rather than also being used as the recipient or addressee.",
            "Verify that report/accession identifiers are not described in clinical prose as MRNs or patient IDs, and that MRNs are not described as report/accession identifiers. Under the fixed suite taxonomy, patient-associated report/accession identifiers are intentionally and correctly annotated as ID:Patient; do not flag that label or the patient.report_id source slot.",
            "Verify that national/public patient identifiers are described with their supplied locale-correct field meaning (for example SSN or NHS number), never as an MRN or report/accession identifier. They intentionally retain the public ID:Patient label and the private patient.national_id source slot.",
            "In laboratory reports, flag the supplied named validating/reporting clinician if that same person is also used as the ordering clinician.",
            "The PII values are deliberately synthetic and may be unusual. Do not flag a name, public institution, occupation, or geographic value merely because it is unfamiliar.",
            "Lowercase, uppercase, first-name-only, initial-only, first-name-plus-initial, and credentialed name forms are deliberate formatting-robustness examples. If the supplied span and surrounding grammar preserve that form consistently, do not request conventional capitalization or expansion.",
            "In the fixed MedDeID taxonomy, Name:Caregiver denotes a healthcare caregiver/clinician (for example a GP, referring doctor, attending, validator, or signatory), not a family caregiver. Relatives and informal carers use Name:Other. A referral signed by a Name:Caregiver span is therefore signed by the supplied referring clinician; do not claim that no clinician is named merely because of the label wording.",
            "Do not demand information that the document type would normally omit. Do not make stylistic preferences into issues unless wording is clearly awkward, contradictory, or unrealistic.",
            "EHR-style completed field labels (for example Occupation:, Employer:, MRN:, Date:) are deliberate formatting diversity and are not template artifacts. Minor correspondence conventions such as Yours sincerely versus Yours faithfully are non-blocking style preferences.",
            "Do not flag debatable guideline choices, reasonable alternative management, or missing rationale merely because another clinician might choose differently. Reserve clinical issues for objective contradictions, clearly unsafe advice, impossible scenarios, or material incoherence.",
            "A structured routine follow-up date may coexist with nearer symptom-based safety-netting. Do not reject a longer routine review interval merely because no scheduled review would also be reasonable, provided urgent or persistence advice is internally coherent.",
            "The structured synthetic case may deliberately use a historical or future encounter year to train date-shift robustness. Treat that value like a pseudonymized calendar shift: surrounding medicine, terminology, and health-service pathways may remain modern. Do not require period-specific drug availability, historical service names, or speculative future practice when dates and ages are internally consistent.",
            "Every issue must quote exact text from the document. If there are no issues, return verdict pass and an empty issues list.",
        ],
        "locale": record["language"],
        "document_type": record["document_type"],
        "structured_clinical_case": _case_payload(record),
        "required_hard_negatives": hard_negative_targets(record),
        "document": doc,
    }
    response = await client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": (
                    "You are the independent final clinical-quality reviewer for a synthetic "
                    "medical de-identification corpus. Be conservative but exact."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        reasoning={"effort": reasoning_effort},
        max_output_tokens=1200,
        text={
            "format": {
                "type": "json_schema",
                "name": "english_synthetic_document_review",
                "strict": True,
                "schema": schema,
            }
        },
        store=False,
    )
    usage = _usage_dict(response)
    response_id = getattr(response, "id", None)
    if usage_recorder is not None:
        usage_recorder(
            {
                "stage": "clinical_review",
                "model": model,
                "response_id": response_id,
                "usage": usage,
            }
        )
    try:
        review = json.loads(str(response.output_text))
    except json.JSONDecodeError as exc:
        raise EnglishLLMRenderError("clinical reviewer returned invalid JSON") from exc
    if not isinstance(review, dict):
        raise EnglishLLMRenderError("clinical reviewer returned a non-object")
    issues = review.get("issues")
    if not isinstance(issues, list):
        raise EnglishLLMRenderError("clinical reviewer omitted its issue list")
    text = str(doc.get("text", ""))
    for issue in issues:
        quote = str(issue.get("exact_quote", ""))
        if quote and quote not in text:
            raise EnglishLLMRenderError(
                f"clinical reviewer cited text that is not in the document: {quote!r}"
            )
    review["model"] = model
    review["response_id"] = response_id
    review["usage"] = usage
    return review
