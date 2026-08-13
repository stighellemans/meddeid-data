"""Synthetic note generation with exact gold annotations."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from meddeid_core import normalize_record
from meddeid_language_nl.date_age_variants import parse_date_text

from .clinical_cases import (
    ClinicalCase,
    LookupSampler,
    caregiver_name_variants,
    case_from_record,
    generate_case,
    person_name_variants,
)
from .identifiers import hard_negative_codes
from .production_postprocess import (
    apply_production_post_process,
    is_relative_date_hard_negative,
)
from .span_builder import SpanBuilder, canonical_pii_metadata
from .synthea_adapter import load_or_generate_synthea_csv_seeds

Renderer = Callable[[str, ClinicalCase, random.Random], dict]


def _sources(case: ClinicalCase, renderer: str) -> dict:
    metadata = {
        "generation_method": "case-model-renderer-v2",
        "renderer": renderer,
        "document_type": case.document_type,
        "note_style": case.note_style,
        "age_context": case.age_context,
        "birthdate_prefix": case.birthdate_prefix,
        "administrative_gender": case.administrative_gender,
        "style_profile": case.style_profile,
        "medical_details": case.medical_details,
        "date_overview": case.date_overview,
        "date_times": case.date_times,
        "date_periods": case.date_periods,
        "date_focus": case.date_focus,
        "date_focus_template": case.date_focus_template,
        "date_focus_style": case.date_focus_style,
        "lang": case.language,
        "synthea_source": case.synthea_source,
        # Every synthetic identity/location value comes from the same published
        # provider. Record its package version, not internal resource paths.
        "lookup_source": case.hospital[1],
    }
    metadata.update(canonical_pii_metadata(case))
    return metadata


def _condition(case: ClinicalCase) -> str:
    return str(case.condition["name"])


def _symptoms(case: ClinicalCase, count: int = 2) -> str:
    return ", ".join(case.condition["symptoms"][:count])


def _medications(case: ClinicalCase, count: int = 3) -> str:
    return ", ".join(case.condition["medications"][:count])


def _catalog_medication_rows(case: ClinicalCase) -> list[dict]:
    details = case.medical_details or {}
    rows = details.get("catalog_medications", [])
    if not isinstance(rows, list):
        return []
    valid_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cnk = str(row.get("cnk", "")).strip()
        name = str(row.get("name", "")).strip()
        if cnk and name:
            valid_rows.append({"cnk": cnk, "name": name})
    return valid_rows


def _catalog_medication_names(case: ClinicalCase) -> list[str]:
    return [row["name"] for row in _catalog_medication_rows(case)]


def _medical_eponym_rows(case: ClinicalCase) -> list[dict]:
    details = case.medical_details or {}
    rows = details.get("medical_eponyms", [])
    if not isinstance(rows, list):
        return []
    valid_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        slug = str(row.get("slug", "")).strip()
        if title:
            valid_rows.append({"slug": slug, "title": title})
    return valid_rows


def _medical_eponym_titles(case: ClinicalCase) -> list[str]:
    return [row["title"] for row in _medical_eponym_rows(case)]


def _numbered_title(case: ClinicalCase, title: str, context: str) -> str:
    seed = _stable_offset(
        case.document_type, case.note_style, case.patient.name, context
    )
    if seed % 4 != 0:
        return title
    chapter = 1 + (seed // 4) % 8
    section = 1 + (seed // 32) % 6
    return f"{chapter}.{section} {title}"


CAPITALIZED_MEDICAL_TITLE_BASES = (
    "ECHOGRAFIE VAN LINKER BEEN",
    "ECHOGRAFIE RECHTER NIERNIVEAU",
    "MR HERSENEN",
    "MRI LUMBALE WERVELZUIL",
    "CT THORAX ABDOMEN",
    "RX THORAX",
    "DUPLEX HALSSLAGADERS",
    "COLONOSCOPIE",
    "GASTROSCOPIE",
    "LONGFUNCTIE",
    "SLAAPONDERZOEK",
    "WONDZORGCONTROLE",
    "PREOPERATIEVE EVALUATIE",
    "POSTOPERATIEVE CONTROLE",
    "MAMMOGRAFIE",
    "BOTDENSITOMETRIE",
    "NEUROLOGISCH CONSULT",
    "CARDIALE ECHOGRAFIE",
    "DIABETESJAARCONTROLE",
    "MEDICATIEVERIFICATIE",
    "METINGEN",
    "MEETRESULTATEN",
)
CAPITALIZED_MEDICAL_TITLE_SUFFIXES = (
    "",
    " REPORT",
    " RAPPORT",
    " VERSLAG",
    " RESULTAAT",
    " AANVRAAG",
    " PROTOCOL",
)


def _capitalized_medical_title(case: ClinicalCase, context: str) -> str:
    seed = _stable_offset(
        case.document_type, case.note_style, case.department, context, _condition(case)
    )
    base = CAPITALIZED_MEDICAL_TITLE_BASES[seed % len(CAPITALIZED_MEDICAL_TITLE_BASES)]
    suffix = CAPITALIZED_MEDICAL_TITLE_SUFFIXES[
        (seed // max(1, len(CAPITALIZED_MEDICAL_TITLE_BASES)))
        % len(CAPITALIZED_MEDICAL_TITLE_SUFFIXES)
    ]
    title = f"{base}{suffix}"
    return _numbered_title(case, title, f"{context}:capitalized-medical-title")


def _add_capitalized_medical_title(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    style = (
        _stable_offset(
            case.document_type, case.note_style, context, "capitalized-title-style"
        )
        % 5
    )
    title = _capitalized_medical_title(case, context)
    if style == 0:
        b.add(title)
        b.add("\n")
    elif style == 1:
        b.add(title)
        b.add("\n")
        b.add("KLINISCHE VRAAG: ")
        b.add(_condition(case).upper())
        b.add("\n")
    elif style == 2:
        b.add("== ")
        b.add(title)
        b.add(" ==\n")
    elif style == 3:
        b.add(title)
        b.add("\nSTATUS: DEFINITIEF\n")
    else:
        b.add(title)
        b.add("\n")
        b.add("MEDISCHE CONTEXT, GEEN IDENTIFICERENDE NAAM\n")


CAREGIVER_FUNCTION_HARD_NEGATIVES = [
    "cardioloog",
    "cardioloog-consulent",
    "interventioneel cardioloog",
    "huisarts",
    "huisarts van wacht",
    "HAIO",
    "arts-assistent",
    "assistent-specialist",
    "ASO geneeskunde",
    "urgentist",
    "spoedarts",
    "pneumoloog",
    "longarts",
    "geriater",
    "arts algemene geneeskunde",
    "nefroloog",
    "endocrinoloog",
    "diabetoloog",
    "neuroloog",
    "psychiater",
    "kinderpsychiater",
    "pediater",
    "kinderarts",
    "gynaecoloog",
    "uroloog",
    "gastro-enteroloog",
    "hepatoloog",
    "hematoloog",
    "oncoloog",
    "radiotherapeut",
    "dermatoloog",
    "reumatoloog",
    "orthopedist",
    "orthopedisch chirurg",
    "algemeen chirurg",
    "vaatchirurg",
    "neurochirurg",
    "plastisch chirurg",
    "anesthesist",
    "intensivist",
    "pijnarts",
    "oogarts",
    "NKO-arts",
    "MKA-chirurg",
    "klinisch apotheker",
    "ziekenhuisapotheker",
    "verpleegkundig specialist",
    "referentieverpleegkundige",
    "wondzorgverpleegkundige",
    "diabeteseducator",
    "oncocoach",
    "liaisonverpleegkundige",
    "hoofdverpleegkundige",
    "zorgcoordinator",
    "sociaal verpleegkundige",
    "klinisch psycholoog",
    "neuropsycholoog",
    "psychotherapeut",
    "kinesitherapeut",
    "ergotherapeut",
    "logopedist",
    "dietetist",
    "vroedvrouw",
    "audioloog",
    "radiologisch technoloog",
    "laboratoriumarts",
    "klinisch bioloog",
    "radioloog",
    "nucleair geneeskundige",
    "patholoog",
    "patholoog-anatoom",
    "revalidatiearts",
    "zaalarts",
    "MDO-voorzitter",
    "supervisor",
    "consulent",
]


def _patient_gender_marker(case: ClinicalCase, context: str) -> str:
    value = str(getattr(case, "administrative_gender", "") or "").strip()
    if value:
        return value
    options = ["M", "V", "X", "man", "vrouw", "onbekend"]
    return options[
        _stable_offset(case.patient.name, case.document_type, context) % len(options)
    ]


def _caregiver_function(case: ClinicalCase, context: str) -> str:
    return CAREGIVER_FUNCTION_HARD_NEGATIVES[
        _stable_offset(case.department, case.document_type, context)
        % len(CAREGIVER_FUNCTION_HARD_NEGATIVES)
    ]


def _add_caregiver_function_suffix(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    role = _caregiver_function(case, context)
    style = _stable_offset(case.document_type, case.note_style, context, role) % 12
    if style == 0:
        b.add(f", {role}")
    elif style == 1:
        b.add(f" ({role})")
    elif style == 2:
        b.add(f", functie {role}")
    elif style == 3:
        b.add(f", rol: {role}")
    elif style == 4:
        b.add(f", discipline {role}")
    elif style == 5:
        b.add(f", als {role}")
    elif style == 6:
        b.add(f", behandelend {role}")
    elif style == 7:
        b.add(f" - {role}")
    elif style == 8:
        b.add(f", team {role}")
    elif style == 9:
        b.add(f", supervisie {role}")
    elif style == 10:
        b.add(f", zorgrol {role}")
    else:
        b.add(f", klinische functie: {role}")


def _add_caregiver_role_hard_negative_line(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    role = _caregiver_function(case, context)
    style = _stable_offset(case.document_type, case.note_style, context, role) % 8
    if style == 0:
        b.add("\nZorgteamrol ")
        b.add(role)
        b.add(" is context bij de behandelaar en geen patientberoep.")
    elif style == 1:
        b.add("\nBehandelcontext: ")
        b.add(role)
        b.add(" hoort bij het zorgteam, niet bij de patientidentiteit.")
    elif style == 2:
        b.add("\nDiscipline behandelaar: ")
        b.add(role)
        b.add("; geen Profession-span.")
    elif style == 3:
        b.add("\nFunctievermelding zorgverlener: ")
        b.add(role)
        b.add(" blijft ongemarkeerde klinische context.")
    elif style == 4:
        b.add("\nTeam/discipline: ")
        b.add(role)
        b.add(" is een zorgverlenersrol en geen beroep van de patient.")
    elif style == 5:
        b.add("\nOndertekenrol: ")
        b.add(role)
        b.add("; alleen de naam/ID/contact van de zorgverlener is PII.")
    elif style == 6:
        b.add("\nConsultfunctie ")
        b.add(role)
        b.add(" niet annoteren als Profession.")
    else:
        b.add("\nKlinische rol rond zorgteam = ")
        b.add(role)
        b.add("; geen PII-label.")


MEDICATION_BLOCK_PREFIXES = (
    "Rx-lijst",
    "Thuismedicatie volgens reconciliatie",
    "Medicatieschema",
    "Farmaca/dosissen",
    "Therapie",
    "eVoorschrift-klinisch",
    "Innamecontrole",
    "Apotheeknota",
    "Toedieningslijst",
    "Medicatiebeleid",
    "Geneesmiddelencheck",
    "Ontslagmedicatie",
)
MEDICATION_ROUTES = (
    "p.o.",
    "oraal",
    "s.c.",
    "i.v.",
    "inhalatie",
    "transdermaal",
    "sublinguaal",
    "lokaal",
    "neusspray",
    "oogdruppels",
)
MEDICATION_DOSE_FRAGMENTS = (
    "1-0-1",
    "1 dd 1",
    "2 dd 1",
    "500 mg",
    "10 mg",
    "zo nodig",
    "voor de nacht",
    "ochtend en avond",
    "wekelijks",
    "volgens afbouwschema",
)
MEDICATION_ACTIONS = (
    "verderzetten",
    "tijdelijk pauzeren",
    "herstarten na labo",
    "niet dubbel voorschrijven",
    "therapietrouw navragen",
    "innamemoment vereenvoudigen",
    "bijwerkingen monitoren",
    "interactiecontrole herhalen",
    "dosis niet wijzigen",
    "meegeven op ontslag",
)


def _medication_fragment(
    case: ClinicalCase, medication: str, context: str, offset: int
) -> str:
    seed_context = f"{context}|{medication}|{offset}"
    route = MEDICATION_ROUTES[
        _stable_offset(case.document_type, seed_context, "route")
        % len(MEDICATION_ROUTES)
    ]
    dose = MEDICATION_DOSE_FRAGMENTS[
        _stable_offset(case.note_style, seed_context, "dose")
        % len(MEDICATION_DOSE_FRAGMENTS)
    ]
    action = MEDICATION_ACTIONS[
        _stable_offset(case.department, seed_context, "action")
        % len(MEDICATION_ACTIONS)
    ]
    forms = [
        f"{medication} {dose}",
        f"{medication} {route}",
        f"{medication} {dose} {route}",
        f"{action}: {medication}",
        f"{medication} - {action}",
        f"{medication} ({dose}, {route})",
        f"{medication}; beleid {action}",
        f"{medication} / {dose} / {route}",
    ]
    return forms[
        _stable_offset(case.document_type, case.note_style, seed_context, "form")
        % len(forms)
    ]


def _medication_examples(case: ClinicalCase, context: str, count: int = 3) -> list[str]:
    details = case.medical_details or {}
    medications = [
        str(item).strip()
        for item in case.condition.get("medications", [])
        if str(item).strip()
    ]
    medications.extend(
        medication
        for medication in _catalog_medication_names(case)
        if medication not in medications
    )
    if medications:
        offset = _stable_offset(
            case.document_type, case.note_style, context, "medication-example-offset"
        ) % len(medications)
        medications = medications[offset:] + medications[:offset]
    order_by_medication: dict[str, str] = {}
    for item in details.get("medication_orders", []) or []:
        if not isinstance(item, str) or ":" not in item:
            continue
        medication, order = item.split(":", 1)
        order_by_medication[medication.strip().casefold()] = order.strip()

    rows = []
    for offset, medication in enumerate(medications[:count]):
        order = order_by_medication.get(medication.casefold())
        if order:
            row_options = [
                f"{medication}: {order}",
                f"{medication} - {order}",
                f"{medication} ({order})",
                f"{medication}; schema {order}",
            ]
            row = row_options[
                _stable_offset(case.document_type, context, medication, "order")
                % len(row_options)
            ]
        else:
            row = _medication_fragment(case, medication, context, offset)
        rows.append(row.strip())
    return rows


def _add_medication_hard_negative_block(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    examples = _medication_examples(case, context)
    if not examples:
        return
    style = (
        _stable_offset(case.document_type, case.note_style, context, _condition(case))
        % 12
    )
    title = MEDICATION_BLOCK_PREFIXES[style]
    if style == 0:
        b.add(f"\n{title}: ")
        b.add("; ".join(examples))
    elif style == 1:
        b.add(f"\n{title}: ")
        b.add(" | ".join(examples))
    elif style == 2:
        b.add(f"\n{title}\n")
        for example in examples:
            b.add(f"- {example}\n")
    elif style == 3:
        b.add(f"\n{title}: ")
        for item_index, example in enumerate(examples, start=1):
            if item_index > 1:
                b.add(" / ")
            b.add(f"R{item_index}={example}")
    elif style == 4:
        b.add(f"\n{title}: ")
        b.add(", ".join(examples))
    elif style == 5:
        b.add(f"\n{title}: ")
        for item_index, example in enumerate(examples, start=1):
            if item_index > 1:
                b.add("; ")
            b.add(f"CNK-vrij voorbeeld {item_index}: {example}")
    elif style == 6:
        b.add(f"\n{title}: ")
        b.add("; ".join(f"inname bevestigd voor {example}" for example in examples))
    elif style == 7:
        b.add(f"\n{title}: ")
        b.add("; ".join(f"apotheek vergeleek {example}" for example in examples))
    elif style == 8:
        b.add(f"\n{title}: middel | schema | opmerking = ")
        b.add(" ; ".join(examples))
    elif style == 9:
        b.add(f"\n{title}: ")
        b.add(
            "; ".join(f"{example} blijft klinisch geneesmiddel" for example in examples)
        )
    elif style == 10:
        b.add(f"\n{title}: ")
        b.add("; ".join(f"geen allergie gemeld voor {example}" for example in examples))
    else:
        b.add(f"\n{title}: ")
        b.add("; ".join(f"{example} meegegeven" for example in examples))
    b.add(
        ". Medicatienamen, merknamen, dosissen, routes en innameschema's blijven klinische inhoud."
    )


def _add_medication_action_line(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    medications = [
        str(item).strip()
        for item in case.condition.get("medications", [])
        if str(item).strip()
    ]
    if not medications:
        return
    offset = _stable_offset(
        case.document_type, case.note_style, context, "med-action"
    ) % len(medications)
    rotated = medications[offset:] + medications[:offset]
    selected = rotated[: min(3, len(rotated))]
    style = _stable_offset(case.department, context, "med-action-style") % 6
    if style == 0:
        b.add("\nMedicatieactie: ")
        b.add(
            "; ".join(f"{medication} niet als PII markeren" for medication in selected)
        )
    elif style == 1:
        b.add("\nInteractienazicht: ")
        b.add(", ".join(f"{medication} gecontroleerd" for medication in selected))
    elif style == 2:
        b.add("\nAllergiecheck geneesmiddelen: ")
        b.add(" / ".join(f"geen reactie op {medication}" for medication in selected))
    elif style == 3:
        b.add("\nVoorschriftregels: ")
        b.add(
            "; ".join(
                _medication_fragment(case, medication, context, index)
                for index, medication in enumerate(selected)
            )
        )
    elif style == 4:
        b.add("\nMedicatiehistoriek: ")
        b.add(
            "; ".join(
                f"{medication} reeds gekend in thuisschema" for medication in selected
            )
        )
    else:
        b.add("\nToedieningscontrole: ")
        b.add(
            "; ".join(
                f"{medication} afgevinkt door verpleegkundige"
                for medication in selected
            )
        )
    b.add(".")


def _add_medical_eponym_block(b: SpanBuilder, case: ClinicalCase, context: str) -> None:
    eponyms = _medical_eponym_titles(case)
    if not eponyms:
        return
    style = (
        _stable_offset(case.document_type, case.note_style, context, "medical-eponyms")
        % 6
    )
    selected = eponyms[
        : 3 + (_stable_offset(case.patient.name, context, "medical-eponym-count") % 2)
    ]
    if style == 0:
        b.add("\nEponiemen in klinische terminologie: ")
        b.add("; ".join(selected))
    elif style == 1:
        b.add("\nMedische eponiemen/lookalikes: ")
        b.add(" | ".join(selected))
    elif style == 2:
        b.add("\nDifferentiaal bevat eponiemen: ")
        b.add(", ".join(f"{title} als klinische term" for title in selected))
    elif style == 3:
        b.add("\nEponiem\tContext\n")
        for title in selected:
            b.add(f"{title}\tklinische benaming, geen persoonsnaam\n")
        return
    elif style == 4:
        b.add("\nOnderzoeksnotitie: ")
        b.add("; ".join(f"{title} niet als Name markeren" for title in selected))
    else:
        b.add("\nTerminologie uit eponiemenlijst: ")
        b.add("; ".join(selected))
    b.add(
        ". Deze medische eponiemen blijven klinische inhoud, ook wanneer ze op familienamen lijken."
    )


def _add_substance_use_context_line(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    seed = _stable_offset(
        case.document_type, case.note_style, case.patient.name, context, "substance-use"
    )
    line = SUBSTANCE_USE_CONTEXT_LINES[seed % len(SUBSTANCE_USE_CONTEXT_LINES)]
    b.add("\n")
    b.add(line)
    b.add(" Termen rond tabak en abusus zijn klinische sociale-anamnese, geen PII.")


def _add_catalog_medication_treatment_plan(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    catalog_rows = _catalog_medication_rows(case)
    if not catalog_rows:
        return
    style = (
        _stable_offset(
            case.document_type, case.note_style, context, "catalog-treatment-plan"
        )
        % 5
    )
    heading = _numbered_title(
        case, "Behandelplan medicatie", f"{context}:catalog-treatment-plan"
    )
    if style == 0:
        b.add(f"\n{heading}: ")
        b.add(" | ".join(f"CNK {row['cnk']} {row['name']}" for row in catalog_rows[:3]))
    elif style == 1:
        b.add(f"\n{heading}\n")
        for row_index, row in enumerate(catalog_rows[:4], start=1):
            b.add(
                f"{row_index}. CNK {row['cnk']} - {row['name']} - evalueren volgens schema\n"
            )
    elif style == 2:
        b.add(f"\n{heading}: ")
        b.add(
            "; ".join(
                f"{row['name']} (CNK {row['cnk']}) niet als PII"
                for row in catalog_rows[:3]
            )
        )
    elif style == 3:
        b.add(f"\n{heading}\nMiddel\tCNK\tAfspraak\n")
        for row in catalog_rows[:3]:
            b.add(f"{row['name']}\t{row['cnk']}\tcontrole in thuisschema\n")
    else:
        b.add(f"\n{heading}: ")
        b.add(
            "; ".join(
                f"{row['name']} opnemen in behandelplan" for row in catalog_rows[:3]
            )
        )
    b.add(". Geneesmiddel- en CNK-productregels blijven klinische inhoud.")


def _add_study_protocol_line(
    b: SpanBuilder, case: ClinicalCase, context: str, *, force: bool = False
) -> None:
    protocol_id = str(case.identifiers.get("study_protocol_id", "")).strip()
    protocol_name = str(case.identifiers.get("study_protocol_name", "")).strip()
    if not protocol_id or not protocol_name:
        return
    if (
        not force
        and _stable_offset(
            case.document_type, case.note_style, context, protocol_id, protocol_name
        )
        % 5
        != 0
    ):
        return
    style = (
        _stable_offset(case.patient.name, case.document_type, context, "study-protocol")
        % 4
    )
    if style == 0:
        b.add("\nStudieprotocol: ")
        b.add(protocol_name, "ID:Patient")
        b.add(" / protocol-ID ")
        b.add(protocol_id, "ID:Patient")
        b.add(" is patientgebonden in dit studiedossier.")
    elif style == 1:
        b.add("\nClinical trial: protocolnaam ")
        b.add(protocol_name, "ID:Patient")
        b.add(", protocolnummer ")
        b.add(protocol_id, "ID:Patient")
        b.add(".")
    elif style == 2:
        b.add("\nOnderzoeksdossier ")
        b.add(protocol_id, "ID:Patient")
        b.add(" (")
        b.add(protocol_name, "ID:Patient")
        b.add(") gekoppeld aan deze patient.")
    else:
        b.add("\nProtocolreferentie: naam=")
        b.add(protocol_name, "ID:Patient")
        b.add("; id=")
        b.add(protocol_id, "ID:Patient")
        b.add("; niet verwarren met klinische terminologiecodes.")


DATE_PERIOD_LABELS = {
    "future_appointment": "afspraak",
    "deadline_control": "deadline",
    "recurrence_interval": "herhaalinterval",
    "treatment_duration": "behandelduur",
    "lookback_window": "klachtduur",
    "monitoring_window": "monitoring",
    "pregnancy_duration": "zwangerschapsduur",
}

MEASUREMENT_CONTEXT_HEADINGS = (
    "Recente parameters",
    "Metingen",
    "Meetresultaten",
    "METINGEN",
    "MEETRESULTATEN",
    "Vitale metingen",
    "Objectieve meetresultaten",
)

SUBSTANCE_USE_CONTEXT_LINES = (
    "Sociaalanamnese: tabak actief, alcoholabusus ontkend.",
    "Middelenanamnese: tabak 10 PY; geen andere abusus.",
    "Leefstijl: tabak gestopt, abusus niet weerhouden.",
    "Risicofactoren: tabak ++; voorgeschiedenis van alcoholabusus.",
    "Context middelengebruik: geen tabak, geen medicatie-abusus.",
    "Sociale context: rookstatus tabak wisselend; abusus bespreekbaar.",
    "Verslavingsanamnese: tabak dagelijks, cannabisabusus ontkend.",
    "Anamnese leefgewoonten: Tabak: ex-roker; abusus: nee.",
    "Probleemlijst: tabaksabusus in remissie, alcohol beperkt.",
    "Screening: nicotine-abusus vroeger, tabak nu nihil.",
)


COMMON_DUTCH_ABBREVIATION_VARIANTS = (
    ("IOM", "i.o.m.", "iom", "I.O.M."),
    ("TAV", "t.a.v.", "tav", "T.A.V."),
    ("IVM", "i.v.m.", "ivm", "I.V.M."),
    ("MBT", "m.b.t.", "mbt", "M.B.T."),
    ("ZN", "z.n.", "zn", "Z.N."),
    ("DD", "d.d.", "dd", "D.D."),
    ("OWV", "o.w.v.", "owv", "O.W.V."),
    ("THV", "t.h.v.", "thv", "T.H.V."),
    ("TOV", "t.o.v.", "tov", "T.O.V."),
    ("VNL", "vnl.", "vnl", "VNL."),
    ("EVT", "evt.", "evt", "EVT."),
    ("IKV", "i.k.v.", "ikv", "I.K.V."),
)
COMMON_DUTCH_ABBREVIATION_CONTEXTS = (
    "overleg",
    "verslag",
    "planning",
    "beleid",
    "controle",
    "opname",
    "verwijzing",
    "afspraak",
)
PRE_ARRIVAL_LINES = (
    "Pre-Arrival Type:  Verwezen",
    "Pre Arrival Type:\tVerwezen",
    "pre-arrival type: verwezen",
    "PRE-ARRIVAL TYPE: VERWEZEN",
    "Prearrival type : verwezen door zorgverlener",
)
CAREGIVER_COORDINATION_PATTERNS = (
    "IOM",
    "i.o.m.",
    "iom",
    "I.O.M.",
    "in overleg met",
    "In overleg met",
)
ASSISTANT_HEAD_PATTERNS = (
    "Assistent diensthoofd",
    "assistent diensthoofd",
    "ASSISTENT DIENSTHOOFD",
    "Ass. diensthoofd",
    "ass. diensthoofd",
    "assistent-diensthoofd",
)


def _common_abbreviation_variant(case: ClinicalCase, context: str, index: int) -> str:
    group = COMMON_DUTCH_ABBREVIATION_VARIANTS[
        _stable_offset(
            case.document_type, case.note_style, context, str(index), "abbr-group"
        )
        % len(COMMON_DUTCH_ABBREVIATION_VARIANTS)
    ]
    return group[
        _stable_offset(
            case.patient.name, case.department, context, str(index), "abbr-style"
        )
        % len(group)
    ]


def _add_common_dutch_abbreviation_line(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    style = (
        _stable_offset(
            case.document_type, case.note_style, context, "common-abbreviations"
        )
        % 4
    )
    count = 4 + (
        _stable_offset(case.patient.name, context, "common-abbreviation-count") % 4
    )
    abbreviations = [
        _common_abbreviation_variant(case, context, index) for index in range(count)
    ]
    if style == 0:
        b.add("\nAfkortingen gewone taal: ")
        b.add(", ".join(abbreviations))
        b.add(" blijven context rond de zorgactie.")
        return
    if style == 1:
        b.add("\nKorte notitie: ")
        for index, abbreviation in enumerate(abbreviations):
            if index:
                b.add("; ")
            label = COMMON_DUTCH_ABBREVIATION_CONTEXTS[
                _stable_offset(case.document_type, context, abbreviation, str(index))
                % len(COMMON_DUTCH_ABBREVIATION_CONTEXTS)
            ]
            b.add(f"{abbreviation} {label}")
        b.add(".")
        return
    if style == 2:
        b.add("\nAdmin afk.: ")
        b.add(
            " | ".join(
                f"{abbr}={COMMON_DUTCH_ABBREVIATION_CONTEXTS[index % len(COMMON_DUTCH_ABBREVIATION_CONTEXTS)]}"
                for index, abbr in enumerate(abbreviations)
            )
        )
        b.add(".")
        return
    b.add("\nAfkortingenlijst\n")
    for abbreviation in abbreviations:
        b.add(f"- {abbreviation} contextterm, geen PII\n")


def _add_referral_coordination_block(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    style = (
        _stable_offset(
            case.document_type, case.note_style, context, "referral-coordination"
        )
        % 6
    )
    pre_arrival = PRE_ARRIVAL_LINES[
        _stable_offset(case.patient.name, context, "pre-arrival")
        % len(PRE_ARRIVAL_LINES)
    ]
    iom = CAREGIVER_COORDINATION_PATTERNS[
        _stable_offset(case.caregiver.name, context, "iom")
        % len(CAREGIVER_COORDINATION_PATTERNS)
    ]
    assistant = ASSISTANT_HEAD_PATTERNS[
        _stable_offset(case.secondary_caregiver.name, context, "assistant-head")
        % len(ASSISTANT_HEAD_PATTERNS)
    ]
    b.add("\n")
    if style == 0:
        b.add(pre_arrival)
        b.add("\n")
        b.add(iom)
        b.add(" ")
        _add_caregiver_name(b, case.caregiver.name, f"{context}:iom")
        b.add(" (in overleg met)")
        b.add("\n")
        b.add(assistant)
        b.add(" ")
        _add_caregiver_name(
            b, case.secondary_caregiver.name, f"{context}:assistant_head"
        )
        return
    if style == 1:
        b.add("Pre-Arrival Type:  Verwezen; ")
        b.add(iom)
        b.add(" ")
        _add_caregiver_name(b, case.secondary_caregiver.name, f"{context}:iom")
        b.add("; ")
        b.add(assistant)
        b.add(": ")
        _add_caregiver_name(b, case.caregiver.name, f"{context}:assistant_head")
        return
    if style == 2:
        b.add("Instroom: verwezen. ")
        b.add(iom)
        b.add(" (")
        _add_caregiver_name(b, case.caregiver.name, f"{context}:iom")
        b.add("). ")
        b.add(assistant)
        b.add(" = ")
        _add_caregiver_name(
            b, case.secondary_caregiver.name, f"{context}:assistant_head"
        )
        return
    if style == 3:
        b.add(pre_arrival)
        b.add(" / ")
        b.add(assistant)
        b.add(" ")
        _add_caregiver_name(b, case.caregiver.name, f"{context}:assistant_head")
        b.add(" / ")
        b.add(iom)
        b.add(" ")
        _add_caregiver_name(b, case.secondary_caregiver.name, f"{context}:iom")
        return
    if style == 4:
        b.add("Verwijscontext: ")
        b.add(pre_arrival.split(":", 1)[-1].strip())
        b.add("; ")
        b.add(iom)
        b.add(" zorgverlener ")
        _add_caregiver_name(b, case.caregiver.name, f"{context}:iom")
        b.add("; ")
        b.add(assistant)
        b.add(" zorgverlener ")
        _add_caregiver_name(
            b, case.secondary_caregiver.name, f"{context}:assistant_head"
        )
        return
    b.add("Overleglijn: ")
    b.add(iom)
    b.add(" ")
    _add_caregiver_name(b, case.caregiver.name, f"{context}:iom")
    b.add(" - ")
    b.add(assistant)
    b.add(" ")
    _add_caregiver_name(b, case.secondary_caregiver.name, f"{context}:assistant_head")
    b.add(" - Pre-Arrival Type:  Verwezen")


def _date_period_items(case: ClinicalCase, context: str) -> list[tuple[str, str]]:
    periods = case.date_periods or {}
    items = [
        (key, str(periods[key]).strip())
        for key in DATE_PERIOD_LABELS
        if str(periods.get(key, "")).strip()
    ]
    if not items:
        return []
    offset = _stable_offset(
        case.document_type, case.note_style, context, "date-period-offset"
    ) % len(items)
    return items[offset:] + items[:offset]


def _add_date_period_value(b: SpanBuilder, value: str) -> None:
    b.add(value, None if is_relative_date_hard_negative(value) else "Date")


def _add_relative_period_block(
    b: SpanBuilder, case: ClinicalCase, context: str
) -> None:
    items = _date_period_items(case, context)
    if not items:
        return
    style = (
        _stable_offset(
            case.document_type, case.note_style, context, "date-period-style"
        )
        % 6
    )
    selected = items[
        : 4 + (_stable_offset(case.patient.name, context, "date-period-count") % 3)
    ]
    title = _numbered_title(case, "Relatieve termijnen", f"{context}:date-periods")
    if style == 0:
        b.add(f"\n{title}: ")
        for item_index, (key, value) in enumerate(selected):
            if item_index:
                b.add("; ")
            b.add(DATE_PERIOD_LABELS[key])
            b.add(" ")
            _add_date_period_value(b, value)
            b.add(" gekoppeld aan ")
            b.add(_condition(case))
        b.add(".")
        return
    if style == 1:
        b.add(f"\n{title}\n")
        for key, value in selected:
            b.add("- ")
            _add_date_period_value(b, value)
            b.add(f": {DATE_PERIOD_LABELS[key]} voor {_condition(case)}\n")
        return
    if style == 2:
        b.add("\nPlanning: nieuwe afspraak ")
        future = dict(items).get("future_appointment")
        deadline = dict(items).get("deadline_control")
        recurrence = dict(items).get("recurrence_interval")
        duration = dict(items).get("treatment_duration")
        if future:
            _add_date_period_value(b, future)
            b.add("; verslag afronden ")
        if deadline:
            _add_date_period_value(b, deadline)
            b.add("; opvolging nadien ")
        if recurrence:
            _add_date_period_value(b, recurrence)
            b.add("; therapie ")
        if duration:
            _add_date_period_value(b, duration)
        b.add(".")
        return
    if style == 3:
        b.add(f"\n{title}: ")
        for item_index, (key, value) in enumerate(selected):
            if item_index:
                b.add(" | ")
            b.add(f"{DATE_PERIOD_LABELS[key]}=")
            _add_date_period_value(b, value)
        b.add(". Dit zijn relatieve datumuitdrukkingen.")
        return
    if style == 4:
        b.add("\nTermijnbeleid: klachten bestaan ")
        lookback = dict(items).get("lookback_window")
        monitoring = dict(items).get("monitoring_window")
        future = dict(items).get("future_appointment")
        recurrence = dict(items).get("recurrence_interval")
        if lookback:
            _add_date_period_value(b, lookback)
        b.add("; controle ")
        if monitoring:
            _add_date_period_value(b, monitoring)
        b.add("; afspraak ")
        if future:
            _add_date_period_value(b, future)
        b.add("; daarna ")
        if recurrence:
            _add_date_period_value(b, recurrence)
        b.add(".")
        return
    b.add(f"\n{title}\nTermijn\tActie\n")
    for key, value in selected:
        _add_date_period_value(b, value)
        b.add(f"\t{DATE_PERIOD_LABELS[key]} bij {case.department}\n")


def _lab_rows(case: ClinicalCase) -> list[tuple[str, str, str]]:
    return [(str(a), str(b), str(c)) for a, b, c in case.condition["labs"][:6]]


def _detail_rows(case: ClinicalCase, key: str) -> list[dict]:
    details = case.medical_details or {}
    rows = details.get(key, [])
    return rows if isinstance(rows, list) else []


def _format_detail_rows(rows: list[dict]) -> str:
    formatted = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        value = str(row.get("value", "")).strip()
        unit = str(row.get("unit", "")).strip()
        if name and value:
            formatted.append(" ".join(part for part in (name, value, unit) if part))
    return "; ".join(formatted)


def _format_procedure_snippets(items: list) -> str:
    formatted = []
    for item in items:
        if isinstance(item, dict):
            procedure = str(item.get("procedure", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            status = str(item.get("status", "")).strip()
            if procedure and snippet:
                formatted.append(
                    f"{procedure}: {snippet}" + (f" ({status})" if status else "")
                )
            elif procedure or snippet:
                formatted.append(procedure or snippet)
        elif item:
            formatted.append(str(item))
    return "; ".join(formatted)


def _format_abbreviations(items: list) -> str:
    values = []
    for item in items:
        if isinstance(item, dict):
            value = str(item.get("abbreviation", "")).strip()
        else:
            value = str(item).strip()
        if value and value not in values:
            values.append(value)
    return ", ".join(values)


def _stable_offset(*values: str) -> int:
    text = "|".join(values)
    return sum(
        (position + 1) * ord(character) for position, character in enumerate(text)
    )


def _patient_display_name(
    case: ClinicalCase, context: str, value: str | None = None
) -> str:
    display_name = value or case.patient.name
    if (
        _stable_offset(case.patient.name, case.document_type, context, display_name) % 4
        == 0
    ):
        return display_name.upper()
    return display_name


def _add_patient_name(
    b: SpanBuilder, case: ClinicalCase, context: str, value: str | None = None
) -> None:
    b.add(_patient_display_name(case, context, value), "Name:Patient")


def _add_patient_inverted_name(
    b: SpanBuilder, case: ClinicalCase, context: str, *, lowercase: bool = False
) -> None:
    variants = person_name_variants(case.patient.name)
    value = variants.get("surname_first") or case.patient.name
    if lowercase:
        value = value.lower()
    elif _stable_offset(case.patient.name, case.document_type, context) % 3 == 0:
        value = value.upper()
    b.add(value, "Name:Patient")


def _caregiver_display_name(name: str, context: str) -> str:
    variants = caregiver_name_variants(name)
    forms = [
        name,
        variants.get("first"),
        variants.get("initial_surname"),
        variants.get("surname"),
        variants.get("first_last_initial"),
        variants.get("first_last_initial_nodot"),
        variants.get("initials_compact"),
        variants.get("initials_compact_nodot"),
        variants.get("initials_spaced"),
        variants.get("initials_spaced_nodot"),
        variants.get("initial_surname_nodot"),
    ]
    values = [value for value in forms if value]
    return values[_stable_offset(name, context) % len(values)] if values else name


CAREGIVER_TITLE_VARIANTS = {
    "dr.": (
        "dr.",
        "Dr.",
        "dr",
        "DR",
        "dokter",
        "Dokter",
    ),
    "prof. dr.": (
        "prof. dr.",
        "Prof. dr.",
        "Prof. Dr.",
        "prof dr",
        "prof.dr.",
        "prof.  dr.",
        "prof. dokter",
        "Prof. dokter",
        "professor dr.",
        "Professor dr.",
        "professor dokter",
        "Professor Dokter",
    ),
}


def _caregiver_title_variants(title: str) -> tuple[str, ...]:
    normalized = " ".join(title.strip().casefold().replace("professor", "prof").split())
    if normalized in {
        "prof dr",
        "prof. dr",
        "prof. dr.",
        "prof dokter",
        "prof. dokter",
    }:
        return CAREGIVER_TITLE_VARIANTS["prof. dr."]
    if normalized in {"dr", "dr.", "dokter", "doctor"}:
        return CAREGIVER_TITLE_VARIANTS["dr."]
    stripped = title.strip()
    return (stripped,) if stripped else ()


def _with_caregiver_title(
    display_name: str, title: str, name: str, context: str
) -> str:
    title_variants = _caregiver_title_variants(title)
    if not title_variants:
        return display_name
    seed = _stable_offset(name, context, title, display_name, "caregiver-title")
    if seed % 7 == 0:
        return display_name
    if title_variants == CAREGIVER_TITLE_VARIANTS["dr."] and seed % 13 == 0:
        title_variants = CAREGIVER_TITLE_VARIANTS["prof. dr."]
    title_value = title_variants[(seed // 7) % len(title_variants)]
    placement = (seed // (7 * max(1, len(title_variants)))) % 6
    if placement == 0:
        return f"{title_value} {display_name}"
    if placement == 1:
        return f"{title_value}  {display_name}"
    if placement == 2:
        return f"{title_value}\t{display_name}"
    if placement == 3:
        return f"{display_name}, {title_value}"
    if placement == 4:
        return f"{display_name} - {title_value}"
    return f"{display_name} ({title_value})"


def _add_caregiver_name(
    b: SpanBuilder,
    name: str,
    context: str,
    title: str = "",
) -> None:
    display_name = _caregiver_display_name(name, context)
    display_name = _with_caregiver_title(display_name, title, name, context)
    b.add(display_name, "Name:Caregiver")


def _add_caregiver_first_name(b: SpanBuilder, name: str, context: str) -> None:
    variants = caregiver_name_variants(name)
    value = variants.get("first") or name.split()[0]
    b.add(value, "Name:Caregiver")


def _add_caregiver_inverted_name(b: SpanBuilder, name: str, context: str) -> None:
    variants = caregiver_name_variants(name)
    value = variants.get("surname_first") or name
    if _stable_offset(name, context) % 3 == 0:
        value = value.upper()
    b.add(value, "Name:Caregiver")


def _add_caregiver_internal_phone(
    b: SpanBuilder, case: ClinicalCase, prefix: str
) -> bool:
    phone = case.contact.get("caregiver_internal_phone")
    if not phone:
        return False
    b.add(prefix)
    b.add(phone, "Contactdetails")
    return True


def _add_caregiver_registry(
    b: SpanBuilder, case: ClinicalCase, prefix: str, fallback_descriptor: str
) -> None:
    registry = case.identifiers["caregiver_registry"]
    b.add(prefix)
    if fallback_descriptor and not registry.startswith("RIZIV Nr. "):
        b.add(fallback_descriptor)
    b.add(registry, "ID:Caregiver")


EXECUTION_AUDIT_LINE_PREFIXES = (
    "Uitgevoerd op:",
    "Geregistreerd op:",
    "Laatste validatie:",
    "Elektronisch afgetekend:",
    "Actie geregistreerd:",
    "Gevalideerd door",
    "Audit:",
)


def _audit_date_value(case: ClinicalCase, context: str) -> str:
    seed = _stable_offset(
        "execution_audit_date",
        case.patient.name,
        case.caregiver.name,
        case.document_type,
        context,
    )
    year = 2014 + seed % 12
    month = 1 + (seed // 12) % 12
    day = 1 + (seed // 144) % 28
    style = (seed // 4032) % 3
    if style == 0:
        return f"{day}/{month:02d}/{year}"
    if style == 1:
        return f"{day:02d}/{month:02d}/{year}"
    return f"{day}/{month}/{year}"


def _audit_time_value(case: ClinicalCase, context: str) -> str:
    seed = _stable_offset(
        "execution_audit_time",
        case.patient.name,
        case.caregiver.name,
        case.document_type,
        context,
    )
    hour = 6 + seed % 18
    minute = (seed // 18) % 60
    return f"{hour:02d}:{minute:02d}"


def _add_audit_date_time(b: SpanBuilder, case: ClinicalCase, context: str) -> None:
    b.add(_audit_date_value(case, context), "Date")
    b.add(" ")
    b.add(_audit_time_value(case, context))


def _add_caregiver_audit_name(
    b: SpanBuilder,
    name: str,
    context: str,
    *,
    prefer_inverted_title: bool = False,
) -> None:
    variants = caregiver_name_variants(name)
    if prefer_inverted_title:
        value = variants.get("surname_first_dr") or variants.get("surname_first")
    else:
        values = [
            variants.get("surname_first_dr"),
            variants.get("surname_first"),
            variants.get("initial_surname"),
            name,
        ]
        filtered = [item for item in values if item]
        value = (
            filtered[_stable_offset(name, context, "audit_name") % len(filtered)]
            if filtered
            else name
        )
    b.add(value or name, "Name:Caregiver")


def _add_execution_audit_line(b: SpanBuilder, case: ClinicalCase, context: str) -> None:
    style = (
        _stable_offset(
            case.document_type, case.note_style, case.caregiver.name, context
        )
        % 8
    )
    if style == 0:
        b.add("\nUitgevoerd op:  ")
        _add_audit_date_time(b, case, f"{context}:performed")
        b.add("  door ")
        _add_caregiver_audit_name(
            b, case.caregiver.name, context, prefer_inverted_title=True
        )
        return
    if style == 1:
        b.add("\n")
        _add_caregiver_audit_name(
            b, case.caregiver.name, context, prefer_inverted_title=True
        )
        b.add(" - ")
        _add_audit_date_time(b, case, f"{context}:signature")
        return
    if style == 2:
        b.add("\nGeregistreerd op: ")
        _add_audit_date_time(b, case, f"{context}:registered")
        b.add(" door ")
        _add_caregiver_audit_name(
            b, case.caregiver.name, context, prefer_inverted_title=True
        )
        return
    if style == 3:
        b.add("\nLaatste validatie: ")
        _add_audit_date_time(b, case, f"{context}:validated")
        b.add(" door ")
        _add_caregiver_audit_name(
            b, case.secondary_caregiver.name, context, prefer_inverted_title=True
        )
        return
    if style == 4:
        b.add("\nElektronisch afgetekend: ")
        _add_caregiver_audit_name(
            b, case.caregiver.name, context, prefer_inverted_title=True
        )
        b.add(" - ")
        _add_audit_date_time(b, case, f"{context}:signed")
        return
    if style == 5:
        b.add("\nActie geregistreerd: ")
        _add_audit_date_time(b, case, f"{context}:action")
        b.add(" / uitvoerder ")
        _add_caregiver_audit_name(
            b, case.caregiver.name, context, prefer_inverted_title=True
        )
        return
    if style == 6:
        b.add("\nGevalideerd door ")
        _add_caregiver_audit_name(
            b, case.secondary_caregiver.name, context, prefer_inverted_title=True
        )
        b.add(" op ")
        _add_audit_date_time(b, case, f"{context}:checked")
        return
    b.add("\nAudit: ")
    _add_caregiver_audit_name(
        b, case.caregiver.name, context, prefer_inverted_title=True
    )
    b.add(" / ")
    _add_audit_date_time(b, case, f"{context}:audit")


def _add_birthdate(
    b: SpanBuilder, case: ClinicalCase, *, include_prefix: bool = True
) -> None:
    prefix = (case.birthdate_prefix or "").strip()
    if include_prefix and prefix:
        b.add(prefix)
        b.add(" ")
    b.add(case.birthdate, "Age_Birthdate")


AGE_CONTEXT_INSIDE_SPAN = {"ongeveer", "bijna", "ca.", "ca", "rond", "±", "+/-"}
AGE_FIELD_CONTEXTS = {"leeftijd", "lft", "lft."}


def _add_age_text(
    b: SpanBuilder, case: ClinicalCase, suppress_label_context: bool = False
) -> None:
    context = (case.age_context or "").strip()
    if context in AGE_CONTEXT_INSIDE_SPAN:
        b.add(f"{context} {case.age_text}", "Age_Birthdate")
        return
    if context and not (
        suppress_label_context and context.rstrip(".").casefold() in AGE_FIELD_CONTEXTS
    ):
        b.add(context)
        b.add(" ")
    b.add(case.age_text, "Age_Birthdate")


def _date_supports_time(value: str) -> bool:
    parsed = parse_date_text(value, label="Date")
    return parsed is not None and parsed.precision in {"day", "month_day"}


def _add_date(
    b: SpanBuilder,
    value: str,
    case: ClinicalCase,
    time_key: str | None = None,
    context: str = "",
) -> None:
    b.add(value, "Date")
    if not time_key or not _date_supports_time(value):
        return
    time_value = str((case.date_times or {}).get(time_key, "")).strip()
    if not time_value:
        return
    joiners = [" om ", " rond ", " omstreeks ", " vanaf "]
    b.add(
        joiners[
            _stable_offset(case.document_type, context, time_key, value) % len(joiners)
        ]
    )
    b.add(time_value)


def _overview_subject(case: ClinicalCase, template_index: int) -> str:
    details = case.medical_details or {}
    timeframe = str(details.get("timeframe", "")).strip()
    severity = str(details.get("severity", "")).strip()
    candidates = [
        _condition(case),
        f"{case.department} follow-up",
        f"{timeframe} rond {_condition(case)}" if timeframe else "",
        f"{severity} traject {_condition(case)}" if severity else "",
    ]
    values = [value for value in candidates if value]
    return values[template_index % len(values)] if values else _condition(case)


def _overview_done_action(case: ClinicalCase) -> str:
    rows = _detail_rows(case, "procedure_snippets")
    for row in rows:
        procedure = str(row.get("procedure", "")).strip()
        status = str(row.get("status", "")).strip()
        if procedure:
            return f"{procedure} {status}".strip()
    return "klinische evaluatie afgerond"


def _add_compact_date_overview(b: SpanBuilder, case: ClinicalCase) -> None:
    overview = case.date_overview or {}
    required = {
        "history_start_year",
        "history_review_year",
        "plan_year",
        "completed_date",
        "todo_date",
    }
    if not required.issubset(overview):
        return

    template_index = (
        _stable_offset(case.document_type, case.note_style, _condition(case)) % 3
    )
    subject = _overview_subject(case, template_index)
    if template_index == 0:
        b.add("\n")
        b.add(subject)
        b.add(": ")
        b.add(overview["history_start_year"], "Date")
        b.add(" - eerste vermelding ")
        b.add(_condition(case))
        b.add("; ")
        b.add(overview["history_review_year"], "Date")
        b.add(" - behandeling en medicatie herbekeken; ")
        b.add(overview["plan_year"], "Date")
        b.add(" - verdere opvolging voorzien.")
        return

    if template_index == 1:
        b.add("\n")
        b.add(subject)
        b.add(": ")
        b.add(overview["history_review_year"], "Date")
        b.add(" gedaan: dossier en VG nagekeken; ")
        b.add(overview["completed_date"], "Date")
        b.add(" gedaan: ")
        b.add(_overview_done_action(case))
        b.add("; ")
        b.add(overview["todo_date"], "Date")
        b.add(" te doen: controle en bijsturing.")
        return

    b.add("\n")
    b.add(subject)
    b.add(": ")
    b.add(overview["history_start_year"], "Date")
    b.add(" startprobleem; ")
    b.add(overview["history_review_year"], "Date")
    b.add(" laatste herziening; ")
    b.add(overview["plan_year"], "Date")
    b.add(" gepland vervolgtraject.")


DATE_FOCUS_LABELS = {
    "aanmelding_numeric_slash_long": "aanmelding",
    "telefonisch_numeric_slash_short": "telefonisch contact",
    "opname_numeric_dash_long": "opname",
    "controle_numeric_dash_short": "controle",
    "staal_numeric_dot_long": "staalafname",
    "consult_textual_full": "consult",
    "brief_textual_abbr_dot": "brief",
    "ingreep_textual_hyphen": "ingreep",
    "mdo_weekday_numeric": "MDO",
    "verpleeg_weekday_textual": "verpleegmoment",
    "dagplan_day_month_numeric": "dagplanning",
    "thuiszorg_day_month_textual": "thuiszorg",
    "revalidatie_month_year": "revalidatieblok",
    "antecedenten_month_year_1": "digestief antecedent",
    "antecedenten_month_year_2": "gynaecologisch antecedent",
    "antecedenten_month_year_3": "reumatologisch antecedent",
    "voorgeschiedenis_year_1": "voorgeschiedenis 1",
    "voorgeschiedenis_year_2": "voorgeschiedenis 2",
    "voorgeschiedenis_year_3": "voorgeschiedenis 3",
    "voorgeschiedenis_year_4": "voorgeschiedenis 4",
    "voorgeschiedenis_year_5": "voorgeschiedenis 5",
    "lange_termijn_year_only": "langetermijnplan",
}


def _date_focus_items(focus: dict[str, str]) -> list[tuple[str, str]]:
    return [(key, str(focus[key])) for key in DATE_FOCUS_LABELS if key in focus]


def _date_focus_variant(
    case: ClinicalCase, focus: dict[str, str], salt: str, modulo: int
) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    style = case.date_focus_style if isinstance(case.date_focus_style, int) else 0
    return (
        style
        + _stable_offset(
            case.document_type,
            case.department,
            _condition(case),
            case.patient.name,
            salt,
            "|".join(str(value) for value in focus.values()),
        )
    ) % modulo


def _date_focus_pick(
    values: list[str], case: ClinicalCase, focus: dict[str, str], salt: str
) -> str:
    return values[_date_focus_variant(case, focus, salt, len(values))]


def _date_focus_style_index(
    case: ClinicalCase, focus: dict[str, str], salt: str, modulo: int
) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    if isinstance(case.date_focus_style, int):
        return case.date_focus_style % modulo
    return _date_focus_variant(case, focus, salt, modulo)


def _date_focus_style_pick(
    values: list[str], case: ClinicalCase, focus: dict[str, str], salt: str
) -> str:
    return values[_date_focus_style_index(case, focus, salt, len(values))]


def _date_focus_clock(case: ClinicalCase, key: str) -> str:
    seed = _stable_offset(case.document_type, case.note_style, _condition(case), key)
    hour = 7 + seed % 11
    minute = (seed // 11) % 60
    second = ((seed // 660) % 4) * 15
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _add_parenthesized_date(
    b: SpanBuilder,
    value: str,
    *,
    trailing_time: str | None = None,
) -> None:
    b.add("(")
    b.add(value, "Date")
    if trailing_time:
        b.add(" ")
        b.add(trailing_time)
    b.add(")")


def _add_focus_remainder(
    b: SpanBuilder,
    focus: dict[str, str],
    used_keys: set[str],
    prefix: str,
    *,
    case: ClinicalCase | None = None,
) -> None:
    remaining = [
        (key, value) for key, value in _date_focus_items(focus) if key not in used_keys
    ]
    if not remaining:
        return
    b.add(prefix)
    for item_index, (key, value) in enumerate(remaining):
        if item_index:
            b.add("; ")
        b.add(DATE_FOCUS_LABELS.get(key, key.replace("_", " ")))
        b.add(" ")
        b.add(value, "Date")
        if case is not None:
            b.add(" (")
            b.add(_timeline_clinical_note(key, case, focus, item_index))
            b.add(")")
    b.add(".")


TIMELINE_LABELS = {
    "aanmelding_numeric_slash_long": [
        "eerste contact",
        "aanmelding via dossier",
        "start observatie",
        "triage-ingang",
    ],
    "telefonisch_numeric_slash_short": [
        "telefoonnotitie",
        "familiecontact",
        "voorbespreking",
        "intake op afstand",
    ],
    "opname_numeric_dash_long": [
        "opnamevenster",
        "klinische opname",
        "dagzaalregistratie",
        "start verblijf",
    ],
    "controle_numeric_dash_short": [
        "controlepunt",
        "tussentijdse check",
        "herbeoordeling",
        "zorgteamcontrole",
    ],
    "staal_numeric_dot_long": [
        "staal verwerkt",
        "labo-afname",
        "biobanknotitie",
        "technische validatie",
    ],
    "consult_textual_full": [
        "raadpleging",
        "consultatie",
        "artscontact",
        "klinische evaluatie",
    ],
    "brief_textual_abbr_dot": [
        "brief klaargezet",
        "verslagversie",
        "correspondentie",
        "ontwerpbrief",
    ],
    "ingreep_textual_hyphen": [
        "procedure",
        "interventie",
        "technische handeling",
        "behandelkamer",
    ],
    "mdo_weekday_numeric": [
        "teamoverleg",
        "MDO",
        "beslisoverleg",
        "bespreking zorgpad",
    ],
    "verpleeg_weekday_textual": [
        "verpleegkundige evaluatie",
        "zorgobservatie",
        "afdelingsnota",
        "parametercontrole",
    ],
    "dagplan_day_month_numeric": [
        "dagplan",
        "toedieningsdag",
        "planning bed",
        "controlelijst",
    ],
    "thuiszorg_day_month_textual": [
        "thuiszorgmoment",
        "mantelzorgafspraak",
        "navraag thuis",
        "zorglijn thuis",
    ],
    "revalidatie_month_year": [
        "revalidatieblok",
        "therapieperiode",
        "trainingsfase",
        "opvolgmaand",
    ],
    "antecedenten_month_year_1": [
        "digestieve voorgeschiedenis",
        "abdominaal antecedent",
        "endoscopische follow-up",
        "maag-darm evaluatie",
    ],
    "antecedenten_month_year_2": [
        "operatief antecedent",
        "gynaecologische voorgeschiedenis",
        "postoperatieve notitie",
        "bekkenchirurgie",
    ],
    "antecedenten_month_year_3": [
        "reumatologisch antecedent",
        "pijncluster",
        "spier-peesklachten",
        "drukpuntenhistoriek",
    ],
    "voorgeschiedenis_year_1": [
        "oudste episode",
        "eerste gekend antecedent",
        "vroeg dossierjaar",
        "oude ingreep",
    ],
    "voorgeschiedenis_year_2": [
        "tweede antecedent",
        "latere complicatie",
        "heropflakkering",
        "beeldvorming historiek",
    ],
    "voorgeschiedenis_year_3": [
        "chronische klachten",
        "diagnostisch spoor",
        "negatieve onderzoeken",
        "recidiefklacht",
    ],
    "voorgeschiedenis_year_4": [
        "parallel probleem",
        "metabole notitie",
        "controlebeeld",
        "nevendiagnose",
    ],
    "voorgeschiedenis_year_5": [
        "handchirurgie",
        "neuropathisch nazicht",
        "orthopedisch dossier",
        "pijnnazorg",
    ],
    "lange_termijn_year_only": [
        "langetermijnplan",
        "verdere follow-up",
        "toekomstlijn",
        "jaardoel",
    ],
}


def _timeline_label(key: str, case: ClinicalCase, focus: dict[str, str]) -> str:
    values = TIMELINE_LABELS.get(key) or [
        DATE_FOCUS_LABELS.get(key, key.replace("_", " "))
    ]
    return _date_focus_pick(values, case, focus, f"timeline-label:{key}")


TIMELINE_EXTRA_DISEASES = [
    "pneumonie",
    "pleura-empyeem",
    "longembolie",
    "niersteenlijden",
    "synovitis",
    "maagulcus",
    "leversteatose",
    "mesenteriale panniculitis",
    "polymyalgia rheumatica",
    "arteriele hypertensie",
    "diabetes mellitus",
    "astma",
    "COPD",
    "voorkamerfibrillatie",
    "migraine",
    "jicht",
    "osteoporose",
    "anemie",
    "hypothyreoidie",
    "chronische diarree",
    "rotator-cufflijden",
    "tendinopathie gluteaal",
    "diplopie-episode",
    "urineweginfectie",
    "galsteenlijden",
    "diverticulitis",
    "eczeem",
    "psoriasis",
    "slaapapneu",
    "perifere neuropathie",
]


TIMELINE_FALLBACK_SYMPTOMS = [
    "koorts",
    "dyspnoe",
    "hoest",
    "buikpijn",
    "diarree",
    "misselijkheid",
    "gewichtsverlies",
    "nachtzweten",
    "vermoeidheid",
    "pijn op de borst",
    "duizeligheid",
    "diplopie",
    "paresthesieen",
    "gewrichtspijn",
    "jeuk",
    "mictieklachten",
    "oedeem",
    "wazig zicht",
]


def _timeline_clinical_note(
    key: str,
    case: ClinicalCase,
    focus: dict[str, str],
    item_index: int,
) -> str:
    condition = _condition(case)
    symptoms = [
        str(symptom) for symptom in case.condition.get("symptoms", []) if symptom
    ]
    if not symptoms:
        symptoms = TIMELINE_FALLBACK_SYMPTOMS
    symptom_1 = symptoms[item_index % len(symptoms)]
    symptom_2 = symptoms[(item_index + 1) % len(symptoms)]
    symptom_3 = symptoms[(item_index + 2) % len(symptoms)]
    details = case.medical_details or {}
    comorbidities = [str(item) for item in details.get("comorbidities", []) if item]
    comorbidity = (
        comorbidities[item_index % len(comorbidities)]
        if comorbidities
        else "geen extra comorbiditeit genoteerd"
    )
    disease = _date_focus_pick(
        TIMELINE_EXTRA_DISEASES, case, focus, f"timeline-disease:{key}:{item_index}"
    )
    disease_2 = _date_focus_pick(
        TIMELINE_EXTRA_DISEASES, case, focus, f"timeline-disease2:{key}:{item_index}"
    )

    if key.startswith("voorgeschiedenis") or key.startswith("antecedenten"):
        templates = [
            "VG {disease}; later {condition} met {symptom_1}",
            "{disease} en {disease_2} in oudere probleemlijst",
            "antecedent {disease}; klachtpatroon {symptom_1}/{symptom_2}",
            "{disease} opgevolgd, nadien {condition} vermeld",
            "oude episode {disease}, restklacht {symptom_1}",
        ]
    elif key in {
        "staal_numeric_dot_long",
        "consult_textual_full",
        "mdo_weekday_numeric",
    }:
        templates = [
            "{condition}: {symptom_1}, {symptom_2}; labo/beleid gekoppeld",
            "herevaluatie {condition} bij {symptom_1} en {symptom_2}",
            "{disease} meegewogen, hoofdprobleem {condition}",
            "bespreking rond {condition}; symptoomfocus {symptom_1}",
            "{condition} met {comorbidity}; {symptom_3} apart genoteerd",
        ]
    elif key in {
        "dagplan_day_month_numeric",
        "thuiszorg_day_month_textual",
        "revalidatie_month_year",
        "lange_termijn_year_only",
    }:
        templates = [
            "planning voor {condition}, opvolgen van {symptom_1}",
            "nazorg bij {condition}; alarmsymptoom {symptom_2}",
            "functionele impact door {symptom_1} en {symptom_2}",
            "revalidatie/thuisspoor na {condition}, ook {disease} in VG",
            "langetermijnbewaking: {condition}, {comorbidity}",
        ]
    else:
        templates = [
            "presentatie {condition}: {symptom_1} en {symptom_2}",
            "{condition} vermoed bij {symptom_1}/{symptom_2}",
            "{disease} in differentiaal, actueel {condition}",
            "klachtencluster {symptom_1}, {symptom_2}, {symptom_3}",
            "{condition} met {comorbidity}",
        ]
    template = templates[
        _date_focus_variant(
            case, focus, f"timeline-clinical:{key}:{item_index}", len(templates)
        )
    ]
    return template.format(
        condition=condition,
        symptom_1=symptom_1,
        symptom_2=symptom_2,
        symptom_3=symptom_3,
        comorbidity=comorbidity,
        disease=disease,
        disease_2=disease_2,
    )


def _date_focus_context_sentence(
    case: ClinicalCase, focus: dict[str, str], salt: str
) -> str:
    symptoms = [
        str(symptom) for symptom in case.condition.get("symptoms", []) if symptom
    ]
    if not symptoms:
        symptoms = TIMELINE_FALLBACK_SYMPTOMS[:3]
    disease = _date_focus_pick(
        TIMELINE_EXTRA_DISEASES, case, focus, f"context-disease:{salt}"
    )
    note = _timeline_clinical_note(
        "consult_textual_full", case, focus, _date_focus_variant(case, focus, salt, 9)
    )
    templates = [
        "{condition}: {symptoms}; {note}",
        "Klinisch spoor {condition} met {symptoms}; {disease} blijft in VG/differentiaal.",
        "Context bij deze datums: {condition}, klachten {symptoms}; {note}.",
        "Datumblok rond {condition}; symptomen {symptoms}; extra aandacht voor {disease}.",
    ]
    template = templates[
        _date_focus_variant(case, focus, f"context-template:{salt}", len(templates))
    ]
    return template.format(
        condition=_condition(case),
        symptoms=", ".join(symptoms[:3]),
        disease=disease,
        note=note,
    )


def _timeline_item_sets(
    focus: dict[str, str], variant: int
) -> list[list[tuple[str, str]]]:
    items = _date_focus_items(focus)
    if variant == 0:
        return [items]
    if variant == 1:
        return [
            [
                (key, value)
                for key, value in items
                if key.startswith("voorgeschiedenis") or key.startswith("antecedenten")
            ],
            [
                (key, value)
                for key, value in items
                if not (
                    key.startswith("voorgeschiedenis") or key.startswith("antecedenten")
                )
            ],
        ]
    if variant == 2:
        return [items[::2], items[1::2]]
    if variant == 3:
        return [items[:7], items[7:14], items[14:]]
    return [list(reversed(items))]


def _add_timeline_value(
    b: SpanBuilder, value: str, *, with_parentheses: bool = False
) -> None:
    if with_parentheses:
        _add_parenthesized_date(b, value)
    else:
        b.add(value, "Date")


def _add_compact_date_focus_block(
    b: SpanBuilder, case: ClinicalCase, focus: dict[str, str]
) -> None:
    variant = _date_focus_style_index(case, focus, "compact-timeline-shape", 5)
    title = _date_focus_style_pick(
        [
            "Datumreconciliatie",
            "Chronologie uit het dossier",
            "Datums en acties",
            "Zorgpad met datumsporen",
            "Kruiscontrole datums",
            "Klinische tijdlijn",
        ],
        case,
        focus,
        "compact-title",
    )

    if variant == 0:
        b.add("\n")
        b.add(title)
        b.add(": ")
        separator = _date_focus_pick(
            ["; ", " | ", " / "], case, focus, "compact-separator"
        )
        for item_index, (key, value) in enumerate(_date_focus_items(focus)):
            if item_index:
                b.add(separator)
            b.add(_timeline_label(key, case, focus))
            b.add(" ")
            b.add(value, "Date")
            b.add(" (")
            b.add(_timeline_clinical_note(key, case, focus, item_index))
            b.add(")")
        b.add(".")
        return

    if variant == 1:
        b.add("\n")
        b.add(title)
        b.add("\n")
        bullet = _date_focus_pick(["- ", "* ", "• "], case, focus, "timeline-bullet")
        for group_index, group in enumerate(_timeline_item_sets(focus, variant)):
            if not group:
                continue
            b.add("Historiek\n" if group_index == 0 else "Actueel en planning\n")
            for item_index, (key, value) in enumerate(group):
                b.add(bullet)
                b.add(_timeline_label(key, case, focus))
                b.add(": ")
                _add_timeline_value(
                    b, value, with_parentheses=key.startswith("voorgeschiedenis")
                )
                b.add(" - ")
                b.add(
                    _timeline_clinical_note(
                        key, case, focus, item_index + group_index * 10
                    )
                )
                b.add("\n")
        return

    if variant == 2:
        b.add("\n")
        b.add(title)
        b.add(" - parallelle notities\n")
        for lane_index, group in enumerate(
            _timeline_item_sets(focus, variant), start=1
        ):
            if not group:
                continue
            b.add(f"Lijn {lane_index}: ")
            for item_index, (key, value) in enumerate(group):
                if item_index:
                    b.add(" -> ")
                b.add(value, "Date")
                b.add(" ")
                b.add(_timeline_label(key, case, focus))
                b.add(" (")
                b.add(
                    _timeline_clinical_note(
                        key, case, focus, item_index + lane_index * 10
                    )
                )
                b.add(")")
            b.add("\n")
        return

    if variant == 3:
        headings = ["Voortraject", "Klinische episode", "Nazorg en historiek"]
        b.add("\n")
        b.add(title)
        b.add("\n")
        for heading, group in zip(headings, _timeline_item_sets(focus, variant)):
            if not group:
                continue
            b.add(heading)
            b.add(": ")
            for item_index, (key, value) in enumerate(group):
                if item_index:
                    b.add(", ")
                b.add(_timeline_label(key, case, focus))
                b.add("=")
                b.add(value, "Date")
                b.add(" [")
                b.add(_timeline_clinical_note(key, case, focus, item_index))
                b.add("]")
            b.add("\n")
        return

    b.add("\n")
    b.add(title)
    b.add(" (laatste naar oudste)\n")
    for item_index, (key, value) in enumerate(_timeline_item_sets(focus, variant)[0]):
        b.add(value, "Date")
        b.add(" - ")
        b.add(_timeline_label(key, case, focus))
        b.add("; ")
        b.add(_timeline_clinical_note(key, case, focus, item_index))
        b.add("\n")


def _add_screening_date_focus_block(
    b: SpanBuilder, case: ClinicalCase, focus: dict[str, str]
) -> None:
    screening_date = str(
        focus.get("telefonisch_numeric_slash_short") or next(iter(focus.values()))
    )
    titles = [
        "Aandachtsscreening delirium",
        "Risico-inschatting bij opname",
        "Kwetsbaarheidsscreening",
        "Observatieblad aandachtspunten",
        "Preventieve verpleegscreening",
        "Opnamecheck verwardheidsrisico",
    ]
    row_sets = [
        (
            "Delirium ",
            [
                ("cognitieve kwetsbaarheid", "nee"),
                ("stemming opvallend afwijkend", "nee"),
                ("acute ziektelast", "nee"),
                ("middelengebruik als risico", "nee"),
                ("recent ingrijpen of trauma", "nee"),
                ("fractuurrisico", "nee"),
                ("risicoscore", "0"),
                ("interpretatie", "geen verhoogd risico"),
            ],
        ),
        (
            "Verwardheid ",
            [
                ("orientatie wisselend", "nee"),
                ("aandacht niet vol te houden", "nee"),
                ("slaap-waakritme verstoord", "nee"),
                ("hallucinaties gemeld", "nee"),
                ("acute infectietekenen", "nee"),
                ("pijn onbehandeld", "nee"),
                ("score", "0"),
                ("beleid", "routine observatie"),
            ],
        ),
        (
            "Kwetsbaarheid ",
            [
                ("geheugenklacht", "nee"),
                ("val in voorbije maand", "nee"),
                ("polyfarmacie actief risico", "nee"),
                ("dehydratatie vermoed", "nee"),
                ("zintuiglijke beperking relevant", "nee"),
                ("mantelzorg overbelast", "nee"),
                ("totaal", "1"),
                ("samenvatting", "geen bijkomend alarm"),
            ],
        ),
        (
            "Observatie ",
            [
                ("CAM kenmerk acuut begin", "afwezig"),
                ("CAM aandachtsstoornis", "afwezig"),
                ("denken ongeorganiseerd", "afwezig"),
                ("bewustzijn veranderd", "afwezig"),
                ("nachtelijke onrust", "niet gezien"),
                ("middelenonttrekking", "niet vermoed"),
                ("score", "0/4"),
                ("duiding", "negatieve screening"),
            ],
        ),
        (
            "Opnamecheck ",
            [
                ("bril of hoorapparaat nodig", "nee"),
                ("urineretentie risico", "nee"),
                ("stoelgangsimpact", "nee"),
                ("koorts of hypoxie", "nee"),
                ("recent anesthesiecontact", "nee"),
                ("fractuur of immobilisatie", "nee"),
                ("risicokleur", "groen"),
                ("actie", "hercontrole bij verandering"),
            ],
        ),
    ]
    title = _date_focus_style_pick(titles, case, focus, "screening-title")
    prefix, rows = row_sets[
        _date_focus_style_index(case, focus, "screening-rows", len(row_sets))
    ]
    b.add("\n")
    b.add(title)
    b.add(":\n")
    b.add("Context: ")
    b.add(_date_focus_context_sentence(case, focus, "screening-context"))
    b.add("\n")
    for label, outcome in rows:
        b.add(prefix)
        b.add(label)
        b.add(": ")
        b.add(outcome)
        b.add(" ")
        _add_parenthesized_date(b, screening_date)
        b.add("\n")
    _add_focus_remainder(
        b,
        focus,
        {"telefonisch_numeric_slash_short"},
        _date_focus_style_pick(
            [
                "Aanvullende datumankers: ",
                "Losse registratiedatums: ",
                "Nog vergeleken datums: ",
                "Andere observatiemomenten: ",
            ],
            case,
            focus,
            "screening-remainder",
        ),
        case=case,
    )


def _add_functional_date_focus_block(
    b: SpanBuilder, case: ClinicalCase, focus: dict[str, str]
) -> None:
    status_date = str(
        focus.get("aanmelding_numeric_slash_long") or next(iter(focus.values()))
    )
    status_time = _date_focus_clock(case, "functional-status")
    titles = [
        "Functionele status",
        "ADL-status bij opname",
        "Zelfredzaamheid en continentie",
        "Thuissituatie functioneren",
        "Observaties dagelijkse zorg",
        "Mobiliteits- en zorgprofiel",
    ]
    row_sets = [
        [
            ("Stoelgang", "continent"),
            ("Urine", "continent"),
            ("Wassen", "zelfstandig"),
            ("Kleden", "zelfstandig"),
            ("Toiletbezoek", "zelfstandig"),
            ("Stappen", "zelfstandig"),
            ("Eten en drinken", "onafhankelijk"),
            ("Extra hulp", "geen bijkomende hulpvraag"),
        ],
        [
            ("Transfer bed-stoel", "zonder hulp"),
            ("Traplopen", "met leuning"),
            ("Hulpmiddel stappen", "geen"),
            ("Douche", "toezicht volstaat"),
            ("Maaltijd klaarmaken", "familie helpt"),
            ("Medicatiebeheer", "weekdoos"),
            ("Nachtelijke oproepen", "niet gemeld"),
            ("Thuiszorgvraag", "te herbekijken"),
        ],
        [
            ("Blaascontrole", "continent"),
            ("Darmcontrole", "regelmatig"),
            ("Zelfzorg bovenlichaam", "zelfstandig"),
            ("Zelfzorg onderlichaam", "lichte hulp"),
            ("Schoeisel", "hulp bij veters"),
            ("Wandelafstand", "kamer-gang"),
            ("Valangst", "beperkt aanwezig"),
            ("Mantelzorg", "beschikbaar"),
        ],
        [
            ("Ochtendtoilet", "alleen gestart"),
            ("Aankleden", "traag maar zelfstandig"),
            ("Toilettransfer", "veilig"),
            ("Eten", "volledig zelfstandig"),
            ("Drinken", "herinnering nodig"),
            ("Mobiliteit kamer", "stabiel"),
            ("Rolstoelgebruik", "niet nodig"),
            ("Beloproepsysteem", "begrepen"),
        ],
        [
            ("ADL-score wassen", "4/4"),
            ("ADL-score kleden", "3/4"),
            ("ADL-score toilet", "4/4"),
            ("ADL-score transfer", "4/4"),
            ("ADL-score continentie", "4/4"),
            ("ADL-score voeding", "4/4"),
            ("Risico hulptekort", "laag"),
            ("Plan thuis", "ongewijzigd"),
        ],
    ]
    rows = row_sets[
        _date_focus_style_index(case, focus, "functional-rows", len(row_sets))
    ]
    title = _date_focus_style_pick(titles, case, focus, "functional-title")
    b.add("\n")
    b.add(title)
    b.add(":\n")
    b.add("Klinische context: ")
    b.add(_date_focus_context_sentence(case, focus, "functional-context"))
    b.add("\n")
    for label, outcome in rows:
        b.add(label)
        b.add(": ")
        b.add(outcome)
        b.add(" ")
        _add_parenthesized_date(b, status_date, trailing_time=status_time)
        b.add("\n")
    _add_focus_remainder(
        b,
        focus,
        {"aanmelding_numeric_slash_long"},
        _date_focus_style_pick(
            [
                "Andere meetmomenten: ",
                "Controledata bij ADL: ",
                "Aanvullende zorgmomenten: ",
                "Datumsporen rond functioneren: ",
            ],
            case,
            focus,
            "functional-remainder",
        ),
        case=case,
    )


def _history_labels(case: ClinicalCase, focus: dict[str, str]) -> list[str]:
    details = case.medical_details or {}
    values = [_condition(case)]
    values.extend(str(item) for item in details.get("comorbidities", [])[:3])
    values.extend(
        str(item.get("procedure", ""))
        for item in _detail_rows(case, "procedure_snippets")[:2]
    )
    values = [value for value in values if value]
    fallback = [
        "pneumonie doorgemaakt",
        "niersteenepisode",
        "tromboseverdenking",
        "orthopedische klacht",
        "cardiovasculaire risicofactor",
        "chronische opvolging",
        "endoscopische controle zonder alarmsignaal",
        "postoperatieve pijnnazorg",
        "neurologische observatie bij diplopie",
        "reumatologische klachtencluster",
        "leververvetting in follow-up",
        "abdominaal littekentraject",
        "handchirurgie met gevoelsklachten",
        "psychosociale kwetsbaarheid",
    ]
    values.extend(item for item in fallback if item not in values)
    offset = _date_focus_variant(case, focus, "history-label-rotation", len(values))
    rotated = values[offset:] + values[:offset]
    return rotated[:6]


YEAR_FIRST_HISTORY_ROW_POOLS = [
    [
        "strabismecorrectie, vooraf esotropiebeeld",
        "urologische poliepingreep, nadien controletraject",
        "pneumonie met pleurale verwikkeling",
    ],
    [
        "galblaasoperatie, nadien buikklachten in opvolging",
        "handingreep links met neuropathische napijn",
        "periode van diplopie, inflammatieparameters verhoogd",
    ],
    [
        "abdominale revisie met doorgemaakte embolische verwikkeling",
        "chronische diarree, endoscopie en biopten geruststellend",
        "leversteatose bij beeldvorming",
    ],
    [
        "orthopedische correctie, revalidatie nadien traag",
        "longinfectie met langdurig herstel",
        "pijnsyndroom na distale ingreep",
    ],
    [
        "blaasletseltje endoscopisch behandeld",
        "pleurale infectie met drainagevermelding",
        "longembolie in oud ontslagverslag",
    ],
    [
        "niersteenepisode met spontane passage",
        "synovitis gevolgd door infiltratietraject",
        "chronische rugklacht conservatief opgevolgd",
    ],
    [
        "oogstandcorrectie in jeugdverslag beschreven",
        "recidiverende luchtweginfectie in oude VG",
        "trombo-embolische gebeurtenis na opname",
    ],
]


def _year_first_history_rows(case: ClinicalCase, focus: dict[str, str]) -> list[str]:
    pool = YEAR_FIRST_HISTORY_ROW_POOLS[
        _date_focus_style_index(
            case, focus, "year-first-history-pool", len(YEAR_FIRST_HISTORY_ROW_POOLS)
        )
    ]
    return pool


def _add_history_date_focus_block(
    b: SpanBuilder, case: ClinicalCase, focus: dict[str, str]
) -> None:
    year_keys = [
        "voorgeschiedenis_year_1",
        "voorgeschiedenis_year_2",
        "voorgeschiedenis_year_3",
        "voorgeschiedenis_year_4",
        "voorgeschiedenis_year_5",
        "lange_termijn_year_only",
    ]
    title = _date_focus_style_pick(
        [
            "Probleemhistoriek",
            "VG in jaartallen",
            "Chronische voorgeschiedenis",
            "Historische probleemlijst",
            "Antecedenten per dossierjaar",
        ],
        case,
        focus,
        "history-title",
    )
    bullet = _date_focus_style_pick(["- ", "• ", "* "], case, focus, "history-bullet")
    b.add("\n")
    b.add(title)
    b.add(":\n")
    used_year_keys: set[str] = set()
    for label, key in zip(_year_first_history_rows(case, focus), year_keys[:3]):
        value = focus.get(key)
        if not value:
            continue
        b.add(bullet)
        b.add(str(value), "Date")
        b.add(" ")
        b.add(label)
        b.add("\n")
        used_year_keys.add(key)

    remaining_year_keys = [key for key in year_keys if key not in used_year_keys]
    row_style_sequences = [
        ["parenthesized", "colon", "dash"],
        ["colon", "parenthesized", "colon"],
        ["dash", "colon", "parenthesized"],
        ["parenthesized", "dash", "colon"],
        ["colon", "parenthesized", "dash"],
    ]
    row_styles = row_style_sequences[
        _date_focus_style_index(
            case, focus, "history-row-style", len(row_style_sequences)
        )
    ]
    for row_index, (label, key) in enumerate(
        zip(_history_labels(case, focus), remaining_year_keys)
    ):
        value = focus.get(key)
        if not value:
            continue
        b.add(bullet)
        row_style = row_styles[row_index % len(row_styles)]
        if row_style == "colon":
            b.add(str(value), "Date")
            b.add(": ")
            b.add(label)
        elif row_style == "dash":
            b.add(str(value), "Date")
            b.add(" - ")
            b.add(label)
        else:
            b.add(label)
            b.add(" ")
            _add_parenthesized_date(b, str(value))
        b.add("\n")
        used_year_keys.add(key)

    _add_focus_remainder(
        b,
        focus,
        used_year_keys,
        _date_focus_style_pick(
            [
                "Recente datumankers: ",
                "Aanvullende episode-datums: ",
                "Niet-jaarlijkse datumverwijzingen: ",
                "Losse historische datums: ",
            ],
            case,
            focus,
            "history-remainder",
        ),
        case=case,
    )


def _add_categorized_antecedents_date_focus_block(
    b: SpanBuilder,
    case: ClinicalCase,
    focus: dict[str, str],
) -> None:
    used_keys: set[str] = set()
    title = _date_focus_style_pick(
        [
            "Antecedentenoverzicht",
            "Persoonlijke voorgeschiedenis",
            "Gegroepeerde antecedenten",
            "Historiek per stelsel",
            "Samengevatte VG",
        ],
        case,
        focus,
        "antecedent-title",
    )
    digestive_heading = _date_focus_style_pick(
        [
            "Abdominaal en digestief",
            "Gastro-enterologisch traject",
            "Buikheelkunde en digestieve opvolging",
            "Maag-darm en lever",
            "Digestieve VG met jaartallen",
        ],
        case,
        focus,
        "antecedent-digestive-heading",
    )
    other_heading = _date_focus_style_pick(
        [
            "Andere voorgeschiedenis",
            "Niet-digestieve antecedenten",
            "Overige medische historiek",
            "Systeemoverschrijdende problemen",
            "Psychisch, vasculair en bewegingsstelsel",
        ],
        case,
        focus,
        "antecedent-other-heading",
    )
    bullet = _date_focus_style_pick(
        ["- ", "• ", "· "], case, focus, "antecedent-bullet"
    )
    b.add("\n")
    b.add(title)
    b.add("\n")
    b.add(digestive_heading)
    b.add("\n")
    b.add(bullet)
    b.add(
        _date_focus_pick(
            [
                "ulcuslijden in voorgeschiedenis",
                "episodische epigastrische klachten",
                "maagpathologie zonder recente bloeding",
                "oud ulcusdossier met medicamenteuze opvolging",
            ],
            case,
            focus,
            "antecedent-undated-digestive",
        )
    )
    b.add("\n")

    key = "voorgeschiedenis_year_1"
    if key in focus:
        b.add(bullet)
        b.add("+ ")
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "galblaasoperatie met uitgebreide abdominale revisie; nadien embolisch event vermeld",
                    "abdominale heelkunde met langere opname; later trombo-embolische verwikkeling genoteerd",
                    "cholecystectomie en darmresectie vermeld in oud verslag",
                    "buikoperatie met postoperatieve longembolie in dossierhistoriek",
                ],
                case,
                focus,
                "antecedent-year1",
            )
        )
        b.add("\n")
        used_keys.add(key)

    key = "voorgeschiedenis_year_2"
    if key in focus:
        b.add(bullet)
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "inflammatoir mesenteriaal beeld op radiologie",
                    "vetweefselinflammatie mesenteriaal beschreven",
                    "radiologische controle voor mesenteriale afwijking",
                    "buikpijntraject met mesenteriale densiteitsverandering",
                ],
                case,
                focus,
                "antecedent-year2",
            )
        )
        b.add("\n")
        used_keys.add(key)

    key = "voorgeschiedenis_year_3"
    if key in focus:
        b.add(bullet)
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "langdurige diarree, endoscopie en biopten zonder alarmsignaal",
                    "chronisch stoelgangsprobleem, scopies zonder duidelijke verklaring",
                    "gastro- en coloscopie geruststellend bij aanhoudende diarree",
                    "diarree-evaluatie met negatieve histologie",
                ],
                case,
                focus,
                "antecedent-year3",
            )
        )
        b.add("\n")
        used_keys.add(key)

    key = "voorgeschiedenis_year_4"
    if key in focus:
        b.add(bullet)
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "steatosebeeld bij leveropvolging",
                    "leververvetting op echografie",
                    "metabool leverbeeld zonder cholestase",
                    "hepatische steatose in opvolgnota",
                ],
                case,
                focus,
                "antecedent-year4",
            )
        )
        b.add("\n")
        used_keys.add(key)

    key = "antecedenten_month_year_1"
    if key in focus:
        b.add(bullet)
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "diarree-opstoot, gastroscopie rustig en coloscopie zonder duidelijke macroscopische afwijking",
                    "heropflakkering stoelgangsklachten, scopiecontrole grotendeels geruststellend",
                    "gastro-enterologische herevaluatie, beperkte microscopische afwijkingen",
                    "chronische diarree opnieuw besproken, biopten zonder majeure afwijking",
                ],
                case,
                focus,
                "antecedent-month1",
            )
        )
        b.add("\n")
        used_keys.add(key)

    b.add("\n")
    b.add(other_heading)
    b.add("\n")
    undated_rows = [
        [
            "recidiverende stemmingsklachten, matig tot ernstig patroon",
            "arteriele hypertensie",
            "degeneratieve lagerugklachten",
            "calcifierende peesproblematiek rond heup en schoudergordel",
        ],
        [
            "stemmingsproblematiek met recidiverend verloop",
            "hypertensie onder medicamenteuze controle",
            "chronisch lageruglijden",
            "peesverkalkingen peri-articulair, rechts meer uitgesproken",
        ],
        [
            "psychische kwetsbaarheid met persoonlijkheidskenmerken in vraagstelling",
            "cardiovasculaire risicofactoren",
            "mechanische rugklachten",
            "schouder- en heuppeeslijden met calcificaties",
        ],
    ]
    for row in undated_rows[
        _date_focus_variant(case, focus, "antecedent-undated-set", len(undated_rows))
    ]:
        b.add(bullet)
        b.add(row)
        b.add("\n")

    key = "antecedenten_month_year_2"
    if key in focus:
        b.add(bullet)
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "gynaecologische ingreep, postpartaal traject complex verlopen",
                    "hysterectomietraject kort na partus vermeld",
                    "bekkenoperatie met volumineuze afwijking volgens oud verslag",
                    "gynaecologische chirurgie met moeilijke herstelperiode",
                ],
                case,
                focus,
                "antecedent-month2",
            )
        )
        b.add("\n")
        used_keys.add(key)

    key = "voorgeschiedenis_year_5"
    if key in focus:
        b.add(bullet)
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "handchirurgie links, neuropathische pijn nadien",
                    "heelkundige ingreep aan linkerduim met pijnsyndroom erna",
                    "excisie handletseltje, nadien langdurige napijn",
                    "perifere zenuwproblematiek na handoperatie",
                ],
                case,
                focus,
                "antecedent-year5",
            )
        )
        b.add("\n")
        used_keys.add(key)

    key = "lange_termijn_year_only"
    if key in focus:
        b.add(bullet)
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "neurologische klachten met diplopie-episodes in observatie",
                    "mogelijke demyeliniserende problematiek ooit geopperd",
                    "periodes van dubbelzien met verhoogde sedimentatie",
                    "neurologische follow-up zonder definitieve conclusie",
                ],
                case,
                focus,
                "antecedent-long-term",
            )
        )
        b.add("\n")
        used_keys.add(key)

    key = "antecedenten_month_year_3"
    if key in focus:
        b.add(bullet)
        b.add(focus[key], "Date")
        b.add(": ")
        b.add(
            _date_focus_pick(
                [
                    "spierreumatische klachten met uitgesproken drukpijnpunten",
                    "polymyalgiform klachtenbeeld met functionele component",
                    "diffuse pees- en spierpijn, meerdere drukpunten positief",
                    "reumatologische episode met brede pijncomponent",
                ],
                case,
                focus,
                "antecedent-month3",
            )
        )
        b.add("\n")
        used_keys.add(key)

    _add_focus_remainder(
        b,
        focus,
        used_keys,
        "Losse datumverwijzingen: ",
        case=case,
    )


def _add_checklist_date_focus_block(
    b: SpanBuilder, case: ClinicalCase, focus: dict[str, str]
) -> None:
    titles = [
        "Zorgpad checklist",
        "Actielijst zorgpad",
        "Controlelijst afspraken",
        "Openstaande punten",
        "Geplande stappen",
        "Te volgen items",
    ]
    action_sets = [
        [
            "dossier geopend",
            "telefoonnotitie nagekeken",
            "opnamegegevens vergeleken",
            "controlepunt bevestigd",
            "staalstatus bekeken",
            "consultverslag gelezen",
            "brief klaargezet",
            "ingreep genoteerd",
            "MDO gepland",
            "verpleegkundige evaluatie",
            "dagplanning afgestemd",
            "thuiszorgmoment",
            "revalidatieblok",
            "eerste antecedent",
            "tweede antecedent",
            "derde antecedent",
            "vierde antecedent",
            "vijfde antecedent",
            "langetermijnlijn",
        ],
        [
            "triageformulier aangevuld",
            "telefonisch akkoord genoteerd",
            "bedplanning naast dossier gelegd",
            "controlebeeld herbekeken",
            "afnameformulier gevalideerd",
            "raadplegingsbrief vergeleken",
            "correspondentie verstuurd",
            "procedureboekje nagekeken",
            "zorgoverleg voorbereid",
            "afdelingsobservatie ingepland",
            "dagzaalplanning vastgelegd",
            "familiezorg verwittigd",
            "therapieblok gereserveerd",
            "oudste VG gecontroleerd",
            "operatieve VG gecontroleerd",
            "reumatologische VG gecontroleerd",
            "historisch jaartal gecontroleerd",
            "oude complicatie gecontroleerd",
            "opvolgjaar in agenda",
        ],
        [
            "administratief startpunt",
            "voorafgaand overleg",
            "klinisch beginmoment",
            "tussencontrole",
            "staal of meting",
            "consultmoment",
            "verslagronde",
            "technische handeling",
            "teamafspraak",
            "verpleegkundig ijkpunt",
            "dagplanning",
            "zorg thuis",
            "revalidatieperiode",
            "maand-jaarantecedent",
            "tweede maand-jaarantecedent",
            "derde maand-jaarantecedent",
            "jaarhistoriek een",
            "jaarhistoriek twee",
            "lange termijn",
        ],
    ]
    actions = action_sets[
        _date_focus_style_index(case, focus, "checklist-actions", len(action_sets))
    ]
    title = _date_focus_style_pick(titles, case, focus, "checklist-title")
    b.add("\n")
    b.add(title)
    b.add(":\n")
    b.add("Klinische insteek: ")
    b.add(_date_focus_context_sentence(case, focus, "checklist-context"))
    b.add("\n")
    for item_index, (key, value) in enumerate(_date_focus_items(focus)):
        if key == "dagplan_day_month_numeric":
            b.add("- beenmergonderzoek ")
            b.add(value, "Date")
            b.add(" met lachgasanalgesie")
            b.add(" bij ")
            b.add(_timeline_clinical_note(key, case, focus, item_index))
            b.add("\n")
            continue
        action = actions[item_index % len(actions)]
        row_variant = (
            item_index + _date_focus_style_index(case, focus, "checklist-row-style", 4)
        ) % 4
        if row_variant == 0:
            b.add("- gedaan ")
            b.add(value, "Date")
            b.add(": ")
            b.add(action)
        elif row_variant == 1:
            b.add("- ")
            b.add(action)
            b.add(" ")
            _add_parenthesized_date(b, value)
        elif row_variant == 2:
            b.add("- te controleren tegen ")
            b.add(value, "Date")
            b.add(": ")
            b.add(action)
        else:
            b.add("- ")
            b.add(value, "Date")
            b.add(" -> ")
            b.add(action)
        b.add(" - ")
        b.add(_timeline_clinical_note(key, case, focus, item_index))
        b.add("\n")


def _decimal_comma(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _add_measurement_date_focus_block(
    b: SpanBuilder, case: ClinicalCase, focus: dict[str, str]
) -> None:
    style = case.date_focus_style if isinstance(case.date_focus_style, int) else 0
    seed = _date_focus_variant(case, focus, "measurement-values", 10_000)
    height_cm = 96 + (seed % 72)
    weight_kg = 18.0 + ((seed // 7) % 980) / 10
    previous_weight = max(3.2, weight_kg - (((seed // 19) % 17) - 8) / 20)
    delta = weight_kg - previous_weight
    bmi = weight_kg / ((height_cm / 100) ** 2)
    bsa_dubois = 0.007184 * (height_cm**0.725) * (weight_kg**0.425)
    bsa_mosteller = ((height_cm * weight_kg) / 3600) ** 0.5
    first_time = _date_focus_clock(case, "measurement-current")
    previous_time = _date_focus_clock(case, "measurement-previous")
    titles = [
        "Antropometrie",
        "Meetwaarden groei en oppervlak",
        "Somatometrie",
        "Parameters met datum",
        "Gewicht-lengte overzicht",
        "Meetmomenten verpleegpost",
    ]
    title = _date_focus_style_pick(titles, case, focus, "measurement-title")
    current_date = str(
        focus.get("aanmelding_numeric_slash_long") or next(iter(focus.values()))
    )
    previous_date = str(focus.get("telefonisch_numeric_slash_short") or current_date)
    length_date = str(focus.get("consult_textual_full") or current_date)
    derived_date = str(focus.get("dagplan_day_month_numeric") or current_date)
    label_sets = [
        [
            (
                "Lichaamsgewicht",
                f"{_decimal_comma(weight_kg, 2)} kg",
                current_date,
                first_time,
            ),
            (
                "Vorige gewichtsnotitie",
                f"{_decimal_comma(previous_weight, 2)} kg",
                previous_date,
                previous_time,
            ),
            ("Lichaamslengte", f"{height_cm} cm", length_date, None),
            ("Gewichtsverschil", f"{_decimal_comma(delta, 2)} kg", derived_date, None),
            ("Body Mass Index", f"{_decimal_comma(bmi, 1)} kg/m2", derived_date, None),
            ("BSA Dubois", f"{_decimal_comma(bsa_dubois, 2)} m2", derived_date, None),
            (
                "BSA Mosteller",
                f"{_decimal_comma(bsa_mosteller, 2)} m2",
                derived_date,
                None,
            ),
        ],
        [
            (
                "Gewicht verpleegpost",
                f"{_decimal_comma(weight_kg, 1)} kg",
                current_date,
                first_time,
            ),
            (
                "Gewicht referentie",
                f"{_decimal_comma(previous_weight, 1)} kg",
                previous_date,
                previous_time,
            ),
            ("Lengte dossier", f"{height_cm} cm", length_date, None),
            ("Delta gewicht", f"{_decimal_comma(delta, 2)} kg", derived_date, None),
            ("BMI herrekend", f"{_decimal_comma(bmi, 1)} kg/m2", derived_date, None),
            (
                "Oppervlak volgens Dubois",
                f"{_decimal_comma(bsa_dubois, 2)} m2",
                derived_date,
                None,
            ),
            (
                "Oppervlak volgens Mosteller",
                f"{_decimal_comma(bsa_mosteller, 2)} m2",
                derived_date,
                None,
            ),
        ],
        [
            (
                "Actueel gewicht",
                f"{_decimal_comma(weight_kg, 2)} kilogram",
                current_date,
                first_time,
            ),
            (
                "Controlegewicht",
                f"{_decimal_comma(previous_weight, 2)} kilogram",
                previous_date,
                previous_time,
            ),
            ("Staande lengte", f"{height_cm} centimeter", length_date, None),
            (
                "Verschil t.o.v. vorige meting",
                f"{_decimal_comma(delta, 2)} kg",
                derived_date,
                None,
            ),
            ("Quetelet-index", f"{_decimal_comma(bmi, 1)} kg/m2", derived_date, None),
            (
                "Berekend BSA Dubois",
                f"{_decimal_comma(bsa_dubois, 2)} m2",
                derived_date,
                None,
            ),
            (
                "Berekend BSA Mosteller",
                f"{_decimal_comma(bsa_mosteller, 2)} m2",
                derived_date,
                None,
            ),
        ],
    ]
    rows = label_sets[style % len(label_sets)]
    used_keys = {
        "aanmelding_numeric_slash_long",
        "telefonisch_numeric_slash_short",
        "consult_textual_full",
        "dagplan_day_month_numeric",
    }
    b.add("\n")
    b.add(title)
    b.add(":\n")
    b.add("Klinische context: ")
    b.add(_date_focus_context_sentence(case, focus, "measurement-context"))
    b.add("\n")
    for label, measurement, value, clock in rows:
        b.add(label)
        b.add(": ")
        b.add(measurement)
        b.add(" ")
        _add_parenthesized_date(b, value, trailing_time=clock)
        b.add("\n")
    _add_focus_remainder(
        b,
        focus,
        used_keys,
        _date_focus_style_pick(
            [
                "Niet-gebruikte datumvelden bij meting: ",
                "Andere datumankers in meetnota: ",
                "Meetcontext elders in dossier: ",
                "Aanvullende datumsporen: ",
            ],
            case,
            focus,
            "measurement-remainder",
        ),
        case=case,
    )


def _add_date_focus_block(b: SpanBuilder, case: ClinicalCase) -> None:
    focus = case.date_focus or {}
    if not focus:
        return

    template = case.date_focus_template or "compact_timeline"
    if template == "screening_rows":
        _add_screening_date_focus_block(b, case, focus)
    elif template == "functional_rows":
        _add_functional_date_focus_block(b, case, focus)
    elif template == "history_list":
        _add_history_date_focus_block(b, case, focus)
    elif template == "categorized_antecedents":
        _add_categorized_antecedents_date_focus_block(b, case, focus)
    elif template == "checklist":
        _add_checklist_date_focus_block(b, case, focus)
    elif template == "measurement_rows":
        _add_measurement_date_focus_block(b, case, focus)
    else:
        _add_compact_date_focus_block(b, case, focus)


def _clinical_context(b: SpanBuilder, case: ClinicalCase, rng: random.Random) -> None:
    negatives = hard_negative_codes(rng)
    details = case.medical_details or {}
    context_prefixes = [
        "\nKlinische context: voorgeschiedenis met ",
        "\nMedische samenvatting: gekend met ",
        "\nProbleemlijst: opvolging voor ",
        "\nContext voor beleid: dossier vermeldt ",
    ]
    style_index = sum(ord(ch) for ch in case.note_style) % len(context_prefixes)
    numbered_prefix = _numbered_title(
        case, context_prefixes[style_index].strip(), "clinical-context-heading"
    )
    b.add("\n")
    b.add(numbered_prefix)
    if not numbered_prefix.endswith(" "):
        b.add(" ")
    b.add(_condition(case))
    b.add("; actuele klachten zijn ")
    b.add(_symptoms(case, 3))
    b.add(". Medicatie: ")
    b.add(_medications(case, 3))
    if details.get("medication_orders"):
        b.add(" (")
        b.add("; ".join(str(item) for item in details["medication_orders"][:2]))
        b.add(")")
    measurement_heading = MEASUREMENT_CONTEXT_HEADINGS[
        _stable_offset(
            case.document_type,
            case.note_style,
            case.patient.name,
            "measurement-heading",
        )
        % len(MEASUREMENT_CONTEXT_HEADINGS)
    ]
    b.add(f".\n{measurement_heading}: ")
    compact_rows = _format_detail_rows(
        _detail_rows(case, "vitals") + _detail_rows(case, "numeric_results")
    )
    if compact_rows:
        b.add(compact_rows)
        b.add("; ")
    else:
        for name, value, unit in _lab_rows(case):
            b.add(f"{name} {value} {unit}; ")
    if details.get("procedure_snippets"):
        procedures = _format_procedure_snippets(details["procedure_snippets"][:2])
        if procedures:
            b.add("acties: ")
            b.add(procedures)
            b.add("; ")
    if details.get("abbreviations"):
        abbreviations = _format_abbreviations(details["abbreviations"][:5])
        if abbreviations:
            b.add("afkortingen: ")
            b.add(abbreviations)
            b.add("; ")
    b.add("terminologie/lookalikes ")
    b.add(", ".join(negatives[:3]))
    b.add(" blijven klinische hard negatives en worden niet als PII gemarkeerd.")
    _add_compact_date_overview(b, case)
    _add_date_focus_block(b, case)
    _add_relative_period_block(b, case, "clinical_context")
    if (
        _stable_offset(
            case.document_type, case.note_style, "clinical_context_capitalized_title"
        )
        % 4
        == 0
    ):
        b.add("\nTitel in bronsysteem: ")
        b.add(_capitalized_medical_title(case, "clinical_context_inline"))
        b.add(" blijft medische inhoud.")
    _add_referral_coordination_block(b, case, "clinical_context")
    _add_common_dutch_abbreviation_line(b, case, "clinical_context")
    if (
        _stable_offset(
            case.document_type,
            case.note_style,
            case.patient.name,
            "clinical_context_eponyms",
        )
        % 3
        != 0
    ):
        _add_medical_eponym_block(b, case, "clinical_context")
    if (
        _stable_offset(
            case.document_type,
            case.note_style,
            case.patient.name,
            "clinical_context_substance",
        )
        % 5
        == 0
    ):
        _add_substance_use_context_line(b, case, "clinical_context")
    _add_medication_hard_negative_block(b, case, "clinical_context")
    _add_medication_action_line(b, case, "clinical_context")
    _add_catalog_medication_treatment_plan(b, case, "clinical_context")
    _add_study_protocol_line(b, case, "clinical_context")
    _add_caregiver_role_hard_negative_line(b, case, "clinical_context")


def _add_structured_demographics(
    b: SpanBuilder, case: ClinicalCase, style_index: int
) -> None:
    gender = _patient_gender_marker(case, "admin")
    pattern = style_index % 9
    if pattern == 0:
        b.add(" | Geboren: ")
        _add_birthdate(b, case, include_prefix=False)
        b.add(" | Geslacht: ")
        b.add(gender)
        b.add(" | Leeftijd: ")
        _add_age_text(b, case, suppress_label_context=True)
        return
    if pattern == 1:
        b.add(" | Geboortedatum: ")
        _add_birthdate(b, case, include_prefix=False)
        b.add(" / ")
        b.add(gender)
        b.add(" / ")
        _add_age_text(b, case, suppress_label_context=True)
        return
    if pattern == 2:
        b.add("\nDemografie: geb.dat. ")
        _add_birthdate(b, case, include_prefix=False)
        b.add("; sexe ")
        b.add(gender)
        b.add("; lft ")
        _add_age_text(b, case, suppress_label_context=True)
        return
    if pattern == 3:
        b.add(" (")
        _add_birthdate(b, case, include_prefix=True)
        b.add(", ")
        b.add(gender)
        b.add(", ")
        _add_age_text(b, case, suppress_label_context=True)
        b.add(")")
        return
    if pattern == 4:
        b.add("\nID-lijn: ° ")
        _add_birthdate(b, case, include_prefix=False)
        b.add(" - ")
        _add_age_text(b, case, suppress_label_context=True)
        b.add(" - ")
        b.add(gender)
        return
    if pattern == 5:
        b.add(" | geb. ")
        _add_birthdate(b, case, include_prefix=False)
        b.add(" | ")
        b.add(gender)
        b.add(" | oud: ")
        _add_age_text(b, case, suppress_label_context=True)
        return
    if pattern == 6:
        b.add("\nPersoonslijn: DOB=")
        _add_birthdate(b, case, include_prefix=False)
        b.add("; gender=")
        b.add(gender)
        b.add("; leeftijd=")
        _add_age_text(b, case, suppress_label_context=True)
        return
    if pattern == 7:
        b.add(" | ")
        b.add(gender)
        b.add(" | GBD ")
        _add_birthdate(b, case, include_prefix=False)
        b.add(" | ")
        _add_age_text(b, case, suppress_label_context=True)
        return
    b.add("\nAdministratief: geboren ")
    _add_birthdate(b, case, include_prefix=False)
    b.add(", geslacht ")
    b.add(gender)
    b.add(", leeftijd ")
    _add_age_text(b, case, suppress_label_context=True)


HIS_PATIENT_TEMPLATE_PREFIXES = (
    "Patient Achternaam:",
    "PATIENT ACHTERNAAM",
    "Achternaam patient:",
    "Patient Achternaam :",
)


def _add_his_patient_template_header(
    b: SpanBuilder, case: ClinicalCase, style_index: int
) -> None:
    variants = person_name_variants(case.patient.name)
    surname = variants.get("surname") or case.patient.name
    first = variants.get("first") or case.patient.name
    his_id = str(
        case.identifiers.get("his_patient_id") or case.identifiers["patient_number"]
    )
    gender = _patient_gender_marker(case, "his_admin")
    pattern = (style_index // 6) % 4

    if pattern == 0:
        b.add("Patient Achternaam:\t")
        _add_patient_name(b, case, "his_patient_surname", surname)
        b.add("\nPatient Voornaam:\t")
        _add_patient_name(b, case, "his_patient_first", first)
        b.add("\nHIS Patient ID:\t\t")
        b.add(his_id, "ID:Patient")
        b.add("\nGeboortedatum:\t")
        _add_birthdate(b, case, include_prefix=False)
        b.add("\nGeslacht:\t")
        b.add(gender)
    elif pattern == 1:
        b.add("PATIENT ACHTERNAAM\t")
        _add_patient_name(b, case, "his_patient_surname", surname)
        b.add("\nPATIENT VOORNAAM\t")
        _add_patient_name(b, case, "his_patient_first", first)
        b.add("\nHIS patient-id\t\t")
        b.add(his_id, "ID:Patient")
        b.add("\nGeboren op\t\t")
        _add_birthdate(b, case, include_prefix=False)
        b.add("\nSexe\t\t\t")
        b.add(gender)
    elif pattern == 2:
        b.add("Patientgegevens\nAchternaam patient: ")
        _add_patient_name(b, case, "his_patient_surname", surname)
        b.add("\nVoornaam patient: ")
        _add_patient_name(b, case, "his_patient_first", first)
        b.add("\nHIS Patient ID: ")
        b.add(his_id, "ID:Patient")
        b.add("\nGeboortedatum: ")
        _add_birthdate(b, case, include_prefix=False)
        b.add("\nAdministratief geslacht: ")
        b.add(gender)
    else:
        b.add("Patient Achternaam : ")
        _add_patient_name(b, case, "his_patient_surname", surname)
        b.add("\nPatient Voornaam   : ")
        _add_patient_name(b, case, "his_patient_first", first)
        b.add("\nHIS Patient ID     : ")
        b.add(his_id, "ID:Patient")
        b.add("\nGeb.datum          : ")
        _add_birthdate(b, case, include_prefix=False)
        b.add("\nGeslacht           : ")
        b.add(gender)

    b.add("\nAdres: ")
    b.add(case.patient_address.text, "Address_Location:Patient")
    b.add("\nContact: ")
    b.add(case.contact["patient_phone"], "Contactdetails")
    b.add(" | ")
    b.add(case.contact["patient_email"], "Contactdetails")
    b.add("\n")


def _admin_header(b: SpanBuilder, case: ClinicalCase, rng: random.Random) -> None:
    header_styles = [
        ("Patient: ", " | Dossier: ", "\nAdres: ", " | Contact: "),
        (
            "Identificatie - naam: ",
            " | patientnummer: ",
            "\nWoonadres: ",
            " | bereikbaarheid: ",
        ),
        (
            "Administratief blok\nNaam patient: ",
            " | dossier-ID: ",
            "\nAdresgegevens: ",
            " | telefoon/e-mail: ",
        ),
        ("Pt ", " | nr. ", "\nAdreslijn: ", " | tel/mail: "),
        ("Identiteit: ", " | dossier ", "\nVerblijfadres: ", " | contactgegevens: "),
    ]
    style_index = _stable_offset(case.note_style, case.document_type, case.patient.name)
    if style_index % 6 == 5:
        _add_his_patient_template_header(b, case, style_index)
        return

    style = header_styles[style_index % len(header_styles)]
    b.add(style[0])
    _add_patient_name(b, case, "admin_header")
    _add_structured_demographics(b, case, style_index)
    b.add(style[1])
    b.add(case.identifiers["patient_number"], "ID:Patient")
    b.add(style[2])
    b.add(case.patient_address.text, "Address_Location:Patient")
    b.add(style[3])
    b.add(case.contact["patient_phone"], "Contactdetails")
    b.add(" | ")
    b.add(case.contact["patient_email"], "Contactdetails")
    b.add("\n")


def _care_context(b: SpanBuilder, case: ClinicalCase, rng: random.Random) -> None:
    b.add("Zorginstelling: ")
    b.add(case.hospital[0], "Organization:Healthcare")
    b.add(" | Behandelaar: ")
    _add_caregiver_name(
        b, case.caregiver.name, f"{case.document_type}:care_context", "dr."
    )
    _add_caregiver_function_suffix(b, case, "care_context")
    _add_caregiver_registry(b, case, " | ", "RIZIV: ")
    _add_caregiver_internal_phone(b, case, " | Rechtstreeks: ")
    b.add("\n")


def render_ai_scribe_note(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    b = SpanBuilder()
    if (
        _stable_offset(case.document_type, case.patient.name, "ai_scribe_title") % 3
        == 0
    ):
        _add_capitalized_medical_title(b, case, "ai_scribe_note")
    else:
        b.add(f"AI-scribe verslag {case.department}\n")
    _admin_header(b, case, rng)
    _care_context(b, case, rng)
    b.add("\nAnamnese: ")
    _add_patient_name(b, case, "ai_scribe_anamnese")
    b.add(f" meldt {_symptoms(case)} sinds ")
    _add_date(b, case.encounter_date, case, "encounter_time", "ai_scribe_anamnese")
    b.add(". Patient werkt als ")
    b.add(case.profession, "Profession")
    b.add(" en werd begeleid door ")
    b.add(case.relative.name, "Name:Other")
    b.add(".\nBeleid: verderzetten van ")
    b.add(_medications(case))
    b.add(". Controle op ")
    _add_date(b, case.followup_date, case, "followup_time", "ai_scribe_controle")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "ai_scribe_note")
    return b.doc(doc_id, _sources(case, "ai_scribe_note"))


def render_discharge_summary(
    doc_id: str, case: ClinicalCase, rng: random.Random
) -> dict:
    b = SpanBuilder()
    b.add("Ontslagbrief\n")
    _admin_header(b, case, rng)
    b.add("Opname in ")
    b.add(case.hospital[0], "Organization:Healthcare")
    b.add(" van ")
    _add_date(b, case.encounter_date, case, "encounter_time", "discharge_from")
    b.add(" tot ")
    _add_date(b, case.followup_date, case, "followup_time", "discharge_until")
    b.add(f" wegens {_condition(case)}. Medicatie bij ontslag: {_medications(case)}.\n")
    b.add("Brief gevalideerd door ")
    _add_caregiver_name(
        b, case.caregiver.name, f"{case.document_type}:validator", "prof. dr."
    )
    _add_caregiver_function_suffix(b, case, "validator")
    if _add_caregiver_internal_phone(b, case, " ("):
        b.add(")")
    b.add(". Kopie naar huisarts ")
    _add_caregiver_name(b, case.secondary_caregiver.name, f"{case.document_type}:copy")
    b.add(" te ")
    b.add(case.caregiver_locality[0], "Address_Location:Caregiver")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "discharge_summary")
    return b.doc(doc_id, _sources(case, "discharge_summary"))


def render_ed_note(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    b = SpanBuilder()
    b.add("Spoednota - triage\n")
    b.add("Aangemeld op ")
    _add_date(b, case.encounter_date, case, "encounter_time", "ed_arrival")
    b.add(" in ")
    b.add(case.hospital[0], "Organization:Healthcare")
    b.add(". Naam ")
    _add_patient_name(b, case, "ed_note_name")
    b.add(", INSZ ")
    b.add(case.identifiers["national_register"], "ID:Patient")
    b.add(", tel. ")
    b.add(case.contact["patient_phone"], "Contactdetails")
    b.add(".\nKlacht: ")
    b.add(_symptoms(case, 3))
    b.add(f". Werkdiagnose: {_condition(case)}. ")
    b.add("Familiecontact ")
    b.add(case.relative.name, "Name:Other")
    b.add(" wacht in de inkomhal. Controle-afspraak ")
    _add_date(b, case.followup_date, case, "followup_time", "ed_followup")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "ed_note")
    return b.doc(doc_id, _sources(case, "ed_note"))


def render_consult_letter(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    b = SpanBuilder()
    b.add("Consultatiebrief\n")
    b.add("Geachte collega,\n")
    b.add("Ik zag ")
    _add_patient_name(b, case, "consult_letter_seen")
    b.add(" op ")
    _add_date(b, case.encounter_date, case, "encounter_time", "consult_seen")
    b.add(" voor evaluatie van ")
    b.add(_condition(case))
    b.add(". Voorgeschiedenis en medicatie werden vergeleken met het dossier ")
    b.add(case.identifiers["patient_number"], "ID:Patient")
    b.add(".\nPatient woont op ")
    b.add(case.patient_address.text, "Address_Location:Patient")
    b.add(" en volgt de opleiding/activiteit ")
    b.add(case.profession, "Profession")
    b.add(". Verslag opgesteld door ")
    _add_caregiver_name(b, case.caregiver.name, f"{case.document_type}:author", "dr.")
    _add_caregiver_function_suffix(b, case, "author")
    b.add(" in ")
    b.add(case.healthcare_institution[0], "Organization:Healthcare")
    _add_caregiver_internal_phone(b, case, ". Overleg via ")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "consult_letter")
    return b.doc(doc_id, _sources(case, "consult_letter"))


def render_lab_report(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    negatives = hard_negative_codes(rng)
    b = SpanBuilder()
    if (
        _stable_offset(case.document_type, case.patient.name, "lab_report_title") % 2
        == 0
    ):
        _add_capitalized_medical_title(b, case, "lab_report")
    else:
        b.add("Laboratoriumrapport\n")
    b.add("Naam: ")
    _add_patient_name(b, case, "lab_report_name")
    b.add(" | Geboortedatum: ")
    _add_birthdate(b, case)
    b.add(" | Laboresultaten ID: ")
    b.add(case.identifiers["lab_accession"], "ID:Patient")
    b.add(" | Aanvraagdatum: ")
    _add_date(b, case.encounter_date, case, "encounter_time", "lab_request")
    b.add("\nAanvrager: ")
    _add_caregiver_name(
        b, case.caregiver.name, f"{case.document_type}:requester", "dr."
    )
    b.add(" (")
    b.add(case.healthcare_institution[0], "Organization:Healthcare")
    b.add("; functie ")
    b.add(_caregiver_function(case, "lab_requester"))
    _add_caregiver_internal_phone(b, case, "; bereikbaar op ")
    b.add(")\n\n")
    b.add("Analyse                 Resultaat      Eenheid      Ref/Codes\n")
    for name, value, unit in _lab_rows(case):
        b.add(f"{name:<23} {value:<12} {unit:<11} {rng.choice(negatives)}\n")
    b.add("Extra identificatie staal: ")
    b.add(case.identifiers["imaging_key"], "ID:Patient")
    b.add(". Opmerking: gen/biomerker ")
    b.add(negatives[0])
    b.add(" is klinische inhoud en geen patient-ID.\n")
    collection_time = str((case.date_times or {}).get("collection_time", "")).strip()
    if collection_time:
        b.add("Technische validatie: hemolyse-index 12, afnametijd ")
        b.add(collection_time)
        b.add(", analyzer QC OK. ")
    else:
        b.add("Technische validatie: hemolyse-index 12, analyzer QC OK. ")
    b.add(
        "LOINC 718-7 en SNOMED 44054006 zijn terminologiecodes zonder directe herleidbaarheid."
    )
    _add_date_focus_block(b, case)
    _add_medication_hard_negative_block(b, case, "lab_report")
    _add_medication_action_line(b, case, "lab_report")
    _add_execution_audit_line(b, case, "lab_report")
    return b.doc(doc_id, _sources(case, "lab_report"))


def render_genetics_report(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    finding = case.genetic_finding
    b = SpanBuilder()
    b.add("Genetisch verslag\n")
    b.add("Proband: ")
    _add_patient_name(b, case, "genetics_report_proband")
    b.add(" (")
    _add_birthdate(b, case)
    b.add("), aanvraag ")
    b.add(case.identifiers["cfdna_reference"], "ID:Patient")
    b.add(".\nFamiliaal staal van ")
    b.add(case.relative.name, "Name:Other")
    b.add(" ontvangen op ")
    _add_date(b, case.encounter_date, case, "collection_time", "genetics_received")
    b.add(".\nResultaat: ")
    b.add(
        f"{finding['gene']} {finding['variant']} {finding['protein']} ({finding['rsid']})"
    )
    b.add(f", interpretatie {finding['interpretation']}. ")
    b.add("Analyse gebeurde op panel NM_004333.6; deze codes zijn geen PII. ")
    b.add("Rapportnummer ")
    b.add(case.identifiers["pathology_accession"], "ID:Patient")
    b.add(
        ".\nAanvullende interpretatie: segregatieanalyse wordt aanbevolen bij relevante familieanamnese. "
    )
    b.add(
        "HGVS-strings, rsIDs en transcriptcodes zoals NM_004333.6 worden bewust niet als PII geannoteerd."
    )
    _add_date_focus_block(b, case)
    _add_medication_hard_negative_block(b, case, "genetics_report")
    _add_medication_action_line(b, case, "genetics_report")
    _add_execution_audit_line(b, case, "genetics_report")
    return b.doc(doc_id, _sources(case, "genetics_report"))


def render_oncology_mdo(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    b = SpanBuilder()
    b.add("Oncologisch MDO\n")
    _admin_header(b, case, rng)
    b.add("Besproken in ")
    b.add(case.hospital[0], "Organization:Healthcare")
    b.add(" op ")
    _add_date(b, case.encounter_date, case, "review_time", "oncology_discussed")
    b.add(". MDO-referentie ")
    b.add(case.identifiers["study_name"], "ID:Patient")
    b.add("; studieprotocol ")
    b.add(case.identifiers["study_protocol_name"], "ID:Patient")
    b.add(" met protocol-ID ")
    b.add(case.identifiers["study_protocol_id"], "ID:Patient")
    b.add(".\nAanwezig: ")
    _add_caregiver_name(b, case.caregiver.name, f"{case.document_type}:attendee", "dr.")
    _add_caregiver_function_suffix(b, case, "attendee")
    b.add(", verpleegkundig specialist ")
    _add_caregiver_name(
        b, case.secondary_caregiver.name, f"{case.document_type}:nurse_specialist"
    )
    b.add(". Plan: kuur met ")
    b.add(_medications(case, 2))
    b.add("; herevaluatie ")
    _add_date(b, case.followup_date, case, "followup_time", "oncology_recheck")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "oncology_mdo")
    return b.doc(doc_id, _sources(case, "oncology_mdo"))


def render_medication_reconciliation(
    doc_id: str, case: ClinicalCase, rng: random.Random
) -> dict:
    b = SpanBuilder()
    b.add("Medicatieverificatie\n")
    b.add("Patient ")
    _add_patient_name(b, case, "medication_reconciliation_patient")
    b.add(", dossier ")
    b.add(case.identifiers["patient_number"], "ID:Patient")
    b.add(", werd gezien door ")
    _add_caregiver_name(b, case.caregiver.name, f"{case.document_type}:pharmacist")
    b.add(", apotheker,")
    b.add(" op ")
    _add_date(b, case.encounter_date, case, "encounter_time", "medrec_seen")
    b.add(".\nThuismedicatie: ")
    b.add(_medications(case))
    b.add(".")
    _add_medication_hard_negative_block(b, case, "medication_reconciliation")
    b.add("\nInteractiecontrole vermeldt ")
    b.add(rng.choice(hard_negative_codes(rng)))
    b.add(" als systeemcode, niet als patient-ID. Partner ")
    b.add(case.relative.name, "Name:Other")
    b.add(" bevestigt inname via ")
    b.add(case.contact["relative_email"], "Contactdetails")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "medication_reconciliation")
    return b.doc(doc_id, _sources(case, "medication_reconciliation"))


def render_nursing_note(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    b = SpanBuilder()
    b.add("Verpleegkundige observatie\n")
    _add_patient_name(b, case, "nursing_note_patient")
    b.add(" verblijft op kamer ")
    b.add(
        f"D{rng.randrange(1, 9)}:BOX{rng.randrange(1, 20)}", "Organization:Healthcare"
    )
    b.add(" in ")
    b.add(case.hospital[0], "Organization:Healthcare")
    b.add(". Wondzorg uitgevoerd op ")
    _add_date(b, case.encounter_date, case, "encounter_time", "nursing_woundcare")
    b.add(" door ")
    _add_caregiver_name(b, case.caregiver.name, f"{case.document_type}:nurse")
    _add_caregiver_function_suffix(b, case, "nurse")
    if _add_caregiver_internal_phone(b, case, " ("):
        b.add(")")
    b.add(". Pols en bloeddruk stabiel; klachten: ")
    b.add(_symptoms(case))
    b.add(". Moeder/partner ")
    b.add(case.relative.name, "Name:Other")
    b.add(" belt via ")
    b.add(case.contact["relative_phone"], "Contactdetails")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "nursing_note")
    return b.doc(doc_id, _sources(case, "nursing_note"))


def render_radiology_summary(
    doc_id: str, case: ClinicalCase, rng: random.Random
) -> dict:
    b = SpanBuilder()
    _add_capitalized_medical_title(b, case, "radiology_summary")
    b.add("Onderzoek voor ")
    _add_patient_name(b, case, "radiology_summary_patient")
    b.add(" op ")
    _add_date(b, case.encounter_date, case, "encounter_time", "radiology_exam")
    b.add(". Beelden via ")
    b.add("https://beelden.example.be/viewer", "Contactdetails")
    b.add(" met sleutel ")
    b.add(case.identifiers["imaging_key"], "ID:Patient")
    b.add(".\nVerslag door ")
    _add_caregiver_name(
        b, case.caregiver.name, f"{case.document_type}:radiologist", "dr."
    )
    _add_caregiver_function_suffix(b, case, "radiologist")
    b.add(" in ")
    b.add(case.hospital[0], "Organization:Healthcare")
    b.add(". Geen pneumonie; bevindingen passen bij ")
    b.add(_condition(case))
    b.add(". Controle ")
    _add_date(b, case.followup_date, case, "followup_time", "radiology_control")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "radiology_summary")
    return b.doc(doc_id, _sources(case, "radiology_summary"))


def render_referral_letter(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    b = SpanBuilder()
    b.add("Verwijsbrief\n")
    b.add("Gelieve ")
    _add_patient_name(b, case, "referral_letter_patient")
    b.add(" te zien wegens ")
    b.add(_condition(case))
    b.add(". Zij/hij woont in ")
    b.add(case.patient_address.text, "Address_Location:Patient")
    b.add(" en werkt/studeert als ")
    b.add(case.profession, "Profession")
    b.add(".\nVerwijzer ")
    _add_caregiver_name(b, case.caregiver.name, f"{case.document_type}:referrer", "dr.")
    _add_caregiver_function_suffix(b, case, "referrer")
    _add_caregiver_registry(b, case, " (", "artsen-ID ")
    b.add(") vraagt consult op ")
    _add_date(b, case.followup_date, case, "followup_time", "referral_requested")
    b.add(" bij ")
    b.add(case.healthcare_institution[0], "Organization:Healthcare")
    _add_caregiver_internal_phone(b, case, "; overleg via ")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "referral_letter")
    return b.doc(doc_id, _sources(case, "referral_letter"))


def render_home_care_report(
    doc_id: str, case: ClinicalCase, rng: random.Random
) -> dict:
    b = SpanBuilder()
    b.add("Thuiszorgverslag\n")
    b.add("Client ")
    _add_patient_name(b, case, "home_care_report_patient")
    b.add(" (")
    _add_age_text(b, case)
    b.add(") op ")
    b.add(case.patient_address.text, "Address_Location:Patient")
    b.add(". Zorgmoment ")
    _add_date(b, case.encounter_date, case, "encounter_time", "home_care_moment")
    b.add(" door ")
    _add_caregiver_name(b, case.caregiver.name, f"{case.document_type}:home_care")
    _add_caregiver_function_suffix(b, case, "home_care")
    b.add(" van ")
    b.add(case.healthcare_institution[0], "Organization:Healthcare")
    b.add(". Zorgplan ")
    b.add(case.identifiers["crisis_card"], "ID:Patient")
    b.add("; contact mantelzorger ")
    b.add(case.relative.name, "Name:Other")
    b.add(" via ")
    b.add(case.contact["relative_email"], "Contactdetails")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "home_care_report")
    return b.doc(doc_id, _sources(case, "home_care_report"))


def render_rehab_progress(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    b = SpanBuilder()
    b.add("Revalidatievoortgang\n")
    _add_patient_name(b, case, "rehab_progress_patient")
    b.add(" werd gezien op ")
    _add_date(b, case.encounter_date, case, "encounter_time", "rehab_seen")
    b.add(" in ")
    b.add(case.healthcare_institution[0], "Organization:Healthcare")
    b.add(". ")
    b.add(f"Route {rng.randrange(10, 99)}", "Organization:Healthcare")
    b.add(". Begeleider ")
    _add_caregiver_name(b, case.caregiver.name, f"{case.document_type}:therapist")
    _add_caregiver_function_suffix(b, case, "therapist")
    b.add(" noteert vooruitgang bij traplopen. Patient is ")
    b.add(case.profession, "Profession")
    b.add(" en wil terug naar activiteiten bij ")
    b.add(case.other_org, "Organization:Other")
    b.add(". Volgende sessie ")
    _add_date(b, case.followup_date, case, "followup_time", "rehab_next")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "rehab_progress")
    return b.doc(doc_id, _sources(case, "rehab_progress"))


def render_pathology_report(
    doc_id: str, case: ClinicalCase, rng: random.Random
) -> dict:
    b = SpanBuilder()
    if (
        _stable_offset(case.document_type, case.patient.name, "pathology_report_title")
        % 2
        == 0
    ):
        _add_capitalized_medical_title(b, case, "pathology_report")
    else:
        b.add("Pathologieverslag\n")
    b.add("Patient ")
    _add_patient_name(b, case, "pathology_report_patient")
    b.add(", geboortedatum ")
    _add_birthdate(b, case)
    b.add(". Materiaal ontvangen ")
    _add_date(b, case.encounter_date, case, "collection_time", "pathology_received")
    b.add(", pathologienummer ")
    b.add(case.identifiers["pathology_accession"], "ID:Patient")
    b.add(".\nMacroscopie en microscopie passend bij ")
    b.add(_condition(case))
    b.add(". Immunokleuringen: HER2, Ki-67 en EGFR zijn biomerkers en geen ID. ")
    b.add("Verslag gevalideerd door ")
    _add_caregiver_name(
        b, case.caregiver.name, f"{case.document_type}:pathologist", "dr."
    )
    _add_caregiver_function_suffix(b, case, "pathologist")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "pathology_report")
    return b.doc(doc_id, _sources(case, "pathology_report"))


def render_device_implant_note(
    doc_id: str, case: ClinicalCase, rng: random.Random
) -> dict:
    b = SpanBuilder()
    b.add("Implantatieverslag\n")
    _add_patient_name(b, case, "device_implant_note_patient")
    b.add(" onderging procedure op ")
    _add_date(b, case.encounter_date, case, "encounter_time", "device_procedure")
    b.add(" in ")
    b.add(f"OK {rng.randrange(1, 12)}", "Organization:Healthcare")
    b.add(" van ")
    b.add(case.hospital[0], "Organization:Healthcare")
    b.add(". Device model Medtronic Inceptiv vermeld in materiaalstaat; serienummer ")
    b.add(case.identifiers["device_serial"], "ID:Patient")
    b.add(" is patientgebonden. LOT-nummer ")
    b.add(case.identifiers["material_lot"], "ID:Caregiver")
    b.add(" hoort bij het gebruikte materiaal. Operator ")
    _add_caregiver_name(b, case.caregiver.name, f"{case.document_type}:operator", "dr.")
    _add_caregiver_function_suffix(b, case, "operator")
    _add_caregiver_internal_phone(b, case, ", bereikbaar via ")
    b.add(".")
    _clinical_context(b, case, rng)
    _add_execution_audit_line(b, case, "device_implant_note")
    return b.doc(doc_id, _sources(case, "device_implant_note"))


def render_secure_email(doc_id: str, case: ClinicalCase, rng: random.Random) -> dict:
    b = SpanBuilder()
    b.add("beste ")
    _add_caregiver_name(
        b, case.caregiver.name, f"{case.document_type}:email_recipient", "dr."
    )
    b.add(",\n\n")
    b.add("Kan u het behandelplan voor ")
    _add_patient_name(b, case, "secure_email_patient")
    b.add(" nakijken? Dossier ")
    b.add(case.identifiers["patient_number"], "ID:Patient")
    b.add(" vermeldt ")
    b.add(_condition(case))
    b.add(" en thuismedicatie ")
    b.add(_medications(case))
    b.add(". De patient is bereikbaar via ")
    b.add(case.contact["patient_email"], "Contactdetails")
    b.add(" en werd gezien op ")
    _add_date(b, case.encounter_date, case, "encounter_time", "secure_email_seen")
    b.add(" in ")
    b.add(case.healthcare_institution[0], "Organization:Healthcare")
    b.add(".")
    _add_catalog_medication_treatment_plan(b, case, "secure_email")
    b.add("\n\nMet vriendelijke groet,\n")
    _add_caregiver_name(
        b, case.secondary_caregiver.name, f"{case.document_type}:email_sender"
    )
    _add_execution_audit_line(b, case, "secure_email")
    return b.doc(doc_id, _sources(case, "secure_email"))


def render_anesthesia_operating_grid(
    doc_id: str, case: ClinicalCase, rng: random.Random
) -> dict:
    b = SpanBuilder()
    if (
        _stable_offset(case.document_type, case.patient.name, "anesthesia-title") % 2
        == 0
    ):
        _add_capitalized_medical_title(b, case, "anesthesia_operating_grid")
    else:
        b.add(
            _numbered_title(
                case, "Anesthesieverslag operatiekwartier", "anesthesia-title"
            )
        )
        b.add("\n")
    b.add(
        "Anesthesist                     ASA #              Operatiedatum     Naam patiënt                               Gewicht     Admission #\n"
    )
    b.add("- ")
    _add_caregiver_inverted_name(
        b, case.caregiver.name, f"{case.document_type}:anesthetist"
    )
    b.add(f"{'':<23}")
    b.add(str(1 + _stable_offset(case.patient.name, case.document_type, "asa") % 4))
    b.add(f"{'':<17}")
    b.add(_audit_date_value(case, "anesthesia_operation"), "Date")
    b.add("         ")
    _add_patient_inverted_name(b, case, "anesthesia_patient")
    b.add(f"{'':<10}")
    weight = _detail_rows(case, "vitals")
    weight_value = next(
        (row for row in weight if str(row.get("name", "")).casefold() == "gewicht"),
        None,
    )
    if weight_value:
        b.add(
            " ".join(
                str(weight_value.get(part, "")).strip()
                for part in ("value", "unit")
                if weight_value.get(part)
            )
        )
    else:
        b.add(
            f"{52 + (_stable_offset(case.patient.name, 'weight') % 54)},{_stable_offset(case.patient.name, 'weight-dec') % 10} kg"
        )
    b.add("     ")
    b.add(case.identifiers["operating_room_case"], "ID:Patient")
    b.add(
        "\nPreoperatieve nota: specialisme anesthesist en ASA-klasse zijn klinische context, geen Profession-span."
    )
    _add_medication_hard_negative_block(b, case, "anesthesia_operating_grid")
    _add_medical_eponym_block(b, case, "anesthesia_operating_grid")
    _add_catalog_medication_treatment_plan(b, case, "anesthesia_operating_grid")
    _add_execution_audit_line(b, case, "anesthesia_operating_grid")
    return b.doc(doc_id, _sources(case, "anesthesia_operating_grid"))


def render_pulmonary_calibration_report(
    doc_id: str, case: ClinicalCase, rng: random.Random
) -> dict:
    b = SpanBuilder()
    b.add("UZA", "Organization:Healthcare")
    b.add(" Longziekten LAST CALIBRATION ")
    b.add(_audit_date_value(case, "pulmonary_last_calibration"), "Date")
    b.add(" ")
    b.add(_audit_time_value(case, "pulmonary_last_calibration"))
    b.add("\nOPERATOR COMMENTS\n\n")
    _add_caregiver_first_name(
        b, case.caregiver.name, f"{case.document_type}:operator_comment"
    )
    b.add("\n\nPre BD\n")
    b.add(case.identifiers["his_patient_id"], "ID:Patient")
    b.add(" -  ")
    _add_patient_inverted_name(b, case, "pulmonary_patient", lowercase=True)
    b.add("\n")
    b.add(_patient_gender_marker(case, "pulmonary_display"))
    b.add("\n")
    age_seed = _stable_offset(case.patient.name, case.birthdate, "pulmonary-age")
    b.add(f"{18 + age_seed % 73},{(age_seed // 73) % 10} yrs", "Age_Birthdate")
    b.add("\n")
    _add_birthdate(b, case, include_prefix=False)
    b.add("\n\nCaucasian\n\n")
    b.add(f"{145 + (_stable_offset(case.patient.name, 'height') % 46)} cm")
    b.add(
        "\nLongfunctieparameters: FEV1, FVC, PEF en kalibratietijd blijven klinische meetcontext."
    )
    _add_medication_action_line(b, case, "pulmonary_calibration_report")
    _add_execution_audit_line(b, case, "pulmonary_calibration_report")
    return b.doc(doc_id, _sources(case, "pulmonary_calibration_report"))


RENDERERS: list[Renderer] = [
    render_ai_scribe_note,
    render_discharge_summary,
    render_ed_note,
    render_consult_letter,
    render_lab_report,
    render_genetics_report,
    render_oncology_mdo,
    render_medication_reconciliation,
    render_nursing_note,
    render_radiology_summary,
    render_referral_letter,
    render_home_care_report,
    render_rehab_progress,
    render_pathology_report,
    render_device_implant_note,
    render_secure_email,
    render_anesthesia_operating_grid,
    render_pulmonary_calibration_report,
]

RENDERERS_BY_DOCUMENT_TYPE: dict[str, Renderer] = {
    renderer.__name__.replace("render_", ""): renderer for renderer in RENDERERS
}


def generate_documents(
    count: int,
    seed: int = 20260508,
    synthea_csv_dir: Path | None = None,
    auto_synthea: bool = False,
    synthea_repo_dir: Path = Path("external/synthea"),
    synthea_population: int | None = None,
    force_synthea: bool = False,
    require_synthea: bool = False,
) -> list[dict]:
    sampler = LookupSampler(seed=seed)
    rng = random.Random(seed)
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
    docs = []
    for index in range(count):
        synthea_seed = (
            synthea_seeds[index % len(synthea_seeds)] if synthea_seeds else None
        )
        case = generate_case(sampler, rng, index, synthea_seed=synthea_seed)
        renderer = RENDERERS[index % len(RENDERERS)]
        doc = renderer(f"synthetic-{index + 1:05d}", case, rng)
        docs.append(apply_production_post_process(doc))
    return docs


def render_documents_from_case_records(
    records: list[dict],
    seed: int = 20260508,
) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for index, record in enumerate(records):
        case = case_from_record(record)
        renderer = RENDERERS_BY_DOCUMENT_TYPE.get(case.document_type)
        if renderer is None:
            renderer = RENDERERS[index % len(RENDERERS)]
        doc_id = record.get("case_id", f"case-{index + 1:05d}").replace(
            "case-", "synthetic-"
        )
        doc = renderer(doc_id, case, rng)
        docs.append(apply_production_post_process(doc))
    return docs


def write_jsonl(docs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(normalize_record(doc), ensure_ascii=False) + "\n")
