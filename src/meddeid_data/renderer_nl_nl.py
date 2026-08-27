"""Netherlands-specific renderers for synthetic Dutch clinical documents.

The Belgian corpus renderer deliberately remains unchanged.  This module owns
the Netherlands document vocabulary and administrative presentation so an
``nl-NL`` case cannot silently acquire Belgian headers, identifiers, or care
system terminology during rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .production_postprocess import apply_production_post_process
from .span_builder import SpanBuilder, split_person_name


PROFILE_ID = "nl-NL"
RENDERER_ID = "netherlands-clinical-renderer-v1"

NETHERLANDS_DOCUMENT_TITLES = {
    "ai_scribe_note": "Poliklinische consultnotitie",
    "discharge_summary": "Ontslagbrief",
    "ed_note": "SEH-verslag",
    "consult_letter": "Specialistenbrief",
    "lab_report": "Laboratoriumuitslag",
    "genetics_report": "Klinisch-genetisch verslag",
    "oncology_mdo": "Oncologisch MDO-verslag",
    "medication_reconciliation": "Medicatieverificatie",
    "nursing_note": "Verpleegkundige rapportage",
    "radiology_summary": "Radiologieverslag",
    "referral_letter": "Verwijsbrief",
    "home_care_report": "Wijkverpleegkundige rapportage",
    "rehab_progress": "Revalidatievoortgang",
    "pathology_report": "Pathologieverslag",
    "device_implant_note": "Implantatieverslag",
    "secure_email": "Beveiligd zorgbericht",
    "anesthesia_operating_grid": "Anesthesieverslag operatiekamer",
    "pulmonary_calibration_report": "Longfunctierapport",
}

NETHERLANDS_SECTION_TEXT = {
    "ai_scribe_note": (
        "SOEP-notitie: patiënt gezien op de polikliniek. Bevindingen en beleid "
        "zijn tijdens het consult vastgelegd."
    ),
    "discharge_summary": (
        "Beloop tijdens opname: klinisch herstel zonder nieuwe complicaties. "
        "De huisarts ontvangt deze ontslaginformatie."
    ),
    "ed_note": (
        "Beoordeling op de spoedeisende hulp. Na triage en aanvullend onderzoek "
        "is het vervolgbeleid afgesproken."
    ),
    "consult_letter": (
        "Samenvatting van het polikliniekbezoek en advies aan de verwijzend "
        "huisarts."
    ),
    "lab_report": (
        "Materiaal ontvangen door het laboratorium; uitslagen zijn beoordeeld "
        "in samenhang met de klinische vraagstelling."
    ),
    "genetics_report": (
        "De uitslag is besproken volgens de werkwijze van de klinische genetica; "
        "familieonderzoek kan worden overwogen."
    ),
    "oncology_mdo": (
        "Conclusie multidisciplinair overleg: diagnostiek en behandelopties zijn "
        "door het behandelteam besproken."
    ),
    "medication_reconciliation": (
        "De thuismedicatie is bij opname geverifieerd met patiënt en apotheek."
    ),
    "nursing_note": (
        "Verpleegkundige observatie volgens het zorgplan; bijzonderheden zijn "
        "overgedragen aan de volgende dienst."
    ),
    "radiology_summary": (
        "Beeldvorming beoordeeld door de radioloog; bevindingen en conclusie "
        "staan hieronder."
    ),
    "referral_letter": (
        "Graag uw beoordeling op de polikliniek. Relevante voorgeschiedenis en "
        "actuele medicatie zijn bijgevoegd."
    ),
    "home_care_report": (
        "Rapportage van de wijkverpleging: zorg uitgevoerd volgens indicatie en "
        "afgestemd met de huisarts."
    ),
    "rehab_progress": (
        "Voortgang binnen het revalidatieplan; belastbaarheid en behandeldoelen "
        "zijn opnieuw geëvalueerd."
    ),
    "pathology_report": (
        "Pathologisch onderzoek verricht; macroscopie, microscopie en conclusie "
        "zijn vastgelegd."
    ),
    "device_implant_note": (
        "Implantatie op de operatiekamer zonder directe complicaties; controle "
        "van het hulpmiddel was technisch akkoord."
    ),
    "secure_email": (
        "Dit beveiligde zorgbericht bevat een korte overdracht aan een betrokken "
        "zorgverlener."
    ),
    "anesthesia_operating_grid": (
        "Preoperatieve beoordeling en anesthesiebeleid voor de operatiekamer."
    ),
    "pulmonary_calibration_report": (
        "Longfunctieonderzoek uitgevoerd na geldige kalibratie; technische "
        "kwaliteit was voldoende voor beoordeling."
    ),
}

TYPE_IDENTIFIER_FIELDS = {
    "ai_scribe_note": "patient_file",
    "discharge_summary": "his_patient_id",
    "ed_note": "crisis_card",
    "consult_letter": "study_reference",
    "lab_report": "lab_accession",
    "genetics_report": "study_protocol_id",
    "oncology_mdo": "study_name",
    "medication_reconciliation": "patient_file",
    "nursing_note": "his_patient_id",
    "radiology_summary": "imaging_key",
    "referral_letter": "study_protocol_name",
    "home_care_report": "patient_file",
    "rehab_progress": "study_reference",
    "pathology_report": "pathology_accession",
    "device_implant_note": "device_serial",
    "secure_email": "patient_file",
    "anesthesia_operating_grid": "operating_room_case",
    "pulmonary_calibration_report": "his_patient_id",
}


def _pair_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def _person_name(record: dict[str, Any], key: str) -> str:
    person = record.get(key, {})
    return str(person.get("name", "")) if isinstance(person, dict) else str(person)


def _metadata(record: dict[str, Any], document_type: str) -> dict[str, Any]:
    patient = split_person_name(_person_name(record, "patient"))
    if record.get("birthdate"):
        patient["birth_date"] = str(record["birthdate"])
    caregivers = [
        split_person_name(_person_name(record, key))
        for key in ("caregiver", "secondary_caregiver")
    ]
    caregivers = [value for value in caregivers if value]
    return {
        "generation_method": RENDERER_ID,
        "renderer": f"nl_nl_{document_type}",
        "document_type": document_type,
        "note_style": record.get("note_style"),
        "lang": PROFILE_ID,
        "generation_profile": PROFILE_ID,
        "document_creation_date": record.get("encounter_date"),
        "lookup_source": _pair_value(record.get("hospital", ("", ""))[1:2]),
        "patient": patient,
        "caregivers": caregivers,
    }


def _clinical_summary(record: dict[str, Any]) -> str:
    condition = record.get("condition", {})
    name = str(condition.get("name", "klinische beoordeling"))
    symptoms = ", ".join(str(value) for value in condition.get("symptoms", [])[:2])
    medications = ", ".join(
        str(value) for value in condition.get("medications", [])[:3]
    )
    parts = [f"Werkdiagnose: {name}."]
    if symptoms:
        parts.append(f"Klachten: {symptoms}.")
    if medications:
        parts.append(f"Medicatie: {medications}.")
    return " ".join(parts)


def _render_document(
    document_id: str,
    record: dict[str, Any],
    document_type: str,
) -> dict[str, Any]:
    b = SpanBuilder()
    identifiers = record["identifiers"]
    contact = record["contact"]

    b.add(NETHERLANDS_DOCUMENT_TITLES[document_type])
    b.add("\nZorginstelling: ")
    b.add(_pair_value(record["hospital"]), "Organization:Healthcare")
    b.add("\nAfdeling: ")
    b.add(str(record["department"]))
    b.add("\nDatum verslag: ")
    b.add(str(record["encounter_date"]), "Date")

    b.add("\n\nPatiënt: ")
    b.add(_person_name(record, "patient"), "Name:Patient")
    b.add("\nGeboortedatum: ")
    b.add(str(record["birthdate"]), "Age_Birthdate")
    b.add(" (leeftijd ")
    b.add(str(record["age_text"]), "Age_Birthdate")
    b.add(")")
    b.add("\nBSN: ")
    b.add(str(identifiers["national_register"]), "ID:Patient")
    b.add("\nEPD-nummer: ")
    b.add(str(identifiers["patient_number"]), "ID:Patient")
    b.add("\nAdres: ")
    b.add(str(record["patient_address"]["text"]), "Address_Location:Patient")
    b.add("\nTelefoon: ")
    b.add(str(contact["patient_phone"]), "Contactdetails")
    b.add(" | e-mail: ")
    b.add(str(contact["patient_email"]), "Contactdetails")

    b.add("\n\n")
    b.add(NETHERLANDS_SECTION_TEXT[document_type])
    b.add(" ")
    b.add(_clinical_summary(record))
    b.add("\nDocumentreferentie: ")
    identifier_key = TYPE_IDENTIFIER_FIELDS[document_type]
    b.add(str(identifiers[identifier_key]), "ID:Patient")

    b.add("\n\nBehandelaar: dr. ")
    b.add(_person_name(record, "caregiver"), "Name:Caregiver")
    b.add(" | BIG-register: ")
    b.add(str(identifiers["caregiver_registry"]), "ID:Caregiver")
    b.add("\nZorgorganisatie: ")
    b.add(_pair_value(record["healthcare_institution"]), "Organization:Healthcare")
    b.add("\nVestigingsplaats zorgverlener: ")
    b.add(_pair_value(record["caregiver_locality"]), "Address_Location:Caregiver")

    b.add("\n\nContactpersoon: ")
    b.add(_person_name(record, "relative"), "Name:Other")
    b.add(" | beroep: ")
    b.add(str(record["profession"]), "Profession")
    b.add("\nPlaats contactpersoon: ")
    b.add(_pair_value(record["other_location"]), "Address_Location:Other")
    b.add("\nWerkgever of vereniging: ")
    b.add(str(record["other_org"]), "Organization:Other")
    b.add("\nVervolgafspraak: ")
    b.add(str(record["followup_date"]), "Date")

    return apply_production_post_process(
        b.doc(document_id, _metadata(record, document_type))
    )


def _renderer(document_type: str) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    def render(document_id: str, record: dict[str, Any]) -> dict[str, Any]:
        return _render_document(document_id, record, document_type)

    render.__name__ = f"render_{document_type}_nl_nl"
    return render


NETHERLANDS_RENDERERS = {
    document_type: _renderer(document_type)
    for document_type in NETHERLANDS_DOCUMENT_TITLES
}


def render_documents_from_case_records_nl_nl(
    records: list[dict[str, Any]], *, seed: int = 20260508
) -> list[dict[str, Any]]:
    """Render Netherlands records without calling the Belgian renderer."""

    documents = []
    for index, record in enumerate(records):
        document_type = str(record["document_type"])
        try:
            renderer = NETHERLANDS_RENDERERS[document_type]
        except KeyError as exc:
            raise ValueError(
                f"unsupported nl-NL document type {document_type!r}; expected one of: "
                f"{', '.join(NETHERLANDS_RENDERERS)}"
            ) from exc
        document_id = str(
            record.get("case_id", f"case-{index + 1:05d}")
        ).replace("case-", "synthetic-")
        documents.append(renderer(document_id, record))
    return documents
