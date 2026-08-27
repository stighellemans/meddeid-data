"""Private generation semantics beneath the public MedDeID labels.

These values are authoring and audit hints.  They deliberately do not extend
the public model taxonomy and are therefore never used as classifier labels.
"""

from __future__ import annotations

from typing import Any


SOURCE_SLOT_SEMANTIC_TYPES: dict[str, str] = {
    "patient.mrn": "patient_mrn",
    "patient.report_id": "patient_report_identifier",
    "patient.national_id": "patient_identifier.national",
    "caregiver.professional_id": "caregiver.professional_identifier",
    "healthcare.organization": "healthcare_organization.unspecified",
    "other.organization": "other_organization.unspecified",
    "patient.address": "patient_address.postal",
    "other.locality": "other_address.locality",
    "caregiver.locality": "caregiver_address.locality",
    "patient.phone": "patient_contact.phone",
    "patient.email": "patient_contact.email",
    "patient.birth_date": "patient_birthdate",
    "patient.age_contextual": "patient_age.contextual",
    "patient.age_hyphenated": "patient_age.hyphenated",
    "patient.age_abbreviated": "patient_age.abbreviated",
}


def semantic_type_for_target(
    target: dict[str, Any],
    *,
    document_family: str | None = None,
) -> str:
    """Resolve a private semantic type without changing its public label.

    New generators should write ``semantic_type`` explicitly.  The source-slot
    fallback keeps already generated corpora auditable and is intentionally
    conservative: an unspecified subtype is preferable to a fabricated one.
    """

    explicit = target.get("semantic_type") or target.get("semantic_kind")
    if explicit:
        return str(explicit)

    source_slot = str(target.get("source_slot") or target.get("slot") or "")
    if source_slot == "patient.report_id" and document_family == "laboratory_report":
        return "laboratory_accession_id"
    if source_slot == "other.organization":
        value = str(target.get("value") or "").casefold()
        if any(token in value for token in ("school", "academy", "college", "university", "nursery")):
            return "other_organization.school"
        if "housing" in value:
            return "other_organization.housing_service"
        if any(token in value for token in ("transit", "transport")):
            return "other_organization.transport_service"
        if any(token in value for token in ("council", "library", "senior center", "senior centre", "sports club")):
            return "other_organization.community_program"
        return "other_organization.unspecified"
    if source_slot in SOURCE_SLOT_SEMANTIC_TYPES:
        return SOURCE_SLOT_SEMANTIC_TYPES[source_slot]

    label = str(target.get("label") or "")
    if label == "Organization:Healthcare":
        return "healthcare_organization.unspecified"
    if label == "Organization:Other":
        return "other_organization.unspecified"
    if label:
        return f"public_label.{label}"
    return "unspecified"


__all__ = ["SOURCE_SLOT_SEMANTIC_TYPES", "semantic_type_for_target"]
