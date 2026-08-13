"""Annotation label definitions shared with the custom annotation platform."""

from __future__ import annotations

LABELS = {
    "Address_Location:Caregiver",
    "Address_Location:Other",
    "Address_Location:Patient",
    "Age_Birthdate",
    "Anonymize_Other",
    "Contactdetails",
    "Date",
    "ID:Caregiver",
    "ID:Patient",
    "Name:Caregiver",
    "Name:Other",
    "Name:Patient",
    "Organization:Healthcare",
    "Organization:Other",
    "Profession",
}


def split_label(label: str) -> tuple[str, str | None]:
    category, sep, subtype = label.partition(":")
    return category, subtype if sep else None
