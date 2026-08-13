"""Helpers for constructing exact character-offset annotations."""

from __future__ import annotations

from dataclasses import dataclass, field

from .labels import split_label


@dataclass
class SpanBuilder:
    parts: list[str] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.parts)

    @property
    def position(self) -> int:
        return len(self.text)

    def add(self, value: str, label: str | None = None) -> None:
        begin = self.position
        self.parts.append(value)
        end = self.position
        if label is None:
            return

        category, subtype = split_label(label)
        self.annotations.append(
            {
                "begin": begin,
                "end": end,
                "label": label,
                "text": value,
                "Category": category,
                "Subtype": subtype,
                "confirmed": True,
            }
        )

    def doc(self, document_id: str, metadata: dict | None = None) -> dict:
        return {
            "document_id": document_id,
            "text": self.text,
            "spans": self.annotations,
            "metadata": metadata or {},
            "annotated": True,
        }


# --- canonical metadata helpers (meddeid-core vocabulary) -----------------

# Dutch/Belgian surname particles: when they appear after the first token they
# start the family name (e.g. "Jan van den Berg" -> given "Jan", family "van den Berg").
_NAME_PARTICLES = {
    "van", "de", "den", "der", "ten", "ter", "het", "op", "in", "aan",
    "du", "le", "la", "el", "da", "di", "von", "vande", "vanden", "vander",
}

# Leading courtesy/professional titles to strip before splitting.
_NAME_TITLES = {
    "dr", "prof", "drs", "ir", "mr", "mevr", "mevrouw", "dhr", "meneer",
    "mw", "mej", "dr.ir", "prof.dr",
}


def split_person_name(full: str | None) -> dict[str, str]:
    """Best-effort split of a full name into ``{given_name, family_name}``.

    ``PersonProfile`` only stores the joined name string, so we split it here.
    Strips leading titles (Dr., Prof., ...), handles Dutch/Belgian particles;
    single-token names become family_name.
    """
    full = (full or "").strip()
    if not full:
        return {}
    parts = full.split()
    while len(parts) > 1 and parts[0].lower().rstrip(".") in _NAME_TITLES:
        parts = parts[1:]
    if len(parts) == 1:
        return {"family_name": parts[0]}
    lower = [p.lower().rstrip(".") for p in parts]
    idx = next((i for i in range(1, len(parts)) if lower[i] in _NAME_PARTICLES), None)
    if idx is None:
        idx = len(parts) - 1
    return {"given_name": " ".join(parts[:idx]), "family_name": " ".join(parts[idx:])}


def canonical_pii_metadata(case) -> dict:
    """Canonical patient/caregiver/date metadata from a case (duck-typed).

    Emits only what is available (``patient``, ``caregivers``,
    ``document_creation_date``) so records without the data stay lean.
    """
    meta: dict = {}

    patient = split_person_name(getattr(getattr(case, "patient", None), "name", None))
    birthdate = getattr(case, "birthdate", None)
    if birthdate:
        patient["birth_date"] = birthdate
    if patient:
        meta["patient"] = patient

    caregivers = []
    for attr in ("caregiver", "secondary_caregiver"):
        profile = getattr(case, attr, None)
        name = split_person_name(getattr(profile, "name", None)) if profile else {}
        if name:
            caregivers.append(name)
    if caregivers:
        meta["caregivers"] = caregivers

    creation_date = getattr(case, "encounter_date", None)
    if creation_date:
        meta["document_creation_date"] = creation_date

    return meta
