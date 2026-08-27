"""Sampling from profile-scoped Dutch lookup resources."""

from __future__ import annotations

import random

from meddeid_language_nl import lookup_source, lookup_values

PROVIDER_TERMS = {
    "nl-BE": (
        "kinesist",
        "podoloog",
        "logopedist",
        "diëtist",
        "dietist",
        "psycholoog ",
        "arts ",
        "dokter ",
        "verpleegkundige ",
    ),
    "nl-NL": (
        "fysiotherapeut",
        "podotherapeut",
        "logopedist",
        "diëtist",
        "dietist",
        "psycholoog ",
        "arts ",
        "dokter ",
        "verpleegkundige ",
    ),
}

ORGANIZATION_TERMS = {
    "nl-BE": (
        "algemeen ziekenhuis",
        "centrum",
        "centre",
        "kliniek",
        "clinique",
        "ziekenhuis",
        "hopital",
        "hôpital",
        "residentie",
        "résidence",
        "huisartsenwachtpost",
        "praktijk",
        "maison médicale",
        "medisch",
        "médical",
        "psychiatr",
        "revalidatie",
        "mutualiteit",
        "zorgcentrum",
        "zorggroep",
        "woonzorg",
        "cm ",
    ),
    "nl-NL": (
        "apotheek",
        "centrum",
        "gezondheidscentrum",
        "ggz",
        "hospice",
        "huisartsenpraktijk",
        "kliniek",
        "medisch",
        "praktijk",
        "psychiatr",
        "revalidatie",
        "stichting",
        "thuiszorg",
        "verpleeghuis",
        "ziekenhuis",
        "zorgcentrum",
        "zorggroep",
    ),
}

NON_HEALTHCARE_TERMS = ("politie", "politiezone", "brandweer")


class LookupSampler:
    """Sample profile-scoped Dutch synthetic PII without checkout fallbacks."""

    def __init__(self, seed: int = 20260508, profile_id: str = "nl-BE") -> None:
        self.random = random.Random(seed)
        self.profile_id = profile_id
        self._value_cache: dict[str, tuple[tuple[str, ...], str]] = {}

    @property
    def source(self) -> str:
        return lookup_source(self.profile_id)

    def _values(self, category: str, predicate=None) -> tuple[tuple[str, ...], str]:
        if category not in self._value_cache:
            values = lookup_values(self.profile_id, category)
            if predicate is not None:
                values = tuple(value for value in values if predicate(value))
            if not values:
                raise RuntimeError(
                    f"no usable values in {self.profile_id} language lookup: {category}"
                )
            self._value_cache[category] = values, self.source
        return self._value_cache[category]

    def first_name(self) -> tuple[str, str]:
        values, source = self._values(
            "first_names", lambda value: 2 <= len(value) <= 14 and value[0].isupper()
        )
        return self.random.choice(values), source

    def family_name(self) -> tuple[str, str]:
        values, source = self._values(
            "family_names", lambda value: 3 <= len(value) <= 22 and value[0].isupper()
        )
        return self.random.choice(values), source

    def street(self) -> tuple[str, str]:
        values, source = self._values(
            "streets",
            lambda value: 5 <= len(value) <= 32
            and not any(char.isdigit() for char in value),
        )
        return self.random.choice(values), source

    def locality(self) -> tuple[str, str]:
        values, source = self._values("localities", lambda value: 3 <= len(value) <= 24)
        return self.random.choice(values), source

    def postal_locality(self) -> tuple[str, str]:
        if self.profile_id == "nl-NL":
            locality, source = self.locality()
            return f"{self.random.randrange(1000, 9999)} {locality}", source
        values, source = self._values(
            "postal_code_localities",
            lambda value: len(value) <= 40 and any(char.isdigit() for char in value),
        )
        return self.random.choice(values), source

    def hospital(self) -> tuple[str, str]:
        provider_terms = PROVIDER_TERMS[self.profile_id]
        values, source = self._values(
            "hospitals",
            lambda value: 4 <= len(value) <= 70
            and not any(term in value.lower() for term in provider_terms),
        )
        return self.random.choice(values), source

    def healthcare_institution(self) -> tuple[str, str]:
        organization_terms = ORGANIZATION_TERMS[self.profile_id]
        provider_terms = PROVIDER_TERMS[self.profile_id]
        values, source = self._values(
            "healthcare_institutions",
            lambda value: 4 <= len(value) <= 70
            and any(term in value.lower() for term in organization_terms)
            and not any(term in value.lower() for term in provider_terms)
            and not any(term in value.lower() for term in NON_HEALTHCARE_TERMS),
        )
        return self.random.choice(values), source


def full_name(sampler: LookupSampler) -> tuple[str, dict]:
    first, first_source = sampler.first_name()
    last, last_source = sampler.family_name()
    return f"{first} {last}", {"first_name": first_source, "family_name": last_source}
