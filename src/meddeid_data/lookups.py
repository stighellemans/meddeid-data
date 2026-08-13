"""Sampling from the lookup resources owned by the ``nl-BE`` profile."""

from __future__ import annotations

import random

from meddeid_language_nl import lookup_source, lookup_values

PROVIDER_TERMS = (
    "kinesist",
    "podoloog",
    "logopedist",
    "diëtist",
    "dietist",
    "psycholoog ",
    "arts ",
    "dokter ",
    "verpleegkundige ",
)

NON_HEALTHCARE_TERMS = ("politie", "politiezone", "brandweer")


class LookupSampler:
    """Sample Belgian synthetic PII without checkout discovery or fallbacks."""

    def __init__(self, seed: int = 20260508) -> None:
        self.random = random.Random(seed)
        self._value_cache: dict[str, tuple[tuple[str, ...], str]] = {}

    @property
    def source(self) -> str:
        return lookup_source()

    def _values(self, category: str, predicate=None) -> tuple[tuple[str, ...], str]:
        if category not in self._value_cache:
            values = lookup_values(category)
            if predicate is not None:
                values = tuple(value for value in values if predicate(value))
            if not values:
                raise RuntimeError(
                    f"no usable values in nl-BE language lookup: {category}"
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
        values, source = self._values(
            "postal_code_localities",
            lambda value: len(value) <= 40 and any(char.isdigit() for char in value),
        )
        return self.random.choice(values), source

    def hospital(self) -> tuple[str, str]:
        values, source = self._values(
            "hospitals",
            lambda value: 4 <= len(value) <= 70
            and not any(term in value.lower() for term in PROVIDER_TERMS),
        )
        return self.random.choice(values), source

    def healthcare_institution(self) -> tuple[str, str]:
        organization_terms = (
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
        )
        values, source = self._values(
            "healthcare_institutions",
            lambda value: 4 <= len(value) <= 70
            and any(term in value.lower() for term in organization_terms)
            and not any(term in value.lower() for term in PROVIDER_TERMS)
            and not any(term in value.lower() for term in NON_HEALTHCARE_TERMS),
        )
        return self.random.choice(values), source


def full_name(sampler: LookupSampler) -> tuple[str, dict]:
    first, first_source = sampler.first_name()
    last, last_source = sampler.family_name()
    return f"{first} {last}", {"first_name": first_source, "family_name": last_source}


# Backwards-compatible name for the small public API introduced in 0.1.0.
FullLookups = LookupSampler
