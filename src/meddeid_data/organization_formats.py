"""Deterministic display-case diversity for synthetic organizations."""

from __future__ import annotations

import hashlib
import re

UPPERCASE_SHARE_PERCENT = 30

_LOWERCASE_WORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to"}
)
_ACRONYMS = frozenset(
    {
        "CHLA",
        "CAH",
        "CDC",
        "CHI",
        "CTR",
        "HCA",
        "HHS",
        "ICASH",
        "LBJ",
        "MD",
        "MUSC",
        "NHS",
        "NIH",
        "NYU",
        "SUNY",
        "SSM",
        "UAB",
        "UCLA",
        "UCSF",
        "UNC",
        "UPMC",
        "USC",
        "VA",
    }
)


def _digest(profile_id: str, identity_key: str) -> bytes:
    return hashlib.sha256(
        f"healthcare-organization-case-v1-1|{profile_id}|{identity_key}".encode()
    ).digest()


def natural_organization_case(value: str) -> str:
    """Convert an uppercase registry value to readable institutional casing."""

    result = str(value).strip().title()
    result = re.sub(r"'S\b", "'s", result)
    for word in _LOWERCASE_WORDS:
        result = re.sub(
            rf"(?<!^)\b{re.escape(word)}\b",
            word,
            result,
            flags=re.IGNORECASE,
        )
    for acronym in _ACRONYMS:
        result = re.sub(
            rf"\b{re.escape(acronym)}\b",
            acronym,
            result,
            flags=re.IGNORECASE,
        )
    result = re.sub(
        r"\b(I|II|III|IV|V|VI|VII|VIII|IX|X)\b",
        lambda match: match.group(0).upper(),
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bMc([a-z])",
        lambda match: f"Mc{match.group(1).upper()}",
        result,
    )
    return result


def format_healthcare_organization(
    profile_id: str, value: str, identity_key: str
) -> str:
    """Return stable uppercase or natural display casing for one organization."""

    if profile_id not in {"en-GB", "en-US"}:
        raise ValueError(f"unsupported English profile: {profile_id!r}")
    selector = int.from_bytes(_digest(profile_id, str(identity_key))[:4], "big") % 100
    if selector < UPPERCASE_SHARE_PERCENT:
        return str(value).strip().upper()
    return natural_organization_case(value)


__all__ = [
    "UPPERCASE_SHARE_PERCENT",
    "format_healthcare_organization",
    "natural_organization_case",
]
