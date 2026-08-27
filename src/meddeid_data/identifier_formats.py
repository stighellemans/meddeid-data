"""Deterministic format diversity for synthetic English identifiers."""

from __future__ import annotations

import hashlib

SUPPORTED_SLOTS = frozenset(
    {"patient.mrn", "patient.report_id", "caregiver.professional_id"}
)

RESERVED_NINO_PREFIXES = ("BG", "GB", "KN", "NK", "NT", "TN", "ZZ")


def _digest(profile_id: str, source_slot: str, identity_key: str) -> bytes:
    return hashlib.sha256(
        f"english-identifier-formats-v1|{profile_id}|{source_slot}|{identity_key}".encode()
    ).digest()


def _digits(digest: bytes, length: int, *, offset: int = 1) -> str:
    """Return a numeric-only synthetic payload with a non-assignable zero lead."""

    width = length - 1
    value = int.from_bytes(digest[offset : offset + 8], "big") % (10**width)
    return f"0{value:0{width}d}"


def format_english_identifier(
    profile_id: str, source_slot: str, identity_key: str
) -> str:
    """Return a stable locale/slot-aware identifier surface.

    Most outputs are digits only. The remaining variants intentionally cover
    common internal-record shapes without returning to a single three-part
    synthetic prefix convention.
    """

    if profile_id not in {"en-GB", "en-US"}:
        raise ValueError(f"unsupported English profile: {profile_id!r}")
    if source_slot not in SUPPORTED_SLOTS:
        raise ValueError(f"unsupported English identifier slot: {source_slot!r}")
    digest = _digest(profile_id, source_slot, str(identity_key))
    percentile = digest[0] % 100

    if source_slot == "patient.mrn":
        if percentile < 65:
            return _digits(digest, 7 + digest[9] % 3)
        if percentile < 75:
            prefix = "H" if profile_id == "en-GB" else "M"
            return f"{prefix}{_digits(digest, 8)[1:]}"
        if percentile < 85:
            value = _digits(digest, 9)
            return f"{value[:3]} {value[3:6]} {value[6:]}"
        if percentile < 93:
            value = _digits(digest, 8)
            return f"{value[:6]}/{value[6:]}"
        return f"MRN-{_digits(digest, 8)}"

    if source_slot == "patient.report_id":
        if percentile < 60:
            return _digits(digest, 8 + digest[9] % 3)
        if percentile < 72:
            value = _digits(digest, 8)
            return f"{20 + digest[10] % 10:02d}/{value[2:]}"
        if percentile < 84:
            prefix = "L" if profile_id == "en-GB" else "A"
            letter = chr(ord("A") + digest[11] % 26)
            return f"{prefix}{20 + digest[10] % 10:02d}{letter}{_digits(digest, 6)[1:]}"
        if percentile < 94:
            return f"LAB-{_digits(digest, 8)}"
        return f"S{20 + digest[10] % 10:02d}-{_digits(digest, 7)}"

    if profile_id == "en-GB":
        value = _digits(digest, 7)
        if percentile < 85:
            return value
        if percentile < 93:
            return f"GMC {value}"
        return f"GMC-TEST-{value}"

    value = _digits(digest, 9)
    if percentile < 85:
        return value
    if percentile < 93:
        return f"LIC{value[1:]}"
    return f"LIC-{value[1:]}"


def _invalid_mod11_number(digest: bytes) -> str:
    """Return an NHS-shaped number that deliberately fails its checksum."""

    base = f"000{int.from_bytes(digest[4:12], 'big') % 10_000_000:07d}"
    weights = range(10, 1, -1)
    total = sum(int(value) * weight for value, weight in zip(base[:9], weights))
    check = 11 - total % 11
    expected = 0 if check == 11 else check
    if check == 10:
        return base
    replacement = (expected + 1) % 10
    return f"{base[:9]}{replacement}"


def synthetic_national_identifier(
    profile_id: str, identity_key: str
) -> dict[str, str]:
    """Return a stable, deliberately non-assignable national/public patient ID.

    The surface shapes mirror identifiers encountered in real English-language
    clinical records, but invalid/reserved components prevent the values from
    representing real people.
    """

    if profile_id not in {"en-GB", "en-US"}:
        raise ValueError(f"unsupported English profile: {profile_id!r}")
    digest = _digest(profile_id, "patient.national_id", str(identity_key))

    if profile_id == "en-US":
        selector = digest[0] % 3
        if selector == 0:
            area = "000"
        elif selector == 1:
            area = "666"
        else:
            area = str(900 + digest[1] % 100)
        group = f"{1 + int.from_bytes(digest[2:4], 'big') % 99:02d}"
        serial = f"{1 + int.from_bytes(digest[4:8], 'big') % 9999:04d}"
        style = digest[8] % 3
        value = (
            f"{area}-{group}-{serial}"
            if style == 0
            else f"{area} {group} {serial}"
            if style == 1
            else f"{area}{group}{serial}"
        )
        return {"value": value, "kind": "ssn", "field_label": "SSN"}

    # Most UK public identifiers are numeric NHS-shaped values. A smaller
    # share uses a National Insurance-shaped value with a reserved prefix.
    if digest[0] % 100 < 70:
        digits = _invalid_mod11_number(digest)
        value = (
            f"{digits[:3]} {digits[3:6]} {digits[6:]}"
            if digest[8] % 3
            else digits
        )
        return {
            "value": value,
            "kind": "nhs_number",
            "field_label": "NHS number",
        }

    prefix = RESERVED_NINO_PREFIXES[digest[1] % len(RESERVED_NINO_PREFIXES)]
    digits = f"{int.from_bytes(digest[2:8], 'big') % 1_000_000:06d}"
    suffix = "ABCD"[digest[8] % 4]
    value = (
        f"{prefix} {digits[:2]} {digits[2:4]} {digits[4:]} {suffix}"
        if digest[9] % 2
        else f"{prefix}{digits}{suffix}"
    )
    return {
        "value": value,
        "kind": "national_insurance_number",
        "field_label": "National Insurance number",
    }
