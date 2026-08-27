"""Deterministic, realistic email-domain selection for synthetic profiles."""

from __future__ import annotations

import hashlib

GLOBAL_EMAIL_DOMAINS_WEIGHTED = (
    ("gmail.com",) * 8
    + ("outlook.com",) * 6
    + ("hotmail.com",) * 5
    + ("yahoo.com",) * 4
    + ("icloud.com",) * 3
    + ("proton.me",)
    + ("fastmail.com",)
)

EMAIL_DOMAINS_BY_PROFILE = {
    "en-GB": (
        ("gmail.com",) * 8
        + ("outlook.com",) * 6
        + ("hotmail.com",) * 5
        + ("yahoo.co.uk",) * 4
        + ("icloud.com",) * 3
        + ("btinternet.com",) * 2
        + ("sky.com",) * 2
        + ("virginmedia.com",) * 2
        + ("talktalk.net",)
        + ("proton.me",)
        + ("live.co.uk",)
        + ("fastmail.com",)
    ),
    "en-US": (
        ("gmail.com",) * 9
        + ("outlook.com",) * 6
        + ("yahoo.com",) * 5
        + ("hotmail.com",) * 4
        + ("icloud.com",) * 3
        + ("aol.com",) * 2
        + ("comcast.net",) * 2
        + ("att.net",)
        + ("verizon.net",)
        + ("proton.me",)
        + ("fastmail.com",)
        + ("me.com",)
    ),
    "nl-BE": (
        ("gmail.com",) * 7
        + ("hotmail.com",) * 5
        + ("outlook.com",) * 5
        + ("telenet.be",) * 5
        + ("skynet.be",) * 4
        + ("proximus.be",) * 3
        + ("icloud.com",) * 2
        + ("outlook.be",) * 2
        + ("hotmail.be",) * 2
        + ("scarlet.be",)
        + ("live.be",)
        + ("yahoo.com",)
    ),
    "nl-NL": (
        ("gmail.com",) * 8
        + ("hotmail.com",) * 5
        + ("outlook.com",) * 5
        + ("ziggo.nl",) * 4
        + ("kpnmail.nl",) * 3
        + ("icloud.com",) * 2
        + ("xs4all.nl",) * 2
        + ("planet.nl",)
        + ("live.nl",)
        + ("proton.me",)
        + ("yahoo.com",)
    ),
}


def choose_email_domain(profile_id: str, local: str) -> str:
    """Choose a stable locale-aware domain, with a safe global fallback."""

    domains = EMAIL_DOMAINS_BY_PROFILE.get(profile_id, GLOBAL_EMAIL_DOMAINS_WEIGHTED)
    domain_index = int.from_bytes(
        hashlib.sha256(f"{profile_id}|{local.lower()}".encode()).digest()[:8], "big"
    )
    return domains[domain_index % len(domains)]


def email_address(profile_id: str, local: str) -> str:
    """Render a synthetic address without using a reserved example domain."""

    return f"{local}@{choose_email_domain(profile_id, local)}"
