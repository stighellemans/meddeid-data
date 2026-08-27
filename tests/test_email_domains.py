from __future__ import annotations

from meddeid_data.email_domains import choose_email_domain, email_address


def test_all_supported_profiles_use_varied_non_example_domains() -> None:
    for profile_id in ("en-GB", "en-US", "nl-BE", "nl-NL"):
        domains = {
            choose_email_domain(profile_id, f"synthetic.person{index}")
            for index in range(100)
        }
        assert len(domains) >= 5
        assert all(not domain.startswith("example.") for domain in domains)


def test_unknown_language_profiles_use_global_provider_mix() -> None:
    addresses = {
        email_address("future-language", f"synthetic.person{index}")
        for index in range(100)
    }
    assert len({address.rsplit("@", 1)[1] for address in addresses}) >= 5
    assert all("@example." not in address for address in addresses)
