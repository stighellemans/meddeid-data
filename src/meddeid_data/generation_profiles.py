"""Pluggable language-and-locale profiles for synthetic generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

GENERATION_PROFILE_CONTRACT = "meddeid.generation-profile.v1"
ENTRY_POINT_GROUP = "meddeid.generation_profiles"

GenerateDocuments = Callable[..., list[dict[str, Any]]]
BuildCaseRecords = Callable[..., list[dict[str, Any]]]
RenderCaseRecords = Callable[..., list[dict[str, Any]]]
ReviewDocuments = Callable[
    ..., tuple[list[dict[str, Any]], list[Any], list[dict[str, Any]]]
]
ManifestProvider = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class GenerationProfile:
    """One locale-specific generation pipeline."""

    profile_id: str
    language_tags: tuple[str, ...]
    description: str
    generate_documents: GenerateDocuments
    build_case_records: BuildCaseRecords
    render_case_records: RenderCaseRecords
    review_documents: ReviewDocuments
    resource_manifest_provider: ManifestProvider
    language_manifest_provider: ManifestProvider | None = None
    default_preannotation_model: str | None = None

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("generation profile ID must be non-empty")
        if not self.language_tags:
            raise ValueError("generation profile must accept at least one language tag")

    @property
    def selection(self) -> str:
        return self.profile_id

    def accepts_language(self, language_tag: str) -> bool:
        normalized = language_tag.strip().replace("_", "-").lower()
        return normalized in {tag.lower() for tag in self.language_tags}

    def manifest(self) -> dict[str, Any]:
        resources = self.resource_manifest_provider()
        if resources.get("profile_id") != self.profile_id:
            raise RuntimeError(
                "generation resource manifest is scoped to another profile"
            )
        manifest = {
            "contract_version": GENERATION_PROFILE_CONTRACT,
            "profile_id": self.profile_id,
            "language_tags": list(self.language_tags),
            "description": self.description,
            "resources": resources,
        }
        if "allowed_labels" in resources:
            manifest["allowed_labels"] = list(resources["allowed_labels"])
        return manifest

    def language_manifest(self) -> dict[str, Any]:
        if self.language_manifest_provider is not None:
            manifest = self.language_manifest_provider()
            if manifest.get("profile_id") != self.profile_id:
                raise RuntimeError("language manifest is scoped to another profile")
            return manifest
        return {
            "profile_id": self.profile_id,
            "language_tags": list(self.language_tags),
            "availability": "generation-only",
        }


def _providers():
    discovered = entry_points()
    if hasattr(discovered, "select"):
        yield from discovered.select(group=ENTRY_POINT_GROUP)
    else:  # pragma: no cover - Python 3.10 importlib compatibility
        yield from discovered.get(ENTRY_POINT_GROUP, ())


def _builtin_profile(profile_id: str) -> GenerationProfile | None:
    normalized = profile_id.strip().replace("_", "-").lower()
    if normalized == "nl-be":
        from .generation_profile_nl import NL_BE_GENERATION_PROFILE

        return NL_BE_GENERATION_PROFILE
    if normalized == "nl-nl":
        from .generation_profile_nl_nl import NL_NL_GENERATION_PROFILE

        return NL_NL_GENERATION_PROFILE
    if normalized == "en-gb":
        from .generation_profile_en import EN_GB_GENERATION_PROFILE

        return EN_GB_GENERATION_PROFILE
    if normalized == "en-us":
        from .generation_profile_en_us import EN_US_GENERATION_PROFILE

        return EN_US_GENERATION_PROFILE
    return None


def resolve_generation_profile(profile_id: str) -> GenerationProfile:
    """Resolve a built-in or installed generation-profile provider."""

    builtin = _builtin_profile(profile_id)
    if builtin is not None:
        return builtin

    attempted: list[str] = []
    for entry_point in _providers():
        attempted.append(entry_point.name)
        provider = entry_point.load()
        try:
            profile = provider(profile_id)
        except ValueError:
            continue
        if not isinstance(profile, GenerationProfile):
            raise TypeError(
                f"generation-profile plugin {entry_point.name!r} returned "
                f"{type(profile).__name__}, expected GenerationProfile"
            )
        if not profile.accepts_language(profile_id):
            raise ValueError(
                f"generation-profile plugin {entry_point.name!r} returned "
                f"{profile.selection}, which does not satisfy {profile_id!r}"
            )
        return profile

    installed = ", ".join(attempted) or "none"
    raise ValueError(
        f"no generation profile provides {profile_id!r}; built-ins: "
        "nl-BE, nl-NL, en-GB, en-US; installed plugins: "
        f"{installed}. Install a matching provider from the "
        f"{ENTRY_POINT_GROUP!r} entry-point group."
    )


def profile_from_case_records(
    records: list[dict[str, Any]],
    *,
    requested_profile_id: str | None = None,
) -> GenerationProfile:
    """Resolve rendering profile from CLI selection or case-record provenance."""

    if not records:
        raise ValueError("cannot render an empty case-record file")
    first = records[0]
    provenance = first.get("generation_profile")
    if not isinstance(provenance, dict):
        provenance = {}
    profile_id = (
        requested_profile_id or provenance.get("profile_id") or first.get("language")
    )
    if not profile_id:
        raise ValueError(
            "case records do not identify a generation profile; pass --language-profile"
        )
    profile = resolve_generation_profile(str(profile_id))

    for index, record in enumerate(records, start=1):
        record_provenance = record.get("generation_profile")
        if isinstance(record_provenance, dict):
            record_id = record_provenance.get("profile_id")
            if record_id and str(record_id).lower() != profile.profile_id.lower():
                raise ValueError(
                    f"case record {index} belongs to {record_id}, "
                    f"not {profile.selection}"
                )
        language = str(record.get("language", ""))
        if language and not profile.accepts_language(language):
            raise ValueError(
                f"case record {index} language {language!r} is incompatible with "
                f"{profile.selection}"
            )
    return profile


def write_review_report(
    results: list[Any], model_reviews: list[dict[str, Any]], path: Path
) -> None:
    """Write the shared human-readable report for any generation profile."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Synthetic DEID Judge Report", ""]
    passed = sum(1 for result in results if result.passed)
    lines.append(f"Documents passed deterministic judge: {passed}/{len(results)}")
    lines.append("")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"## {result.document_id}: {status}")
        if result.corrections:
            lines.append("Corrections:")
            lines.extend(f"- {item}" for item in result.corrections)
        if result.issues:
            lines.append("Issues:")
            lines.extend(f"- {item}" for item in result.issues)
        if not result.corrections and not result.issues:
            lines.append("No issues.")
        lines.append("")
    if model_reviews:
        lines.append("## Optional Judge Reviews")
        lines.append("```json")
        lines.append(json.dumps(model_reviews, ensure_ascii=False, indent=2))
        lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
