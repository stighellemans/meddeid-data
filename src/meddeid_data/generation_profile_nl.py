"""Adapter exposing the legacy Dutch generator through the profile contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meddeid_language_nl import get_profile

from .clinical_cases import generate_case_records
from .generation_profiles import GenerationProfile
from .generator import generate_documents, render_documents_from_case_records
from .judge import judge_documents


def _resource_manifest() -> dict[str, Any]:
    language_manifest = get_profile("nl-BE").manifest()
    return {
        "manifest_version": "meddeid.generation-resources.v1",
        "package": "meddeid-data",
        "profile_id": "nl-BE",
        "language_resources": language_manifest.get("resources", {}),
        "generator": "case-model-renderer-v2",
        "document_types": 18,
    }


def _build_cases(
    count: int,
    *,
    seed: int = 20260508,
    synthea_seeds: list[dict[str, Any]] | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    records = generate_case_records(
        count,
        seed=seed,
        synthea_seeds=synthea_seeds,
        start_index=start_index,
    )
    for record in records:
        record["generation_profile"] = {
            "contract_version": "meddeid.generation-profile.v1",
            "profile_id": "nl-BE",
        }
    return records


def _generate(
    count: int,
    *,
    seed: int = 20260508,
    synthea_csv_dir: Path | None = None,
    auto_synthea: bool = False,
    synthea_repo_dir: Path = Path("external/synthea"),
    synthea_population: int | None = None,
    force_synthea: bool = False,
    require_synthea: bool = False,
) -> list[dict[str, Any]]:
    documents = generate_documents(
        count,
        seed=seed,
        synthea_csv_dir=synthea_csv_dir,
        auto_synthea=auto_synthea,
        synthea_repo_dir=synthea_repo_dir,
        synthea_population=synthea_population,
        force_synthea=force_synthea,
        require_synthea=require_synthea,
    )
    return _tag_documents(documents)


def _tag_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for document in documents:
        metadata = document.setdefault("metadata", {})
        metadata["lang"] = "nl-BE"
        metadata["generation_profile"] = "nl-BE"
    return documents


def _render_cases(records: list[dict[str, Any]], *, seed: int = 20260508) -> list[dict[str, Any]]:
    return _tag_documents(render_documents_from_case_records(records, seed=seed))


NL_BE_GENERATION_PROFILE = GenerationProfile(
    profile_id="nl-BE",
    language_tags=("nl-BE",),
    description="Dutch clinical notes in the Belgian setting",
    generate_documents=_generate,
    build_case_records=_build_cases,
    render_case_records=_render_cases,
    review_documents=judge_documents,
    resource_manifest_provider=_resource_manifest,
    language_manifest_provider=lambda: get_profile("nl-BE").manifest(),
    default_preannotation_model="stighellemans/meddeid-dutch-synth",
)
