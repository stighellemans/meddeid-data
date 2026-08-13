import pytest
from meddeid_language_nl import __version__ as language_pack_version
from meddeid_core import validate_record

from meddeid_data.clinical_cases import _load_jsonl_resource
from meddeid_data.generate import generate_example
from meddeid_data.generator import generate_documents
from meddeid_data.lookups import LookupSampler


def test_sampler_uses_nl_be_language_profile_lookups() -> None:
    sampler = LookupSampler()
    expected_source = (
        f"meddeid-language-nl {language_pack_version} nl-BE lookup resources"
    )
    assert sampler.first_name()[1] == expected_source
    assert sampler.family_name()[1] == expected_source
    assert sampler.street()[1] == expected_source
    assert "fallback" not in sampler.source.lower()


def test_full_generator_has_exact_valid_offsets() -> None:
    rows = generate_documents(2, seed=20260508, require_synthea=False)
    assert len(rows) == 2
    assert all(validate_record(row) == [] for row in rows)
    assert all(
        row["metadata"]["generation_method"] == "case-model-renderer-v2" for row in rows
    )
    assert all(
        row["metadata"]["lookup_source"]
        == f"meddeid-language-nl {language_pack_version} nl-BE lookup resources"
        for row in rows
    )


def test_missing_clinical_resource_fails_instead_of_falling_back() -> None:
    with pytest.raises(RuntimeError, match="missing packaged clinical resource"):
        _load_jsonl_resource("does-not-exist.jsonl")


def test_generate_example_uses_full_generator() -> None:
    row = generate_example(3)
    assert row["document_id"] == "synthetic-00004"
    assert validate_record(row) == []
