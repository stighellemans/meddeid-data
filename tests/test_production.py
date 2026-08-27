from __future__ import annotations

import json

import pytest

from meddeid_core import BERT_ENTITY_LABELS, ProfileRef
from meddeid_data.production import (
    ProductionPlan,
    apply_document_edit,
    audit_batch,
    batch_paths,
    finalize_production,
    initialize_production,
    invalidate_document,
    production_status,
    record_review,
    register_batch_documents,
    sign_off_batch,
)
from meddeid_data.production_cli import main as production_main
from meddeid_data.production_backends import resolve_production_backend
from meddeid_data.cli import main as data_main


def _document(document_id: str, profile: str, family: str, value: str) -> dict:
    pathway = {
        "gb-1": "respiratory",
        "us-1": "orthopedic",
        "gb-2": "neurology",
        "us-2": "dermatology",
    }.get(document_id, "general")
    text = (
        f"Patient {value} attended the {family.replace('_', ' ')}. "
        f"The {pathway} pathway was completed."
    )
    begin = text.index(value)
    return {
        "document_id": document_id,
        "text": text,
        "spans": [
            {
                "begin": begin,
                "end": begin + len(value),
                "text": value,
                "label": "Name:Patient",
            }
        ],
        "metadata": {
            "lang": profile,
            "generation_profile": profile,
            "document_type": family,
        },
    }


def _plan() -> ProductionPlan:
    return ProductionPlan(
        profiles=(ProfileRef.parse("en-GB"), ProfileRef.parse("en-US")),
        target_documents=4,
        batch_size=2,
        benchmark_documents=2,
        seed=7,
        document_families=("clinic_note",),
        require_full_label_coverage=False,
        near_duplicate_distance=None,
    )


def test_production_plan_serializes_only_unversioned_locale_identities() -> None:
    payload = _plan().to_dict()
    assert [profile["selection"] for profile in payload["profiles"]] == [
        "en-GB",
        "en-US",
    ]
    assert all("version" not in profile for profile in payload["profiles"])
    payload["profiles"][0]["version"] = "1"
    with pytest.raises(ValueError, match="unversioned"):
        ProductionPlan.from_dict(payload)


def _accept_and_sign(workspace, batch_index: int) -> None:
    audit = audit_batch(workspace, batch_index)
    assert audit["gate_passed"] is True
    for document in (
        json.loads(line)
        for line in batch_paths(workspace, batch_index)["documents"]
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        record_review(
            workspace,
            batch_index,
            document_id=document["document_id"],
            decision="accept",
            reviewer="fixture-reviewer",
        )
    sign_off_batch(
        workspace,
        batch_index,
        reviewer="fixture-reviewer",
        notes="all documents reviewed",
    )


def test_production_state_requires_audit_review_and_content_bound_signoff(
    tmp_path,
) -> None:
    initialize_production(tmp_path, _plan())
    assert production_status(tmp_path)["next"] == {
        "action": "generate",
        "batch_index": 0,
    }
    register_batch_documents(
        tmp_path,
        0,
        documents=(
            _document("gb-1", "en-GB", "clinic_note", "Alice North"),
            _document("us-1", "en-US", "clinic_note", "Brian West"),
        ),
    )
    assert production_status(tmp_path)["next"] == {"action": "audit", "batch_index": 0}
    audit_batch(tmp_path, 0)
    assert production_status(tmp_path)["next"] == {"action": "review", "batch_index": 0}
    for document_id in ("gb-1", "us-1"):
        record_review(
            tmp_path,
            0,
            document_id=document_id,
            decision="accept",
            reviewer="reviewer",
        )
    assert production_status(tmp_path)["next"] == {
        "action": "sign-off",
        "batch_index": 0,
    }
    sign_off_batch(tmp_path, 0, reviewer="reviewer", notes="checked")
    assert production_status(tmp_path)["next"] == {
        "action": "generate",
        "batch_index": 1,
    }

    documents_path = batch_paths(tmp_path, 0)["documents"]
    rows = [json.loads(line) for line in documents_path.read_text().splitlines()]
    rows[0]["text"] += " changed"
    documents_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    assert production_status(tmp_path)["batches"][0]["state"] == "needs_audit"


def test_finalization_builds_balanced_deterministic_sealed_split(tmp_path) -> None:
    initialize_production(tmp_path, _plan())
    batches = (
        (
            _document("gb-1", "en-GB", "clinic_note", "Alice North"),
            _document("us-1", "en-US", "clinic_note", "Brian West"),
        ),
        (
            _document("gb-2", "en-GB", "clinic_note", "Claire Green"),
            _document("us-2", "en-US", "clinic_note", "Daniel Stone"),
        ),
    )
    for batch_index, documents in enumerate(batches):
        register_batch_documents(tmp_path, batch_index, documents=documents)
        _accept_and_sign(tmp_path, batch_index)
    manifest = finalize_production(tmp_path)
    assert manifest["total_documents"] == 4
    assert manifest["development_documents"] == 2
    assert manifest["benchmark_documents"] == 2
    assert manifest["benchmark_profiles"] == {"en-GB": 1, "en-US": 1}
    assert manifest["allowed_labels"] == list(BERT_ENTITY_LABELS)
    assert "Anonymize_Other" in manifest["forbidden_generated_labels"]
    assert production_status(tmp_path)["complete"] is True


def test_invalid_generated_labels_are_rejected_before_persistence(tmp_path) -> None:
    initialize_production(tmp_path, _plan())
    document = _document("gb-1", "en-GB", "clinic_note", "Alice North")
    document["spans"][0]["label"] = "Anonymize_Other"
    with pytest.raises(RuntimeError, match="Anonymize_Other"):
        register_batch_documents(
            tmp_path,
            0,
            documents=(document, _document("us-1", "en-US", "clinic_note", "B West")),
        )


def test_production_cli_requires_explicit_remote_authorization(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit authorization"):
        production_main(
            [
                "init",
                str(tmp_path),
                "--profile",
                "en-GB,en-US",
                "--count",
                "7000",
                "--mode",
                "remote",
                "--author-model",
                "gpt-5.6-luna",
            ]
        )


def test_remote_backend_resolution_is_profile_explicit() -> None:
    assert (
        resolve_production_backend(None, {"en-GB", "en-US"}).backend_id
        == "english-luna"
    )
    with pytest.raises(ValueError, match="does not support"):
        resolve_production_backend("english-luna", {"en-GB"})


def test_invalidation_is_recoverable_and_returns_batch_to_generation(tmp_path) -> None:
    initialize_production(tmp_path, _plan())
    register_batch_documents(
        tmp_path,
        0,
        documents=(
            _document("gb-1", "en-GB", "clinic_note", "Alice North"),
            _document("us-1", "en-US", "clinic_note", "Brian West"),
        ),
    )
    _accept_and_sign(tmp_path, 0)
    result = invalidate_document(
        tmp_path,
        0,
        document_id="gb-1",
        reason="PII boundary must be regenerated",
    )
    assert result["document_id"] == "gb-1"
    assert production_status(tmp_path)["next"] == {
        "action": "generate",
        "batch_index": 0,
    }
    archive = batch_paths(tmp_path, 0)["invalidated"].read_text(encoding="utf-8")
    assert "PII boundary must be regenerated" in archive


def test_manual_edit_invalidates_quality_and_only_resets_changed_review(
    tmp_path,
) -> None:
    initialize_production(tmp_path, _plan())
    documents = (
        _document("gb-1", "en-GB", "clinic_note", "Alice North"),
        _document("us-1", "en-US", "clinic_note", "Brian West"),
    )
    register_batch_documents(tmp_path, 0, documents=documents)
    audit_batch(tmp_path, 0)
    for document_id in ("gb-1", "us-1"):
        record_review(
            tmp_path,
            0,
            document_id=document_id,
            decision="accept",
            reviewer="reviewer",
        )
    edited = _document("gb-1", "en-GB", "clinic_note", "Alicia North")
    result = apply_document_edit(
        tmp_path,
        0,
        document=edited,
        editor="reviewer",
        reason="corrected patient-name boundary",
    )
    assert result["before_sha256"] != result["after_sha256"]
    assert production_status(tmp_path)["next"] == {
        "action": "audit",
        "batch_index": 0,
    }
    audit_batch(tmp_path, 0)
    decisions = {
        row["document_id"]: row
        for row in (
            json.loads(line)
            for line in batch_paths(tmp_path, 0)["review"]
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    assert decisions["gb-1"]["decision"] == "pending"
    assert decisions["us-1"]["decision"] == "accept"


def test_attempt_cost_counts_api_ledger_events_including_nonaccepted_work(
    tmp_path,
) -> None:
    initialize_production(tmp_path, _plan())
    register_batch_documents(
        tmp_path,
        0,
        documents=(
            _document("gb-1", "en-GB", "clinic_note", "Alice North"),
            _document("us-1", "en-US", "clinic_note", "Brian West"),
        ),
        attempts=(
            {
                "stage": "author",
                "outcome": "validation_failed",
                "model": "gpt-5.6-luna",
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            },
        ),
    )
    assert production_status(tmp_path)["estimated_attempt_cost_usd"] == 0.0008


def test_production_commands_are_available_through_main_data_cli(tmp_path) -> None:
    assert (
        data_main(
            [
                "production",
                "init",
                str(tmp_path),
                "--profile",
                "en-GB,en-US",
                "--count",
                "4",
                "--batch-size",
                "2",
                "--allow-incomplete-label-coverage",
                "--near-duplicate-distance",
                "3",
            ]
        )
        == 0
    )
    assert data_main(["production", "status", str(tmp_path)]) == 0
