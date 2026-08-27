"""Generic, resumable production state for synthetic MedDeID corpora.

Locale packages provide cases, rendering, resources, and validators.  This
module owns the language-neutral transition rules: batching, locking, attempt
accounting, quality gates, content-bound review, sign-off, and sealed splits.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from meddeid_core import (
    BERT_ENTITY_LABELS,
    AttemptRecord,
    BatchManifest,
    ProfileRef,
    normalize_record,
    validate_record,
)

from .corpus_quality import CorpusDiversityContract, audit_corpus_diversity
from .generation_profiles import resolve_generation_profile
from .production_backends import estimate_attempt_cost, resolve_production_backend


PRODUCTION_PLAN_CONTRACT = "meddeid.production-plan.v1"
PRODUCTION_STATUS_CONTRACT = "meddeid.production-status.v1"
FINAL_CORPUS_CONTRACT = "meddeid.synthetic-corpus.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_sha256(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        normalize_record(dict(document)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plan_profile(value: object) -> ProfileRef:
    if isinstance(value, Mapping):
        if value.get("version") is not None:
            raise ValueError(
                "production plans use unversioned locale profiles; remove profile.version"
            )
        return ProfileRef(str(value["selection"]))
    return ProfileRef.parse(str(value))


@dataclass(frozen=True)
class ProductionPlan:
    profiles: tuple[ProfileRef, ...]
    target_documents: int
    batch_size: int = 500
    benchmark_documents: int = 0
    seed: int = 20260508
    mode: str = "local"
    remote_authorized: bool = False
    backend: str | None = None
    author_model: str | None = None
    reasoning_effort: str | None = None
    max_cost_usd: float | None = None
    document_families: tuple[str, ...] = ()
    require_full_label_coverage: bool = True
    near_duplicate_distance: int | None = 3

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError("production requires at least one profile")
        if len({profile.selection for profile in self.profiles}) != len(self.profiles):
            raise ValueError("production profiles must be unique")
        if self.target_documents < 2:
            raise ValueError("target_documents must be at least two")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.benchmark_documents < self.target_documents:
            raise ValueError(
                "benchmark_documents must be smaller than target_documents"
            )
        if self.mode not in {"local", "remote"}:
            raise ValueError("production mode must be local or remote")
        if self.mode == "remote" and not self.author_model:
            raise ValueError("remote production requires an author_model")
        if self.mode == "remote" and not self.remote_authorized:
            raise ValueError("remote production requires explicit authorization")
        if self.max_cost_usd is not None and self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductionPlan":
        return cls(
            profiles=tuple(_plan_profile(item) for item in value["profiles"]),
            target_documents=int(value["target_documents"]),
            batch_size=int(value.get("batch_size", 500)),
            benchmark_documents=int(value.get("benchmark_documents", 0)),
            seed=int(value.get("seed", 20260508)),
            mode=str(value.get("mode", "local")),
            remote_authorized=bool(value.get("remote_authorized", False)),
            backend=value.get("backend"),
            author_model=value.get("author_model"),
            reasoning_effort=value.get("reasoning_effort"),
            max_cost_usd=(
                float(value["max_cost_usd"])
                if value.get("max_cost_usd") is not None
                else None
            ),
            document_families=tuple(value.get("document_families", ())),
            require_full_label_coverage=bool(
                value.get("require_full_label_coverage", True)
            ),
            near_duplicate_distance=value.get("near_duplicate_distance", 3),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_version"] = PRODUCTION_PLAN_CONTRACT
        payload["profiles"] = [profile.to_dict() for profile in self.profiles]
        payload["allowed_labels"] = list(BERT_ENTITY_LABELS)
        payload["forbidden_generated_labels"] = ["Anonymize_Other"]
        payload["independently_authored"] = True
        return payload


def plan_path(workspace: Path) -> Path:
    return workspace / "production.json"


def batch_paths(workspace: Path, batch_index: int) -> dict[str, Path]:
    directory = workspace / "batches" / f"batch-{batch_index + 1:02d}"
    return {
        "directory": directory,
        "cases": directory / "cases.jsonl",
        "documents": directory / "documents.jsonl",
        "marked": directory / "marked.jsonl",
        "attempts": directory / "usage-attempts.jsonl",
        "failures": directory / "failures.jsonl",
        "invalidated": directory / "invalidated-documents.jsonl",
        "edits": directory / "document-edits.jsonl",
        "quality": directory / "quality-report.json",
        "review": directory / "personal-review-decisions.jsonl",
        "signoff": directory / "personal-review-signoff.json",
        "manifest": directory / "manifest.json",
    }


@contextmanager
def production_lock(workspace: Path):
    """Prevent paid work in two batches of the same corpus at once."""

    workspace.mkdir(parents=True, exist_ok=True)
    lock_path = workspace / ".production.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "this corpus already has an active production writer"
            ) from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def initialize_production(workspace: Path, plan: ProductionPlan) -> Path:
    workspace = workspace.expanduser().resolve()
    target = plan_path(workspace)
    if target.is_file():
        existing = ProductionPlan.from_dict(
            json.loads(target.read_text(encoding="utf-8"))
        )
        if existing != plan:
            raise RuntimeError(
                "production workspace already contains a different immutable plan"
            )
        return target
    workspace.mkdir(parents=True, exist_ok=True)
    _atomic_json(target, plan.to_dict())
    return target


def load_plan(workspace: Path) -> ProductionPlan:
    target = plan_path(workspace)
    if not target.is_file():
        raise RuntimeError(f"missing production plan: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("contract_version") != PRODUCTION_PLAN_CONTRACT:
        raise RuntimeError(
            f"unsupported production contract: {value.get('contract_version')!r}"
        )
    if tuple(value.get("allowed_labels", ())) != tuple(BERT_ENTITY_LABELS):
        raise RuntimeError("production plan does not pin exact BERT_ENTITY_LABELS")
    if "Anonymize_Other" not in value.get("forbidden_generated_labels", ()):
        raise RuntimeError("production plan does not forbid Anonymize_Other")
    return ProductionPlan.from_dict(value)


def batch_count(plan: ProductionPlan) -> int:
    return (plan.target_documents + plan.batch_size - 1) // plan.batch_size


def expected_batch_documents(plan: ProductionPlan, batch_index: int) -> int:
    if not 0 <= batch_index < batch_count(plan):
        raise IndexError(f"batch index {batch_index} is outside this production plan")
    return min(plan.batch_size, plan.target_documents - batch_index * plan.batch_size)


def _profile_for_document(document: Mapping[str, Any]) -> str:
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    value = metadata.get("generation_profile") or metadata.get("lang")
    return ProfileRef.parse(str(value)).selection if value else ""


def validate_generated_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    plan: ProductionPlan,
    expected: int,
) -> list[str]:
    failures: list[str] = []
    if len(documents) != expected:
        failures.append(f"expected {expected} documents, found {len(documents)}")
    identifiers = [str(document.get("document_id") or "") for document in documents]
    if any(not identifier for identifier in identifiers):
        failures.append("one or more documents have no document_id")
    if len(identifiers) != len(set(identifiers)):
        failures.append("document IDs are not unique")
    allowed_profiles = {profile.selection for profile in plan.profiles}
    for index, document in enumerate(documents, start=1):
        document_id = str(document.get("document_id") or f"row-{index}")
        for problem in validate_record(dict(document), strict_taxonomy=True):
            failures.append(f"{document_id}: {problem}")
        profile = _profile_for_document(document)
        if profile not in allowed_profiles:
            failures.append(f"{document_id}: profile {profile!r} is outside the plan")
        for span in document.get("spans", ()):
            if not isinstance(span, Mapping):
                continue
            label = str(span.get("label") or "")
            if label == "Anonymize_Other":
                failures.append(f"{document_id}: forbidden Anonymize_Other span")
            elif label not in BERT_ENTITY_LABELS:
                failures.append(
                    f"{document_id}: generated label {label!r} is not allowed"
                )
    return list(dict.fromkeys(failures))


def register_batch_documents(
    workspace: Path,
    batch_index: int,
    *,
    documents: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]] = (),
    attempts: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Atomically register one backend's output after canonical validation."""

    plan = load_plan(workspace)
    expected = expected_batch_documents(plan, batch_index)
    failures = validate_generated_documents(documents, plan=plan, expected=expected)
    if failures:
        raise RuntimeError("generated batch is invalid: " + "; ".join(failures[:5]))
    paths = batch_paths(workspace, batch_index)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    prior_attempts = _read_jsonl(paths["attempts"])
    _write_jsonl(paths["cases"], cases)
    _write_jsonl(paths["documents"], (normalize_record(dict(row)) for row in documents))
    _write_jsonl(paths["attempts"], (*prior_attempts, *attempts))
    for stale in (paths["quality"], paths["signoff"], paths["manifest"]):
        if stale.exists():
            stale.unlink()


def _invalidate_derived_artifacts(workspace: Path, paths: Mapping[str, Path]) -> None:
    """Remove reproducible outputs that no longer describe current documents."""

    for key in ("quality", "signoff", "manifest"):
        if paths[key].exists():
            paths[key].unlink()
    final = workspace / "final"
    for name in ("development.jsonl", "benchmark.jsonl", "manifest.json"):
        target = final / name
        if target.exists():
            target.unlink()


def _prior_documents(workspace: Path, batch_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(batch_index):
        rows.extend(_read_jsonl(batch_paths(workspace, index)["documents"]))
    return rows


def _profile_allocations(
    plan: ProductionPlan, batch_index: int
) -> list[tuple[ProfileRef, int, int]]:
    expected = expected_batch_documents(plan, batch_index)
    base, remainder = divmod(expected, len(plan.profiles))
    allocations: list[tuple[ProfileRef, int, int]] = []
    for profile_index, profile in enumerate(plan.profiles):
        count = base + (1 if profile_index < remainder else 0)
        prior = 0
        for earlier in range(batch_index):
            earlier_expected = expected_batch_documents(plan, earlier)
            earlier_base, earlier_remainder = divmod(
                earlier_expected, len(plan.profiles)
            )
            prior += earlier_base + (1 if profile_index < earlier_remainder else 0)
        allocations.append((profile, count, prior))
    return allocations


def _generate_local_batch(
    workspace: Path, plan: ProductionPlan, batch_index: int
) -> None:
    paths = batch_paths(workspace, batch_index)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for profile_index, (profile_ref, count, start_index) in enumerate(
        _profile_allocations(plan, batch_index)
    ):
        if count == 0:
            continue
        profile = resolve_generation_profile(profile_ref.selection)
        profile_cases = profile.build_case_records(
            count,
            seed=plan.seed + batch_index * len(plan.profiles) + profile_index,
            start_index=start_index,
        )
        rendered = profile.render_case_records(
            profile_cases,
            seed=plan.seed + batch_index * len(plan.profiles) + profile_index,
        )
        reviewed, results, _ = profile.review_documents(rendered)
        failures = [result for result in results if not result.passed]
        if failures:
            raise RuntimeError(
                f"{profile_ref.selection} local judge rejected {len(failures)} documents"
            )
        cases.extend(profile_cases)
        documents.extend(reviewed)
        for document in reviewed:
            attempts.append(
                AttemptRecord(
                    document_id=str(document["document_id"]),
                    attempt=1,
                    role="local",
                    outcome="accepted",
                    created_at=_now(),
                    details={"profile": profile_ref.identifier},
                ).to_dict()
            )
    problems = validate_generated_documents(
        documents, plan=plan, expected=expected_batch_documents(plan, batch_index)
    )
    if problems:
        raise RuntimeError("generated batch is invalid: " + "; ".join(problems[:5]))
    register_batch_documents(
        workspace,
        batch_index,
        documents=documents,
        cases=cases,
        attempts=attempts,
    )


def _generate_remote_batch(
    workspace: Path, plan: ProductionPlan, batch_index: int
) -> None:
    backend = resolve_production_backend(
        plan.backend, {profile.selection for profile in plan.profiles}
    )
    backend.generate_batch(workspace, plan, batch_index)
    paths = batch_paths(workspace, batch_index)
    documents = _read_jsonl(paths["documents"])
    failures = validate_generated_documents(
        documents,
        plan=plan,
        expected=expected_batch_documents(plan, batch_index),
    )
    if failures:
        raise RuntimeError(
            "remote batch remains incomplete or invalid and must be resumed: "
            + "; ".join(failures[:5])
        )
    # Backends may create an immediate diagnostic report.  A production gate
    # is always rebuilt through audit_batch so shared and locale-specific
    # checks are bound to the same current document bytes.
    _invalidate_derived_artifacts(workspace, paths)


def _attempt_cost(workspace: Path) -> float:
    total = 0.0
    for index in range(batch_count(load_plan(workspace))):
        for attempt in _read_jsonl(batch_paths(workspace, index)["attempts"]):
            total += estimate_attempt_cost(attempt)
    return total


def _projected_completion_cost(
    workspace: Path, plan: ProductionPlan, batch_index: int
) -> float | None:
    """Project the next generation step from real prior attempt spend."""

    attempts: list[dict[str, Any]] = []
    for index in range(batch_count(plan)):
        attempts.extend(_read_jsonl(batch_paths(workspace, index)["attempts"]))
    attempted_documents = {
        str(attempt.get("document_id"))
        for attempt in attempts
        if attempt.get("document_id")
    }
    current_cost = sum(estimate_attempt_cost(attempt) for attempt in attempts)
    if not attempted_documents or current_cost <= 0:
        return None
    current_documents = len(
        _read_jsonl(batch_paths(workspace, batch_index)["documents"])
    )
    missing = max(0, expected_batch_documents(plan, batch_index) - current_documents)
    return current_cost + missing * (current_cost / len(attempted_documents))


def append_attempt(workspace: Path, batch_index: int, attempt: AttemptRecord) -> None:
    plan = load_plan(workspace)
    prospective = _attempt_cost(workspace) + attempt.estimated_cost_usd
    if plan.max_cost_usd is not None and prospective > plan.max_cost_usd:
        raise RuntimeError(
            f"attempt would exceed the configured USD {plan.max_cost_usd:.2f} cost ceiling"
        )
    _append_jsonl(batch_paths(workspace, batch_index)["attempts"], attempt.to_dict())


def audit_batch(workspace: Path, batch_index: int) -> dict[str, Any]:
    plan = load_plan(workspace)
    paths = batch_paths(workspace, batch_index)
    documents = _read_jsonl(paths["documents"])
    cases = _read_jsonl(paths["cases"])
    expected = expected_batch_documents(plan, batch_index)
    failures = validate_generated_documents(documents, plan=plan, expected=expected)
    contract = CorpusDiversityContract(
        expected_documents=expected,
        profiles=tuple(profile.selection for profile in plan.profiles),
        document_families=plan.document_families,
        allowed_labels=(
            tuple(BERT_ENTITY_LABELS) if plan.require_full_label_coverage else ()
        ),
        forbidden_labels=("Anonymize_Other",),
        required_hard_negative_categories=(),
        require_balanced_profile_family_cells=bool(plan.document_families),
        near_duplicate_distance=plan.near_duplicate_distance,
    )
    diversity = audit_corpus_diversity(
        documents,
        contract=contract,
        prior_records=_prior_documents(workspace, batch_index),
        case_records=cases,
    )
    failures.extend(diversity["failures"])
    backend_report: dict[str, Any] | None = None
    if plan.mode == "remote":
        backend = resolve_production_backend(
            plan.backend, {profile.selection for profile in plan.profiles}
        )
        if backend.audit_batch is not None:
            backend_report = backend.audit_batch(workspace, plan, batch_index)
            failures.extend(
                f"backend quality: {failure}"
                for failure in backend_report.get("gate_failures", ())
            )
    report = {
        "contract_version": "meddeid.production-quality.v1",
        "created_at": _now(),
        "batch_index": batch_index,
        "documents": len(documents),
        "expected_documents": expected,
        "documents_sha256": (
            _sha256_file(paths["documents"]) if paths["documents"].is_file() else None
        ),
        "cases_sha256": (
            _sha256_file(paths["cases"]) if paths["cases"].is_file() else None
        ),
        "allowed_labels": list(BERT_ENTITY_LABELS),
        "forbidden_generated_labels": ["Anonymize_Other"],
        "diversity": diversity,
        "backend_quality": backend_report,
        "gate_failures": list(dict.fromkeys(failures)),
        "gate_passed": not failures,
    }
    _atomic_json(paths["quality"], report)
    existing = {
        str(row.get("document_id")): row for row in _read_jsonl(paths["review"])
    }
    review_rows = []
    for document in documents:
        document_id = str(document["document_id"])
        digest = document_sha256(document)
        row = existing.get(document_id)
        if row is None or row.get("document_sha256") != digest:
            row = {
                "contract_version": "meddeid.review-decision.v1",
                "document_id": document_id,
                "document_sha256": digest,
                "decision": "pending",
                "reviewer": None,
                "notes": "",
                "reviewed_at": None,
            }
        review_rows.append(row)
    _write_jsonl(paths["review"], review_rows)
    if paths["signoff"].exists():
        paths["signoff"].unlink()
    if paths["manifest"].exists():
        paths["manifest"].unlink()
    return report


def _quality_is_current(paths: Mapping[str, Path]) -> bool:
    if not paths["quality"].is_file() or not paths["documents"].is_file():
        return False
    try:
        report = json.loads(paths["quality"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if report.get("contract_version") != "meddeid.production-quality.v1":
        return False
    if report.get("documents_sha256") != _sha256_file(paths["documents"]):
        return False
    recorded_cases = report.get("cases_sha256")
    if recorded_cases is not None:
        if not paths["cases"].is_file() or recorded_cases != _sha256_file(
            paths["cases"]
        ):
            return False
    return True


def record_review(
    workspace: Path,
    batch_index: int,
    *,
    document_id: str,
    decision: str,
    reviewer: str,
    notes: str = "",
) -> dict[str, Any]:
    paths = batch_paths(workspace, batch_index)
    documents = {
        str(row.get("document_id")): row for row in _read_jsonl(paths["documents"])
    }
    if document_id not in documents:
        raise KeyError(f"unknown batch document {document_id!r}")
    aliases = {"pass": "accept", "fail": "reject"}
    decision = aliases.get(decision, decision)
    if decision not in {"accept", "edit", "regenerate", "reject"}:
        raise ValueError(f"unsupported review decision: {decision!r}")
    rows = _read_jsonl(paths["review"])
    replacement = {
        "contract_version": "meddeid.review-decision.v1",
        "document_id": document_id,
        "document_sha256": document_sha256(documents[document_id]),
        "decision": decision,
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_at": _now(),
    }
    updated = False
    for index, row in enumerate(rows):
        if str(row.get("document_id")) == document_id:
            rows[index] = replacement
            updated = True
            break
    if not updated:
        rows.append(replacement)
    _write_jsonl(paths["review"], rows)
    if paths["signoff"].exists():
        paths["signoff"].unlink()
    return replacement


def invalidate_document(
    workspace: Path,
    batch_index: int,
    *,
    document_id: str,
    reason: str,
) -> dict[str, Any]:
    """Archive one document and return its case to the generation queue."""

    if not reason.strip():
        raise ValueError("invalidation requires a reason for the attempt history")
    plan = load_plan(workspace)
    paths = batch_paths(workspace, batch_index)
    result: dict[str, Any] | None = None
    if plan.mode == "remote":
        backend = resolve_production_backend(
            plan.backend, {profile.selection for profile in plan.profiles}
        )
        if backend.invalidate_document is not None:
            result = backend.invalidate_document(
                workspace, batch_index, document_id, reason
            )
    if result is None:
        documents = _read_jsonl(paths["documents"])
        document = next(
            (row for row in documents if str(row.get("document_id")) == document_id),
            None,
        )
        if document is None:
            raise KeyError(f"unknown batch document {document_id!r}")
        cases = _read_jsonl(paths["cases"])
        case = next(
            (
                row
                for row in cases
                if str(row.get("document_id") or row.get("case_id")) == document_id
            ),
            None,
        )
        invalidated_at = _now()
        _append_jsonl(
            paths["invalidated"],
            {
                "document_id": document_id,
                "reason": reason,
                "invalidated_at": invalidated_at,
                "document_sha256": document_sha256(document),
                "document": document,
                "case": case,
            },
        )
        _write_jsonl(
            paths["documents"],
            (row for row in documents if str(row.get("document_id")) != document_id),
        )
        marked = _read_jsonl(paths["marked"])
        if marked:
            _write_jsonl(
                paths["marked"],
                (row for row in marked if str(row.get("document_id")) != document_id),
            )
        result = {
            "document_id": document_id,
            "reason": reason,
            "invalidated_at": invalidated_at,
        }
    reviews = [
        row
        for row in _read_jsonl(paths["review"])
        if str(row.get("document_id")) != document_id
    ]
    _write_jsonl(paths["review"], reviews)
    _invalidate_derived_artifacts(workspace, paths)
    return result


def apply_document_edit(
    workspace: Path,
    batch_index: int,
    *,
    document: Mapping[str, Any],
    editor: str,
    reason: str,
) -> dict[str, Any]:
    """Replace one document while retaining an immutable edit audit trail."""

    if not editor.strip() or not reason.strip():
        raise ValueError("document edits require both editor and reason")
    plan = load_plan(workspace)
    normalized = normalize_record(dict(document))
    document_id = str(normalized.get("document_id") or "")
    if not document_id:
        raise ValueError("edited document has no document_id")
    failures = validate_generated_documents((normalized,), plan=plan, expected=1)
    if failures:
        raise RuntimeError("edited document is invalid: " + "; ".join(failures[:5]))
    paths = batch_paths(workspace, batch_index)
    documents = _read_jsonl(paths["documents"])
    prior = next(
        (row for row in documents if str(row.get("document_id")) == document_id),
        None,
    )
    if prior is None:
        raise KeyError(f"unknown batch document {document_id!r}")
    edited_at = _now()
    _append_jsonl(
        paths["edits"],
        {
            "contract_version": "meddeid.document-edit.v1",
            "document_id": document_id,
            "editor": editor,
            "reason": reason,
            "edited_at": edited_at,
            "before_sha256": document_sha256(prior),
            "after_sha256": document_sha256(normalized),
            "before": prior,
        },
    )
    _write_jsonl(
        paths["documents"],
        (
            normalized if str(row.get("document_id")) == document_id else row
            for row in documents
        ),
    )
    marked = _read_jsonl(paths["marked"])
    if marked:
        _write_jsonl(
            paths["marked"],
            (row for row in marked if str(row.get("document_id")) != document_id),
        )
    reviews = [
        row
        for row in _read_jsonl(paths["review"])
        if str(row.get("document_id")) != document_id
    ]
    _write_jsonl(paths["review"], reviews)
    _invalidate_derived_artifacts(workspace, paths)
    return {
        "document_id": document_id,
        "editor": editor,
        "reason": reason,
        "edited_at": edited_at,
        "before_sha256": document_sha256(prior),
        "after_sha256": document_sha256(normalized),
    }


def sign_off_batch(
    workspace: Path,
    batch_index: int,
    *,
    reviewer: str,
    notes: str,
) -> dict[str, Any]:
    paths = batch_paths(workspace, batch_index)
    if not _quality_is_current(paths):
        raise RuntimeError(
            "cannot sign off a missing or stale automated quality report"
        )
    report = json.loads(paths["quality"].read_text(encoding="utf-8"))
    if not report.get("gate_passed"):
        raise RuntimeError("cannot sign off a batch whose automated gate fails")
    documents = _read_jsonl(paths["documents"])
    decisions = {
        str(row.get("document_id")): row for row in _read_jsonl(paths["review"])
    }
    failures: list[str] = []
    for document in documents:
        document_id = str(document["document_id"])
        decision = decisions.get(document_id)
        if decision is None or decision.get("decision") not in {"accept", "pass"}:
            failures.append(f"{document_id}: review is not accepted")
        elif decision.get("document_sha256") != document_sha256(document):
            failures.append(f"{document_id}: review belongs to older content")
    if failures or len(decisions) != len(documents):
        raise RuntimeError(
            "cannot sign off incomplete or stale personal review: "
            + "; ".join(failures[:5])
        )
    signoff = {
        "contract_version": "meddeid.production-signoff.v1",
        "batch_index": batch_index,
        "decision": "pass",
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_at": _now(),
        "reviewed_documents": len(documents),
        "documents_sha256": _sha256_file(paths["documents"]),
        "quality_report_sha256": _sha256_file(paths["quality"]),
        "review_decisions_sha256": _sha256_file(paths["review"]),
    }
    _atomic_json(paths["signoff"], signoff)
    plan = load_plan(workspace)
    manifest = BatchManifest(
        batch_index=batch_index,
        profiles=plan.profiles,
        expected_documents=expected_batch_documents(plan, batch_index),
        documents_sha256=signoff["documents_sha256"],
        quality_report_sha256=signoff["quality_report_sha256"],
        signoff_sha256=_sha256_file(paths["signoff"]),
    ).to_dict()
    _atomic_json(paths["manifest"], manifest)
    return signoff


def _valid_signoff(paths: Mapping[str, Path]) -> bool:
    if not paths["signoff"].is_file() or not paths["manifest"].is_file():
        return False
    try:
        signoff = json.loads(paths["signoff"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = {
        "documents_sha256": paths["documents"],
        "quality_report_sha256": paths["quality"],
        "review_decisions_sha256": paths["review"],
    }
    if signoff.get("decision") != "pass" or not all(
        path.is_file() and signoff.get(key) == _sha256_file(path)
        for key, path in required.items()
    ):
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("documents_sha256") == signoff.get("documents_sha256")
        and manifest.get("quality_report_sha256")
        == signoff.get("quality_report_sha256")
        and manifest.get("signoff_sha256") == _sha256_file(paths["signoff"])
        and tuple(manifest.get("allowed_labels", ())) == tuple(BERT_ENTITY_LABELS)
        and "Anonymize_Other" in manifest.get("forbidden_generated_labels", ())
    )


def production_status(workspace: Path) -> dict[str, Any]:
    plan = load_plan(workspace)
    batches: list[dict[str, Any]] = []
    next_action: dict[str, Any] | None = None
    for batch_index in range(batch_count(plan)):
        paths = batch_paths(workspace, batch_index)
        documents = _read_jsonl(paths["documents"])
        expected = expected_batch_documents(plan, batch_index)
        state = "signed"
        action = None
        if len(documents) != expected:
            state, action = "needs_generation", "generate"
        elif not _quality_is_current(paths):
            state, action = "needs_audit", "audit"
        else:
            report = json.loads(paths["quality"].read_text(encoding="utf-8"))
            if not report.get("gate_passed"):
                state, action = "quality_failed", "repair"
            else:
                decisions = _read_jsonl(paths["review"])
                by_id = {str(row.get("document_id")): row for row in decisions}
                if any(
                    by_id.get(str(document["document_id"]), {}).get("decision")
                    not in {"accept", "pass"}
                    or by_id.get(str(document["document_id"]), {}).get(
                        "document_sha256"
                    )
                    != document_sha256(document)
                    for document in documents
                ):
                    state, action = "needs_review", "review"
                elif not _valid_signoff(paths):
                    state, action = "needs_signoff", "sign-off"
        batch_status = {
            "batch_index": batch_index,
            "expected_documents": expected,
            "documents": len(documents),
            "state": state,
        }
        batches.append(batch_status)
        if action is not None and next_action is None:
            next_action = {"action": action, "batch_index": batch_index}
    final_manifest = workspace / "final" / "manifest.json"
    complete = next_action is None and final_manifest.is_file()
    if next_action is None and not complete:
        next_action = {"action": "finalize", "batch_index": None}
    attempt_cost = _attempt_cost(workspace)
    projected_cost = None
    if next_action and next_action.get("action") == "generate":
        projected_cost = _projected_completion_cost(
            workspace, plan, int(next_action["batch_index"])
        )
    return {
        "contract_version": PRODUCTION_STATUS_CONTRACT,
        "profiles": [profile.identifier for profile in plan.profiles],
        "target_documents": plan.target_documents,
        "estimated_attempt_cost_usd": round(attempt_cost, 6),
        "remaining_cost_guardrail_usd": (
            round(max(0.0, plan.max_cost_usd - attempt_cost), 6)
            if plan.max_cost_usd is not None
            else None
        ),
        "projected_cost_after_next_generation_usd": (
            round(projected_cost, 6) if projected_cost is not None else None
        ),
        "max_cost_usd": plan.max_cost_usd,
        "batches": batches,
        "next": next_action,
        "complete": complete,
    }


def run_next(workspace: Path) -> dict[str, Any]:
    with production_lock(workspace):
        plan = load_plan(workspace)
        status = production_status(workspace)
        action = status["next"]
        if action is None:
            return status
        batch_index = action.get("batch_index")
        if action["action"] == "generate":
            if (
                plan.max_cost_usd is not None
                and _attempt_cost(workspace) >= plan.max_cost_usd
            ):
                raise RuntimeError("production cost ceiling has been reached")
            projected = _projected_completion_cost(workspace, plan, int(batch_index))
            if (
                plan.max_cost_usd is not None
                and projected is not None
                and projected > plan.max_cost_usd
            ):
                raise RuntimeError(
                    f"projected generation cost USD {projected:.2f} exceeds the "
                    f"configured USD {plan.max_cost_usd:.2f} guardrail"
                )
            if plan.mode == "remote":
                _generate_remote_batch(workspace, plan, int(batch_index))
            else:
                _generate_local_batch(workspace, plan, int(batch_index))
        elif action["action"] == "audit":
            audit_batch(workspace, int(batch_index))
        elif action["action"] in {"review", "sign-off", "repair"}:
            return status
        elif action["action"] == "finalize":
            finalize_production(workspace)
        return production_status(workspace)


def _verified_documents(workspace: Path) -> list[dict[str, Any]]:
    plan = load_plan(workspace)
    documents: list[dict[str, Any]] = []
    for batch_index in range(batch_count(plan)):
        paths = batch_paths(workspace, batch_index)
        if not _valid_signoff(paths):
            raise RuntimeError(f"batch {batch_index + 1} has no current sign-off")
        documents.extend(_read_jsonl(paths["documents"]))
    if len(documents) != plan.target_documents:
        raise RuntimeError(
            f"expected {plan.target_documents} signed documents, found {len(documents)}"
        )
    return documents


def _benchmark_quotas(
    groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]], total: int
) -> dict[tuple[str, str], int]:
    if total == 0:
        return {key: 0 for key in groups}
    keys = sorted(groups)
    base, remainder = divmod(total, len(keys))
    quotas = {
        key: base + (1 if index < remainder else 0) for index, key in enumerate(keys)
    }
    for key, quota in quotas.items():
        if len(groups[key]) < quota:
            raise RuntimeError(
                f"benchmark stratum {key} contains fewer than {quota} documents"
            )
    return quotas


def finalize_production(workspace: Path) -> dict[str, Any]:
    plan = load_plan(workspace)
    documents = _verified_documents(workspace)
    final_audit = audit_corpus_diversity(
        documents,
        contract=CorpusDiversityContract(
            expected_documents=plan.target_documents,
            profiles=tuple(profile.selection for profile in plan.profiles),
            allowed_labels=(
                tuple(BERT_ENTITY_LABELS) if plan.require_full_label_coverage else ()
            ),
            forbidden_labels=("Anonymize_Other",),
            near_duplicate_distance=plan.near_duplicate_distance,
        ),
    )
    validation_failures = validate_generated_documents(
        documents, plan=plan, expected=plan.target_documents
    )
    if validation_failures or not final_audit["passed"]:
        failures = [*validation_failures, *final_audit["failures"]]
        raise RuntimeError("final corpus validation failed: " + "; ".join(failures[:5]))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        metadata = document.get("metadata") or {}
        groups[
            (
                _profile_for_document(document),
                str(metadata.get("document_type") or "unknown"),
            )
        ].append(document)
    quotas = _benchmark_quotas(groups, plan.benchmark_documents)
    benchmark_ids: set[str] = set()
    for key, rows in groups.items():
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{plan.seed}|sealed-benchmark|{row['document_id']}".encode("utf-8")
            ).hexdigest(),
        )
        benchmark_ids.update(str(row["document_id"]) for row in ranked[: quotas[key]])
    development = sorted(
        (row for row in documents if str(row["document_id"]) not in benchmark_ids),
        key=lambda row: str(row["document_id"]),
    )
    benchmark = sorted(
        (row for row in documents if str(row["document_id"]) in benchmark_ids),
        key=lambda row: str(row["document_id"]),
    )
    final = workspace / "final"
    development_path = final / "development.jsonl"
    benchmark_path = final / "benchmark.jsonl"
    _write_jsonl(development_path, development)
    _write_jsonl(benchmark_path, benchmark)
    profile_counts = Counter(_profile_for_document(row) for row in documents)
    benchmark_profiles = Counter(_profile_for_document(row) for row in benchmark)
    manifest = {
        "contract_version": FINAL_CORPUS_CONTRACT,
        "created_at": _now(),
        "plan_sha256": _sha256_file(plan_path(workspace)),
        "total_documents": len(documents),
        "development_documents": len(development),
        "benchmark_documents": len(benchmark),
        "profiles": dict(sorted(profile_counts.items())),
        "benchmark_profiles": dict(sorted(benchmark_profiles.items())),
        "benchmark_cells": {
            f"{profile}:{family}": quota
            for (profile, family), quota in sorted(quotas.items())
        },
        "allowed_labels": list(BERT_ENTITY_LABELS),
        "forbidden_generated_labels": ["Anonymize_Other"],
        "independently_authored": True,
        "quality": final_audit,
        "files": {
            "development": {
                "path": development_path.name,
                "sha256": _sha256_file(development_path),
            },
            "benchmark": {
                "path": benchmark_path.name,
                "sha256": _sha256_file(benchmark_path),
            },
        },
    }
    _atomic_json(final / "manifest.json", manifest)
    return manifest


__all__ = [
    "FINAL_CORPUS_CONTRACT",
    "PRODUCTION_PLAN_CONTRACT",
    "PRODUCTION_STATUS_CONTRACT",
    "ProductionPlan",
    "apply_document_edit",
    "append_attempt",
    "audit_batch",
    "batch_count",
    "batch_paths",
    "document_sha256",
    "expected_batch_documents",
    "finalize_production",
    "initialize_production",
    "invalidate_document",
    "load_plan",
    "production_lock",
    "production_status",
    "register_batch_documents",
    "record_review",
    "run_next",
    "sign_off_batch",
    "validate_generated_documents",
]
