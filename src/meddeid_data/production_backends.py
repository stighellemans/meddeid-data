"""Pluggable paid/remote authoring backends for generic corpus production."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable, Mapping


PRODUCTION_BACKEND_CONTRACT = "meddeid.production-backend.v1"
ENTRY_POINT_GROUP = "meddeid.production_backends"

GenerateBatch = Callable[[Path, Any, int], None]
AuditBatch = Callable[[Path, Any, int], dict[str, Any]]
InvalidateDocument = Callable[[Path, int, str, str], dict[str, Any]]


# Standard, short-context GPT-5.6 Luna prices per 1M tokens.  Production
# manifests pin the applicable price date; the provider account ledger remains
# authoritative for billing.
LUNA_STANDARD_PRICING = {
    "input": 0.50,
    "cached_input": 0.05,
    "cache_write": 0.625,
    "output": 3.00,
}


def estimate_usage_cost(
    usage: Mapping[str, Any] | None,
    pricing: Mapping[str, float],
) -> float:
    """Estimate one recorded API attempt without dropping failed attempts."""

    if not usage:
        return 0.0
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    details = usage.get("input_tokens_details") or {}
    if not isinstance(details, Mapping):
        details = {}
    cached_tokens = int(details.get("cached_tokens", 0) or 0)
    cache_write_tokens = int(details.get("cache_write_tokens", 0) or 0)
    uncached_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)
    return (
        uncached_tokens * float(pricing["input"])
        + cached_tokens * float(pricing["cached_input"])
        + cache_write_tokens * float(pricing["cache_write"])
        + output_tokens * float(pricing["output"])
    ) / 1_000_000


def estimate_attempt_cost(attempt: Mapping[str, Any]) -> float:
    """Read self-contained cost data, with a Luna fallback for older ledgers."""

    for field in (
        "estimated_cost_usd",
        "estimated_usd",
        "estimated_usd_at_2026_08_20_list_price",
    ):
        if attempt.get(field) is not None:
            return float(attempt.get(field) or 0.0)
    pricing = attempt.get("pricing") or {}
    if isinstance(pricing, Mapping):
        rates = pricing.get("usd_per_million_tokens")
        if isinstance(rates, Mapping):
            return estimate_usage_cost(attempt.get("usage"), rates)
    if str(attempt.get("model") or "").lower() == "gpt-5.6-luna":
        return estimate_usage_cost(attempt.get("usage"), LUNA_STANDARD_PRICING)
    return 0.0


@dataclass(frozen=True)
class ProductionBackend:
    backend_id: str
    supported_profiles: tuple[str, ...]
    generate_batch: GenerateBatch
    audit_batch: AuditBatch | None = None
    invalidate_document: InvalidateDocument | None = None
    external: bool = True

    def supports(self, profiles: set[str]) -> bool:
        return profiles == set(self.supported_profiles)


def _english_luna_generate(workspace: Path, plan: Any, batch_index: int) -> None:
    from .english_production import generate_batch

    expected = min(
        int(plan.batch_size),
        int(plan.target_documents) - batch_index * int(plan.batch_size),
    )
    return_code = asyncio.run(
        generate_batch(
            output_dir=workspace,
            batch_index=batch_index,
            batch_size=expected,
            seed=int(plan.seed),
            model=str(plan.author_model),
            concurrency=8,
            max_output_tokens=1800,
            reasoning_effort=str(plan.reasoning_effort or "low"),
            validation_retries=4,
            api_retries=5,
            resume=True,
            model_review=False,
        )
    )
    # A non-zero result can mean an incomplete batch or a quality failure.  The
    # generic state machine inspects the persisted documents next and either
    # resumes generation or runs the authoritative combined audit.
    if return_code not in {0, 1}:
        raise RuntimeError(f"English production backend exited with {return_code}")


def _english_luna_audit(workspace: Path, plan: Any, batch_index: int) -> dict[str, Any]:
    from .english_production import audit_batch, write_batch_reports, _batch_paths

    expected = min(
        int(plan.batch_size),
        int(plan.target_documents) - batch_index * int(plan.batch_size),
    )
    paths = _batch_paths(workspace, batch_index)
    report = audit_batch(
        output_dir=workspace,
        batch_index=batch_index,
        expected=expected,
    )
    documents = []
    if paths["docs"].is_file():
        import json

        documents = [
            json.loads(line)
            for line in paths["docs"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    write_batch_reports(paths=paths, docs=documents, report=report)
    return report


def _english_luna_invalidate(
    workspace: Path, batch_index: int, document_id: str, reason: str
) -> dict[str, Any]:
    from .english_production import invalidate_document

    return invalidate_document(
        output_dir=workspace,
        batch_index=batch_index,
        document_id=document_id,
        reason=reason,
    )


ENGLISH_LUNA_BACKEND = ProductionBackend(
    backend_id="english-luna",
    supported_profiles=("en-GB", "en-US"),
    generate_batch=_english_luna_generate,
    audit_batch=_english_luna_audit,
    invalidate_document=_english_luna_invalidate,
)


def _providers():
    discovered = entry_points()
    if hasattr(discovered, "select"):
        yield from discovered.select(group=ENTRY_POINT_GROUP)
    else:  # pragma: no cover - Python 3.10 importlib compatibility
        yield from discovered.get(ENTRY_POINT_GROUP, ())


def resolve_production_backend(
    backend_id: str | None, profiles: set[str]
) -> ProductionBackend:
    candidates = [ENGLISH_LUNA_BACKEND]
    for entry_point in _providers():
        provider = entry_point.load()
        backend = provider() if callable(provider) else provider
        if not isinstance(backend, ProductionBackend):
            raise TypeError(
                f"production backend {entry_point.name!r} returned "
                f"{type(backend).__name__}, expected ProductionBackend"
            )
        candidates.append(backend)
    if backend_id:
        matches = [
            backend for backend in candidates if backend.backend_id == backend_id
        ]
        if not matches:
            available = ", ".join(sorted(backend.backend_id for backend in candidates))
            raise ValueError(
                f"unknown production backend {backend_id!r}; available: {available or 'none'}"
            )
        backend = matches[0]
        if not backend.supports(profiles):
            raise ValueError(
                f"backend {backend.backend_id!r} does not support profiles {sorted(profiles)}"
            )
        return backend
    matches = [backend for backend in candidates if backend.supports(profiles)]
    if len(matches) != 1:
        available = (
            ", ".join(sorted(backend.backend_id for backend in matches)) or "none"
        )
        raise ValueError(
            "production backend is ambiguous or unavailable for profiles "
            f"{sorted(profiles)}; matching backends: {available}; pass --backend"
        )
    return matches[0]


__all__ = [
    "ENTRY_POINT_GROUP",
    "ENGLISH_LUNA_BACKEND",
    "LUNA_STANDARD_PRICING",
    "PRODUCTION_BACKEND_CONTRACT",
    "ProductionBackend",
    "estimate_attempt_cost",
    "estimate_usage_cost",
    "resolve_production_backend",
]
