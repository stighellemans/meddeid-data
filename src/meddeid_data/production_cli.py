"""Command-line interface for the generic synthetic production state machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meddeid_core import ProfileRef

from .production import (
    ProductionPlan,
    apply_document_edit,
    audit_batch,
    finalize_production,
    initialize_production,
    invalidate_document,
    production_lock,
    production_status,
    record_review,
    run_next,
    sign_off_batch,
)


DEFAULT_ENGLISH_FAMILIES = (
    "clinic_note",
    "discharge_summary",
    "emergency_note",
    "referral_letter",
    "laboratory_report",
    "nursing_note",
)


def _profiles(values: list[str]) -> tuple[ProfileRef, ...]:
    flattened = [part for value in values for part in value.split(",") if part.strip()]
    return tuple(ProfileRef.parse(value.strip()) for value in flattened)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meddeid-data production")
    sub = parser.add_subparsers(dest="production_command", required=True)

    init = sub.add_parser("init", help="create an immutable corpus production plan")
    init.add_argument("workspace", type=Path)
    init.add_argument("--profile", action="append", required=True)
    init.add_argument("--count", type=int, required=True)
    init.add_argument("--batch-size", type=int, default=500)
    init.add_argument("--benchmark-count", type=int, default=0)
    init.add_argument("--seed", type=int, default=20260508)
    init.add_argument("--mode", choices=("local", "remote"), default="local")
    init.add_argument("--allow-remote", action="store_true")
    init.add_argument("--backend")
    init.add_argument("--author-model")
    init.add_argument("--reasoning-effort", default="low")
    init.add_argument("--max-cost-usd", type=float)
    init.add_argument("--document-family", action="append", default=[])
    init.add_argument(
        "--allow-incomplete-label-coverage",
        action="store_true",
        help="intended only for pilots smaller than a full 500-document gate",
    )
    init.add_argument("--near-duplicate-distance", type=int, default=3)

    status = sub.add_parser("status", help="show the next safe production transition")
    status.add_argument("workspace", type=Path)

    run = sub.add_parser("run", help="execute the next automatic transition")
    run.add_argument("workspace", type=Path)
    run.add_argument(
        "--until-gate",
        action="store_true",
        help="continue through generation and audit, then stop before human review",
    )

    audit = sub.add_parser("audit", help="rerun one immutable batch quality gate")
    audit.add_argument("workspace", type=Path)
    audit.add_argument("--batch-index", type=int, required=True)

    review = sub.add_parser("review", help="record a content-bound document decision")
    review.add_argument("workspace", type=Path)
    review.add_argument("--batch-index", type=int, required=True)
    review.add_argument("--document-id", required=True)
    review.add_argument(
        "--decision", choices=("accept", "edit", "regenerate", "reject"), required=True
    )
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", default="")

    invalidate = sub.add_parser(
        "invalidate", help="archive one document and regenerate it on the next run"
    )
    invalidate.add_argument("workspace", type=Path)
    invalidate.add_argument("--batch-index", type=int, required=True)
    invalidate.add_argument("--document-id", required=True)
    invalidate.add_argument("--reason", required=True)

    replace = sub.add_parser(
        "replace", help="apply one manually corrected canonical JSON document"
    )
    replace.add_argument("workspace", type=Path)
    replace.add_argument("--batch-index", type=int, required=True)
    replace.add_argument("--document", type=Path, required=True)
    replace.add_argument("--editor", required=True)
    replace.add_argument("--reason", required=True)

    signoff = sub.add_parser("sign-off", help="bind personal review to batch content")
    signoff.add_argument("workspace", type=Path)
    signoff.add_argument("--batch-index", type=int, required=True)
    signoff.add_argument("--reviewer", required=True)
    signoff.add_argument("--notes", required=True)

    finalize = sub.add_parser("finalize", help="create deterministic sealed splits")
    finalize.add_argument("workspace", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.production_command == "init":
        profiles = _profiles(args.profile)
        english_preset = {profile.selection for profile in profiles} == {
            "en-GB",
            "en-US",
        } and args.count == 7_000
        families = tuple(args.document_family) or (
            DEFAULT_ENGLISH_FAMILIES if english_preset else ()
        )
        benchmark_count = args.benchmark_count or (300 if english_preset else 0)
        plan = ProductionPlan(
            profiles=profiles,
            target_documents=args.count,
            batch_size=args.batch_size,
            benchmark_documents=benchmark_count,
            seed=args.seed,
            mode=args.mode,
            remote_authorized=args.allow_remote,
            backend=args.backend,
            author_model=args.author_model,
            reasoning_effort=args.reasoning_effort,
            max_cost_usd=args.max_cost_usd,
            document_families=families,
            require_full_label_coverage=not args.allow_incomplete_label_coverage,
            near_duplicate_distance=args.near_duplicate_distance,
        )
        with production_lock(args.workspace):
            path = initialize_production(args.workspace, plan)
        print(path)
        print(
            json.dumps(production_status(args.workspace), ensure_ascii=False, indent=2)
        )
        return 0
    if args.production_command == "status":
        print(
            json.dumps(production_status(args.workspace), ensure_ascii=False, indent=2)
        )
        return 0
    if args.production_command == "run":
        status = run_next(args.workspace)
        if args.until_gate:
            while status.get("next", {}).get("action") in {"generate", "audit"}:
                status = run_next(args.workspace)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    if args.production_command == "audit":
        with production_lock(args.workspace):
            report = audit_batch(args.workspace, args.batch_index)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["gate_passed"] else 1
    if args.production_command == "review":
        with production_lock(args.workspace):
            decision = record_review(
                args.workspace,
                args.batch_index,
                document_id=args.document_id,
                decision=args.decision,
                reviewer=args.reviewer,
                notes=args.notes,
            )
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0
    if args.production_command == "invalidate":
        with production_lock(args.workspace):
            result = invalidate_document(
                args.workspace,
                args.batch_index,
                document_id=args.document_id,
                reason=args.reason,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.production_command == "replace":
        values = [
            json.loads(line)
            for line in args.document.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError("--document must contain exactly one JSON object")
        with production_lock(args.workspace):
            result = apply_document_edit(
                args.workspace,
                args.batch_index,
                document=values[0],
                editor=args.editor,
                reason=args.reason,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.production_command == "sign-off":
        with production_lock(args.workspace):
            signoff = sign_off_batch(
                args.workspace,
                args.batch_index,
                reviewer=args.reviewer,
                notes=args.notes,
            )
        print(json.dumps(signoff, ensure_ascii=False, indent=2))
        return 0
    with production_lock(args.workspace):
        manifest = finalize_production(args.workspace)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
