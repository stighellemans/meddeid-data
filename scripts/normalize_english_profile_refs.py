#!/usr/bin/env python3
"""Remove profile release suffixes from English corpus metadata and refresh hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PROFILE_PATTERN = re.compile(
    r'("generation_profile"\s*:\s*")(?P<profile>en-(?:GB|US))@[^"\s]+(")'
)
STABLE_PROFILES = ["en-GB", "en-US"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def jsonl_stats(path: Path) -> tuple[int, int]:
    documents = 0
    spans = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            documents += 1
            spans += len(row.get("spans", []))
            metadata = row.get("metadata") or {}
            language = str(metadata.get("lang") or "").replace("_", "-")
            profile = str(metadata.get("generation_profile") or "").replace("_", "-")
            if profile not in STABLE_PROFILES or profile != language:
                raise ValueError(
                    f"{path}:{documents} has incompatible lang/profile {language!r}/{profile!r}"
                )
    return documents, spans


def refresh_view_manifest(path: Path, final_root: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["profiles"] = STABLE_PROFILES
    for info in manifest["files"].values():
        artifact = path.parent / info["filename"]
        documents, spans = jsonl_stats(artifact)
        info.update(sha256=sha256(artifact), documents=documents, spans=spans)
    for source, source_path in {
        "development": final_root / "development.jsonl",
        "benchmark": final_root / "benchmark.jsonl",
    }.items():
        if source in manifest.get("sources", {}):
            manifest["sources"][source]["sha256"] = sha256(source_path)
    write_json(path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    final_root = root / "final"
    views_root = root / "training-views"

    changed_files = 0
    changed_refs = 0
    for base in (final_root, views_root):
        for path in sorted(base.rglob("*.jsonl")):
            original = path.read_text(encoding="utf-8")
            updated, replacements = PROFILE_PATTERN.subn(
                lambda match: f'{match.group(1)}{match.group("profile")}{match.group(3)}',
                original,
            )
            if replacements:
                path.write_text(updated, encoding="utf-8")
                changed_files += 1
                changed_refs += replacements
            jsonl_stats(path)

    final_manifest_path = final_root / "manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    for role in ("development", "benchmark"):
        final_manifest["files"][role]["sha256"] = sha256(final_root / f"{role}.jsonl")
    write_json(final_manifest_path, final_manifest)

    view_manifests = {
        phase: refresh_view_manifest(views_root / phase / "manifest.json", final_root)
        for phase in ("selection", "refit", "fit")
    }
    summary_path = views_root / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["profiles"] = STABLE_PROFILES
    summary["sources"]["development"]["sha256"] = sha256(final_root / "development.jsonl")
    summary["sources"]["benchmark"]["sha256"] = sha256(final_root / "benchmark.jsonl")
    for phase, manifest in view_manifests.items():
        summary["views"][phase] = manifest["files"]
    write_json(summary_path, summary)

    if args.benchmark_root is not None:
        benchmark_root = args.benchmark_root.expanduser().resolve()
        benchmark_source_hash = sha256(final_root / "benchmark.jsonl")
        gold_manifest_path = benchmark_root / "gold" / "manifest.json"
        gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
        gold_manifest["source_sha256"] = benchmark_source_hash
        write_json(gold_manifest_path, gold_manifest)

        validation_path = benchmark_root / "validation_report.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["source_sha256"] = benchmark_source_hash
        generated = validation.get("generated_files", {})
        manifest_key = "gold/manifest.json"
        if manifest_key in generated:
            generated[manifest_key].update(
                bytes=gold_manifest_path.stat().st_size,
                sha256=sha256(gold_manifest_path),
            )
        write_json(validation_path, validation)

    print(
        f"normalized {changed_refs} profile references across {changed_files} JSONL files; "
        "refreshed final, training-view, and benchmark provenance manifests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
