"""Checksum-pinned one- and two-stage training views from reviewed project splits."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from meddeid_core.artifacts import OFFSET_UNIT, SCHEMA_VERSION, validate_document_set
from meddeid_core.validate import validate_record

from .projects import load_project


TRAINING_VIEWS_VERSION = "meddeid.training-views.v2"


def _read_jsonl(path: Path, *, require_completed: bool) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        problems = validate_record(row, strict_taxonomy=True)
        if problems:
            raise ValueError(f"{path}:{line_number}: {problems}")
        if require_completed and not (
            row.get("annotated") is True or row.get("completed") is True
        ):
            raise ValueError(
                f"{path}:{line_number}: document is not marked annotated/completed"
            )
        rows.append(row)
    validate_document_set(rows)
    return rows


def _assert_same_documents(role: str, expected: list[dict], actual: list[dict]) -> None:
    expected_by_id = {row["document_id"]: row for row in expected}
    actual_by_id = {row["document_id"]: row for row in actual}
    missing = sorted(set(expected_by_id) - set(actual_by_id))
    extra = sorted(set(actual_by_id) - set(expected_by_id))
    if missing or extra:
        raise ValueError(
            f"{role} document IDs do not match the project split "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    mismatched = sorted(
        document_id
        for document_id, expected_row in expected_by_id.items()
        if actual_by_id[document_id]["text"] != expected_row["text"]
    )
    if mismatched:
        raise ValueError(
            f"{role} text differs from the project split for {mismatched[:5]}"
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _file_entry(path: Path, rows: list[dict]) -> dict:
    return {
        "filename": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "documents": len(rows),
        "spans": sum(len(row.get("spans", [])) for row in rows),
    }


def _source_entry(path: Path) -> dict:
    return {
        "filename": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _write_view_manifest(
    directory: Path,
    *,
    phase: str,
    files: dict[str, tuple[Path, list[dict]]],
    sources: dict[str, Path],
) -> dict:
    manifest = {
        "manifest_version": TRAINING_VIEWS_VERSION,
        "phase": phase,
        "contracts": {
            "schema_version": SCHEMA_VERSION,
            "offset_unit": OFFSET_UNIT,
        },
        "files": {
            role: _file_entry(path, rows)
            for role, (path, rows) in files.items()
        },
        "sources": {
            role: _source_entry(path) for role, path in sources.items()
        },
    }
    _write_json(directory / "manifest.json", manifest)
    return manifest


def prepare_training_views(
    root: Path,
    *,
    test_gold: Path,
    development: Path | None = None,
    selection_train: Path | None = None,
    selection_validation: Path | None = None,
    output: Path | None = None,
) -> tuple[Path, dict]:
    """Create fit/selection/refit views without mutating reviewed source files.

    ``development`` is the simpler input: one reviewed file containing the union
    of the project's train and validation splits. The split-specific inputs remain
    available for workflows that reviewed those assignments separately.
    """

    root, project = load_project(root)
    split_paths = {
        role: root / "splits" / f"{role}.jsonl"
        for role in ("train", "validation", "test")
    }
    missing_splits = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing_splits:
        raise FileNotFoundError(
            "missing project split files; run meddeid-data project split first: "
            + ", ".join(missing_splits)
        )

    if development is not None and (
        selection_train is not None or selection_validation is not None
    ):
        raise ValueError(
            "use either development or the selection train/validation pair, not both"
        )
    if development is None and (
        selection_train is None or selection_validation is None
    ):
        raise ValueError(
            "provide development or both selection_train and selection_validation"
        )

    source_paths = {"test_gold": test_gold.expanduser().resolve()}
    if development is not None:
        source_paths["development"] = development.expanduser().resolve()
    else:
        assert selection_train is not None and selection_validation is not None
        source_paths.update(
            {
                "selection_train": selection_train.expanduser().resolve(),
                "selection_validation": selection_validation.expanduser().resolve(),
            }
        )
    for role, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {role}: {path}")

    expected_train = _read_jsonl(split_paths["train"], require_completed=False)
    expected_validation = _read_jsonl(
        split_paths["validation"], require_completed=False
    )
    expected_test = _read_jsonl(split_paths["test"], require_completed=False)
    reviewed_test = _read_jsonl(source_paths["test_gold"], require_completed=True)
    _assert_same_documents("test gold", expected_test, reviewed_test)

    if development is not None:
        reviewed_development = _read_jsonl(
            source_paths["development"], require_completed=True
        )
        expected_development = sorted(
            [*expected_train, *expected_validation],
            key=lambda row: row["document_id"],
        )
        _assert_same_documents(
            "development", expected_development, reviewed_development
        )
        reviewed_by_id = {
            row["document_id"]: row for row in reviewed_development
        }
        reviewed_train = [
            reviewed_by_id[row["document_id"]] for row in expected_train
        ]
        reviewed_validation = [
            reviewed_by_id[row["document_id"]] for row in expected_validation
        ]
    else:
        reviewed_train = _read_jsonl(
            source_paths["selection_train"], require_completed=True
        )
        reviewed_validation = _read_jsonl(
            source_paths["selection_validation"], require_completed=True
        )
        _assert_same_documents("selection train", expected_train, reviewed_train)
        _assert_same_documents(
            "selection validation", expected_validation, reviewed_validation
        )

    development_rows = sorted(
        [*reviewed_train, *reviewed_validation], key=lambda row: row["document_id"]
    )
    validate_document_set(development_rows)
    target = (
        output.expanduser().resolve()
        if output is not None
        else root / "prepared"
    )
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"training-view output directory is not empty: {target}")
    selection_dir = target / "selection"
    refit_dir = target / "refit"
    fit_dir = target / "fit"

    fit_files = {
        "train": (fit_dir / "train.jsonl", reviewed_train),
        "val": (fit_dir / "val.jsonl", reviewed_validation),
        "test": (fit_dir / "test.jsonl", reviewed_test),
    }
    selection_files = {
        "train": (selection_dir / "train.jsonl", reviewed_train),
        "val": (selection_dir / "val.jsonl", reviewed_validation),
        "test": (selection_dir / "test.jsonl", []),
    }
    refit_files = {
        "train": (refit_dir / "train.jsonl", development_rows),
        "val": (refit_dir / "val.jsonl", reviewed_validation),
        "test": (refit_dir / "test.jsonl", reviewed_test),
    }
    for path, rows in [
        *fit_files.values(),
        *selection_files.values(),
        *refit_files.values(),
    ]:
        _write_jsonl(path, rows)

    source_manifest = {
        **source_paths,
        "project_split_manifest": root / "manifests" / "splits.json",
    }
    selection_manifest = _write_view_manifest(
        selection_dir,
        phase="epoch_selection",
        files=selection_files,
        sources=source_manifest,
    )
    refit_manifest = _write_view_manifest(
        refit_dir,
        phase="full_development_refit",
        files=refit_files,
        sources=source_manifest,
    )
    fit_manifest = _write_view_manifest(
        fit_dir,
        phase="single_fit",
        files=fit_files,
        sources=source_manifest,
    )
    summary = {
        "manifest_version": TRAINING_VIEWS_VERSION,
        "project_version": project["project_version"],
        "development_documents": len(development_rows),
        "test_documents": len(reviewed_test),
        "fit_manifest": "fit/manifest.json",
        "selection_manifest": "selection/manifest.json",
        "refit_manifest": "refit/manifest.json",
        "fit": fit_manifest["files"],
        "selection": selection_manifest["files"],
        "refit": refit_manifest["files"],
    }
    _write_json(target / "manifest.json", summary)
    return target, summary
