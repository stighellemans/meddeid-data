"""Hospital-friendly canonical project import and deterministic splitting."""

from __future__ import annotations

import csv
import datetime as dt
import decimal
import json
import os
import re
import secrets
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import yaml
from meddeid_core.artifacts import (
    OFFSET_UNIT,
    SCHEMA_VERSION,
    build_artifact_manifest,
    stable_document_id,
    validate_document_set,
)
from meddeid_core.normalize import normalize_metadata, normalize_name
from meddeid_core.taxonomy import CONTRACT_VERSION, TAXONOMY_VERSION
from meddeid_core.validate import validate_record

PROJECT_VERSION = "meddeid.project.v1"
IMPORT_MAPPING_VERSION = "meddeid.import-mapping.v1"
IMPORT_MAPPING_FIELDS = {
    "text_column",
    "id_column",
    "metadata_columns",
    "metadata_json_column",
    "include_metadata",
    "patient_name_column",
    "patient_given_name_column",
    "patient_family_name_column",
    "caregiver_columns",
    "caregiver_delimiter",
    "caregivers",
}


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    text: str
    metadata: dict[str, Any]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_import_mapping(path: Path) -> dict[str, Any]:
    """Load one strict, reusable YAML or JSON table-column mapping."""

    path = path.expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: import mapping must be an object")
    version = payload.pop("version", None)
    if version != IMPORT_MAPPING_VERSION:
        raise ValueError(
            f"{path}: expected version {IMPORT_MAPPING_VERSION!r}, found {version!r}"
        )
    unknown = sorted(set(payload) - IMPORT_MAPPING_FIELDS)
    if unknown:
        raise ValueError(f"{path}: unknown import mapping field(s) {unknown}")
    for key in (
        "text_column",
        "id_column",
        "metadata_json_column",
        "patient_name_column",
        "patient_given_name_column",
        "patient_family_name_column",
        "caregiver_delimiter",
    ):
        if payload.get(key) is not None and not isinstance(payload[key], str):
            raise ValueError(f"{path}: {key} must be a string or null")
    for key in ("metadata_columns", "caregiver_columns"):
        value = payload.get(key)
        if value is not None and (
            not isinstance(value, list)
            or not all(isinstance(item, str) and item for item in value)
        ):
            raise ValueError(f"{path}: {key} must be a list of column names or null")
    caregivers = payload.get("caregivers")
    if caregivers is not None and payload.get("caregiver_columns") is not None:
        raise ValueError(
            f"{path}: use either caregivers or caregiver_columns, not both"
        )
    if caregivers is not None:
        if not isinstance(caregivers, list):
            raise ValueError(f"{path}: caregivers must be a list or null")
        allowed = {
            "full_name_column",
            "given_name_column",
            "family_name_column",
            "delimiter",
        }
        for index, caregiver in enumerate(caregivers):
            if not isinstance(caregiver, dict):
                raise ValueError(f"{path}: caregivers[{index}] must be an object")
            unknown_caregiver = sorted(set(caregiver) - allowed)
            if unknown_caregiver:
                raise ValueError(
                    f"{path}: caregivers[{index}] has unknown field(s) "
                    f"{unknown_caregiver}"
                )
            if not caregiver or any(
                not isinstance(value, str) or not value
                for value in caregiver.values()
            ):
                raise ValueError(
                    f"{path}: caregivers[{index}] values must be non-empty strings"
                )
            if caregiver.get("full_name_column") and (
                caregiver.get("given_name_column")
                or caregiver.get("family_name_column")
            ):
                raise ValueError(
                    f"{path}: caregivers[{index}] cannot combine full and "
                    "given/family name columns"
                )
            if not any(
                caregiver.get(key)
                for key in (
                    "full_name_column",
                    "given_name_column",
                    "family_name_column",
                )
            ):
                raise ValueError(
                    f"{path}: caregivers[{index}] needs a name column"
                )
            if caregiver.get("delimiter") and not caregiver.get("full_name_column"):
                raise ValueError(
                    f"{path}: caregivers[{index}].delimiter requires "
                    "full_name_column"
                )
    if "include_metadata" in payload and not isinstance(
        payload["include_metadata"], bool
    ):
        raise ValueError(f"{path}: include_metadata must be true or false")
    return payload


def init_project(root: Path, *, namespace: str, language_profile: str) -> dict:
    root = root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"project directory is not empty: {root}")
    for relative in ("artifacts", "manifests", "splits", "private"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    secret_path = root / "private" / "document-id.key"
    secret_path.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    os.chmod(secret_path, 0o600)
    (root / ".gitignore").write_text("private/\n", encoding="utf-8")
    project = {
        "project_version": PROJECT_VERSION,
        "namespace": namespace.strip(),
        "language_profile": language_profile.strip(),
        "contracts": {"schema_version": SCHEMA_VERSION, "offset_unit": OFFSET_UNIT},
        "paths": {
            "canonical_documents": "artifacts/annotations.jsonl",
            "artifact_manifest": "manifests/input-documents.json",
        },
    }
    if not project["namespace"] or not project["language_profile"]:
        raise ValueError("namespace and language_profile must be non-empty")
    _write_json(root / "project.json", project)
    return project


def load_project(root: Path) -> tuple[Path, dict]:
    root = root.expanduser().resolve()
    manifest_path = root / "project.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path}; run: meddeid-data project init {root} --namespace <hospital>"
        )
    project = json.loads(manifest_path.read_text(encoding="utf-8"))
    if project.get("project_version") != PROJECT_VERSION:
        raise ValueError(f"unsupported project version: {project.get('project_version')!r}")
    return root, project


def _txt_records(source: Path) -> list[SourceRecord]:
    files = sorted(path for path in source.rglob("*.txt") if path.is_file())
    if not files:
        raise ValueError(f"no .txt files found under {source}")
    return [
        SourceRecord(
            source_id=path.relative_to(source).as_posix(),
            text=path.read_text(encoding="utf-8"),
            metadata={},
        )
        for path in files
    ]


def _json_safe(value: Any, *, context: str) -> Any:
    """Convert common table scalar/container values to deterministic JSON values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, context=f"{context}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item(), context=context)
    raise ValueError(
        f"{context}: value of type {type(value).__name__} is not JSON-compatible"
    )


def _metadata_object(value: Any, *, source: Path, row_number: int, column: str) -> dict:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source} row {row_number}: {column!r} is not valid JSON"
            ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"{source} row {row_number}: {column!r} must contain a JSON object"
        )
    return _json_safe(value, context=f"{source} row {row_number} metadata")


def _mapped_columns(
    *,
    patient_name_column: str | None,
    patient_given_name_column: str | None,
    patient_family_name_column: str | None,
    caregiver_columns: list[str] | None,
    caregiver_mappings: list[dict[str, str]] | None,
) -> list[str]:
    structured_caregiver_columns = [
        caregiver[key]
        for caregiver in caregiver_mappings or []
        for key in (
            "full_name_column",
            "given_name_column",
            "family_name_column",
        )
        if caregiver.get(key)
    ]
    return [
        column
        for column in (
            patient_name_column,
            patient_given_name_column,
            patient_family_name_column,
            *(caregiver_columns or []),
            *structured_caregiver_columns,
        )
        if column is not None
    ]


def _resolve_table_mapping(
    fieldnames: list[str],
    *,
    text_column: str,
    id_column: str | None,
    patient_name_column: str | None,
    patient_given_name_column: str | None,
    patient_family_name_column: str | None,
    caregiver_columns: list[str] | None,
    caregiver_mappings: list[dict[str, str]] | None,
) -> dict[str, Any]:
    """Apply the documented zero-configuration column conventions."""

    resolved_id = id_column
    if resolved_id is None and "source_id" in fieldnames:
        resolved_id = "source_id"

    resolved_patient_name = patient_name_column
    resolved_patient_given = patient_given_name_column
    resolved_patient_family = patient_family_name_column
    if not any(
        (resolved_patient_name, resolved_patient_given, resolved_patient_family)
    ):
        if "patient" in fieldnames:
            resolved_patient_name = "patient"
        else:
            if "patient_given_name" in fieldnames:
                resolved_patient_given = "patient_given_name"
            if "patient_family_name" in fieldnames:
                resolved_patient_family = "patient_family_name"

    resolved_caregiver_columns = caregiver_columns
    resolved_caregivers = caregiver_mappings
    if resolved_caregiver_columns is None and resolved_caregivers is None:
        resolved_caregiver_columns = []
        if "caregivers" in fieldnames:
            resolved_caregiver_columns.append("caregivers")

        grouped: dict[int, dict[str, str]] = {}
        aliases = {
            "full": "full_name_column",
            "given": "given_name_column",
            "first": "given_name_column",
            "family": "family_name_column",
            "last": "family_name_column",
        }
        for column in fieldnames:
            match = re.fullmatch(
                r"caregiver(?:_(\d+))?_(full|given|first|family|last)_name",
                column,
            )
            if not match:
                continue
            number = int(match.group(1) or 1)
            grouped.setdefault(number, {})[aliases[match.group(2)]] = column
        resolved_caregivers = [grouped[number] for number in sorted(grouped)]

    return {
        "text_column": text_column,
        "id_column": resolved_id,
        "patient_name_column": resolved_patient_name,
        "patient_given_name_column": resolved_patient_given,
        "patient_family_name_column": resolved_patient_family,
        "caregiver_columns": resolved_caregiver_columns or None,
        "caregivers": resolved_caregivers or None,
    }


def _caregiver_values(value: Any, *, delimiter: str | None) -> list[Any]:
    if value in (None, ""):
        return []
    if delimiter is None and isinstance(value, str) and value.lstrip().startswith("["):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            value = decoded
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if delimiter is None:
        return values
    split_values: list[Any] = []
    for item in values:
        if isinstance(item, str):
            split_values.extend(part.strip() for part in item.split(delimiter))
        else:
            split_values.append(item)
    return [item for item in split_values if item not in (None, "")]


def _apply_name_column_mapping(
    metadata: dict[str, Any],
    row: dict[str, Any],
    *,
    patient_name_column: str | None,
    patient_given_name_column: str | None,
    patient_family_name_column: str | None,
    caregiver_columns: list[str] | None,
    caregiver_delimiter: str | None,
    caregiver_mappings: list[dict[str, str]] | None,
) -> None:
    if patient_name_column:
        patient_name = normalize_name(row.get(patient_name_column))
        if patient_name:
            metadata["patient"] = patient_name
    elif patient_given_name_column or patient_family_name_column:
        patient_name = normalize_name(
            {
                "given_name": row.get(patient_given_name_column)
                if patient_given_name_column
                else None,
                "family_name": row.get(patient_family_name_column)
                if patient_family_name_column
                else None,
            }
        )
        if patient_name:
            metadata["patient"] = patient_name

    caregivers: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_caregiver(value: Any) -> None:
        caregiver = normalize_name(value)
        if not caregiver:
            return
        identity = json.dumps(caregiver, ensure_ascii=False, sort_keys=True)
        if identity not in seen:
            caregivers.append(caregiver)
            seen.add(identity)

    for column in caregiver_columns or []:
        for value in _caregiver_values(row.get(column), delimiter=caregiver_delimiter):
            add_caregiver(value)
    for caregiver_mapping in caregiver_mappings or []:
        full_name_column = caregiver_mapping.get("full_name_column")
        if full_name_column:
            for value in _caregiver_values(
                row.get(full_name_column),
                delimiter=caregiver_mapping.get("delimiter"),
            ):
                add_caregiver(value)
            continue
        add_caregiver(
            {
                "given_name": row.get(caregiver_mapping.get("given_name_column"))
                if caregiver_mapping.get("given_name_column")
                else None,
                "family_name": row.get(caregiver_mapping.get("family_name_column"))
                if caregiver_mapping.get("family_name_column")
                else None,
            }
        )
    if caregivers:
        metadata["caregivers"] = caregivers


def _records_from_rows(
    source: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
    text_column: str,
    id_column: str | None,
    metadata_columns: list[str] | None,
    metadata_json_column: str | None,
    include_metadata: bool,
    patient_name_column: str | None,
    patient_given_name_column: str | None,
    patient_family_name_column: str | None,
    caregiver_columns: list[str] | None,
    caregiver_delimiter: str | None,
    caregiver_mappings: list[dict[str, str]] | None,
) -> list[SourceRecord]:
    if text_column not in fieldnames:
        raise ValueError(f"{source}: missing text column {text_column!r}")
    if id_column and id_column not in fieldnames:
        raise ValueError(f"{source}: missing ID column {id_column!r}")

    unknown_metadata = sorted(set(metadata_columns or []) - set(fieldnames))
    if unknown_metadata:
        raise ValueError(f"{source}: missing metadata column(s) {unknown_metadata}")
    reserved_metadata = sorted(
        set(metadata_columns or []) & {text_column, id_column}
    )
    if reserved_metadata:
        raise ValueError(
            f"{source}: text/ID columns cannot also be metadata columns: "
            f"{reserved_metadata}"
        )
    if metadata_json_column and metadata_json_column not in fieldnames:
        raise ValueError(
            f"{source}: missing metadata JSON column {metadata_json_column!r}"
        )
    mapped_columns = _mapped_columns(
        patient_name_column=patient_name_column,
        patient_given_name_column=patient_given_name_column,
        patient_family_name_column=patient_family_name_column,
        caregiver_columns=caregiver_columns,
        caregiver_mappings=caregiver_mappings,
    )
    unknown_mapped = sorted(set(mapped_columns) - set(fieldnames))
    if unknown_mapped:
        raise ValueError(
            f"{source}: missing mapped metadata column(s) {unknown_mapped}"
        )
    reserved_mapped = sorted(set(mapped_columns) & {text_column, id_column})
    if reserved_mapped:
        raise ValueError(
            f"{source}: text/ID columns cannot also be name columns: "
            f"{reserved_mapped}"
        )
    auto_json_column = metadata_json_column
    if include_metadata and auto_json_column is None:
        auto_json_column = next(
            (name for name in ("metadata", "metadata_json") if name in fieldnames),
            None,
        )
    if include_metadata:
        selected_columns = (
            list(metadata_columns)
            if metadata_columns is not None
            else [
                name
                for name in fieldnames
                if name not in {text_column, id_column, auto_json_column}
            ]
        )
    else:
        selected_columns = []
        auto_json_column = None

    records: list[SourceRecord] = []
    for row_index, row in enumerate(rows, start=1):
        display_row = row_index + 1 if source.suffix.lower() in {".csv", ".tsv"} else row_index
        text = row.get(text_column)
        if text is None:
            raise ValueError(f"{source} row {display_row}: missing text")
        if not isinstance(text, str):
            text = str(text)
        source_id = row.get(id_column, "") if id_column else f"row-{row_index}"
        if not str(source_id).strip():
            raise ValueError(f"{source} row {display_row}: empty source ID")

        metadata: dict[str, Any] = {}
        if auto_json_column:
            metadata.update(
                _metadata_object(
                    row.get(auto_json_column),
                    source=source,
                    row_number=display_row,
                    column=auto_json_column,
                )
            )
        for column in selected_columns:
            value = row.get(column)
            if value in (None, ""):
                continue
            metadata[column] = _json_safe(
                value, context=f"{source} row {display_row} column {column!r}"
            )
        _apply_name_column_mapping(
            metadata,
            row,
            patient_name_column=patient_name_column,
            patient_given_name_column=patient_given_name_column,
            patient_family_name_column=patient_family_name_column,
            caregiver_columns=caregiver_columns,
            caregiver_delimiter=caregiver_delimiter,
            caregiver_mappings=caregiver_mappings,
        )
        metadata = normalize_metadata(metadata)
        records.append(
            SourceRecord(source_id=str(source_id), text=text, metadata=metadata)
        )
    if not records:
        raise ValueError(f"{source}: table contains no records")
    return records


def _table_records(
    source: Path,
    *,
    text_column: str,
    id_column: str | None,
    delimiter: str,
    metadata_columns: list[str] | None,
    metadata_json_column: str | None,
    include_metadata: bool,
    patient_name_column: str | None,
    patient_given_name_column: str | None,
    patient_family_name_column: str | None,
    caregiver_columns: list[str] | None,
    caregiver_delimiter: str | None,
    caregiver_mappings: list[dict[str, str]] | None,
) -> tuple[list[SourceRecord], dict[str, Any]]:
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{source}: table has no header")
        rows = list(reader)
    fieldnames = list(reader.fieldnames)
    resolved = _resolve_table_mapping(
        fieldnames,
        text_column=text_column,
        id_column=id_column,
        patient_name_column=patient_name_column,
        patient_given_name_column=patient_given_name_column,
        patient_family_name_column=patient_family_name_column,
        caregiver_columns=caregiver_columns,
        caregiver_mappings=caregiver_mappings,
    )
    records = _records_from_rows(
        source,
        rows,
        fieldnames=fieldnames,
        text_column=resolved["text_column"],
        id_column=resolved["id_column"],
        metadata_columns=metadata_columns,
        metadata_json_column=metadata_json_column,
        include_metadata=include_metadata,
        patient_name_column=resolved["patient_name_column"],
        patient_given_name_column=resolved["patient_given_name_column"],
        patient_family_name_column=resolved["patient_family_name_column"],
        caregiver_columns=resolved["caregiver_columns"],
        caregiver_delimiter=caregiver_delimiter,
        caregiver_mappings=resolved["caregivers"],
    )
    return records, resolved


def _parquet_records(
    source: Path,
    *,
    text_column: str,
    id_column: str | None,
    metadata_columns: list[str] | None,
    metadata_json_column: str | None,
    include_metadata: bool,
    patient_name_column: str | None,
    patient_given_name_column: str | None,
    patient_family_name_column: str | None,
    caregiver_columns: list[str] | None,
    caregiver_delimiter: str | None,
    caregiver_mappings: list[dict[str, str]] | None,
) -> tuple[list[SourceRecord], dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Parquet import requires pyarrow; install meddeid-data[parquet]"
        ) from exc
    table = parquet.read_table(source)
    fieldnames = list(table.column_names)
    resolved = _resolve_table_mapping(
        fieldnames,
        text_column=text_column,
        id_column=id_column,
        patient_name_column=patient_name_column,
        patient_given_name_column=patient_given_name_column,
        patient_family_name_column=patient_family_name_column,
        caregiver_columns=caregiver_columns,
        caregiver_mappings=caregiver_mappings,
    )
    records = _records_from_rows(
        source,
        table.to_pylist(),
        fieldnames=fieldnames,
        text_column=resolved["text_column"],
        id_column=resolved["id_column"],
        metadata_columns=metadata_columns,
        metadata_json_column=metadata_json_column,
        include_metadata=include_metadata,
        patient_name_column=resolved["patient_name_column"],
        patient_given_name_column=resolved["patient_given_name_column"],
        patient_family_name_column=resolved["patient_family_name_column"],
        caregiver_columns=resolved["caregiver_columns"],
        caregiver_delimiter=caregiver_delimiter,
        caregiver_mappings=resolved["caregivers"],
    )
    return records, resolved


def import_documents(
    root: Path,
    source: Path,
    *,
    text_column: str = "text",
    id_column: str | None = None,
    metadata_columns: list[str] | None = None,
    metadata_json_column: str | None = None,
    include_metadata: bool = True,
    patient_name_column: str | None = None,
    patient_given_name_column: str | None = None,
    patient_family_name_column: str | None = None,
    caregiver_columns: list[str] | None = None,
    caregiver_delimiter: str | None = None,
    caregivers: list[dict[str, str]] | None = None,
) -> tuple[Path, dict]:
    root, project = load_project(root)
    source = source.expanduser().resolve()
    effective_mapping: dict[str, Any] = {
        "text_column": text_column,
        "id_column": id_column,
        "metadata_columns": metadata_columns,
        "metadata_json_column": metadata_json_column,
        "include_metadata": include_metadata,
        "patient_name_column": patient_name_column,
        "patient_given_name_column": patient_given_name_column,
        "patient_family_name_column": patient_family_name_column,
        "caregiver_columns": caregiver_columns,
        "caregiver_delimiter": caregiver_delimiter,
        "caregivers": caregivers,
    }
    if patient_name_column and (
        patient_given_name_column or patient_family_name_column
    ):
        raise ValueError(
            "--patient-name-column cannot be combined with patient given/family "
            "name column mappings"
        )
    if caregiver_delimiter is not None and not caregiver_columns:
        raise ValueError(
            "--caregiver-delimiter requires at least one --caregiver-column"
        )
    if caregiver_delimiter == "":
        raise ValueError("caregiver delimiter must be non-empty")
    if source.is_dir():
        has_column_options = bool(
            metadata_columns
            or metadata_json_column
            or caregiver_delimiter is not None
            or _mapped_columns(
                patient_name_column=patient_name_column,
                patient_given_name_column=patient_given_name_column,
                patient_family_name_column=patient_family_name_column,
                caregiver_columns=caregiver_columns,
                caregiver_mappings=caregivers,
            )
        )
        if has_column_options:
            raise ValueError(
                "metadata column options apply only to CSV, TSV, or Parquet sources"
            )
        raw_records = _txt_records(source)
        source_kind = "txt-directory"
    elif source.suffix.lower() in {".csv", ".tsv"}:
        raw_records, resolved = _table_records(
            source,
            text_column=text_column,
            id_column=id_column,
            delimiter="\t" if source.suffix.lower() == ".tsv" else ",",
            metadata_columns=metadata_columns,
            metadata_json_column=metadata_json_column,
            include_metadata=include_metadata,
            patient_name_column=patient_name_column,
            patient_given_name_column=patient_given_name_column,
            patient_family_name_column=patient_family_name_column,
            caregiver_columns=caregiver_columns,
            caregiver_delimiter=caregiver_delimiter,
            caregiver_mappings=caregivers,
        )
        effective_mapping.update(resolved)
        source_kind = source.suffix.lower().lstrip(".")
    elif source.suffix.lower() in {".parquet", ".pq"}:
        raw_records, resolved = _parquet_records(
            source,
            text_column=text_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
            metadata_json_column=metadata_json_column,
            include_metadata=include_metadata,
            patient_name_column=patient_name_column,
            patient_given_name_column=patient_given_name_column,
            patient_family_name_column=patient_family_name_column,
            caregiver_columns=caregiver_columns,
            caregiver_delimiter=caregiver_delimiter,
            caregiver_mappings=caregivers,
        )
        effective_mapping.update(resolved)
        source_kind = "parquet"
    else:
        raise ValueError(
        "source must be a directory of .txt files or a .csv/.tsv/.parquet file"
        )

    key_path = root / "private" / "document-id.key"
    secret = key_path.read_text(encoding="utf-8").strip()
    seen_source_ids: set[str] = set()
    seen_content: dict[str, str] = {}
    rows: list[dict] = []
    source_map: list[dict] = []
    for source_record in raw_records:
        source_id = source_record.source_id
        text = source_record.text
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate source ID: {source_id!r}")
        seen_source_ids.add(source_id)
        content_hash = sha256(text.encode("utf-8")).hexdigest()
        if content_hash in seen_content:
            raise ValueError(
                f"duplicate document content for {source_id!r} and {seen_content[content_hash]!r}"
            )
        seen_content[content_hash] = source_id
        document_id = stable_document_id(project["namespace"], source_id, secret=secret)
        rows.append(
            {
                "document_id": document_id,
                "text": text,
                "spans": [],
                "metadata": {
                    **source_record.metadata,
                    "lang": project["language_profile"],
                    "source_content_sha256": content_hash,
                },
                "annotated": False,
            }
        )
        source_map.append({"document_id": document_id, "source_id": source_id})

    rows.sort(key=lambda row: row["document_id"])
    validate_document_set(rows)
    artifact_path = root / project["paths"]["canonical_documents"]
    _write_jsonl(artifact_path, rows)
    manifest = build_artifact_manifest(
        role="input_documents",
        artifact_path=artifact_path,
        records=rows,
        producer={"name": "meddeid-data", "version": "0.2.1"},
        contracts={"language_profile": project["language_profile"]},
    )
    manifest["source"] = {
        "kind": source_kind,
        "metadata": "none" if not include_metadata else "selected" if metadata_columns else "auto",
    }
    if metadata_columns:
        manifest["source"]["metadata_columns"] = list(metadata_columns)
    if metadata_json_column:
        manifest["source"]["metadata_json_column"] = metadata_json_column
    name_mapping = {
        key: value
        for key, value in {
            "patient_name_column": effective_mapping["patient_name_column"],
            "patient_given_name_column": effective_mapping[
                "patient_given_name_column"
            ],
            "patient_family_name_column": effective_mapping[
                "patient_family_name_column"
            ],
            "caregiver_columns": effective_mapping["caregiver_columns"],
            "caregiver_delimiter": caregiver_delimiter,
            "caregivers": effective_mapping["caregivers"],
        }.items()
        if value is not None
    }
    if name_mapping:
        manifest["source"]["name_mapping"] = name_mapping
    _write_json(root / project["paths"]["artifact_manifest"], manifest)
    _write_json(
        root / "manifests" / "import-mapping.json",
        {"version": IMPORT_MAPPING_VERSION, **effective_mapping},
    )
    _write_jsonl(root / "private" / "source-map.jsonl", source_map)
    return artifact_path, manifest


def split_project(
    root: Path,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict:
    root, project = load_project(root)
    input_path = root / project["paths"]["canonical_documents"]
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    validate_document_set(rows)
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must satisfy train > 0, validation >= 0, and train + validation < 1")
    ranked = sorted(
        rows,
        key=lambda row: sha256(f"{seed}\0{row['document_id']}".encode("utf-8")).hexdigest(),
    )
    count = len(ranked)
    train_end = int(count * train_fraction)
    validation_end = train_end + int(count * validation_fraction)
    if count >= 3:
        train_end = max(1, min(train_end, count - 2))
        validation_end = max(train_end + 1, min(validation_end, count - 1))
    partitions = {
        "train": ranked[:train_end],
        "validation": ranked[train_end:validation_end],
        "test": ranked[validation_end:],
    }
    files: dict[str, dict] = {}
    for role, split_rows in partitions.items():
        path = root / "splits" / f"{role}.jsonl"
        _write_jsonl(path, split_rows)
        files[role] = {
            "filename": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "documents": len(split_rows),
        }
    manifest = {
        "manifest_version": "meddeid.split-manifest.v1",
        "contracts": {"schema_version": SCHEMA_VERSION, "offset_unit": OFFSET_UNIT},
        "input_sha256": sha256(input_path.read_bytes()).hexdigest(),
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1 - train_fraction - validation_fraction,
        },
        "files": files,
    }
    _write_json(root / "manifests" / "splits.json", manifest)
    return manifest


def package_annotation_set(
    root: Path,
    annotations_path: Path,
    *,
    annotation_set_id: str,
    annotator_id: str | None = None,
) -> tuple[Path, dict]:
    """Validate a completed assignment and write the manifest used by curation."""

    root, _project = load_project(root)
    annotations_path = annotations_path.expanduser().resolve()
    rows = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_document_set(rows)
    for index, row in enumerate(rows, start=1):
        problems = validate_record(row, strict_taxonomy=True)
        if problems:
            raise ValueError(f"{annotations_path} line {index}: {problems}")
        if row.get("annotated") is not True and row.get("completed") is not True:
            raise ValueError(
                f"{annotations_path} line {index}: annotation set is incomplete; "
                "mark every reviewed row annotated:true or completed:true"
            )
    set_id = annotation_set_id.strip()
    if not set_id:
        raise ValueError("annotation_set_id must be non-empty")
    manifest = {
        "manifest_version": "meddeid.annotation-set.v1",
        "annotation_set_id": set_id,
        "status": "completed",
        "contracts": {
            "schema_version": SCHEMA_VERSION,
            "offset_unit": OFFSET_UNIT,
            "taxonomy_contract_version": CONTRACT_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
        },
        "files": {"annotations": annotations_path.name},
        "hashes": {"annotations_sha256": sha256(annotations_path.read_bytes()).hexdigest()},
        "counts": {
            "documents": len(rows),
            "spans": sum(len(row.get("spans", [])) for row in rows),
        },
    }
    if annotator_id and annotator_id.strip():
        manifest["annotator_id"] = annotator_id.strip()
    output = annotations_path.with_suffix(".manifest.json")
    _write_json(output, manifest)
    # Keep a copy in the project manifest collection without changing the
    # portable pair's declared basename.
    _write_json(root / "manifests" / f"annotation-set-{set_id}.json", manifest)
    return output, manifest
