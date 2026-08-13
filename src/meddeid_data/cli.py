from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from meddeid_core import build_artifact_manifest, validate_record
from meddeid_language_nl import get_profile

from .clinical_cases import generate_case_records
from .generator import (
    generate_documents,
    render_documents_from_case_records,
    write_jsonl,
)
from .judge import judge_documents, write_report
from .projects import (
    import_documents,
    init_project,
    load_import_mapping,
    package_annotation_set,
    split_project,
)
from .synthea_adapter import load_or_generate_synthea_csv_seeds
from .training_views import prepare_training_views


def _add_import_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--mapping-config",
        type=Path,
        help=(
            "reusable YAML/JSON import mapping; the effective mapping is saved "
            "in the project and reused on later imports"
        ),
    )
    parser.add_argument("--text-column", help="text column (default: text)")
    parser.add_argument("--id-column")
    parser.add_argument(
        "--metadata-column",
        action="append",
        dest="metadata_columns",
        help=(
            "table column to copy into metadata; repeat as needed. By default all "
            "columns other than text/ID are copied"
        ),
    )
    parser.add_argument(
        "--metadata-json-column",
        help=(
            "table column containing a JSON object to merge into metadata; "
            "metadata or metadata_json is detected automatically"
        ),
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help=(
            "discard unselected raw table columns; explicitly mapped canonical "
            "name metadata is retained"
        ),
    )
    parser.add_argument(
        "--patient-name-column",
        help="source column containing one patient's full name",
    )
    parser.add_argument(
        "--patient-given-name-column",
        help="source column containing the patient's given name(s)",
    )
    parser.add_argument(
        "--patient-family-name-column",
        help="source column containing the patient's family name",
    )
    parser.add_argument(
        "--caregiver-column",
        action="append",
        dest="caregiver_columns",
        help=(
            "source column containing complete caregiver name(s); repeat for "
            "several full-name columns"
        ),
    )
    parser.add_argument(
        "--caregiver-delimiter",
        help=(
            "literal delimiter used for multiple caregiver names in a cell; "
            "omit when each cell contains at most one name"
        ),
    )


def _run_import(args: argparse.Namespace) -> tuple[Path, dict]:
    if args.no_metadata and (args.metadata_columns or args.metadata_json_column):
        raise ValueError(
            "--no-metadata cannot be combined with --metadata-column or "
            "--metadata-json-column"
        )
    saved_mapping = (
        args.directory.expanduser().resolve()
        / "manifests"
        / "import-mapping.json"
    )
    if args.mapping_config:
        mapping = load_import_mapping(args.mapping_config)
    elif saved_mapping.is_file():
        mapping = load_import_mapping(saved_mapping)
    else:
        mapping = {}

    def selected(name: str, default=None):
        value = getattr(args, name)
        return value if value is not None else mapping.get(name, default)

    if args.no_metadata:
        include_metadata = False
    elif args.metadata_columns or args.metadata_json_column:
        include_metadata = True
    else:
        include_metadata = bool(mapping.get("include_metadata", True))
    caregiver_mappings = (
        None if args.caregiver_columns is not None else mapping.get("caregivers")
    )
    return import_documents(
        args.directory,
        args.source,
        text_column=selected("text_column", "text"),
        id_column=selected("id_column"),
        metadata_columns=selected("metadata_columns"),
        metadata_json_column=selected("metadata_json_column"),
        include_metadata=include_metadata,
        patient_name_column=selected("patient_name_column"),
        patient_given_name_column=selected("patient_given_name_column"),
        patient_family_name_column=selected("patient_family_name_column"),
        caregiver_columns=selected("caregiver_columns"),
        caregiver_delimiter=selected("caregiver_delimiter"),
        caregivers=caregiver_mappings,
    )


def _print_annotation_next_steps(directory: Path, artifact: Path) -> None:
    assignment = directory.expanduser().resolve() / "assignments" / "primary.jsonl"
    artifact_arg = shlex.quote(str(artifact))
    assignment_arg = shlex.quote(str(assignment))
    print("\nNext: create an annotation assignment initialized by the local model:")
    print(
        f"  meddeid batch {artifact_arg} --output {assignment_arg} "
        "--model stighellemans/meddeid-dutch-synth --device cpu"
    )
    print("\nThen annotate that current state:")
    print(
        f"  MEDDEID_ANNOTATIONS_PATH={assignment_arg} "
        "npm --prefix /path/to/meddeid-annotate run dev"
    )


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty-output", type=Path)
    parser.add_argument("--judge-report", type=Path)
    parser.add_argument("--synthea-csv-dir", type=Path)
    parser.add_argument("--auto-synthea", action="store_true")
    parser.add_argument(
        "--synthea-repo-dir", type=Path, default=Path("external/synthea")
    )
    parser.add_argument("--synthea-population", type=int)
    parser.add_argument("--force-synthea", action="store_true")
    parser.add_argument("--require-synthea", action="store_true")


def _write_pretty(rows: list[dict], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl_raw(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_dataset_manifest(rows: list[dict], path: Path, *, role: str) -> None:
    profile = get_profile("nl-BE", version="1")
    manifest = build_artifact_manifest(
        role=role,
        artifact_path=path,
        records=rows,
        producer={"name": "meddeid-data", "version": "0.2.0"},
        contracts={"language_profile": "nl-BE", "language_profile_version": "1"},
    )
    manifest["language_profile"] = profile.manifest()
    _write_pretty(manifest, path.with_suffix(path.suffix + ".manifest.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meddeid-data")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser(
        "generate", help="generate synthetic Belgian clinical notes"
    )
    _add_generation_options(generate)

    sample = sub.add_parser(
        "sample", help="alias for an offline synthetic generation run"
    )
    _add_generation_options(sample)

    cases = sub.add_parser(
        "build-cases", help="write structured synthetic case records"
    )
    _add_generation_options(cases)
    cases.add_argument("--start-index", type=int, default=0)

    render = sub.add_parser("render-cases", help="render structured case JSONL")
    render.add_argument("input", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--pretty-output", type=Path)
    render.add_argument("--judge-report", type=Path)
    render.add_argument("--seed", type=int, default=20260508)

    validate = sub.add_parser(
        "validate", help="validate canonical JSONL offsets and labels"
    )
    validate.add_argument("path", type=Path)

    project = sub.add_parser("project", help="create/import/split a canonical hospital project")
    project_sub = project.add_subparsers(dest="project_command", required=True)
    project_create = project_sub.add_parser(
        "create",
        help="create a project and import TXT, CSV, TSV, or Parquet in one step",
    )
    project_create.add_argument("directory", type=Path)
    project_create.add_argument("--namespace", required=True)
    project_create.add_argument("--language-profile", default="nl-BE")
    _add_import_options(project_create)
    project_init = project_sub.add_parser("init", help="create a canonical project directory")
    project_init.add_argument("directory", type=Path)
    project_init.add_argument("--namespace", required=True)
    project_init.add_argument("--language-profile", default="nl-BE")
    project_import = project_sub.add_parser(
        "import", help="import TXT, CSV, TSV, or Parquet documents"
    )
    project_import.add_argument("directory", type=Path)
    _add_import_options(project_import)
    project_split = project_sub.add_parser("split", help="create deterministic train/validation/test files")
    project_split.add_argument("directory", type=Path)
    project_split.add_argument("--seed", type=int, default=20260508)
    project_split.add_argument("--train", type=float, default=0.8)
    project_split.add_argument("--validation", type=float, default=0.1)
    project_package = project_sub.add_parser(
        "package-annotation", help="validate a completed assignment and emit its curation manifest"
    )
    project_package.add_argument("directory", type=Path)
    project_package.add_argument("annotations", type=Path)
    project_package.add_argument("--annotation-set-id", required=True)
    project_package.add_argument("--annotator-id")
    project_training = project_sub.add_parser(
        "prepare-training",
        help="create checksum-pinned one-stage and publication training views",
    )
    project_training.add_argument("directory", type=Path)
    development_inputs = project_training.add_mutually_exclusive_group(required=True)
    development_inputs.add_argument(
        "--development",
        type=Path,
        help="one reviewed file containing both train and validation documents",
    )
    development_inputs.add_argument(
        "--selection-train",
        type=Path,
        help="reviewed training split (requires --selection-validation)",
    )
    project_training.add_argument(
        "--selection-validation",
        type=Path,
        help="reviewed validation split used with --selection-train",
    )
    project_training.add_argument("--test-gold", type=Path, required=True)
    project_training.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "project":
        if args.project_command == "create":
            init_project(
                args.directory,
                namespace=args.namespace,
                language_profile=args.language_profile,
            )
            artifact, manifest = _run_import(args)
            print(
                f"Created {args.directory.expanduser().resolve()} with "
                f"{manifest['counts']['documents']} annotation-ready documents."
            )
            print(f"Canonical dataset: {artifact}")
            print(f"Manifest: {args.directory.expanduser().resolve() / 'manifests' / 'input-documents.json'}")
            import_mapping = (
                args.directory.expanduser().resolve()
                / "manifests"
                / "import-mapping.json"
            )
            print(
                f"Reusable import mapping: {import_mapping}"
            )
            _print_annotation_next_steps(args.directory, artifact)
            return 0
        if args.project_command == "init":
            init_project(
                args.directory,
                namespace=args.namespace,
                language_profile=args.language_profile,
            )
            print(f"Created MedDeID project at {args.directory.expanduser().resolve()}")
            return 0
        if args.project_command == "import":
            artifact, manifest = _run_import(args)
            print(f"Imported {manifest['counts']['documents']} documents into {artifact}")
            import_mapping = (
                args.directory.expanduser().resolve()
                / "manifests"
                / "import-mapping.json"
            )
            print(
                f"Reusable import mapping: {import_mapping}"
            )
            _print_annotation_next_steps(args.directory, artifact)
            return 0
        if args.project_command == "package-annotation":
            manifest_path, manifest = package_annotation_set(
                args.directory,
                args.annotations,
                annotation_set_id=args.annotation_set_id,
                annotator_id=args.annotator_id,
            )
            print(f"Packaged {manifest['counts']['documents']} completed documents: {manifest_path}")
            print("Select this manifest and its JSONL together in meddeid-curate.")
            return 0
        if args.project_command == "prepare-training":
            if bool(args.selection_train) != bool(args.selection_validation):
                parser.error(
                    "prepare-training requires --selection-train and "
                    "--selection-validation together"
                )
            if args.development and args.selection_validation:
                parser.error(
                    "--development cannot be combined with --selection-validation"
                )
            output, manifest = prepare_training_views(
                args.directory,
                development=args.development,
                selection_train=args.selection_train,
                selection_validation=args.selection_validation,
                test_gold=args.test_gold,
                output=args.output,
            )
            print(
                f"Prepared {manifest['development_documents']} development and "
                f"{manifest['test_documents']} test documents in {output}"
            )
            print(
                "One-time fit: "
                f"meddeid-train fit --data {output / 'fit'} ..."
            )
            print(
                "Epoch selection: "
                f"meddeid-train select-epochs --data {output / 'selection'} ..."
            )
            print(
                "Full refit: "
                f"meddeid-train refit --data {output / 'refit'} ..."
            )
            return 0
        manifest = split_project(
            args.directory,
            seed=args.seed,
            train_fraction=args.train,
            validation_fraction=args.validation,
        )
        print(json.dumps(manifest["files"], indent=2))
        return 0

    if args.command in {"generate", "sample"}:
        docs = generate_documents(
            args.count,
            seed=args.seed,
            synthea_csv_dir=args.synthea_csv_dir,
            auto_synthea=args.auto_synthea,
            synthea_repo_dir=args.synthea_repo_dir,
            synthea_population=args.synthea_population,
            force_synthea=args.force_synthea,
            require_synthea=args.require_synthea,
        )
        docs, results, model_reviews = judge_documents(docs)
        write_jsonl(docs, args.output)
        _write_dataset_manifest(docs, args.output, role="synthetic_corpus")
        _write_pretty(docs, args.pretty_output)
        if args.judge_report:
            write_report(results, model_reviews, args.judge_report)
        return 1 if any(not result.passed for result in results) else 0

    if args.command == "build-cases":
        synthea_seeds, _ = load_or_generate_synthea_csv_seeds(
            args.synthea_csv_dir,
            limit=args.count,
            auto_generate=args.auto_synthea,
            synthea_repo_dir=args.synthea_repo_dir,
            population=args.synthea_population,
            seed=args.seed,
            force=args.force_synthea,
            require=args.require_synthea,
        )
        rows = generate_case_records(
            args.count,
            seed=args.seed,
            synthea_seeds=synthea_seeds,
            start_index=args.start_index,
        )
        _write_jsonl_raw(rows, args.output)
        _write_pretty(rows, args.pretty_output)
        return 0

    if args.command == "render-cases":
        docs = render_documents_from_case_records(
            _read_jsonl(args.input), seed=args.seed
        )
        docs, results, model_reviews = judge_documents(docs)
        write_jsonl(docs, args.output)
        _write_dataset_manifest(docs, args.output, role="synthetic_corpus")
        _write_pretty(docs, args.pretty_output)
        if args.judge_report:
            write_report(results, model_reviews, args.judge_report)
        return 1 if any(not result.passed for result in results) else 0

    errors = 0
    for line_number, row in enumerate(_read_jsonl(args.path), start=1):
        problems = validate_record(row)
        if problems:
            errors += 1
            print(f"line {line_number}: {problems}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
