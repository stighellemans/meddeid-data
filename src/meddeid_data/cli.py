from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from meddeid_core import build_artifact_manifest, normalize_record, validate_record

from .generation_profiles import (
    GenerationProfile,
    profile_from_case_records,
    resolve_generation_profile,
    write_review_report,
)
from .projects import (
    ColumnMappingError,
    import_documents,
    init_project,
    load_import_mapping,
    package_annotation_set,
    resume_empty_project,
    split_project,
)
from .synthea_adapter import load_or_generate_synthea_csv_seeds
from .training_views import prepare_training_views


def _display_path(path: Path) -> str:
    """Prefer a copyable path relative to the current working directory."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _is_annotation_checkout(path: Path) -> bool:
    package_path = path / "package.json"
    if not package_path.is_file():
        return False
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return package.get("name") == "meddeid-annotate"


def _find_annotation_checkout() -> Path | None:
    configured = os.environ.get("MEDDEID_ANNOTATE_DIR")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if _is_annotation_checkout(candidate):
            return candidate
    current = Path.cwd().resolve()
    for parent in (current, *current.parents):
        candidate = parent / "repos" / "meddeid-annotate"
        if _is_annotation_checkout(candidate):
            return candidate
    return None


def _docker_bind_argument(path: Path, container_path: str) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return shlex.quote(f"{resolved}:{container_path}")
    escaped = (
        relative.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"$PWD/{escaped}:{container_path}"'


def _print_annotation_app_next_step(annotation_path: Path) -> None:
    checkout = _find_annotation_checkout()
    annotation_arg = shlex.quote(_display_path(annotation_path))
    if checkout is not None:
        checkout_arg = shlex.quote(_display_path(checkout))
        print("\nA local meddeid-annotate checkout was detected.")
        if shutil.which("npm") is None:
            print("Node.js 20+ with npm is required to run this source checkout.")
        if not (checkout / "node_modules").is_dir():
            print("Install its locked application dependencies once:")
            print(f"  npm ci --prefix {checkout_arg}")
        print("Start the review application:")
        print(
            f"  MEDDEID_ANNOTATIONS_PATH={annotation_arg} "
            f"npm --prefix {checkout_arg} run dev"
        )
        print("Then open the local URL printed by the application.")
        return

    image = "ghcr.io/stighellemans/meddeid-annotate:0.3.0"
    container_path = f"/input/{annotation_path.name}"
    if shutil.which("docker") is None:
        print(
            "\nmeddeid-annotate is a separate application, not a pip package. "
            "Docker was not found on PATH; install Docker Desktop or Docker "
            "Engine first."
        )
    else:
        print("\nNo source checkout is needed for the review application.")
    print("Start the public image (it is downloaded automatically on first use):")
    print("  docker run --rm -p 127.0.0.1:8787:8787 \\")
    print("    --read-only --cap-drop ALL --security-opt no-new-privileges \\")
    print(f"    -e MEDDEID_ANNOTATIONS_PATH={container_path} \\")
    print(f"    -v {_docker_bind_argument(annotation_path, container_path)} \\")
    print(f"    {image}")
    print("Then open http://127.0.0.1:8787.")


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


def _import_retry_command(
    args: argparse.Namespace, *, replacements: dict[str, str] | None = None
) -> str:
    """Render the equivalent import-only command after create initialized a project."""

    replacements = replacements or {}
    parts = [
        "meddeid-data",
        "project",
        "import",
        str(args.directory),
        str(args.source),
    ]
    scalar_options = (
        ("mapping_config", "--mapping-config"),
        ("text_column", "--text-column"),
        ("id_column", "--id-column"),
        ("metadata_json_column", "--metadata-json-column"),
        ("patient_name_column", "--patient-name-column"),
        ("patient_given_name_column", "--patient-given-name-column"),
        ("patient_family_name_column", "--patient-family-name-column"),
        ("caregiver_delimiter", "--caregiver-delimiter"),
    )
    for attribute, option in scalar_options:
        value = replacements.get(attribute, getattr(args, attribute))
        if value is not None:
            parts.extend((option, str(value)))
    for value in args.metadata_columns or []:
        parts.extend(("--metadata-column", value))
    for value in args.caregiver_columns or []:
        parts.extend(("--caregiver-column", value))
    if args.no_metadata:
        parts.append("--no-metadata")
    return shlex.join(parts)


def _print_annotation_next_steps(
    directory: Path, artifact: Path, *, language_profile: str
) -> None:
    assignment = (
        directory.expanduser().resolve()
        / "assignments"
        / "model-assisted-review.jsonl"
    )
    artifact_arg = shlex.quote(_display_path(artifact))
    assignment_arg = shlex.quote(_display_path(assignment))
    default_models = {
        "nl": "stighellemans/meddeid-dutch-synth",
        "nl-be": "stighellemans/meddeid-dutch-synth",
    }
    model = default_models.get(language_profile.strip().replace("_", "-").lower())
    if model:
        if shutil.which("meddeid") is None:
            print("\nModel-assisted review requires the MedDeID inference package:")
            print("  python -m pip install 'meddeid>=0.3,<0.4'")
        print("\nNext: create a model-assisted review assignment:")
        print(
            f"  meddeid batch {artifact_arg} --output {assignment_arg} "
            f"--model {model} --device cpu"
        )
        annotation_path = assignment
    else:
        print(
            f"\nNo released local pre-annotation model is configured for "
            f"{language_profile}. Start with empty spans and annotate the imported artifact:"
        )
        annotation_path = artifact
    print("\nThen review and correct that assignment:")
    _print_annotation_app_next_step(annotation_path)


def _add_generation_profile_options(
    parser: argparse.ArgumentParser, *, default_profile: str | None = "nl-BE"
) -> None:
    parser.add_argument(
        "--language-profile",
        default=default_profile,
        help="generation locale profile (built-ins: nl-BE, nl-NL, en-GB, en-US)",
    )


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    _add_generation_profile_options(parser)
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


def _write_jsonl(rows: list[dict], path: Path) -> None:
    for row in rows:
        profile = str(row.get("metadata", {}).get("generation_profile", ""))
        if profile in {"en-GB", "en-US"} and any(
            span.get("label") == "Anonymize_Other"
            for span in row.get("spans", ())
            if isinstance(span, dict)
        ):
            raise ValueError(
                f"{profile} synthetic export must not contain Anonymize_Other"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_record(row), ensure_ascii=False) + "\n")


def _write_dataset_manifest(
    rows: list[dict],
    path: Path,
    *,
    role: str,
    profile: GenerationProfile,
) -> None:
    manifest = build_artifact_manifest(
        role=role,
        artifact_path=path,
        records=rows,
        producer={"name": "meddeid-data", "version": "0.4.1"},
        contracts={
            "language_profile": profile.profile_id,
            "generation_profile": "meddeid.generation-profile.v1",
        },
    )
    manifest["language_profile"] = profile.language_manifest()
    manifest["generation_profile"] = profile.manifest()
    _write_pretty(manifest, path.with_suffix(path.suffix + ".manifest.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meddeid-data")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser(
        "generate", help="generate synthetic clinical notes with a locale profile"
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
    _add_generation_profile_options(render, default_profile=None)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--pretty-output", type=Path)
    render.add_argument("--judge-report", type=Path)
    render.add_argument("--seed", type=int, default=20260508)

    validate = sub.add_parser(
        "validate", help="validate canonical JSONL offsets and labels"
    )
    validate.add_argument("path", type=Path)

    production = sub.add_parser(
        "production",
        help="run resumable, batch-gated synthetic corpus production",
    )
    production.add_argument("production_args", nargs=argparse.REMAINDER)

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
    project_training.add_argument(
        "--output",
        type=Path,
        help=("destination for fit/selection/refit views (default: PROJECT/prepared)"),
    )
    args = parser.parse_args(argv)

    if args.command == "production":
        from .production_cli import main as production_main

        return production_main(args.production_args)

    if args.command == "project":
        if args.project_command == "create":
            resumed = False
            try:
                init_project(
                    args.directory,
                    namespace=args.namespace,
                    language_profile=args.language_profile,
                )
            except ValueError as exc:
                try:
                    root, _ = resume_empty_project(
                        args.directory,
                        namespace=args.namespace,
                        language_profile=args.language_profile,
                    )
                except ValueError as resume_error:
                    parser.error(str(resume_error))
                except FileNotFoundError:
                    parser.error(
                        f"{exc}. Choose an empty project directory; no files "
                        "were changed."
                    )
                print(
                    f"Continuing existing empty MedDeID project at "
                    f"{_display_path(root)}; "
                    "the private document-ID key is unchanged."
                )
                resumed = True
            try:
                artifact, manifest = _run_import(args)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"Dataset import failed: {exc}", file=sys.stderr)
                replacements = {}
                if isinstance(exc, ColumnMappingError) and exc.suggestion:
                    replacements[exc.argument] = exc.suggestion
                    instruction = (
                        f"If {exc.suggestion!r} is the intended column, run:"
                    )
                elif isinstance(exc, ColumnMappingError):
                    instruction = (
                        f"Choose the correct value for {exc.option} from the "
                        "available columns and retry."
                    )
                else:
                    instruction = "Correct the import options and run:"
                print(
                    "The initialized project and its private document-ID key "
                    f"were kept. {instruction}",
                    file=sys.stderr,
                )
                if not isinstance(exc, ColumnMappingError) or exc.suggestion:
                    print(
                        f"  {_import_retry_command(args, replacements=replacements)}",
                        file=sys.stderr,
                    )
                return 2
            print(
                f"{'Completed' if resumed else 'Created'} "
                f"{_display_path(args.directory)} with "
                f"{manifest['counts']['documents']} annotation-ready documents."
            )
            print(f"Canonical dataset: {_display_path(artifact)}")
            print(
                "Manifest: "
                f"{_display_path(args.directory / 'manifests' / 'input-documents.json')}"
            )
            import_mapping = (
                args.directory.expanduser().resolve()
                / "manifests"
                / "import-mapping.json"
            )
            print(f"Reusable import mapping: {_display_path(import_mapping)}")
            _print_annotation_next_steps(
                args.directory,
                artifact,
                language_profile=manifest["contracts"]["language_profile"],
            )
            return 0
        if args.project_command == "init":
            init_project(
                args.directory,
                namespace=args.namespace,
                language_profile=args.language_profile,
            )
            print(f"Created MedDeID project at {_display_path(args.directory)}")
            return 0
        if args.project_command == "import":
            artifact, manifest = _run_import(args)
            print(
                f"Imported {manifest['counts']['documents']} documents into "
                f"{_display_path(artifact)}"
            )
            import_mapping = (
                args.directory.expanduser().resolve()
                / "manifests"
                / "import-mapping.json"
            )
            print(f"Reusable import mapping: {_display_path(import_mapping)}")
            _print_annotation_next_steps(
                args.directory,
                artifact,
                language_profile=manifest["contracts"]["language_profile"],
            )
            return 0
        if args.project_command == "package-annotation":
            manifest_path, manifest = package_annotation_set(
                args.directory,
                args.annotations,
                annotation_set_id=args.annotation_set_id,
                annotator_id=args.annotator_id,
            )
            print(
                f"Packaged {manifest['counts']['documents']} completed documents: "
                f"{_display_path(manifest_path)}"
            )
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
                f"{manifest['test_documents']} test documents in "
                f"{_display_path(output)}"
            )
            output_display = Path(_display_path(output))
            print(
                "One-time fit: "
                f"meddeid-train fit --data {output_display / 'fit'} ..."
            )
            print(
                "Epoch selection: "
                f"meddeid-train select-epochs --data "
                f"{output_display / 'selection'} ..."
            )
            print(
                "Full refit: "
                f"meddeid-train refit --data {output_display / 'refit'} ..."
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
        profile = resolve_generation_profile(args.language_profile)
        docs = profile.generate_documents(
            args.count,
            seed=args.seed,
            synthea_csv_dir=args.synthea_csv_dir,
            auto_synthea=args.auto_synthea,
            synthea_repo_dir=args.synthea_repo_dir,
            synthea_population=args.synthea_population,
            force_synthea=args.force_synthea,
            require_synthea=args.require_synthea,
        )
        docs, results, model_reviews = profile.review_documents(docs)
        _write_jsonl(docs, args.output)
        _write_dataset_manifest(
            docs, args.output, role="synthetic_corpus", profile=profile
        )
        _write_pretty(docs, args.pretty_output)
        if args.judge_report:
            write_review_report(results, model_reviews, args.judge_report)
        return 1 if any(not result.passed for result in results) else 0

    if args.command == "build-cases":
        profile = resolve_generation_profile(args.language_profile)
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
        rows = profile.build_case_records(
            args.count,
            seed=args.seed,
            synthea_seeds=synthea_seeds,
            start_index=args.start_index,
        )
        _write_jsonl_raw(rows, args.output)
        _write_pretty(rows, args.pretty_output)
        return 0

    if args.command == "render-cases":
        records = _read_jsonl(args.input)
        profile = profile_from_case_records(
            records,
            requested_profile_id=args.language_profile,
        )
        docs = profile.render_case_records(records, seed=args.seed)
        docs, results, model_reviews = profile.review_documents(docs)
        _write_jsonl(docs, args.output)
        _write_dataset_manifest(
            docs, args.output, role="synthetic_corpus", profile=profile
        )
        _write_pretty(docs, args.pretty_output)
        if args.judge_report:
            write_review_report(results, model_reviews, args.judge_report)
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
