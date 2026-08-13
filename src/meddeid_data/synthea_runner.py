"""Run Synthea locally when CSV clinical seeds are not already available."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .synthea_adapter import has_synthea_csv_seeds

SYNTHEA_REPO_URL = "https://github.com/synthetichealth/synthea.git"


def _run(command: list[str], cwd: Path | None = None) -> None:
    where = f" in {cwd}" if cwd else ""
    print(f"Running: {' '.join(command)}{where}")
    subprocess.run(command, cwd=cwd, check=True)


def _clone_synthea(synthea_repo_dir: Path) -> None:
    synthea_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", SYNTHEA_REPO_URL, str(synthea_repo_dir)])


def _build_synthea(synthea_repo_dir: Path, *, force: bool) -> None:
    jar_files = list((synthea_repo_dir / "build" / "libs").glob("synthea*.jar"))
    if jar_files and not force:
        return
    _run(["./gradlew", "build", "-x", "test"], cwd=synthea_repo_dir)


def _synthea_base_dir_for_csv(csv_dir: Path) -> Path:
    if csv_dir.name != "csv":
        raise ValueError(
            "--auto-synthea with --synthea-csv-dir expects a path ending in 'csv', "
            "for example external/synthea/output/csv."
        )
    return csv_dir.parent


def ensure_synthea_csv_dir(
    *,
    csv_dir: Path | None,
    synthea_repo_dir: Path,
    population: int,
    seed: int | None = None,
    force: bool = False,
) -> Path:
    synthea_repo_dir = synthea_repo_dir.expanduser().resolve()
    resolved_csv_dir = (
        csv_dir.expanduser().resolve()
        if csv_dir
        else synthea_repo_dir / "output" / "csv"
    )

    if has_synthea_csv_seeds(resolved_csv_dir) and not force:
        print(f"Using existing Synthea CSV seeds from {resolved_csv_dir}")
        return resolved_csv_dir

    if not synthea_repo_dir.exists():
        _clone_synthea(synthea_repo_dir)
    if not (synthea_repo_dir / "run_synthea").exists():
        raise FileNotFoundError(f"Could not find run_synthea in {synthea_repo_dir}")

    _build_synthea(synthea_repo_dir, force=force)

    base_dir = _synthea_base_dir_for_csv(resolved_csv_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "./run_synthea",
        "-p",
        str(population),
        "--exporter.fhir.export=false",
        "--exporter.csv.export=true",
        f"--exporter.baseDirectory={base_dir}",
    ]
    if seed is not None:
        command.extend(["-s", str(seed)])
    _run(command, cwd=synthea_repo_dir)

    if not has_synthea_csv_seeds(resolved_csv_dir):
        raise RuntimeError(f"Synthea finished, but no seed CSV files were found in {resolved_csv_dir}")
    return resolved_csv_dir
