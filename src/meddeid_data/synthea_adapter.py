"""Optional Synthea CSV adapter.

Synthea is used only for clinical content/story seeds. Belgian PII is generated
separately from local Belgian lookup lists.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

REQUIRED_SYNTHEA_CSVS = ("conditions.csv", "medications.csv", "observations.csv")
DEFAULT_SYNTHEA_CSV_DIR = Path("external/synthea/output/csv")
DEFAULT_SYNTHEA_POPULATION = 1000
NON_CLINICAL_CONDITION_TERMS = (
    "medication review due",
    "employment",
    "educated",
    "education",
    "social contact",
    "social isolation",
    "criminal record",
    "transport problem",
    "lack of access to transportation",
    "housing",
    "homeless",
    "refugee",
    "military service",
    "violence in the environment",
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _condition_priority(description: str) -> int | None:
    lower = description.lower()
    if lower.endswith("(situation)") or lower.endswith("(person)"):
        return None
    if any(term in lower for term in NON_CLINICAL_CONDITION_TERMS):
        return None
    if lower.endswith("(disorder)") or lower.endswith("(morphologic abnormality)"):
        return 0
    if lower.endswith("(finding)"):
        return 1
    return 2


def _age_group(age: int) -> str:
    if age < 1:
        return "infant"
    if age < 3:
        return "toddler"
    if age < 6:
        return "preschool_child"
    if age < 13:
        return "school_age_child"
    if age < 18:
        return "adolescent"
    if age < 40:
        return "young_adult"
    if age < 65:
        return "adult"
    return "older_adult"


def _synthea_patient_metadata(
    csv_dir: Path,
    reference_date: date = date(2026, 1, 1),
) -> dict[str, dict]:
    rows = _read_csv(csv_dir / "patients.csv")
    if not rows:
        return {}
    metadata: dict[str, dict] = {}
    for row in rows:
        patient = row.get("Id") or row.get("ID") or row.get("id")
        birthdate = row.get("BIRTHDATE") or row.get("birthdate")
        if not patient or not birthdate:
            continue
        try:
            year, month, day = (int(part) for part in birthdate.split("-"))
            born = date(year, month, day)
        except ValueError:
            continue
        age = reference_date.year - born.year - (
            (reference_date.month, reference_date.day) < (born.month, born.day)
        )
        age = max(0, age)
        metadata[patient] = {
            "synthea_age_years": age,
            "synthea_age_group": _age_group(age),
        }
    return metadata


def load_synthea_csv_seeds(csv_dir: Path, limit: int | None = None) -> list[dict]:
    conditions = defaultdict(list)
    medications = defaultdict(list)
    observations = defaultdict(list)
    patient_metadata = _synthea_patient_metadata(csv_dir)
    patient_order = list(patient_metadata)

    def include_patient(patient: str | None) -> bool:
        return bool(patient)

    condition_order = 0
    for row in _read_csv(csv_dir / "conditions.csv"):
        patient = row.get("PATIENT") or row.get("patient")
        description = row.get("DESCRIPTION") or row.get("description")
        priority = _condition_priority(description or "")
        if include_patient(patient) and description and priority is not None:
            conditions[patient].append((priority, condition_order, description))
            condition_order += 1

    for row in _read_csv(csv_dir / "medications.csv"):
        patient = row.get("PATIENT") or row.get("patient")
        description = row.get("DESCRIPTION") or row.get("description")
        if include_patient(patient) and description:
            medications[patient].append(description)

    for row in _read_csv(csv_dir / "observations.csv"):
        patient = row.get("PATIENT") or row.get("patient")
        description = row.get("DESCRIPTION") or row.get("description")
        value = row.get("VALUE") or row.get("value") or ""
        units = row.get("UNITS") or row.get("units") or ""
        if include_patient(patient) and description:
            observations[patient].append((description, value, units))

    all_patients = set(conditions) | set(medications) | set(observations)
    ordered_patients = [patient for patient in patient_order if patient in all_patients]
    ordered_patients.extend(sorted(all_patients - set(ordered_patients)))

    seeds = []
    for patient in ordered_patients:
        condition = None
        if conditions[patient]:
            condition = sorted(conditions[patient], key=lambda item: (item[0], item[1]))[0][2]
        seed = {
            "synthea_patient_id": patient,
            "condition": condition,
            "medications": medications[patient][:4],
            "observations": observations[patient][:6],
        }
        seed.update(patient_metadata.get(patient, {}))
        seeds.append(seed)
        if limit is not None and len(seeds) >= limit:
            break
    return seeds


def has_synthea_csv_seeds(csv_dir: Path) -> bool:
    return any(
        (csv_dir / name).is_file() and (csv_dir / name).stat().st_size > 0
        for name in REQUIRED_SYNTHEA_CSVS
    )


def discover_synthea_csv_dir(
    csv_dir: Path | None,
    *,
    synthea_repo_dir: Path = Path("external/synthea"),
) -> Path | None:
    """Return the first usable Synthea CSV directory from explicit/default paths."""

    candidates = []
    if csv_dir:
        candidates.append(csv_dir)
    candidates.append(synthea_repo_dir / "output" / "csv")
    candidates.append(DEFAULT_SYNTHEA_CSV_DIR)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = resolved.resolve() if resolved.exists() else resolved
        if key in seen:
            continue
        seen.add(key)
        if has_synthea_csv_seeds(resolved):
            return resolved
    return None


def default_synthea_population(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_SYNTHEA_POPULATION
    return max(DEFAULT_SYNTHEA_POPULATION, int(limit * 1.5) + 1)


def default_generated_synthea_csv_dir(
    *,
    synthea_repo_dir: Path = Path("external/synthea"),
    population: int,
    seed: int | None,
) -> Path:
    """Return the default auto-Synthea output dir for one generation seed."""

    seed_part = str(seed) if seed is not None else "unseeded"
    return synthea_repo_dir / f"output_seed_{seed_part}_population_{population}" / "csv"


def load_or_generate_synthea_csv_seeds(
    csv_dir: Path | None,
    *,
    limit: int | None = None,
    auto_generate: bool = False,
    synthea_repo_dir: Path = Path("external/synthea"),
    population: int | None = None,
    seed: int | None = None,
    force: bool = False,
    require: bool = False,
) -> tuple[list[dict], Path | None]:
    if auto_generate:
        from .synthea_runner import ensure_synthea_csv_dir

        resolved_input_csv_dir = csv_dir or default_generated_synthea_csv_dir(
            synthea_repo_dir=synthea_repo_dir,
            population=population or default_synthea_population(limit),
            seed=seed,
        )
        resolved_csv_dir = ensure_synthea_csv_dir(
            csv_dir=resolved_input_csv_dir,
            synthea_repo_dir=synthea_repo_dir,
            population=population or default_synthea_population(limit),
            seed=seed,
            force=force,
        )
        seeds = load_synthea_csv_seeds(resolved_csv_dir, limit=None)
        if limit is not None and len(seeds) < limit and not force:
            print(
                f"Existing Synthea CSV seeds contain {len(seeds)} usable patients, "
                f"but {limit} were requested; regenerating Synthea output."
            )
            resolved_csv_dir = ensure_synthea_csv_dir(
                csv_dir=resolved_input_csv_dir,
                synthea_repo_dir=synthea_repo_dir,
                population=population or default_synthea_population(limit),
                seed=seed,
                force=True,
            )
            seeds = load_synthea_csv_seeds(resolved_csv_dir, limit=None)
    else:
        resolved_csv_dir = discover_synthea_csv_dir(
            csv_dir,
            synthea_repo_dir=synthea_repo_dir,
        )
        seeds = load_synthea_csv_seeds(resolved_csv_dir, limit=None) if resolved_csv_dir else []

    if not resolved_csv_dir:
        if require:
            raise RuntimeError(
                "No Synthea CSV seeds were found. Pass --synthea-csv-dir, place CSV files under "
                "external/synthea/output/csv, or enable automatic Synthea generation."
            )
        return [], None

    if (auto_generate or require) and not seeds:
        raise RuntimeError(f"No Synthea clinical seeds could be loaded from {resolved_csv_dir}")
    if require and limit is not None and len(seeds) < limit:
        raise RuntimeError(
            f"Only {len(seeds)} usable Synthea clinical seeds were loaded from {resolved_csv_dir}, "
            f"but {limit} cases were requested. Use a larger --synthea-population, "
            "or --force-synthea to regenerate the CSV files."
        )
    return seeds, resolved_csv_dir
