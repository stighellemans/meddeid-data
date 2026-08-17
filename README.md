# meddeid-data

Generate synthetic Belgian clinical notes with exact character-offset
de-identification annotations. The package includes the structured case model,
deterministic Dutch renderers, clinical resource pools, Synthea integration and
dataset-quality checks.

Belgian names, addresses, hospitals and healthcare institutions are sampled
from the versioned `nl-BE` resources distributed by `meddeid-language-nl`.
Lookup sources and attribution are distributed with the language package.
Belgian DEDUCE is not a runtime dependency.

Start with the suite guide to
[preparing and annotating data](https://meddeid.github.io/workflows/prepare-and-annotate/).
This repository remains authoritative for import, project, split, generation,
and validation commands.

## Installation

`meddeid-data` is not on PyPI yet. Install the current public source release:

```bash
git clone https://github.com/stighellemans/meddeid-data.git
cd meddeid-data
python -m pip install .
```

Install Parquet support with `python -m pip install '.[parquet]'` from the same
checkout.

## Generate synthetic data

Generate an offline dataset using the built-in clinical catalog:

```bash
meddeid-data generate --count 100 --output synthetic.jsonl \
  --pretty-output synthetic.pretty.json \
  --judge-report synthetic-report.md
meddeid-data validate synthetic.jsonl
```

Use existing Synthea CSV output or ask the tool to create it:

```bash
meddeid-data generate --count 1000 --output synthetic.jsonl \
  --synthea-csv-dir path/to/synthea/output/csv --require-synthea

meddeid-data generate --count 1000 --output synthetic.jsonl \
  --auto-synthea --require-synthea
```

The two-stage workflow keeps structured cases available for inspection:

```bash
meddeid-data build-cases --count 100 --output cases.jsonl
meddeid-data render-cases cases.jsonl --output synthetic.jsonl
```

`meddeid-data sample` provides a small offline generation run using the same
generator and Belgian lookup collection.

## Create an annotation-ready dataset

No custom conversion script is required for a folder of plain UTF-8 text files
or a CSV, TSV, or Parquet table. Create the project and import its first dataset
in one command:

```bash
meddeid-data project create my-project notes.parquet \
  --namespace hospital-study --language-profile nl-BE \
  --id-column note_id --text-column note_text
```

The same command accepts `notes.csv`, `notes.tsv`, or a directory containing
`.txt` files. Install `meddeid-data[parquet]` for Parquet support.

For tables, every column other than the text and ID columns is copied into the
document's `metadata` by default. A column named `metadata` or `metadata_json`
is interpreted as an object and merged, so structured fields such as
`patient`, `caregivers`, and `known_values` reach inference and
post-processing intact. Control this explicitly with repeatable
`--metadata-column`, `--metadata-json-column`, or `--no-metadata`.

### Zero-configuration table shape

A data scientist can avoid mapping options by producing these conventional
columns:

```csv
source_id,text,patient_given_name,patient_family_name,caregiver_1_given_name,caregiver_1_family_name,caregiver_2_given_name,caregiver_2_family_name
n-1,Jan zag Alice en Bob,Jan,Peeters,Alice,Vermeulen,Bob,Janssens
```

`source_id` is used automatically when present. Patient names may instead use a
single `patient` column. Each caregiver may similarly use paired
`caregiver_1_given_name` / `caregiver_1_family_name` columns. The aliases
`first_name` / `last_name` are also accepted. If only a complete name is
available, use the explicit `caregiver_1_full_name` convention. A
`caregivers` column accepts a real list in Parquet or a JSON-array cell in
CSV. Any unrelated columns are preserved as ordinary metadata, but none are
required. No `metadata_json` construction is needed. A complete sample is
available as `examples/zero-config-notes.csv`.

### Reusable mapping for an existing export

Column names do not otherwise need to follow the MedDeID schema. Put the
hospital export mapping in a YAML or JSON file:

```yaml
version: meddeid.import-mapping.v1
text_column: note_body
id_column: export_key
include_metadata: true
patient_given_name_column: subject_first_name
patient_family_name_column: subject_family_name
caregivers:
  - given_name_column: primary_author_first_name
    family_name_column: primary_author_family_name
  - given_name_column: coauthor_first_name
    family_name_column: coauthor_family_name
```

Each `caregivers` entry represents one person. It may instead contain
`full_name_column`; add `delimiter` to that entry only when the same source cell
contains several complete names.

Use it on the first import:

```bash
meddeid-data project create my-project notes.csv \
  --namespace hospital-study --mapping-config import-mapping.yaml
```

MedDeID writes the normalized effective mapping to
`my-project/manifests/import-mapping.json`. Later exports with the same schema
reuse it automatically:

```bash
meddeid-data project import my-project next-export.csv
```

The effective mapping is saved even when the first import used CLI flags or
only the zero-configuration conventions, so mapping remains a one-time project
step. The external YAML is useful when the same hospital export schema should
be reused across several projects.

Supplying a new `--mapping-config` replaces the saved mapping for that import;
individual CLI mapping flags override its fields. The complete example is in
`examples/import-mapping.yaml`.

For a one-off mapping, flags remain available, and the original columns still
remain in metadata:

```bash
meddeid-data project create my-project notes.csv \
  --namespace hospital-study --id-column note_id --text-column note_text \
  --patient-given-name-column patient_first \
  --patient-family-name-column patient_last \
  --caregiver-column author_1_full_name \
  --caregiver-column author_2_full_name
```

Use `--patient-name-column patient_full_name` when the patient name is stored in
one column. Full names use a best-effort comma/whitespace split, so prefer the
separate given/family options when those source columns exist. For one caregiver
column containing several names, add a literal delimiter:

```bash
meddeid-data project create my-project notes.parquet \
  --namespace hospital-study --id-column note_id --text-column note_text \
  --caregiver-column treating_clinicians --caregiver-delimiter ";"
```

The `--caregiver-column` flag always means a complete-name column. Repeat it for
several full-name columns, or add `--caregiver-delimiter` for several complete
names in one cell. Use the structured `caregivers` config shown above when an
existing export stores given and family names separately. Parquet list-valued
caregiver columns are also accepted directly. Arbitrary names such as `author`
or `validator` are never guessed.

Add `--no-metadata` if only the mapped canonical names should be retained and
the original source columns should be discarded. The mapped `patient` and
`caregivers` are still kept because they were explicitly requested.

The project keeps the source-to-document mapping and HMAC key under the
gitignored `private/` directory. Canonical artifacts contain pseudonymous stable
IDs, empty span lists, imported metadata, hashes, and the selected language
profile. The command prints the exact next steps to generate local model
pre-annotations as the initial state of an ordinary annotation assignment:

```bash
meddeid batch my-project/artifacts/annotations.jsonl \
  --output my-project/assignments/primary.jsonl \
  --model stighellemans/meddeid-dutch-synth --device cpu

MEDDEID_ANNOTATIONS_PATH="$PWD/my-project/assignments/primary.jsonl" \
npm --prefix /path/to/meddeid-annotate run dev
```

The model spans in `primary.jsonl` are not a separate suggestion type. They are
the current spans in an assignment whose documents are still unreviewed. The
reviewer edits, deletes, and adds spans using the normal annotation controls.

Model downloading happens during `meddeid batch`, not data import. After the
bundle is downloaded, inference is local and document text is not sent to a
server.

The separate lifecycle commands remain available when project initialization
and import need to happen at different times:

```bash
meddeid-data project init my-project --namespace hospital-study --language-profile nl-BE
meddeid-data project import my-project path/to/txt-notes
# or: ... notes.csv --id-column note_id --text-column note_text
# or: ... notes.parquet --id-column note_id --text-column note_text
meddeid-data project package-annotation my-project reviewer-a.jsonl \
  --annotation-set-id hospital-study-round-1 --annotator-id reviewer-7
meddeid-data project split my-project --seed 42 --train .8 --validation .1
```

Reusing the same project preserves document IDs; moving or reordering source
files does not affect downstream identity.

## Prepare training views

If development was reviewed as one assignment, pass it once. The command uses
the project's split manifest to recover the train and validation subsets:

```bash
meddeid-data project prepare-training my-project \
  --development assignments/development-reviewer-a.jsonl \
  --test-gold subannotation/evaluation-bundle/benchmark.jsonl
```

For workflows that reviewed the two development subsets separately, the
equivalent inputs are `--selection-train ... --selection-validation ...`.

This validates completion, canonical labels, exact split membership, and text
identity before writing:

```text
my-project/prepared/selection/{train,val,test}.jsonl
my-project/prepared/refit/{train,val,test}.jsonl
my-project/prepared/fit/{train,val,test}.jsonl
```

Use `prepared/fit` with `meddeid-train fit` for one-time training. The selection
test file is intentionally empty. The refit train file recombines
the complete reviewed development pool, while its test file contains only the
sealed gold set. Each directory has a checksum and lineage manifest; existing
non-empty output is never overwritten.

## Development

```bash
pip install -e '.[dev]'
pytest
```

## Licence

Code is AGPL-3.0-only. Generated datasets and incorporated resources retain the
terms stated with their respective artifacts and source notices.
