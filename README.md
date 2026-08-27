# meddeid-data

Generate synthetic clinical notes with exact character-offset de-identification
annotations. The package includes locale generation profiles, structured case
models, deterministic renderers, clinical resource pools, Synthea integration,
and dataset-quality checks.

The `nl-BE` profile samples Belgian resources from `meddeid-language-nl`.
The `nl-NL` profile shares only the locale-neutral clinical case structure. It
has a dedicated Netherlands renderer and judge, Netherlands-specific names,
addresses and institutions, and BSN, BIG, EPD/ZIS, postcode, +31 phone, and
`example.nl` conventions. It is available as generation code; this migration
does not generate or publish a corpus.
The unversioned `en-GB` and `en-US` locale profiles sample independently released, audited
name, address, institution, occupation, and terminology records from
`meddeid-language-en`. Bare `en` is rejected because date order, addresses,
identifiers, phone ranges, and clinical terminology are locale-specific.
Belgian DEDUCE is not a runtime dependency.

Start with the suite guide to
[preparing and annotating data](https://stighellemans.github.io/meddeid.github.io/workflows/prepare-and-annotate/).
This repository remains authoritative for import, project, split, generation,
and validation commands.

## Installation

```bash
python -m pip install meddeid-data
```

Install Parquet support with
`python -m pip install 'meddeid-data[parquet]'`.

## Generate synthetic data

Generation defaults to the released Dutch/Belgian profile:

```bash
meddeid-data generate --count 100 --output synthetic.jsonl \
  --pretty-output synthetic.pretty.json \
  --judge-report synthetic-report.md
meddeid-data validate synthetic.jsonl
```

Select a language and locale explicitly for multilingual generation:

```bash
meddeid-data generate --language-profile en-GB \
  --count 100 --output synthetic.en-gb.jsonl \
  --judge-report synthetic.en-gb-report.md
```

Use `--language-profile en-US` for the separate United States and territories
profile. English synthetic generation permits exactly the 14 token-classifier
labels and rejects `Anonymize_Other` during rendering, review, and export. The
full taxonomy remains valid for non-generation interoperability.

The local `nl-NL` profile can be selected in the same way. Its availability
does not imply that a Netherlands corpus exists or that the released Dutch
model has been trained or evaluated on Netherlands clinical text.

The ordered English synthetic allowlist is exactly
`Address_Location:Caregiver`, `Address_Location:Other`,
`Address_Location:Patient`, `Age_Birthdate`, `Contactdetails`, `Date`,
`ID:Caregiver`, `ID:Patient`, `Name:Caregiver`, `Name:Other`, `Name:Patient`,
`Organization:Healthcare`, `Organization:Other`, and `Profession`.

Every dataset manifest pins both `language_profile` and the complete
`meddeid.generation-profile.v1` descriptor, including resource hashes. Unknown
profiles fail before generation with the installed-provider discovery result.

Use existing Synthea CSV output or ask the tool to create it:

```bash
meddeid-data generate --count 1000 --output synthetic.jsonl \
  --synthea-csv-dir path/to/synthea/output/csv --require-synthea

meddeid-data generate --count 1000 --output synthetic.jsonl \
  --auto-synthea --require-synthea
```

The two-stage workflow keeps structured cases available for inspection:

```bash
meddeid-data build-cases --language-profile en-GB \
  --count 100 --output cases.jsonl
meddeid-data render-cases cases.jsonl --output synthetic.jsonl
```

Case records pin their generation profile, so `render-cases` resolves it without
requiring the selection again. Passing an incompatible profile is rejected.
`meddeid-data sample` provides a small offline generation run through the same
profile contract.

### Batch-gated production corpora

Use the production state machine for paid generation or any corpus that needs
personal review and a sealed benchmark. Unlike `generate`, it records every
batch transition and refuses to finalize unsigned or changed documents:

```bash
meddeid-data production init ./english-production \
  --profile en-GB,en-US \
  --count 7000 --batch-size 500 --benchmark-count 300 \
  --mode remote --allow-remote --backend english-luna \
  --author-model gpt-5.6-luna --reasoning-effort low \
  --max-cost-usd 32.68

meddeid-data production status ./english-production
meddeid-data production run ./english-production --until-gate
```

`run --until-gate` stops before personal review. Record one content-bound
decision for every document, then sign off that batch:

```bash
meddeid-data production review ./english-production \
  --batch-index 0 --document-id en-gb-synthetic-00001 \
  --decision accept --reviewer reviewer-id
meddeid-data production sign-off ./english-production \
  --batch-index 0 --reviewer reviewer-id --notes 'All 500 reviewed'
```

Decisions are `accept`, `edit`, `regenerate`, or `reject`. Changing a document
invalidates its decision and the batch sign-off. The corpus-wide writer lock
prevents a second batch from spending concurrently. The attempt ledger counts
failed and superseded API requests. `--max-cost-usd` stops new work when the
recorded spend reaches the guardrail and, after the first attempts exist,
before a batch whose projected completion would cross it. The provider account
ledger remains authoritative, so leave headroom for requests already in flight.

Use `replace` for a manually corrected canonical document; the old version and
both hashes remain in the audit trail. Use `invalidate` when the document must
be independently re-authored on the next `run`:

```bash
meddeid-data production replace ./english-production \
  --batch-index 0 --document corrected-document.json \
  --editor reviewer-id --reason 'Corrected two PII boundaries'
meddeid-data production invalidate ./english-production \
  --batch-index 0 --document-id en-gb-synthetic-00001 \
  --reason 'Regenerate with a less common address format'
```

Both operations invalidate the old automated report, document review, batch
sign-off, and any derived final split. After every batch is accepted:

```bash
meddeid-data production finalize ./english-production
```

Finalization repeats canonical-label and diversity checks, refuses exact,
PII-normalized, or configured near duplicates, and creates deterministic
profile/document-family-stratified development and benchmark files. The
English 7,000-document preset resolves to 6,700 development documents and a
300-document benchmark with 150 documents per locale and 25 per
locale/document-family cell.

Remote authoring is a plugin boundary (`meddeid.production_backends`). The
built-in `english-luna` backend implements the current English pipeline;
another language can provide a backend without adding locale branches to the
production state machine.

### Add another language or locale

Generation profiles keep locale-specific case construction, rendering, review,
resources, and provenance outside the CLI. A provider returns a
`meddeid_data.generation_profiles.GenerationProfile` and registers it through
the `meddeid.generation_profiles` entry-point group:

```toml
[project.entry-points."meddeid.generation_profiles"]
fr = "example_meddeid_language_fr.generation:get_generation_profile"
```

The provider function receives `profile_id` and keyword-only `version`. It
returns a profile when supported and raises `ValueError` otherwise. A profile
must implement direct generation, two-stage case building/rendering,
deterministic review, and a hashed resource manifest scoped to the same profile
ID and version. Tests should verify deterministic output, all document types,
exact Unicode-code-point offsets, language metadata, profile inference from case
records, resource hashes, and CLI manifest provenance. The built-in `en-GB`
and `en-US` profiles are the executable conformance examples.

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

For a profile without a released inference model, the CLI does not recommend
the Dutch model. It instead prints the command for starting primary annotation
directly from the imported empty-span artifact.

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
