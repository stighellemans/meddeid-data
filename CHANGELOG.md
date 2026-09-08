# Changelog

All notable user-visible changes are recorded here. This project follows
semantic versioning while pre-1.0 versions may still refine public contracts.

## [Unreleased]

## [0.4.1] - 2026-09-08

- Documented and tested explicit output directories for prepared training
  views, making it clear how to keep multiple immutable data snapshots apart.

## [0.4.0] - 2026-09-05

- Made `project create` safely retryable after a failed first import. It now
  continues only a matching untouched scaffold, preserves the private
  document-ID key, prints an actionable `project import` recovery command, and
  continues to refuse projects that already contain data or custom content.
- Validate project identity before creating directories, avoiding partial
  scaffolds for empty namespace or language-profile values.
- Show project artifacts and generated follow-up commands relative to the
  current working directory when possible, while retaining absolute paths for
  projects outside it.
- Report invalid text/ID column selections as mapping errors, list available
  source columns, suggest a close header match, and use that suggestion in the
  printed recovery command instead of repeating the invalid option.
- Rename the recommended model-initialized assignment from the opaque
  `primary.jsonl` to `model-assisted-review.jsonl` and describe the next step
  explicitly as reviewing and correcting that assignment.
- Make the annotation handoff self-contained for new researchers: detect a
  configured or suite-local `meddeid-annotate` checkout and its dependencies,
  otherwise explain that the UI is not a pip package and print a public Docker
  command whose image is downloaded automatically. Also print the compatible
  inference install command when `meddeid` is not available. The printed
  commands use the coordinated `meddeid` 0.3 and `meddeid-annotate` 0.2 release
  lines.

## [0.3.0] - 2026-08-27

- Added the generic `meddeid-data production` state machine with immutable
  plans, corpus/batch locking, pluggable remote backends, attempt cost ceilings,
  500-document gates, content-bound review/sign-off, and deterministic sealed
  splits.
- Production plans now use unversioned locale identities such as `en-GB` and
  reject `en-GB@1`; package releases, resource manifests, model bundles, and
  Git history provide independent provenance.
- Production now retains invalidated and manually edited document versions,
  forces a fresh locale-specific and shared audit after every content change,
  estimates cost from every API attempt, and blocks projected over-budget
  batches after real usage is available.
- Personal English review and finalization now reject stale decisions and
  sign-offs after document, quality-report, or decision-file changes.
- Added the versioned `meddeid.generation-profile.v1` plugin contract and
  provider discovery for language- and locale-specific synthetic generation.
- Added `--language-profile` to direct and
  two-stage generation, with profile-pinned case records and dataset manifests.
- Added deterministic `en-GB` and `en-US` profiles with six document
  types, audited language-pack resources, Synthea seeding, and locale-specific
  conformance review.
- English generation uses the exact 14-label token-classifier allowlist and
  rejects `Anonymize_Other` at render, review, and export boundaries.
- Stopped recommending the Dutch inference model for non-Dutch import projects.
- Added end-to-end English generation, case round-trip, offset, determinism,
  manifest, resource-hash, and annotation-handoff tests.
- Added a dedicated `nl-NL` renderer and deterministic/optional-LLM judge with
  Netherlands healthcare terminology and administrative formats. No
  Netherlands corpus was generated or published.

## [0.2.1] - 2026-08-17

- Published the first externally supported MedDeID data release.
- Added public installation, compatibility, licensing, and verification
  metadata.
- Established independent CI and immutable release artifacts.

For earlier migration history, consult the repository's Git history.
