# Changelog

All notable user-visible changes are recorded here. This project follows
semantic versioning while pre-1.0 versions may still refine public contracts.

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
