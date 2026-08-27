# Contributing

Thank you for helping improve MedDeID data.

## Before opening a change

- Use an issue for substantial behavior or contract changes.
- Do not include patient text, credentials, private infrastructure, or
  restricted datasets.
- Preserve the canonical MedDeID schema, Unicode code-point offsets, immutable
  artifact revisions, and component dependency boundaries.
- Keep changes focused and update the authoritative documentation and
  changelog when user-visible behavior changes.

## Development workflow

Create a branch, install the development dependencies described in
[README.md](README.md), run the repository's complete test/build commands, and
open a pull request. CI must pass before merge. Security reports belong in the
private Security reporting flow described in [SECURITY.md](SECURITY.md), not in
a public issue.

The suite-wide architecture, data contract, privacy boundary, and compatibility
matrix are documented at
<https://stighellemans.github.io/meddeid.github.io/>.

## Generation profiles

Keep language- and locale-specific synthetic behavior behind the
`GenerationProfile` contract. A new provider must include deterministic direct
and two-stage generation, a locale-specific reviewer, hashed and attributed
resources, exact-offset tests, all-renderer coverage, and profile provenance in
case and dataset manifests. Do not treat translated templates alone as a new
locale: identifiers, names, addresses, contacts, date/age forms, clinical style,
and annotation boundary policy all require explicit review.
