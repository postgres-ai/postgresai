# Changelog

## Unreleased

### Fixed

- `checkup --markdown` previously performed server-side conversion and sent the
  full report JSON to the PostgresAI API even when `--no-upload` was set. The
  flags are now mutually exclusive, and `--no-upload` prevents report data from
  being sent to the PostgresAI API. Use `--json` or `--output` for local-only
  output.
