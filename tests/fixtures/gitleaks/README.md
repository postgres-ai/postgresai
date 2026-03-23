# Gitleaks Test Fixtures

Synthetic secrets used to verify that gitleaks rules fire correctly.
These are intentionally fake — not real credentials.

This directory is allowlisted in `gitleaks.toml` so the CI scan ignores it.

## Verify the rules work

```bash
# Should exit 1 (secrets detected)
gitleaks detect --no-git --source tests/fixtures/gitleaks/ --config gitleaks.toml --verbose

# Should exit 0 (allowlisted path)
gitleaks detect --source . --config gitleaks.toml --verbose
```
