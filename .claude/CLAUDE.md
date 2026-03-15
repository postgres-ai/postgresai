# postgresai

AI-native PostgreSQL observability — monitoring, health checks, and root cause analysis.

## Development Guidelines

### Rules and Standards

This project uses shared rules from the `.cursor` submodule. Always follow these rules and keep the submodule updated:

```bash
# Update the rules submodule to latest
git submodule update --remote .cursor
```

### Documentation

- **README.md** — Project overview, features, and quick start
- **CONTRIBUTING.md** — Local development workflow, Docker setup, debugging
- **quality/QUALITY_ENGINEERING_GUIDE.md** — Quality standards, processes, automated gates
- **quality/pr-review-prompt.md** — AI PR review system prompt (PostgreSQL-specific)
- **quality/failure-modes.md** — Critical failure modes with required test coverage
- **quality/checklists/** — PR review and release checklists

### Commands

- `/pgai:issues [id]` — Work with Issues from console.postgres.ai
- `/pgai:checkup <connection>` — Run health checks

## CLI

```bash
# Health check (no Docker required)
postgresai checkup postgresql://user@host:5432/db

# Specific check (e.g., unused indexes)
postgresai checkup --check-id H002 postgresql://...

# Full monitoring stack
postgresai mon local-install --demo
```

## Checks

| ID | Finds |
|----|-------|
| H002 | Unused indexes |
| H004 | Redundant indexes |
| F004 | Table bloat |
| K003 | Top queries |

## Quality

```bash
# Run release readiness check
./quality/scripts/release-readiness.sh

# Full check (includes integration tests)
./quality/scripts/release-readiness.sh --full
```

See `quality/QUALITY_ENGINEERING_GUIDE.md` for the full quality framework.
