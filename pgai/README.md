# pgai

`pgai` is a thin wrapper around the [`postgresai`](../cli/README.md) CLI, intended to provide a short command name.

## Usage

Run without installing:

```bash
npx pgai --help
npx pgai prepare-db postgresql://admin@host:5432/dbname
```

This is equivalent to:

```bash
npx postgresai --help
npx postgresai prepare-db postgresql://admin@host:5432/dbname
```
