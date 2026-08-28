# Changelog

## Unreleased

- Fixed index health checks recommending that built-in catalog indexes be dropped. H001 (invalid indexes), H002 (unused indexes) and H004 (redundant indexes) — and the `unused_indexes`, `redundant_indexes`, `rarely_used_indexes` and `index_definitions` pgwatch metrics behind them — no longer report indexes in `pg_catalog`, `information_schema`, `pg_toast` or per-backend temp schemas. Such indexes cannot be dropped, so recommending their removal was always wrong; `pg_catalog.pg_class_tblspc_relfilenode_index` was the case that surfaced it. Expect a one-off step drop in unused/redundant index counts and total sizes on the first report after upgrading: the catalog rows that were being counted are simply gone. Bloat (F004/F005) and wraparound (F002) still include catalogs on purpose.
- Fixed express checkup on databases without the optional `postgres_ai` schema. F004/F005 now degrade with machine-readable status and warning summaries instead of reporting a misleading healthy empty result.
