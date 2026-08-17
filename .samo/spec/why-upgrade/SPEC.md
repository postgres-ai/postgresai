# SPEC — `pgai why-upgrade`

**Version:** 0.1.0
**Status:** Draft for review
**Slug:** `why-upgrade`
**Scope:** PostgreSQL **minor** releases, lines **17.x** and **18.x**
**Date:** 2026-08-17

> Authoring note: this v0.1 was hand-authored from a three-expert research panel
> (PostgreSQL domain research, competitive/GTM analysis, and a codebase
> architecture audit) plus first-hand verification against the PostgreSQL source
> tree and a live cluster. It is laid out for `samospec iterate why-upgrade` to
> pick up and run review rounds on top.

---

## 1. Goal & why

### 1.1 The problem, in one number

Upgrading PostgreSQL 17.2 → 17.11 crosses **496 release-note entries**. Nobody
reads 496 entries, so nobody reads any of them, so the handful that genuinely
matter get missed.

Measured from the source tree (`doc/src/sgml/release-1[78].sgml` on
`REL_17_STABLE` / `REL_18_STABLE`, 2026-08-17):

| Line | Releases | Items |
|---|---|---|
| 17.x | 17.1 … 17.11 (11) | **545** |
| 18.x | 18.1, 18.2, 18.3, 18.4, 18.6 (5) | **351** |

55 distinct CVEs are referenced across the 17.x line alone.

The items that matter are not cosmetic. Verified examples:

- **CVE-2026-6471 / `output_plugin_libraries` (17.11 and 18.6).** A new GUC
  whitelists logical-decoding output plugins, defaulting to `pgoutput,
  test_decoding`. Anyone using `wal2json`, `decoderbufs`, or another
  third-party decoder finds **logical replication broken after a patch-level
  upgrade** until `postgresql.conf` is edited. `pg_upgrade --check` fails too.
- **17.1** — catalog and/or data corruption for anyone who ever detached a
  partition from a partitioned table holding an FK reference; manual repair
  required. Separately: `LC_CTYPE=C` with non-C `LC_COLLATE` ⇒ reindex text indexes.
- **17.5** — self-referential FK constraints on partitioned tables may not be
  enforced; recreate them. Also reindex BRIN bloom indexes.
- **17.6** — reindex BRIN `numeric_minmax_multi_ops` indexes.
- **18.2** — reindex indexes on `ltree` columns.
- **18.6** — possibly-corrupt `reltuples` on tables with GIN indexes; reindex
  `btree_gist` / `ltree` indexes.

Every one of those is **deterministically detectable from the catalog** — proven
in §4.6. The release notes tell you a fix exists. They never tell you whether it
is *your* problem, and never in priority order.

### 1.2 What we build

One curated knowledge base and one analysis engine, exposed through three
surfaces at three levels of personalization:

1. **CLI** — `npx pgai@latest why-upgrade` — connects to a live database, answers
   *"which of these affect me, and what do I do?"* Free, no signup.
2. **Console** (console.postgres.ai → DB health) — the same engine, continuous
   and fleet-wide, with evidence retention. Paid.
3. **Public web** (postgres.ai/why-upgrade) — the same engine with no database:
   full enriched content, nothing personalized. Free, no login.

### 1.3 Why it is defensible

The summarization layer is not defensible — release notes are free and any LLM
can summarize them. **The join against a live catalog is.** "3 of 47 affect you,
because you run logical replication with `wal2json`, have 4 BRIN bloom indexes,
and one self-referential FK on a partitioned table" requires catalog
introspection an external model cannot perform. That is the product.

### 1.4 Honest verdict

As a public content site this is a **vitamin**: release notes are free, LLMs
summarize them, and interest is spiky (quarterly release waves). It becomes a
**painkiller** only at the moment of connection to a real database with a clock
running — a maintenance window, a CVE ticket, an audit. Therefore: build it as a
tightly-scoped wedge (~1–2 engineer-quarters), judge it on qualified CLI
conversions rather than standalone ARR, and let the Console carry monetization.

### 1.5 Non-goals for v1

- Major-version upgrades, and feature-level ("should I adopt X") analysis. The
  unit of analysis in v1 is a **fix**. Features belong with majors, in v2, and
  only if this earns it.
- PG lines other than 17.x and 18.x.
- A full third-party extension compatibility matrix (see §4.8 for what we do
  instead).
- Any write to the user's database, ever.

---

## 2. User stories

**US-1 — DBA preparing a maintenance window.** *As a DBA on 17.2 planning a jump
to 17.11, I want a ranked list of what actually affects my database so I can
write a change ticket that survives a change-advisory board.*
Accept: output separates before-restart actions from after-upgrade actions;
every claim names concrete objects; a copy-pasteable summary is available.

**US-2 — Platform engineer facing forced cloud maintenance.** *As an RDS user
whose provider scheduled a minor upgrade, I want to know what will change under
me and whether anything will break.*
Accept: works when `version()` is vendor-mangled; states clearly which checks
could not run without superuser; never reports a false "you're fine".

**US-3 — Security engineer with a scanner ticket.** *As a security engineer
whose scanner flagged CVE-2026-6471, I want to know whether we are reachable.*
Accept: CVE lookup resolves to the fixing minor per line; exposure verdict is
backed by named evidence; a documented "not exposed, here's the proof" is a
valid, defensible outcome.

**US-4 — Anonymous visitor on release day.** *As someone who saw "PostgreSQL
18.6 released" on HN, I want to know what's in it.*
Accept: no login, no database, permalinked and shareable; every conditional item
shows the check that would resolve it.

**US-5 — Fleet owner.** *As the owner of 62 instances, I want to know which are
affected by a newly-published fix without re-running anything.*
Accept: Console rollup; new releases light up affected instances automatically.

**US-6 — Auditor-response.** *As the person answering a SOC 2 patch-management
request, I want a durable artifact showing exposure window and remediation date.*

**US-7 — CI gate.** *As a platform team, I want `--format json` so a pipeline can
fail when an instance is exposed to an unpatched critical fix.*

---

## 3. Architecture

<!-- architecture:begin -->
```text
  UPSTREAM SOURCES                    CURATION (platform-all)            CONSUMERS
 ┌───────────────────────┐          ┌──────────────────────────┐
 │ release-1[78].sgml    │          │  ingest: parse <listitem>│      ┌──────────────────┐
 │  · Author:            │──────────▶│   · master commit hashes │      │ CLI  pgai        │
 │  · Branch: X [sha]    │          │     = canonical fix id   │      │ why-upgrade      │
 │  · CVE-XXXX-NNNN      │          │   · per-branch fixed_in  │      │                  │
 │  · migration sect2    │          ├──────────────────────────┤      │ ┌──────────────┐ │
 ├───────────────────────┤          │  enrich (AI + human gate)│      │ │capability    │ │
 │ git commits           │──────────▶│   · why / who / thread   │      │ │profile (1×)  │ │
 │  Discussion: <thread> │          │   · severity             │      │ └──────┬───────┘ │
 ├───────────────────────┤          │   · DETECTOR (probes+fn) │      │        ▼         │
 │ CVE list + CVSS       │──────────▶│   · remediation protocol │      │ ┌──────────────┐ │
 ├───────────────────────┤          ├──────────────────────────┤      │ │probe runner  │ │
 │ extension changelogs  │──────────▶│  review gate: 2-person   │      │ │ dedup, RR tx │ │
 └───────────────────────┘          │  sign-off on any item    │      │ │ SET LOCAL to │ │
                                    │  with a detector         │      │ └──────┬───────┘ │
                                    └────────────┬─────────────┘      │        ▼         │
                                                 │                    │ ┌──────────────┐ │
                                      kb.json (versioned)             │ │ evaluate →   │ │
                                                 │                    │ │ exposure +   │ │
                          ┌──────────────────────┼──────────┐         │ │ evidence     │ │
                          ▼                      ▼          ▼         │ └──────────────┘ │
              ┌───────────────────┐  ┌────────────────┐ ┌───────────┐ └────────▲─────────┘
              │ GET /api/general/ │  │ console        │ │ website   │          │
              │ upgrade_advisories│  │ DB health      │ │ /why-     │  build-time embed
              │  (anonymous)      │  │ (fleet, cont.) │ │ upgrade   │──────────┘
              └───────────────────┘  └────────────────┘ └───────────┘   (offline-first)
```
<!-- architecture:end -->

### 3.1 Repository split (verified)

The console and the website are **not** in `postgres-ai/postgresai` — there are
zero `.tsx/.vue/.html` files outside `node_modules`. Both live in
`gitlab.com/postgres-ai/platform-all`, reached over PostgREST
(`POST {base}/rpc/{name}`). This therefore ships as **two coordinated MRs**:

| Repo | Owns |
|---|---|
| `postgresai` | ingest+enrich pipeline, KB schema, CLI command, detection engine, build-time embed |
| `platform-all` | `upgrade_advisories` table + anonymous endpoint, Console DB-health module, public page |

### 3.2 Data model

Verified from the source tree: for CVE-2026-6471, the 17.11 and 18.6 entries
cite the **identical master commit hashes** (`226e49cbe`, `20e9dcefd`). The set
of master commit hashes is therefore a **stable canonical identity for a logical
fix across every branch that received it** — dedup is deterministic, not fuzzy
text matching. Hence two tables:

- **`fix`** — PK = master commit hash(es). Holds all enrichment.
- **`fix_release`** — `(fix_id, major_line, fixed_in_minor, branch_commit)`; one
  row per back-patched branch.

Interval logic falls out with no special cases:
- 17.2 → 17.11: include iff a 17-line release exists with `17.2 < fixed_in ≤ 17.11`.
- 17.11 → 18.6: **exclude** CVE-2026-6471 — its 17-line release is 17.11, which
  the user already has. This cross-major case is what a naive
  concatenate-the-notes implementation gets wrong.

Two real edge cases the model must survive:
- **Minor versions are not contiguous.** 18.5 was stamped and never released (a
  regression found post-wrap); the notes say so in prose. Never assume `n+1`.
- **Upstream chains its own migration advice** ("if upgrading from a version
  earlier than 17.6, see …") — upstream confirming that accumulation over an
  interval, not per-release display, is the correct model.

### 3.3 The core mechanic: **exposure**, not "matching score"

For each fix we answer *"how likely is this hitting us?"* in two stages.

**Stage 1 — capability gate (cheap, kills most items).** Compute a **capability
profile** once: logical replication in use? partitioning? BRIN/GIN/GiST? which
extensions at which versions? FDW? JIT? non-C collations? Each fix declares the
capabilities it requires. No logical replication ⇒ every logical-decoding fix
collapses to `NOT_EXPOSED` instantly. This is what makes ~900 items cheap to
evaluate against one profile.

**Stage 2 — trigger check (expensive, item-specific, sometimes impossible).**
You use logical replication — but do your circumstances trigger *this* bug?
Sometimes trivial; sometimes a race under concurrent DDL that cannot be observed
from SQL at all. Product integrity depends on saying which.

**Naming.** "Matching score" is rejected on two counts: *matching* reads as
search relevance, and *score* implies a continuous number — a bare `0.73` is
false precision we cannot defend in a safety-adjacent judgment. Operators
already have the term of art from the CVE world: **exposure**. It is expressed
as an **ordinal ladder with evidence**, never a naked float:

| Level | Meaning |
|---|---|
| `CONFIRMED` | Proven from the catalog; evidence attached |
| `LIKELY` | Subsystem in use, most preconditions match, one unprovable |
| `POSSIBLE` | Subsystem in use, specific trigger not checkable |
| `UNLIKELY` | Subsystem in use, preconditions clearly absent |
| `NOT_EXPOSED` | Capability gate proves the feature is unused |
| `UNKNOWN` | Could not check — permission, data, or access missing |

Two invariants, or the product is worse than useless:

1. **`UNKNOWN` is never folded into `NOT_EXPOSED`.** They are opposite claims;
   conflating them is how someone gets paged at 3am.
2. **The ladder is deliberately asymmetric.** A false `NOT_EXPOSED` is far worse
   than a false `POSSIBLE`. Claim `NOT_EXPOSED` only on *positive proof of
   absence*, never on absence of evidence.

**Severity stays orthogonal** (data loss / corruption / crash / wrong results /
security / performance / cosmetic). Exposure is likelihood; severity is impact;
priority is their product — mirroring how CVSS separates them. A `POSSIBLE`
data-corruption fix should outrank a `CONFIRMED` cosmetic one. Sorting may use a
derived rank; the *displayed* artifact is always level + evidence + severity.

**Without a database**, every conditional item is `UNKNOWN` plus its precondition
text — which is exactly the public web page, from the same code path. The gap
between "47 UNKNOWN" and "7 CONFIRMED" *is* the conversion pitch.

### 3.4 Generic engine, Postgres as vertical #1

The engine shape — *upstream fix corpus × capability profile of a live system →
ranked exposure with evidence* — is not Postgres-specific. Keeping the core types
free of Postgres nouns costs almost nothing now and preserves headroom for other
verticals later. Postgres-specific logic lives behind a `SystemAdapter`.

---

## 4. Implementation details

### 4.1 Ingest

Parse `<listitem>` blocks from `release-17.sgml` / `release-18.sgml` on the
stable branches. Each carries an HTML comment with `Author:` and one
`Branch: <name> [<sha>]` line per back-patched branch, then prose, attribution,
and any CVE ID. Also parse the `<sect2 id="release-NN-M-migration">` block — the
highest-value content in the corpus, since it is upstream explicitly saying
"you must act".

Ingest is **deterministic and re-runnable**; it never calls an LLM. Output is
`fix` + `fix_release` rows with verbatim upstream text preserved separately from
enrichment (see §9 on licensing).

### 4.2 Enrichment (AI-assisted, human-gated)

~900 items is too many to hand-write and too dangerous to fully automate. So:

- LLM drafts `why`, `who`, severity, capability requirements, and a *proposed*
  detector, grounded in the commit diff and the `Discussion:` thread.
- **Any item carrying a detector or a remediation protocol requires two-person
  human sign-off before publication.** A wrong detector produces a false
  `NOT_EXPOSED`, which is the one failure this product must not have.
- Items without sign-off ship as content-only (no exposure verdict), never as
  `NOT_EXPOSED`.

### 4.3 Knowledge base distribution

Precedent already exists in-repo: `cli/scripts/embed-checkup-dictionary.ts`
fetches from the postgres.ai API **at build time** and codegens an embedded TS
module — with an SSRF host allowlist, a timeout, and a non-fatal fallback. Its
docstring states the intent: *"no API calls are made at runtime while keeping the
data up-to-date."* That is exactly the why-upgrade shape.

- Author platform-side; serve `GET /api/general/upgrade_advisories` (anonymous).
- Embed at build time via `cli/scripts/embed-upgrade-kb.ts`; register it in
  `embed-all`, `.gitignore`, and `.npmignore` (omitting any of the three breaks
  `typecheck` on a clean checkout — an existing trap).
- **Offline-first default: zero network calls.** `--refresh` and `--kb <path>`
  (airgap escape hatch) are explicit opt-ins. `--no-upload` must continue to mean
  *no network, period* — the one open CHANGELOG entry is a fix for exactly this
  class of leak.
- Staleness is surfaced, not hidden: every item carries `kb_generated_at`; when
  the target version is newer than the snapshot, say so in text and JSON.
- `kb_contract_version` follows the existing `CONTRACT_VERSION` policy
  (MINOR = additive; consumer accepts same MAJOR, MINOR ≥ built-against).
- Guard the build-time fetch: commit a fallback snapshot, assert `items.length > 0`
  in a test, and fail *release* builds (not dev builds) on an empty payload.

Measured budget: the published `postgresai` tarball is **312 KB**; a 17.x+18.x KB
adds roughly **15–90 KB gzipped**. Affordable. It would not be if extended to
every major back to PG 12.

### 4.4 Detection engine

Reuse the existing rule-engine idiom (`AutovacuumRule` / `F001_RULES` in
`cli/lib/checkup.ts`: `appliesTo` version gate + `predicate` + `message`, with
severity roll-up). One structural change: F001 resolves a single context up
front, but ~900 fixes have heterogeneous evidence needs. So **separate "what
evidence does this fix need" (probes) from "does the evidence indict you"
(evaluate), and deduplicate probes across fixes.** Checks run serially on a
single `pg` Client — naive per-item probing is minutes on a real database, so
dedup is design, not optimization.

```ts
export interface Probe {
  id: string;
  label: string;
  /** SQL by minimum PG major; select highest key <= actual (reuse metrics-loader). */
  sqls: Record<number, string>;
  params?: unknown[];
  timeoutMs: number;
  requires?: Array<"pg_read_all_stats" | "postgres_ai.pg_statistic" | "superuser">;
  cost: "cheap" | "catalog-scan" | "heavy";
}

export type ProbeResult =
  | { status: "ok"; rows: Record<string, unknown>[] }
  | { status: "skipped"; reason: "version-gate" | "insufficient-permission" | "cost-budget" }
  | { status: "failed"; reason: "timeout" | "error"; detail: string };

export type Exposure =
  | { level: "CONFIRMED" | "LIKELY" | "POSSIBLE" | "UNLIKELY"; evidence: string; subjects?: string[] }
  | { level: "NOT_EXPOSED"; evidence: string }
  | { level: "UNKNOWN"; reason: string };

export interface Detector {
  probes: Probe[];
  appliesTo?: (ctx: DetectContext) => boolean;   // cheap pre-filter, no I/O
  evaluate: (ctx: DetectContext) => Exposure;
}
```

Safety properties:
- Probes are declarative `sqls` maps — **no code path can emit a write.**
- Wrap the run in `begin isolation level repeatable read` … `rollback` (an
  existing idiom in `init.ts`): consistent snapshot, writes structurally
  impossible.
- Per-probe `SET LOCAL statement_timeout` inside that transaction — `LOCAL`
  needs no restore and cannot leak. (Note the current global timeout is a single
  30s and `metrics.yml`'s per-metric timeout is embedded but never applied —
  don't repeat that.)
- `cost` + a `--fast` flag caps catalog-heavy probes on very large databases.
- Permission failures degrade to `UNKNOWN` with copy-pasteable `grant` SQL,
  matching `formatPermissionCheckMessages`.
- Probe SQL lives in `sqls` maps with `$1` params, keeping the
  `quality:sql-safety` CI gate (which greps for `${…}` near SQL keywords) happy.

### 4.5 CLI surface

```
pgai why-upgrade [conn]              # detect current, target = latest in line
pgai why-upgrade --from 17.2 --to 17.11   # no DB; same renderer as the web page
pgai why-upgrade --protocol          # full step-by-step runbook
pgai why-upgrade --verify            # post-upgrade: did the remediations happen?
  --format json | --fast | --refresh | --kb <path> | --show-sql
```

Output is ordered by **what the operator must do**, not by release or severity
label: *act before you restart* → *act after you upgrade* → *affects you, no
action* → the reduction headline ("489 of 496 do not apply"), with
permission-limited and workload-dependent unknowns listed **separately**.

Naming: files must be `why-upgrade-*`. `upgrade` already means *monitoring-stack*
upgrade in this repo (`mon update`, `cli/test/upgrade.test.ts`).

Wiring: library-first in `cli/lib/why-upgrade*.ts`, consumed by **both** a
top-level command and a thin `REPORT_GENERATORS` entry. `A013` ("Postgres minor
version") is the natural anchor — it already collects the version fields and its
summary is currently a stub. Registering there inherits `--json`, `--output`,
console upload, and — via `generate_issue: true` — auto-created Issues.
**Ship the library and the command in one MR**: `cli/lib/checkup-baseline.ts` is
239 fully-tested lines wired to nothing, the cautionary precedent.

### 4.6 Detectors — validated, not asserted

All five were executed against a live cluster (PG 16.13, purpose-built fixture)
during authoring. Results below are observed, not predicted.

| Fix | Detection | Result |
|---|---|---|
| BRIN bloom / `numeric_minmax_multi_ops` (17.5, 17.6) | `pg_index` → `unnest(indclass)` → `pg_opclass`, `amname='brin'`, opcname `LIKE '%_bloom_ops' OR '%_minmax_multi_ops'` | found both; **correctly excluded** the plain-minmax BRIN index |
| `btree_gist` / `ltree` indexes (17.11, 18.2, 18.6) | same join + `pg_depend`→`pg_extension` on the opclass | found all three, across both gist and btree |
| Self-referential FK on partitioned table (17.5) | `pg_constraint` where `contype='f' AND conrelid=confrelid` and `relkind='p'` | found |
| `LC_CTYPE=C` + non-C `LC_COLLATE` (17.1) | `pg_database` | correct **true negative** (0 rows) |
| `output_plugin_libraries` (17.11/18.6, CVE-2026-6471) | `pg_replication_slots` where `plugin NOT IN ('pgoutput','test_decoding')` | classification verified |

The precision property matters as much as the recall property: telling someone to
`REINDEX` an index that is not affected burns trust as fast as missing one.

### 4.7 Managed services

`version()` on RDS/Aurora/Cloud SQL/AlloyDB does not map 1:1 to community minors,
and superuser is unavailable. Requirements: parse vendor version strings into a
community-minor floor; mark vendor-patched uncertainty explicitly; ensure every
probe degrades to `UNKNOWN` rather than throwing when privileges are missing.
This is the largest install base and the most on-message segment (the provider
decides your window and tells you nothing) — but also the hardest to be *correct*
about. See open question OQ-3.

### 4.8 Third-party extensions

Not a full compatibility matrix in v1 — that is an unbounded standing
commitment. Instead: a **curated known-incidents list** for the top ~10
(pg_cron, pgvector, PostGIS, TimescaleDB, pg_partman, pg_repack, Citus,
pg_stat_statements, pgBouncer, Patroni), plus the cheap and genuinely useful
part — read installed extension versions from `pg_extension` and flag any that
require an `ALTER EXTENSION UPDATE` after the target minor.

### 4.9 Public web

- **Do not** generate the version-pair matrix (~700 versions ⇒ ~240k near-duplicate
  pages). In the current search regime that is scaled-content abuse and a
  deindexing risk.
- Index a small, high-authority set (~600 pages): one per release
  (`/why-upgrade/18.6`), one per CVE (`/cve/CVE-2026-6471`), one evergreen per
  supported major. Arbitrary pair diffs are fully supported, shareable, and
  `noindex`.
- Optimize for **machine legibility**, which is where the incumbents are weakest
  (the leading tool has one `<title>` for every page and no API; pgpedia 403s
  bots): server-rendered HTML, stable canonical URLs, per-page metadata, JSON-LD,
  an open JSON API with CORS, `llms.txt`, and the dataset mirrored to a public
  repo. Publish within the hour of the pgsql-announce email — the release
  calendar is known a year ahead.

---

## 5. Test plan (red/green TDD)

Written red first, in the repo's existing idioms (`bun:test`, `createMockClient`
fixture routing, Ajv 2020 schema validation, ephemeral-cluster integration tests
that shell out to `initdb`/`postgres` and skip when absent or running as root).

**Interval logic (pure, no DB) — the highest-risk unit.**
1. RED: 17.2→17.11 includes a fix whose `fixed_in=17.5`. GREEN.
2. RED: 17.2→17.11 **excludes** `fixed_in=17.1`. GREEN.
3. RED: 17.11→18.6 **excludes** CVE-2026-6471 (17-line release at 17.11 already
   held). GREEN. ← the cross-major dedup case
4. RED: a fix back-patched to both lines appears **once**, not twice. GREEN.
5. RED: 18.4→18.6 does not fabricate 18.5. GREEN. ← non-contiguous versions
6. RED: target < current is rejected with a clear error. GREEN.

**Exposure semantics — the safety-critical invariants.**
7. RED: a probe returning `insufficient-permission` yields `UNKNOWN`, and
   `UNKNOWN` never renders in a "not affected" bucket in any of text, JSON, or
   markdown. GREEN.
8. RED: a probe that times out yields `UNKNOWN`, not `NOT_EXPOSED`. GREEN.
9. RED: `NOT_EXPOSED` is only ever produced with non-empty evidence. GREEN.
10. RED: priority ordering places a `POSSIBLE` data-corruption fix above a
    `CONFIRMED` cosmetic one. GREEN.

**Detection (integration, real cluster with the §4.6 fixture).**
11–15. One test per validated detector, asserting both the positive case and the
true negative (e.g. the plain-minmax BRIN index must **not** be reported).
16. RED: the whole analysis runs inside a transaction that is rolled back;
    asserted by checking no relation was created/modified. GREEN.
17. RED: every probe respects its own `SET LOCAL statement_timeout`. GREEN.

**Distribution / contract.**
18. RED: `kb.json` validates against `reporter/schemas/upgrade_kb.schema.json`
    (Ajv 2020, `additionalProperties: false`). GREEN.
19. RED: embedded KB has `items.length > 0` — guards the silent-empty-fetch
    failure mode. GREEN.
20. RED: with no flags, the command performs **zero** network calls (spy on
    fetch). GREEN. ← the `--no-upload` class of leak
21. RED: version-pair mode with no connection returns all-`UNKNOWN` and never
    touches `pg`. GREEN.
22. CLI-level via `Bun.spawnSync`: `--format json` emits schema-valid JSON; exit
    codes are stable.

---

## 6. Team of veteran experts

| Seat | Why needed |
|---|---|
| **PostgreSQL hacker / curator** | Owns detector correctness and the two-person sign-off gate. The single most important seat — a wrong detector is the product's one unacceptable failure. |
| **CLI/TypeScript engineer** | Engine, probes, renderers, embed script; lives in `postgresai`. |
| **Platform engineer (`platform-all`)** | KB table + anonymous endpoint, Console DB-health module, public page. |
| **Content/curation ops** | Release-day freshness; quarterly waves plus out-of-cycle emergencies. This is a standing commitment, not a launch task. |
| **Technical writer / DevRel** | Enrichment prose voice; launch essay; community relations. |

---

## 7. Sprint plan

**S1 — Ingest + data model.** Parser for `release-1[78].sgml`; `fix`/`fix_release`
schema keyed by master commit; interval + dedup logic with tests 1–6 green.
*Exit:* correct item sets for any 17.x/18.x pair, verified against upstream's own
chained migration advice.

**S2 — Engine + detectors for the action items.** Probe/Detector abstraction;
implement the ~15 highest-value detectors (every "you must act" item in §1.1);
tests 7–17 green. *Exit:* `--from/--to` and connected mode both produce ranked
exposure with evidence on a real cluster.

**S3 — CLI polish + distribution.** Renderers (text/JSON), `--protocol`,
`--verify`, embed script, schema, staleness UX; tests 18–22 green.
*Exit:* `npx pgai@latest why-upgrade` is shippable and offline-first.

**S4 — Public page (`platform-all`).** Endpoint + ~600-page render + machine
legibility. *Exit:* live, permalinked, citable.

**S5 — Console DB health.** Fleet rollup, continuous re-evaluation, Issues
integration, audit artifact. *Exit:* paid tier has something the CLI structurally
cannot do.

**S6 — Curation to depth.** Broaden enrichment beyond the action items toward the
full corpus, at whatever depth OQ-1 settles on.

Launch is wrapped in the broader narrative rather than shipped as a routine
release, and credits the prior art loudly (§10).

---

## 8. Changelog

- **0.1.0** (2026-08-17) — Initial draft. Three-expert panel synthesis; corpus
  and migration sections measured from the source tree; canonical fix identity
  verified via cross-branch master commit hashes; five detectors validated on a
  live cluster; "exposure" adopted over "matching score"; repo split confirmed.

---

## 9. Licensing & correctness risk

Upstream release-note text is under the PostgreSQL License. Keep verbatim
upstream text in a **separate field** from our enrichment so attribution is
mechanical and unambiguous, and carry the required notice. Legal review before
the public page ships.

The dominant correctness risk is a **false `NOT_EXPOSED`**. Mitigations, in
order of strength: positive-proof-only for that verdict; two-person sign-off on
every detector; `UNKNOWN` as a first-class, visually distinct outcome; and true
negatives asserted in the integration tests, not just positives.

---

## 10. Prior art & community stance

`why-upgrade.depesz.com` (source: GitLab `depesz/pgVersions`) is real, beloved,
and maintained by hand on release day. It solved a subtle problem we must not
regress — deduplicating back-patched fixes along an upgrade path — and its
GUC-default data is derived by *compiling every PostgreSQL version*, which is
ground truth and expensive to re-derive.

Stance: credit it visibly, contact the author **before** launch, and prefer
collaboration over re-derivation. In a community this small, doing it quietly is
a self-inflicted wound. Keep `why-upgrade` as the command name (it is a superb
imperative); let page titles do the search-matching work.
