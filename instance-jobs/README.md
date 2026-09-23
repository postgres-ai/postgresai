# instance-jobs

Collection runs **on the monitoring instance** and the results are posted out.
The platform opens no connection to the instance to collect anything.

Every connection this process makes is outbound: it asks the platform for work
(`v1.instance_job_poll`), runs it against the metric store on its own compose
network, and posts the answer back (`v1.instance_job_submit`).

Issue [#366](https://gitlab.com/postgres-ai/postgresai/-/issues/366), part of
[platform-all#730](https://gitlab.com/postgres-ai/platform-all/-/issues/730).

## What runs

```
poll -> one job -> query the local store -> submit -> sleep
```

The sleep is the platform's `next_poll_ms`, clamped locally to [1s, 1h] -- with
a zero, absent or negative value taking the 10-minute default rather than the
1s floor, since "no interval" is not a request to poll every second -- with
±20% jitter so a fleet provisioned together does not poll in lockstep. A poll
that fails, or a platform that has no channel at this `api_base_url`, backs off
ten minutes instead.

Four job kinds, in two groups.

**Collection** — real work the pull path does today, applied by the platform to
`checkup_reports`:

| kind | payload | how it averages |
|---|---|---|
| `aas_collect` | `AAS.json` (Average Active Sessions) | over the slots the window should have contained |
| `tempfile_collect` | `TEMPFILE.json` (temp-file write rate) | over the samples actually observed |

Those two rows are not a typo. See **The contract** below.

**Query** — a PromQL expression somebody typed, run against the same store and
read back by a human through `pgai promql`. The result is stored on the job row
and is **never applied to anything**: `kind` is not in the platform's
`c_collection_kinds`, so `instance_job_apply_collection` is never reached.

| kind | endpoint | shape |
|---|---|---|
| `promql_instant` | `api/v1/query` | a vector at one instant |
| `promql_range` | `api/v1/query_range` | a matrix over a window at a fixed step |

The box picks the endpoint from the **kind**; the caller supplies only the
expression and the time parameters (`at`, or `start`/`end`/`step_s`), never a
path or a host. A result is trimmed to an
850000-byte budget (not a point cap) and flagged `truncated` rather than
refused — see **Caps on a query result** below.

## Two gates, both closed by default

An existing instance keeps behaving exactly as it does today unless **both** are
open, and neither happens as a side effect of an upgrade:

1. **On the machine:** the `instance-jobs` compose profile is enabled, so the
   container exists at all. `mon update` pulls the definition and starts
   nothing.
2. **On the platform:** `app.settings.instance_jobs_enabled` is on. With it off
   the poll authenticates, is handed no work, and routing stays on the pull
   path — even for a machine whose container is running and polling.

## Configuration

Everything comes from `.pgwatch-config`, the file the reporter container already
mounts. There is no new credential and no new file.

| key | what |
|---|---|
| `api_key` | the org API token, the same one the reporter uploads with |
| `instance_id` | this monitoring instance's id; falls back to `PGAI_INSTANCE_ID`, which `mon local-install` writes into `.env` and compose passes through, then to `PGAI_MONITORING_INSTANCE_ID`, which nothing on the box writes (it is the name the telemetry service uses, accepted here so the two agree) |
| `api_base_url` | the platform that provisioned this instance (optional; else `PGAI_API_BASE_URL`, else production). Must be `https`, or a loopback `http` for a local rig: the token goes on the wire either way |

`mon local-install` records the instance id in `.pgwatch-config` on **both**
registration paths: the console-provisioned one (`--instance-id`, where the
platform adopts the provisioned row) and self-registration (where the platform
mints a row and returns its id). It also writes `PGAI_INSTANCE_ID` into `.env`
when the provisioning flow supplied one, which is why the fallback above exists.

This was not always so, and the failure was silent: self-registration discarded
the id the platform returned -- the call was fire-and-forget -- so every box
that was not console-provisioned came up with no identity and idled forever
while the install printed success. If you are looking at a box in that state,
it was installed before that fix; re-running `mon local-install` records the id,
and the file is re-read on the next tick.

The file is re-read on every tick, so an id written after the container started
is picked up without a restart. If anything needed is missing or unusable the
process **idles** — one log line an hour, no crash loop — and reports
**unhealthy** with the reason, because a container someone enabled on purpose
that is collecting nothing is exactly what must not look green in `mon health`.

The metric store is `PROMETHEUS_URL` with `VM_AUTH_USERNAME` /
`VM_AUTH_PASSWORD`. The compose service pins `PROMETHEUS_URL` to
`http://sink-prometheus:9090`, the sink on its own network; the default in the
code is the same value and applies only to the binary run outside compose.

Two hooks exist for tests and local runs and are not operator knobs:
`INSTANCE_JOBS_CONFIG_PATH` overrides the config file, and
`INSTANCE_JOBS_HEALTH_FILE` the health file.

`.pgwatch-config` is 0600 owned by the install user, so the container has to run
as that owner to read it. `mon local-install` stats the file and writes its
`uid:gid` to `INSTANCE_JOBS_USER` in `.env`; the AWS bootstrap writes the same
thing. With no value the container falls back to the image's own unprivileged
user, cannot read the credential, and says so — a loud failure, rather than the
silently-privileged one a root default would have given every box that has not
been re-provisioned.

`mon update` does **not** backfill this key: it is derived from the file's owner
on the box, not a default that can be generated, so it is written by
`mon local-install` only. A box upgraded with `mon update` alone therefore has
no `INSTANCE_JOBS_USER`, and enabling the profile there fails loudly until
`mon local-install` is re-run. That is the intended order — and it is the same
command that enables the profile (see **Enabling the profile on a machine**
below), so the two cannot come apart.

The token is never passed on argv, never put in a URL, and never logged.

## What this is not

The pull path in credential-service reaches a customer's Grafana across the
public internet and carries the machinery that needs: a Grafana datasource proxy
with a pinned uid, a dial-time SSRF guard on the post-DNS address, a host
allowlist and a decrypted service-account bearer token. None of that exists
here, because the store is a container on this instance's own compose network.
That is ~230 lines this implementation simply does not have, and it is why this
is a second implementation rather than a copy.

Both HTTP clients do refuse to follow redirects, which is not about SSRF: Go
drops only `Authorization` across hosts, so the `access-token` header would be
replayed verbatim to a redirect target, and the store request carries the job's
labels inside the PromQL in its URL. Neither endpoint legitimately redirects.

## What goes on the wire

Every submit **names its outcome**. The RPC used to infer it from whether
`error` was non-null, which left a legitimate skip with no way to say so — a
skip is neither a payload nor a failure, and about one collection in three is
one.

| `outcome` | carries | platform |
|---|---|---|
| `ok` | `result`: the **bare** payload, `{checkId, results}` | applies it |
| `skipped` | `skip_reason`: `retention`, `density` or `no_data` | records `*_last_error = 'skipped:<reason>'`, does not apply, job ends `done` |
| `error` | `error` and `failure_class` | job ends `failed` |

The payload is bare because `public.instance_job_apply_collection` reads
`checkId` and `results` at the top level and takes the pin, granularity and
project from the **job row** — the point of that function being that a box which
could choose the pin could delete hand-ingested history at a colliding terminal
timestamp. The pull path's `{status, payload}` envelope belongs to its HTTP
transport and is unwrapped by its own consumer.

Any contradiction — `ok` with an error, `skipped` without a reason, a reason
without `skipped` — is a `PT400` and **does not consume the job**. It is a bug
on this side, not a lost collection.

The reply is `{job_id, status, outcome, accepted_at, error}`. `status` is the
row's lifecycle; `outcome` is what we told the platform, echoed back, and a
different one means it did not understand us. `error` is always present and
NULL unless the platform **refused** the payload — a rejection is recorded on
the job row rather than raised, so a box that does not read that value reports a
collection as landed when it was not. It counts as a job failure here, and three
in a row flip the container unhealthy.

Four SQLSTATEs from a submit are retried: `55P03` and `40P01` (the platform
hitting its own lock bound behind a pull-path consumer, or deadlocking with
one), plus `40001` and `57014`. In each case it **reverts the claim**, so the
job is queued again and the answer is not lost — they are retried, never
recorded as a failed collection. A `PT404` is the opposite: unknown, foreign,
already-answered or expired, and never worth retrying.

## The contract

Each of these fails into a plausible wrong number rather than an error, which is
why they are pinned by name in `internal/collect/contract_test.go` and by data in
`internal/collect/testdata/fixtures/`.

- pgwatch **sparse-emits**: a sample exists only in a slot that had waiters. The
  gaps are real zeros and are **never materialised**.
- Averages divide by `window / 60s`, **not** by the samples returned. Production
  measured 501 present against 4184 expected — dividing by the wrong one
  overstates AAS by 5-10x.
- Percentiles zero-pad up to that same expected count before interpolating.
- Density gate: `present` is the length of the total series clamped to the
  expected count; `present == 0` skips. Zero-filling the gaps would make
  `present == expected` always, so the gate could never fire and a monitoring
  outage would be stored as a confident AAS of 0.
- Window clamp: start is the later of the period start and the retention floor,
  end the earlier of the period end and now. Expected slots at or below zero is
  a retention skip, not a failure.
- The `total` class of the per-queryid ranking uses the five-type regex, not a
  bare selector — the metric also carries idle wait types.
- Label values escape backslash **before** quote, and the wait-event-type regex
  keeps its two backslashes: PromQL's string parser turns them into one, leaving
  a literal-asterisk regex. One backslash and the store rejects the query.
- Non-finite values reach the AAS samples unfiltered. **Only** the tempfile path
  drops them.
- **TEMPFILE divides by the observed sample count** — deliberately the opposite
  of AAS. Its series is dense, so a gap means "not observed", not "no writes".
- Query-text enrichment is best-effort and must never fail a collection. Its
  lookup is evaluated at the window **end**; anywhere else resolves nothing
  while every other number still looks right.
- Errors discriminate on the response body's `code`, **never** on the HTTP
  status: PostgREST maps our `PTxxx` onto the matching status, so a `PT404` from
  submit (swept, foreign, answered or expired) and a `PGRST202` from an
  un-migrated platform both arrive as 404 and mean opposite things.

## Caps on a query result

**Bytes are the only real bound.** `maxPromQLResultBytes` is **850000**, and it
is not the platform's 1 MiB limit rounded down for comfort: the platform
measures `octet_length(result::text)` on the **jsonb** it stored, and Postgres's
jsonb text output inflates our JSON by up to 19.6% on realistic label sets. 900000
bytes of ours would be refused at 1048576 of theirs. The trim drops whole series
from the end until the encoded result fits, and the answer comes back flagged
`truncated: true` rather than as an error — a partial answer is more use than
none to someone reading a chart.

There is deliberately **no point cap**. One was tried and removed: the platform
validates 11000 points *per series* and does not bound the series count, so a
total cap truncated a wide result at three series where the byte budget keeps
the whole answer. `maxPromQLSeries` (1000) still bounds the series count as
well — what changed is that the budget deciding how much comes back is the byte
one.

## Known ceiling

The store response is capped at 8 MiB per request, and `maxPointsPerRange` bounds
points *per series*, not the body — which grows with series cardinality. Measured
for a 30-day window (two slices, 21600 points per series, ~22 bytes per point):
16 series ≈ 7.25 MiB, 20 ≈ 9.07 MiB, 30 ≈ 13.6 MiB. So roughly 18 distinct
`(wait_event_type, wait_event)` pairs is the ceiling for a month window, and a
busy database has more. Such a window fails with `store_error`, once — the cap
overflow is deliberately not retried, because the same window overflows it every
time.

The pull path has the identical cap and the identical limit
([platform-all#681](https://gitlab.com/postgres-ai/platform-all/-/issues/681)),
so raising it here alone would make the two paths disagree about which windows
they can collect. Slicing by series count belongs in its own issue.

## Timing, and what a restart does

One job is claimed at a time, and the platform fails a job still `running` after
an hour (`public.instance_jobs_expire_sweep`, `stuck_after`). The local ceiling
under that is the collection budget (15 min) plus the worst case for answering
it — every submit attempt timing out, plus every backoff between them, 8m40s.
A refused answer costs that budget TWICE, because the terminal close that
follows is a second submit on the same ladder — so 32m20s against the hour.

Answering retries for 4m40s of backoff — 8m40s including every attempt's own timeout — rather than ~20 seconds on purpose.
`v1.instance_job_submit` waits out its own 5s `lock_timeout` for the
`monitoring_instances` row, and the pull path's consumer can hold that row for a
whole PgQ batch: one transaction, one outbound call per event. A budget shorter
than a batch is a retry loop that loses the only race it was written for. It is
still a local budget, not a guarantee: a pathological batch outlasts it and the
job is then closed by the platform's sweep.

A crash is covered: the container is `restart: unless-stopped`, comes back in
~3s and is healthy in ~7s, and the platform re-queues whatever it had claimed
once the sweep retires it. `docker kill` is **not** a crash for this purpose —
the API kill is recorded as an operator stop, so the restart policy does not
fire and the container stays down until someone starts it. Test recovery by
killing the process inside the container, not the container.

## Enabling the profile on a machine

**Through the ansible playbook, per machine** — set `instance_jobs_enabled: true`
(an SI extraVar). Not by hand: a hand-run `mon local-install` without the
instance id self-registers a *new* monitoring instance and splits the health
matrix across two projects (the
[platform-all#311](https://gitlab.com/postgres-ai/platform-all/-/issues/311)
failure). The role exports `PGAI_INSTANCE_JOBS` to the install and persists
`COMPOSE_PROFILES` in the stack `.env` afterwards, so it also works against a
CLI older than that variable.

Directly, on a machine you are already installing by hand:

```bash
postgresai mon local-install --instance-jobs   # or PGAI_INSTANCE_JOBS=true
postgresai mon local-install --no-instance-jobs   # off, and remove the container
```

These two are **not peers of the ansible variable on a machine ansible manages.**
A deploy that sets `instance_jobs_enabled: false` writes the profile out of
`.env` and removes the container, discarding a hand-set value — and because it
runs before anyone looks, the first sign is collection having stopped. It does
that only on an **explicit** `false`: the role's default is unset, which touches
neither the key nor the container, so a hand-enabled box is left alone by an
ordinary deploy. If the channel is meant to stay on, set
`instance_jobs_enabled: true` for that machine rather than relying on the flag
you typed on the box.

`--no-instance-jobs` says which of the two things it did, because they are not
the same event: `⚠ instance-jobs was RUNNING and has been removed — outbound
collection on this machine is now OFF` versus
`✓ instance-jobs is disabled (no container was present)`. `docker rm --force`
exits 0 either way, so the container's state is read before the removal, not
inferred from it.

**What it writes is `COMPOSE_PROFILES=instance-jobs` in the stack `.env`, not a
`--profile` argument.** Compose reads that key by itself, so a plain `up -d`,
`pull` or `down` covers the service and no `mon` command has to learn about the
profile:

- `mon stop` removes the container and `mon start` brings it back;
- `mon update` pulls its image, and then runs a scoped
  `up -d --no-deps instance-jobs` — `pull` cannot create a service that was not
  there before, and the `mon restart` it suggests is `docker compose restart`,
  which re-runs containers as recorded and would leave the old image running.
  That scoped `up` runs only when the stack is actually up: against a stopped
  project it would create the network and volumes and start this container by
  itself, leaving a box the operator believes is stopped with one process
  polling;
- `mon health` reports the container as a **fault** when the profile is on and
  it is absent, and as `- not enabled` when it is off. `mon stop` deleting the
  container is the most likely way this channel stops, so that line must not be
  green for it.

`mon local-install` preserves every `.env` key it does not own, and merges
rather than replaces this one: a run that passes neither flag nor env var leaves
the profile exactly as it was, in either direction. An upgrade therefore never
turns the channel on, and never turns it off.

Turning it **off** removes the container, not just the key. Taking the profile
out of `.env` does not stop one that is already running — a later
`up -d --force-recreate` leaves it up — so the container is removed by name,
with `COMPOSE_PROFILES` supplied to that one call so it does not depend on what
`.env` says by then.

Do not rely on `down` or `down --remove-orphans` to reach a profile-gated
container either way: two reviewers saw it both reach and not reach one, on the
same engine and compose versions, with the profile absent from `.env` and the
environment alike. The explicit removal is what makes the outcome the same
regardless.

One more thing worth knowing when a box behaves unexpectedly: the CLI spawns
compose with the process environment, so an operator who has `COMPOSE_PROFILES`
exported in their shell changes what **every** `mon` command sees, not just the
install. That is compose's own precedence rule — the environment wins over
`.env` — and the CLI follows it.

Two things do not follow from the profile alone. `INSTANCE_JOBS_USER` is
written by `mon local-install` only (it is derived from the credential file's
owner, not a default), so a box upgraded with `mon update` alone has none and
enabling the profile there fails loudly until the install is re-run. And the
platform gate — `app.settings.instance_jobs_enabled` — is separate: with it off
the poll authenticates and is handed no work.

## Tests

```bash
cd instance-jobs
test -z "$(gofmt -l .)" && go vet ./... && go test ./... -race
```

CI gates on all three.

The fixtures are `(recorded store responses + the request params that produced
them) -> the expected payload`, keyed by the **exact** query text. A query the
fixture does not carry fails the test, and a recorded response nothing asks for
fails it too, so the query set is pinned in both directions.
