package collect

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// StepSeconds is the fixed evaluation step (1 minute). It is the bucket size of
// the peak (worst1m = max over 1-minute samples) AND the divisor in
// expectedSlots = window / StepSeconds, so it also fixes the resolution of
// avg / p99 / p999. Widening it would silently relabel the stored peak.
const StepSeconds = 60

const stepMs = StepSeconds * 1000

// maxPointsPerRange bounds the steps one query evaluates. The store holds the
// reader to a memory cap and each answer to a byte cap, so a long window is
// fetched in slices of at most this many points and stitched back together. It
// was VictoriaMetrics' -search.maxPointsPerTimeseries when the store was
// VictoriaMetrics, and the slices kept the same size so a month is still two
// requests. See https://gitlab.com/postgres-ai/platform-all/-/issues/681
const maxPointsPerRange = 30000

// maxSlicesPerRange bounds the fan-out of one logical query. 8 slices is ~166
// days, far past any period we collect (a month needs 2); beyond that the
// window is a bug in the job args, not work to do.
const maxSlicesPerRange = 8

// maxResponseBytes caps the body we read. Slicing bounds points per series, not
// body size, which scales with series cardinality.
const maxResponseBytes = 8 << 20

// maxErrorBodyBytes bounds the NON-2xx read, which is a different question from
// the success path: no store error message is anywhere near 64 KiB, and
// decoding the full 8 MiB cap on every non-2xx would cost ~40 MiB of garbage --
// on the collection path too, once per slice per attempt, in a 256 MiB
// container.
const maxErrorBodyBytes = 64 << 10

// ErrWindowTooLong marks a window needing more than maxSlicesPerRange requests.
// It is a local input error: nothing is queried and the job fails permanently.
var ErrWindowTooLong = errors.New("collection window is too long")

// UpstreamError is a non-2xx or unusable answer from the metric store.
type UpstreamError struct {
	StatusCode int
	Message    string
	// StoreMessage is the message of the store's own JSON error, when it sent
	// one. Empty for every non-JSON or bodyless failure. On a 4xx it names the
	// syntax error, which is the whole diagnostic to someone who has just typed
	// a query -- so runSQL forwards it rather than replacing it with a constant
	// (#378).
	StoreMessage string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("metric store error %d: %s", e.StatusCode, e.Message)
}

// Retryable reports whether re-running the same query could plausibly succeed.
// 401 is retryable here on purpose: the store is ours on the same compose
// network, so a rejected token means the token and the store's own config are
// momentarily out of step (a restart mid-rotation), not a revoked credential. A
// platform 401 is the opposite and is classified elsewhere. 408 is the store's
// own query timeout, which a less loaded moment can clear.
func (e *UpstreamError) Retryable() bool {
	return e.StatusCode >= 500 || e.StatusCode == http.StatusUnauthorized ||
		e.StatusCode == http.StatusTooManyRequests || e.StatusCode == http.StatusRequestTimeout
}

// Client queries the monitoring instance's own metric store, pgai-mon, directly:
// one read-only SELECT per request to its /sql endpoint. The store sits on the
// compose network, so none of the pull path's hostile-endpoint machinery
// (Grafana datasource proxy, pinned datasource uid, dial-time SSRF guard, host
// allowlist) exists here -- see instance-jobs/README.md. Redirects ARE refused,
// for a different reason: the request carries the store's bearer token and a
// body built from the job's labels.
type Client struct {
	httpClient   *http.Client
	baseURL      string
	token        string
	maxRespBytes int64

	// OnEnrichmentError, when set, is called with a dropped query-text lookup
	// error. It is the only thing this package reports outward: nothing here
	// logs, because the job args and the result must never reach a log line and
	// a package that cannot log cannot leak them.
	OnEnrichmentError func(error)
}

// NewClient builds a store client. token is pgai-mon's bearer token; empty
// means the request goes out unauthenticated.
func NewClient(baseURL, token string, timeout time.Duration) *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: timeout,
			// The request carries the token and the job's cluster and node labels.
			// Following a redirect would hand them to whatever host Location names;
			// the store never redirects.
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		baseURL:      strings.TrimRight(baseURL, "/"),
		token:        token,
		maxRespBytes: maxResponseBytes,
	}
}

// MaxResponseBytes reports the body cap this client was built with, so a caller
// can assert the constant it actually wired.
func (c *Client) MaxResponseBytes() int64 {
	return c.maxRespBytes
}

// RequestTimeout reports the per-request deadline this client was built with.
// It exists so the runner can assert what it actually wired: a client given the
// whole job budget makes the first hung request eat every retry.
func (c *Client) RequestTimeout() time.Duration {
	return c.httpClient.Timeout
}

// SeriesFilter identifies the target series.
type SeriesFilter struct {
	Cluster  string
	NodeName string
}

// waitEventTypeRegex is the literal set of wait_event_type values. pgwatch
// emits the literal string "CPU*", so the regex must match a literal asterisk:
// `\*`. It goes into a SQL string literal, where a backslash is an ordinary
// character, so it is written once (the PromQL form needed two).
const waitEventTypeRegex = `CPU\*|IO|IPC|Lock|LWLock`

// matcher is one [label, op, value] selector term of the store's evaluator.
type matcher [3]string

// baseMatchers is the cluster/node selector shared by every query, with no
// wait_event_type filter.
func (f SeriesFilter) baseMatchers() []matcher {
	return []matcher{{"cluster", "=", f.Cluster}, {"node_name", "=", f.NodeName}}
}

// labelMatchers adds the five-type wait_event_type regex to baseMatchers.
func (f SeriesFilter) labelMatchers() []matcher {
	return append([]matcher{{"wait_event_type", "=~", waitEventTypeRegex}}, f.baseMatchers()...)
}

// queryIDClass pairs an output class key with its wait_event_type selector. An
// empty WaitEventType means the "total" class, which reuses the SAME five-type
// regex as the headline AAS total -- NOT a bare no-filter. The metric also
// carries idle wait types (Activity, Client, Timeout, ...); folding those in
// would diverge from the total and surface idle-wait queries as top consumers.
type queryIDClass struct {
	Key           string
	WaitEventType string
}

var queryIDClasses = []queryIDClass{
	{"total", ""},
	{"io", TypeIO},
	{"ipc", TypeIPC},
	{"lock", TypeLock},
	{"lwlock", TypeLWLock},
	{"cpu", TypeCPU},
}

// QueryPerType returns one TypeSeries per wait_event_type present in the window.
// PromQL: sum by (wait_event_type)(pgwatch_wait_events_total{<five types>, cluster, node_name})
func (c *Client) QueryPerType(ctx context.Context, f SeriesFilter, start, end time.Time) ([]TypeSeries, error) {
	matrix, err := c.queryRange(ctx, start, end, func(s, e int64) string {
		return sumBySQL("aas_per_type", "last", "pgwatch_wait_events_total", f.labelMatchers(), s, e,
			[]string{"wait_event_type"}, 0, "")
	})
	if err != nil {
		return nil, err
	}
	out := make([]TypeSeries, 0, len(matrix))
	for _, s := range matrix {
		out = append(out, TypeSeries{Type: s.labels["wait_event_type"], Samples: s.values})
	}
	return out, nil
}

// QueryPerEvent returns one EventSeries per (wait_event_type, wait_event).
// PromQL: sum by (wait_event_type, wait_event)(pgwatch_wait_events_total{...})
func (c *Client) QueryPerEvent(ctx context.Context, f SeriesFilter, start, end time.Time) ([]EventSeries, error) {
	matrix, err := c.queryRange(ctx, start, end, func(s, e int64) string {
		return sumBySQL("aas_per_event", "last", "pgwatch_wait_events_total", f.labelMatchers(), s, e,
			[]string{"wait_event_type", "wait_event"}, 0, "")
	})
	if err != nil {
		return nil, err
	}
	out := make([]EventSeries, 0, len(matrix))
	for _, s := range matrix {
		out = append(out, EventSeries{
			Type:    s.labels["wait_event_type"],
			Event:   s.labels["wait_event"],
			Samples: s.values,
		})
	}
	return out, nil
}

// QueryPerQueryID returns the per-class (query_id, datname) series. The top-k is
// the cardinality guard: query_id x datname is otherwise unbounded. query_id
// stays a STRING so an id above 2^53 round-trips byte-identical; blank ids are
// dropped -- AFTER the ranking, as PromQL's topk() kept them in it.
//
// The top-k is taken PER TIMESTAMP, as topk() inside a range query is, so this
// returns the union of every id that was top-N in any single slot -- more than
// TopQueryIDN series -- and each is missing the slots where it ranked lower,
// which the aggregator then zero-pads. p99/p999 is therefore a slight
// underestimate for a borderline series. That is a known consequence of the
// guard, not a padding bug, and the pull path does exactly the same: changing
// it on this side alone would make the two paths disagree on the same data.
// PromQL: topk(N, sum by (query_id, datname)(pgwatch_wait_events_total{...}))
func (c *Client) QueryPerQueryID(ctx context.Context, f SeriesFilter, start, end time.Time) ([]QueryIDSeries, error) {
	var out []QueryIDSeries
	for _, cls := range queryIDClasses {
		matchers := f.labelMatchers()
		if cls.WaitEventType != "" {
			matchers = append(f.baseMatchers(), matcher{"wait_event_type", "=", cls.WaitEventType})
		}
		matrix, err := c.queryRange(ctx, start, end, func(s, e int64) string {
			return sumBySQL("aas_per_query_id_"+cls.Key, "last", "pgwatch_wait_events_total", matchers, s, e,
				[]string{"query_id", "datname"}, TopQueryIDN, "")
		})
		if err != nil {
			return nil, err
		}
		for _, s := range matrix {
			qid := s.labels["query_id"]
			if qid == "" {
				continue
			}
			out = append(out, QueryIDSeries{
				Class:   cls.Key,
				QueryID: qid,
				Datname: s.labels["datname"],
				Samples: s.values,
			})
		}
	}
	return out, nil
}

// QueryTotalSeries returns the per-timestamp sum across all five types. Its
// length is also what the density gate counts.
// PromQL: sum(pgwatch_wait_events_total{...})
func (c *Client) QueryTotalSeries(ctx context.Context, f SeriesFilter, start, end time.Time) ([]float64, error) {
	matrix, err := c.queryRange(ctx, start, end, func(s, e int64) string {
		return sumBySQL("aas_total", "last", "pgwatch_wait_events_total", f.labelMatchers(), s, e, nil, 0, "")
	})
	if err != nil {
		return nil, err
	}
	if len(matrix) == 0 {
		return nil, nil
	}
	return matrix[0].values, nil
}

// QueryTempfileWriteRate returns instance-wide temporary-file write throughput
// in MiB/s. pgwatch_db_stats_temp_bytes is a cumulative counter; the rate
// handles resets and the sum combines the databases on the selected
// cluster/node.
// PromQL: sum(rate(pgwatch_db_stats_temp_bytes{cluster, node_name}[5m])) / 1024 / 1024
func (c *Client) QueryTempfileWriteRate(ctx context.Context, f SeriesFilter, start, end time.Time) ([]float64, error) {
	matrix, err := c.queryRange(ctx, start, end, func(s, e int64) string {
		return sumBySQL("tempfile_write_rate", "rate", "pgwatch_db_stats_temp_bytes", f.baseMatchers(), s, e,
			nil, 0, "300000") // [5m]
	})
	if err != nil {
		return nil, err
	}
	if len(matrix) == 0 {
		return nil, nil
	}
	values := matrix[0].values
	for i := range values {
		values[i] = values[i] / 1024 / 1024
	}
	return values, nil
}

// alignRange is what VictoriaMetrics' /api/v1/query_range does to the window
// before evaluating it (AdjustStartEnd, without nocache=1): a range of 50 points
// or more is snapped to the step grid, start down and end up, keeping its point
// count. The store's evaluator takes the window as given, so the snap is made
// here and the steps stay where the PromQL path put them.
func alignRange(startMs, endMs int64) (int64, int64) {
	if (endMs-startMs)/stepMs+1 < 50 {
		return startMs, endMs
	}
	down := startMs - startMs%stepMs
	up := endMs
	if r := endMs % stepMs; r > 0 {
		up = endMs + stepMs - r
	}
	extra := (up-down)/stepMs - (endMs-startMs)/stepMs
	if extra < 0 {
		extra = 0
	}
	return down, up - stepMs*extra
}

// sumBySQL renders `sum by (by...)(<fn>(metric{matchers}[window]))` evaluated at
// every step of [startMs, endMs] -- or topk(k, ...) of it per step when k > 0 --
// as one SELECT returning a row per output series: its `by` labels as columns
// and its values, in time order, as the list `vals`. A series has values only
// at the steps where it exists, as in a Prometheus matrix.
//
// fn is the store's evaluator: "last" is a bare selector (VictoriaMetrics'
// default_rollup), "rate" takes window (milliseconds, as SQL text), "timestamp"
// is timestamp(). The leading comment names the query, so a test double can
// tell the queries apart; the store ignores it.
func sumBySQL(name, fn, metric string, matchers []matcher, startMs, endMs int64, by []string, k int, window string) string {
	s, e := alignRange(startMs, endMs)
	args := fmt.Sprintf("%s, %s, %d, %d, %d", sqlString(metric), matchersSQL(matchers), s, e, stepMs)
	if fn == "rate" {
		args += ", " + window
	}
	var cols, keys []string
	for _, l := range by {
		cols = append(cols, fmt.Sprintf("coalesce(labels[%s], '') AS %s", sqlString(l), l))
		keys = append(keys, l)
	}
	sel := strings.Join(append(cols, "t", "sum(value) AS v"), ", ")
	var b strings.Builder
	fmt.Fprintf(&b, "-- instance-jobs: %s\n", name)
	fmt.Fprintf(&b, "WITH a AS (\n  SELECT %s\n  FROM prom_%s(%s)\n  WHERE NOT isnan(value)\n  GROUP BY ALL\n)", sel, fn, args)
	from := "a"
	if k > 0 {
		order := strings.Join(append([]string{"v DESC"}, keys...), ", ")
		fmt.Fprintf(&b, ", k AS (\n  SELECT * FROM a\n  QUALIFY row_number() OVER (PARTITION BY t ORDER BY %s) <= %d\n)", order, k)
		from = "k"
	}
	out := strings.Join(append(append([]string{}, keys...), "list(v ORDER BY t) AS vals"), ", ")
	if len(keys) > 0 {
		fmt.Fprintf(&b, "\nSELECT %s\nFROM %s\nGROUP BY ALL\nORDER BY ALL", out, from)
	} else {
		// one series, or none at all: no row rather than a row with a NULL list
		fmt.Fprintf(&b, "\nSELECT %s\nFROM %s\nHAVING count(*) > 0", out, from)
	}
	return b.String()
}

// sqlString quotes s as a SQL string literal. The store's dialect (DuckDB) takes
// a backslash literally, so doubling the quote is the whole escape.
func sqlString(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

// matchersSQL renders a selector as the evaluator's [[label, op, value], ...].
func matchersSQL(ms []matcher) string {
	parts := make([]string, 0, len(ms))
	for _, m := range ms {
		parts = append(parts, fmt.Sprintf("[%s, %s, %s]", sqlString(m[0]), sqlString(m[1]), sqlString(m[2])))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// promSeries is one parsed series.
type promSeries struct {
	labels map[string]string
	values []float64
}

// timeRange is one closed [start, end] slice of a query window.
type timeRange struct {
	start time.Time
	end   time.Time
}

// splitWindow divides [start, end] into the fewest equal slices whose point
// counts (window/step + 1) each stay at or below maxPointsPerRange. Slice N+1
// starts exactly one step after slice N ends, so every step timestamp --
// boundaries included -- belongs to exactly one slice and merging never
// double-counts.
func splitWindow(start, end time.Time) ([]timeRange, error) {
	whole := []timeRange{{start: start, end: end}}
	if !end.After(start) {
		return whole, nil
	}
	// Unix seconds, not end.Sub(start): a time.Duration saturates at ~292 years,
	// which would under-count the slices an absurd window really needs.
	points := (end.Unix()-start.Unix())/StepSeconds + 1
	if points <= maxPointsPerRange {
		return whole, nil
	}
	slices := (points + maxPointsPerRange - 1) / maxPointsPerRange
	if slices > maxSlicesPerRange {
		return nil, ErrWindowTooLong
	}
	perSlice := (points + slices - 1) / slices

	step := time.Duration(StepSeconds) * time.Second
	out := make([]timeRange, 0, slices)
	for s := start; !s.After(end); s = s.Add(time.Duration(perSlice) * step) {
		e := s.Add(time.Duration(perSlice-1) * step)
		if e.After(end) {
			e = end
		}
		out = append(out, timeRange{start: s, end: e})
	}
	return out, nil
}

// labelFingerprint canonicalizes a label set into a merge key. Lengths are
// encoded alongside names and values, so no two distinct label sets collide
// through a separator character appearing inside a value.
func labelFingerprint(labels map[string]string) string {
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "%d:%s=%d:%s,", len(k), k, len(labels[k]), labels[k])
	}
	return b.String()
}

// queryRange fetches the series for [start, end] at StepSeconds, slicing a long
// window and stitching the results back together BY LABEL SET, never by
// position: pgwatch sparse-emits, so a series present in one slice can be
// absent from the next. build renders the SELECT for one slice, in epoch ms.
func (c *Client) queryRange(ctx context.Context, start, end time.Time, build func(startMs, endMs int64) string) ([]promSeries, error) {
	parts, err := splitWindow(start, end)
	if err != nil {
		return nil, err
	}
	merged := []promSeries{}
	at := make(map[string]int)
	for _, sl := range parts {
		slice, err := c.querySeries(ctx, build(sl.start.UnixMilli(), sl.end.UnixMilli()))
		if err != nil {
			return nil, err
		}
		if len(parts) == 1 {
			return slice, nil
		}
		for _, s := range slice {
			key := labelFingerprint(s.labels)
			if i, ok := at[key]; ok {
				merged[i].values = append(merged[i].values, s.values...)
				continue
			}
			at[key] = len(merged)
			merged = append(merged, s)
		}
	}
	return merged, nil
}
