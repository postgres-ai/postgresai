package collect

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"
)

// StepSeconds is the fixed query_range step (1 minute). It is the bucket size of
// the peak (worst1m = max over 1-minute samples) AND the divisor in
// expectedSlots = window / StepSeconds, so it also fixes the resolution of
// avg / p99 / p999. Widening it would silently relabel the stored peak.
const StepSeconds = 60

// maxPointsPerRange mirrors VictoriaMetrics' DEFAULT
// -search.maxPointsPerTimeseries=30000. The sink's compose command does not set
// that flag, so the default is what is in force; a range asking for more comes
// back 422 -- the failure that lost every monthly row once. See
// https://gitlab.com/postgres-ai/platform-all/-/issues/681
const maxPointsPerRange = 30000

// maxSlicesPerRange bounds the fan-out of one logical query. 8 slices is ~166
// days, far past any period we collect (a month needs 2); beyond that the
// window is a bug in the job args, not work to do.
const maxSlicesPerRange = 8

// maxResponseBytes caps the body we read. Slicing bounds points per series, not
// body size, which scales with series cardinality.
const maxResponseBytes = 8 << 20

// ErrWindowTooLong marks a window needing more than maxSlicesPerRange requests.
// It is a local input error: nothing is queried and the job fails permanently.
var ErrWindowTooLong = errors.New("collection window is too long")

// UpstreamError is a non-2xx or unusable answer from the metric store.
type UpstreamError struct {
	StatusCode int
	Message    string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("metric store error %d: %s", e.StatusCode, e.Message)
}

// Retryable reports whether re-running the same query could plausibly succeed.
// 401 is retryable here on purpose: the store is ours on the same compose
// network, so a rejected basic auth means the credentials and the store's own
// config are momentarily out of step (a restart mid-rotation), not a revoked
// credential. A platform 401 is the opposite and is classified elsewhere.
func (e *UpstreamError) Retryable() bool {
	return e.StatusCode >= 500 || e.StatusCode == http.StatusUnauthorized ||
		e.StatusCode == http.StatusTooManyRequests
}

// Client queries the monitoring instance's own metric store directly. The store
// sits on the compose network, so none of the pull path's hostile-endpoint
// machinery (Grafana datasource proxy, pinned datasource uid, dial-time SSRF
// guard, host allowlist, bearer service-account token) exists here -- see
// instance-jobs/README.md. Redirects ARE refused, for a different reason: the
// request URL carries the job's labels inside the PromQL.
type Client struct {
	httpClient   *http.Client
	baseURL      string
	username     string
	password     string
	maxRespBytes int64

	// OnEnrichmentError, when set, is called with a dropped query-text lookup
	// error. It is the only thing this package reports outward: nothing here
	// logs, because the job args and the result must never reach a log line and
	// a package that cannot log cannot leak them.
	OnEnrichmentError func(error)
}

// NewClient builds a store client. username/password are the VictoriaMetrics
// basic-auth pair; both empty means the request goes out unauthenticated.
func NewClient(baseURL, username, password string, timeout time.Duration) *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: timeout,
			// The request URL carries the PromQL, and the PromQL carries the job's
			// cluster and node labels. Following a redirect would hand that URL to
			// whatever host Location names; the store never redirects.
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		baseURL:      strings.TrimRight(baseURL, "/"),
		username:     username,
		password:     password,
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
// emits the literal string "CPU*", so the regex must match a literal asterisk.
// The PromQL text carries TWO backslashes: PromQL's string-literal parser
// unescapes `\\` -> `\`, leaving the regex `\*`. A single backslash makes the
// store reject the query with `unknown escape sequence U+002A '*'`.
const waitEventTypeRegex = `CPU\\*|IO|IPC|Lock|LWLock`

// escapeLabelValue escapes a value for a PromQL double-quoted label. Backslash
// first, or the quote's own escape would be escaped again.
func escapeLabelValue(v string) string {
	v = strings.ReplaceAll(v, `\`, `\\`)
	v = strings.ReplaceAll(v, `"`, `\"`)
	return v
}

// baseMatcher is the cluster/node selector shared by every query, with no
// wait_event_type filter.
func (f SeriesFilter) baseMatcher() string {
	return fmt.Sprintf(`cluster="%s", node_name="%s"`,
		escapeLabelValue(f.Cluster), escapeLabelValue(f.NodeName))
}

// labelMatcher adds the five-type wait_event_type regex to baseMatcher.
func (f SeriesFilter) labelMatcher() string {
	return fmt.Sprintf(`wait_event_type=~"%s", %s`, waitEventTypeRegex, f.baseMatcher())
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
func (c *Client) QueryPerType(ctx context.Context, f SeriesFilter, start, end time.Time) ([]TypeSeries, error) {
	q := fmt.Sprintf(`sum by (wait_event_type)(pgwatch_wait_events_total{%s})`, f.labelMatcher())
	matrix, err := c.queryRange(ctx, q, start, end)
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
func (c *Client) QueryPerEvent(ctx context.Context, f SeriesFilter, start, end time.Time) ([]EventSeries, error) {
	q := fmt.Sprintf(`sum by (wait_event_type, wait_event)(pgwatch_wait_events_total{%s})`, f.labelMatcher())
	matrix, err := c.queryRange(ctx, q, start, end)
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

// QueryPerQueryID returns the per-class (query_id, datname) series. topk() is
// the cardinality guard: query_id x datname is otherwise unbounded. query_id
// stays a STRING so an id above 2^53 round-trips byte-identical; blank ids are
// dropped.
//
// topk() inside a range query is evaluated PER TIMESTAMP, so this returns the
// union of every id that was top-N in any single slot -- more than TopQueryIDN
// series -- and each is missing the slots where it ranked lower, which the
// aggregator then zero-pads. p99/p999 is therefore a slight underestimate for
// a borderline series. That is a known consequence of the guard, not a padding
// bug, and the pull path does exactly the same: changing it on this side alone
// would make the two paths disagree on the same data.
func (c *Client) QueryPerQueryID(ctx context.Context, f SeriesFilter, start, end time.Time) ([]QueryIDSeries, error) {
	base := f.baseMatcher()
	var out []QueryIDSeries
	for _, cls := range queryIDClasses {
		selector := f.labelMatcher()
		if cls.WaitEventType != "" {
			selector = fmt.Sprintf(`%s, wait_event_type="%s"`, base, escapeLabelValue(cls.WaitEventType))
		}
		q := fmt.Sprintf(`topk(%d, sum by (query_id, datname)(pgwatch_wait_events_total{%s}))`,
			TopQueryIDN, selector)
		matrix, err := c.queryRange(ctx, q, start, end)
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
func (c *Client) QueryTotalSeries(ctx context.Context, f SeriesFilter, start, end time.Time) ([]float64, error) {
	q := fmt.Sprintf(`sum(pgwatch_wait_events_total{%s})`, f.labelMatcher())
	matrix, err := c.queryRange(ctx, q, start, end)
	if err != nil {
		return nil, err
	}
	if len(matrix) == 0 {
		return nil, nil
	}
	return matrix[0].values, nil
}

// QueryTempfileWriteRate returns instance-wide temporary-file write throughput
// in MiB/s. pgwatch_db_stats_temp_bytes is a cumulative counter; rate() handles
// resets and sum() combines the databases on the selected cluster/node.
func (c *Client) QueryTempfileWriteRate(ctx context.Context, f SeriesFilter, start, end time.Time) ([]float64, error) {
	q := fmt.Sprintf(`sum(rate(pgwatch_db_stats_temp_bytes{%s}[5m])) / 1024 / 1024`, f.baseMatcher())
	matrix, err := c.queryRange(ctx, q, start, end)
	if err != nil {
		return nil, err
	}
	if len(matrix) == 0 {
		return nil, nil
	}
	return matrix[0].values, nil
}

// promSeries is one parsed matrix series.
type promSeries struct {
	labels map[string]string
	values []float64
}

// promResponse is the subset of the query_range JSON we parse.
type promResponse struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string `json:"resultType"`
		Result     []struct {
			Metric map[string]string `json:"metric"`
			Values [][]any           `json:"values"`
		} `json:"result"`
	} `json:"data"`
	ErrorType string `json:"errorType"`
	Error     string `json:"error"`
}

// countingReader counts bytes read so doQuery can tell a body truncated at the
// cap from one that is genuinely malformed.
type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// doQuery issues one GET against the store and returns the parsed response.
func (c *Client) doQuery(ctx context.Context, subpath string, params url.Values) (*promResponse, error) {
	endpoint := c.baseURL + "/" + subpath + "?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	// Either half alone still authenticates: a half-configured box sending no
	// header at all would 401 every query with nothing to point at.
	if c.username != "" || c.password != "" {
		req.SetBasicAuth(c.username, c.password)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// One byte past the cap, so a truncated body is distinguishable below.
	limited := &countingReader{r: io.LimitReader(resp.Body, c.maxRespBytes+1)}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &UpstreamError{
			StatusCode: resp.StatusCode,
			Message:    fmt.Sprintf("metric store returned status %d", resp.StatusCode),
		}
	}

	var pr promResponse
	if err := json.NewDecoder(limited).Decode(&pr); err != nil {
		if limited.n > c.maxRespBytes {
			// 413, deliberately: the body scales with series cardinality, so a
			// window that overflows the cap overflows it every time. A 502
			// would be retried three times for the same answer, and would look
			// like a store fault instead of a window we cannot fetch.
			return nil, &UpstreamError{
				StatusCode: http.StatusRequestEntityTooLarge,
				Message:    "response exceeded the size limit",
			}
		}
		return nil, &UpstreamError{StatusCode: http.StatusBadGateway, Message: "failed to parse the response"}
	}
	if pr.Status != "success" {
		return nil, &UpstreamError{StatusCode: http.StatusBadGateway, Message: "query returned a non-success status"}
	}
	return &pr, nil
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

// queryRange fetches the matrix for [start, end] at StepSeconds, slicing a long
// window and stitching the results back together BY LABEL SET, never by
// position: pgwatch sparse-emits, so a series present in one slice can be
// absent from the next.
func (c *Client) queryRange(ctx context.Context, query string, start, end time.Time) ([]promSeries, error) {
	parts, err := splitWindow(start, end)
	if err != nil {
		return nil, err
	}
	if len(parts) == 1 {
		return c.queryRangeOnce(ctx, query, start, end)
	}

	merged := []promSeries{}
	at := make(map[string]int)
	for _, sl := range parts {
		slice, err := c.queryRangeOnce(ctx, query, sl.start, sl.end)
		if err != nil {
			return nil, err
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

// queryRangeOnce performs a single GET /api/v1/query_range and parses the
// matrix. Values that do not parse are skipped; +Inf/NaN DO parse and are kept
// -- only the tempfile path filters them (see AggregateTempfile).
func (c *Client) queryRangeOnce(ctx context.Context, query string, start, end time.Time) ([]promSeries, error) {
	params := url.Values{}
	params.Set("query", query)
	params.Set("start", strconv.FormatInt(start.Unix(), 10))
	params.Set("end", strconv.FormatInt(end.Unix(), 10))
	params.Set("step", strconv.Itoa(StepSeconds))

	pr, err := c.doQuery(ctx, "api/v1/query_range", params)
	if err != nil {
		return nil, err
	}
	// A vector where a matrix was asked for decodes to zero values per series,
	// which the density gate then reads as "no data" -- a silent wrong answer
	// rather than an error.
	if pr.Data.ResultType != "matrix" {
		return nil, &UpstreamError{
			StatusCode: http.StatusBadGateway,
			Message:    "query_range did not return a matrix",
		}
	}

	out := make([]promSeries, 0, len(pr.Data.Result))
	for _, r := range pr.Data.Result {
		vals := make([]float64, 0, len(r.Values))
		for _, pair := range r.Values {
			if len(pair) != 2 {
				continue
			}
			// pair[1] is the value as a string per the Prometheus JSON encoding.
			str, ok := pair[1].(string)
			if !ok {
				continue
			}
			f, err := strconv.ParseFloat(str, 64)
			if err != nil {
				continue
			}
			vals = append(vals, f)
		}
		out = append(out, promSeries{labels: r.Metric, values: vals})
	}
	return out, nil
}

// queryInstant performs a single GET /api/v1/query at `at` and returns the label
// set of every series. It is a label-only lookup: the caller (query-text
// enrichment) reads a constant-1 gauge whose information is all in its labels.
func (c *Client) queryInstant(ctx context.Context, query string, at time.Time) ([]map[string]string, error) {
	params := url.Values{}
	params.Set("query", query)
	params.Set("time", strconv.FormatInt(at.Unix(), 10))

	pr, err := c.doQuery(ctx, "api/v1/query", params)
	if err != nil {
		return nil, err
	}
	if pr.Data.ResultType != "vector" {
		return nil, &UpstreamError{
			StatusCode: http.StatusBadGateway,
			Message:    "query did not return a vector",
		}
	}
	out := make([]map[string]string, 0, len(pr.Data.Result))
	for _, r := range pr.Data.Result {
		out = append(out, r.Metric)
	}
	return out, nil
}
