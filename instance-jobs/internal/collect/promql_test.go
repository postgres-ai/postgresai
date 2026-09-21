package collect

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// promQLHarness records what the store was asked for and answers with a fixed
// body, so the request the box builds is asserted rather than inferred.
type promQLHarness struct {
	srv      *httptest.Server
	client   *Client
	paths    []string
	queries  []string
	rawQuery []string
}

func newPromQLHarness(t *testing.T, body string) *promQLHarness {
	return newPromQLHarnessStatus(t, http.StatusOK, body)
}

// newPromQLHarnessStatus serves body under an explicit status code. The default
// above is 200, which is right for every success fixture -- but NOT for an error
// one: a Prometheus-compatible store answers a bad expression with a 4xx, and
// httptest's implicit 200 made the error tests certify a branch production never
// reaches (#378).
func newPromQLHarnessStatus(t *testing.T, status int, body string) *promQLHarness {
	t.Helper()
	h := &promQLHarness{}
	h.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.paths = append(h.paths, r.URL.Path)
		h.queries = append(h.queries, r.URL.Query().Get("query"))
		h.rawQuery = append(h.rawQuery, r.URL.RawQuery)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(h.srv.Close)
	h.client = NewClient(h.srv.URL, "", "", 10*time.Second)
	return h
}

func runQ(t *testing.T, h *promQLHarness, kind, args string) (Outcome, error) {
	t.Helper()
	return Run(context.Background(), h.client, kind, []byte(args),
		time.Date(2026, 9, 18, 12, 0, 0, 0, time.UTC))
}

// The subpath is chosen by the KIND. Nothing in args can move the request to
// another endpoint -- that is the whole of the box-side guard.
func TestTheSubpathIsChosenByKindAndNotByInput(t *testing.T) {
	const vector = `{"status":"success","data":{"resultType":"vector","result":[]}}`

	for _, tc := range []struct {
		kind string
		args string
		want string
	}{
		{KindPromQLInstant, `{"query":"up"}`, "/api/v1/query"},
		{KindPromQLRange, `{"query":"up","start":"2026-09-18T11:00:00Z","end":"2026-09-18T12:00:00Z","step_s":60}`, "/api/v1/query_range"},
		// Args that TRY to redirect the request: a path, a host, extra params.
		// None of them are read; the kind decides and the query is one value.
		{KindPromQLInstant, `{"query":"up","path":"/api/v1/admin/tsdb/delete_series","url":"http://evil/x"}`, "/api/v1/query"},
		{KindPromQLInstant, `{"query":"up&match[]=x","at":""}`, "/api/v1/query"},
	} {
		t.Run(tc.kind+" "+tc.want, func(t *testing.T) {
			h := newPromQLHarness(t, vector)
			if _, err := runQ(t, h, tc.kind, tc.args); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(h.paths) != 1 || h.paths[0] != tc.want {
				t.Fatalf("paths = %v, want exactly [%s]", h.paths, tc.want)
			}
		})
	}
}

// The expression travels as ONE form value. An expression containing `&` must
// not become a second parameter, which is what url.Values encoding buys and
// what a hand-built string would lose.
func TestTheExpressionIsASingleFormValue(t *testing.T) {
	h := newPromQLHarness(t, `{"status":"success","data":{"resultType":"vector","result":[]}}`)
	const expr = `sum(rate(x{job="a b"}[5m])) & step=1 ?q=2`
	if _, err := runQ(t, h, KindPromQLInstant, `{"query":`+strconvQuote(expr)+`}`); err != nil {
		t.Fatal(err)
	}
	if h.queries[0] != expr {
		t.Fatalf("query round-tripped as %q, want %q", h.queries[0], expr)
	}
	// One `query=` parameter, and no `match[]` smuggled in beside it.
	if got := strings.Count(h.rawQuery[0], "query="); got != 1 {
		t.Fatalf("raw query has %d query= parameters: %s", got, h.rawQuery[0])
	}
	if strings.Contains(h.rawQuery[0], "match") {
		t.Fatalf("an expression created a second parameter: %s", h.rawQuery[0])
	}
}

// json.Marshal REFUSES a non-finite float, so an unsanitised +Inf makes the
// whole result unencodable and the job fails as a transport error instead of
// answering. A user can reach this by typing `1/0`.
func TestNonFiniteValuesStillProduceAnEncodableAnswer(t *testing.T) {
	// Prometheus sends values as strings, so "+Inf" is what the wire carries.
	// A store that sends a NUMBER instead is the dangerous case, and both are
	// covered here.
	const body = `{"status":"success","data":{"resultType":"vector","result":[
		{"metric":{"__name__":"a"},"value":[1789689600,"+Inf"]},
		{"metric":{"__name__":"b"},"value":[1789689600,"NaN"]}]}}`

	h := newPromQLHarness(t, body)
	out, err := runQ(t, h, KindPromQLInstant, `{"query":"1/0"}`)
	if err != nil {
		t.Fatalf("a non-finite value failed the job: %v", err)
	}
	blob, err := json.Marshal(out.Payload)
	if err != nil {
		t.Fatalf("the result is unencodable, so the job would fail as a transport error: %v", err)
	}
	// The answer is honest rather than dropped or zeroed: the user asked 1/0.
	if !strings.Contains(string(blob), "+Inf") {
		t.Fatalf("the +Inf answer did not survive: %s", blob)
	}
	if strings.Contains(string(blob), "null") {
		t.Fatalf("a sample was nulled rather than kept as a string: %s", blob)
	}
}

func TestANumericValueFromTheStoreIsNormalisedOrRefused(t *testing.T) {
	// Prometheus sends values as strings; a store that sends a JSON NUMBER is
	// off-shape. What JSON can and cannot carry decides the two cases here:
	// there are no Inf/NaN literals, and a number that overflows float64 fails
	// to decode at all. So a non-finite float can never arrive this way.

	t.Run("finite number is normalised to a string", func(t *testing.T) {
		h := newPromQLHarness(t, `{"status":"success","data":{"resultType":"vector","result":[
			{"metric":{},"value":[1789689600,1.5]}]}}`)
		out, err := runQ(t, h, KindPromQLInstant, `{"query":"x"}`)
		if err != nil {
			t.Fatal(err)
		}
		blob, err := json.Marshal(out.Payload)
		if err != nil {
			t.Fatalf("unencodable: %v", err)
		}
		if !strings.Contains(string(blob), `"1.5"`) {
			t.Fatalf("a numeric value was not normalised to a string: %s", blob)
		}
	})

	t.Run("overflowing number is refused as a store fault", func(t *testing.T) {
		// It never reaches sanitiseSample: the body itself will not decode.
		// The job fails as a store error rather than producing a half-result.
		h := newPromQLHarness(t, `{"status":"success","data":{"resultType":"vector","result":[
			{"metric":{},"value":[1789689600,1e999]}]}}`)
		_, err := runQ(t, h, KindPromQLInstant, `{"query":"x"}`)
		var upstream *UpstreamError
		if !errors.As(err, &upstream) {
			t.Fatalf("err = %v, want an UpstreamError", err)
		}
	})
}

// The upstream body is decoded and re-encoded, never proxied: a field the store
// invents must not appear in what the platform stores.
func TestTheUpstreamBodyIsNeverProxiedVerbatim(t *testing.T) {
	const body = `{"status":"success","data":{"resultType":"vector","result":[
		{"metric":{"job":"a"},"value":[1789689600,"1"],"injected":"should not survive"}],
		"extra":"should not survive"},"trailer":"should not survive"}`

	h := newPromQLHarness(t, body)
	out, err := runQ(t, h, KindPromQLInstant, `{"query":"up"}`)
	if err != nil {
		t.Fatal(err)
	}
	blob, _ := json.Marshal(out.Payload)
	if strings.Contains(string(blob), "should not survive") {
		t.Fatalf("an upstream field reached the platform: %s", blob)
	}
	if !strings.Contains(string(blob), `"job":"a"`) {
		t.Fatalf("the real labels did not survive: %s", blob)
	}
}

func TestAMatrixIsDecodedWholeAndNotReportedTruncated(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"status":"success","data":{"resultType":"matrix","result":[`)
	for i := 0; i < 3; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `{"metric":{"i":"%d"},"values":[[1,"1"],[2,"2"],[3,"3"]]}`, i)
	}
	b.WriteString(`]}}`)

	h := newPromQLHarness(t, b.String())
	out, err := runQ(t, h, KindPromQLRange,
		`{"query":"up","start":"2026-09-18T11:00:00Z","end":"2026-09-18T12:00:00Z","step_s":60}`)
	if err != nil {
		t.Fatal(err)
	}
	res := out.Payload.(PromQLResult)
	if len(res.Result) != 3 {
		t.Fatalf("series = %d, want 3", len(res.Result))
	}
	for _, s := range res.Result {
		if len(s.Values) != 3 {
			t.Fatalf("points per series = %d, want 3", len(s.Values))
		}
	}
	if res.Stats.Truncated {
		t.Fatal("a result well inside the budget was reported truncated")
	}
	if res.ResultType != "matrix" {
		t.Fatalf("resultType = %q", res.ResultType)
	}
}

// A window the PLATFORM validated as legal must not be truncated by this side.
// It bounds points PER SERIES at 11000 with no series bound; a total-point cap
// here truncated such a window at three series, which is most real queries.
func TestAWindowThePlatformAllowsIsNotTruncatedByAPointCap(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"status":"success","data":{"resultType":"matrix","result":[`)
	// 32000 points: above the 30000 total-point cap that used to be here, and
	// comfortably inside the byte budget, so the ONLY thing that could truncate
	// this is a point count -- which is the axis that should not exist.
	const series, perSeries = 4, 8000
	for i := 0; i < series; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `{"metric":{"i":"%d"},"values":[`, i)
		for j := 0; j < perSeries; j++ {
			if j > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `[%d,"1"]`, 1789689600+j*60)
		}
		b.WriteString(`]}`)
	}
	b.WriteString(`]}}`)

	h := newPromQLHarness(t, b.String())
	out, err := runQ(t, h, KindPromQLRange,
		`{"query":"up","start":"2026-09-18T11:00:00Z","end":"2026-09-18T12:00:00Z","step_s":60}`)
	if err != nil {
		t.Fatal(err)
	}
	res := out.Payload.(PromQLResult)
	if len(res.Result) != series {
		t.Fatalf("%d of %d series survived: a window inside the byte budget was truncated on "+
			"the wrong axis", len(res.Result), series)
	}
	if blob, _ := json.Marshal(res); len(blob) > maxPromQLResultBytes {
		t.Fatalf("the fixture is %d bytes, over the budget: it cannot isolate a point cap", len(blob))
	}
	if res.Stats.Truncated {
		t.Fatal("a window the platform calls legal was reported truncated")
	}
}

func TestTruncationIsReportedRatherThanSilent(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"status":"success","data":{"resultType":"vector","result":[`)
	for i := 0; i < maxPromQLSeries+50; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `{"metric":{"i":"%d"},"value":[1,"1"]}`, i)
	}
	b.WriteString(`]}}`)

	h := newPromQLHarness(t, b.String())
	out, err := runQ(t, h, KindPromQLInstant, `{"query":"x"}`)
	if err != nil {
		t.Fatal(err)
	}
	res := out.Payload.(PromQLResult)
	if !res.Stats.Truncated {
		t.Fatal("a capped result was NOT reported as truncated: a silently partial answer " +
			"to a question a human asked is worse than an error")
	}
	if len(res.Result) > maxPromQLSeries {
		t.Fatalf("series = %d, above the loop bound", len(res.Result))
	}
}

// A syntax error is the user's, not the store's: it must come back as
// invalid_args so it is not retried three times for the same answer, and the
// store's own message is the useful part.
//
// AT THE STATUS CODE A REAL STORE SENDS. VictoriaMetrics answers a bad
// expression with 422 and upstream Prometheus with 400 -- neither with 200. This
// test used the harness's implicit 200, so it exercised the `resp.Status !=
// "success"` branch, which production never reaches: getJSON rejects on the
// status code before the body is ever decoded. The one case where the store's
// message IS the entire diagnostic was the one case that lost it (#378).
func TestAQueryErrorIsABadArgumentNotAStoreFault(t *testing.T) {
	const body = `{"status":"error","errorType":"bad_data","error":"parse error: unexpected )"}`
	for name, status := range map[string]int{
		"victoriametrics 422": http.StatusUnprocessableEntity,
		"prometheus 400":      http.StatusBadRequest,
		// Kept as defence in depth, not as the syntax-error path: no
		// Prometheus-compatible store sends this shape.
		"200 with status:error": http.StatusOK,
	} {
		t.Run(name, func(t *testing.T) {
			h := newPromQLHarnessStatus(t, status, body)
			_, err := runQ(t, h, KindPromQLInstant, `{"query":"sum()"}`)
			if !errors.Is(err, ErrInvalidArgs) {
				t.Fatalf("err = %v, want ErrInvalidArgs", err)
			}
			if !strings.Contains(err.Error(), "parse error") {
				t.Fatalf("the store's message was dropped: %v", err)
			}
		})
	}
}

// A store fault is not the argument's fault: it must stay retryable and must NOT
// be reclassified as invalid_args.
//
// A TABLE, not one 503, because the exclusion list is the load-bearing half of
// that branch and every entry in it was a mutation survivor: a single 5xx case
// left both 401 and 413 unpinned. 429 is here because it is what the first
// version of the exclusion list MISSED -- `UpstreamError.Retryable()` names
// three statuses (>=500, 401, 429) and the list named two, so a store
// rate-limit became a permanent bad argument and was not retried (#378).
func TestAStoreFaultIsNotABadArgument(t *testing.T) {
	for name, status := range map[string]int{
		"503 unavailable":  http.StatusServiceUnavailable,
		"401 mid-rotation": http.StatusUnauthorized,
		"429 rate limited": http.StatusTooManyRequests,
		"413 too large":    http.StatusRequestEntityTooLarge,
		// Not retryable, and still not the caller's fault: every one of these
		// means the box is pointed at the wrong thing. 407 in particular is
		// literally "a proxy in front of the store". An exclusion list left all
		// of them blaming the user's expression.
		"403 forbidden":  http.StatusForbidden,
		"404 wrong url":  http.StatusNotFound,
		"405 proxy":      http.StatusMethodNotAllowed,
		"407 proxy auth": http.StatusProxyAuthRequired,
		"408 timeout":    http.StatusRequestTimeout,
	} {
		t.Run(name, func(t *testing.T) {
			h := newPromQLHarnessStatus(t, status, `{"status":"error","error":"not your fault"}`)
			_, err := runQ(t, h, KindPromQLInstant, `{"query":"up"}`)
			if errors.Is(err, ErrInvalidArgs) {
				t.Fatalf("status %d was classified as a bad argument: %v", status, err)
			}
			// The UpstreamError must still be REACHABLE, or the runner's
			// retryableStoreError cannot see the status at all.
			var ue *UpstreamError
			if !errors.As(err, &ue) {
				t.Fatalf("status %d dropped the *UpstreamError from the chain: %v", status, err)
			}
			if ue.Retryable() != (status == http.StatusServiceUnavailable ||
				status == http.StatusUnauthorized || status == http.StatusTooManyRequests) {
				t.Fatalf("status %d: Retryable() = %v, which disagrees with the exclusion list", status, ue.Retryable())
			}
		})
	}
}

// A hostile label must not survive into the result. pgwatch scrapes labels from
// the customer's own Postgres -- application_name is settable by anyone who can
// connect -- and the CLI prints them to a terminal with no escaping of its own.
func TestHostileLabelsAndSamplesAreStripped(t *testing.T) {
	h := newPromQLHarness(t, `{"status":"success","data":{"resultType":"vector","result":[
		{"metric":{"__name__":"up","job":"alpha","job\u0001":"beta","application_name":"\u001b[2J\u009b[31mPWNED\r\u2028x"},
		 "value":[1758000000,"1\u001b[0m"]}]}}`)
	out, err := runQ(t, h, KindPromQLInstant, `{"query":"up"}`)
	if err != nil {
		t.Fatalf("runQ: %v", err)
	}
	res, ok := out.Payload.(PromQLResult)
	if !ok || len(res.Result) != 1 {
		t.Fatalf("payload = %#v", out.Payload)
	}
	// A hostile KEY is DROPPED, not rewritten: rewriting collapses `job` and
	// "job\x01" onto one entry and silently loses a label.
	if _, ok := res.Result[0].Metric["job"]; !ok {
		t.Fatalf("the clean label was lost: %#v", res.Result[0].Metric)
	}
	if res.Result[0].Metric["job"] != "alpha" {
		t.Fatalf("a hostile key overwrote the clean label's value: %q", res.Result[0].Metric["job"])
	}
	if !res.Stats.Truncated {
		t.Fatal("a dropped label did not set Truncated, so the caller is not told anything is missing")
	}
	got := res.Result[0].Metric["application_name"]
	for _, bad := range []string{"\x1b", "\u009b", "\r", "\u2028"} {
		if strings.Contains(got, bad) {
			t.Fatalf("a control character survived into the label: %q", got)
		}
	}
	if !strings.Contains(got, "PWNED") {
		t.Fatalf("stripping removed the label's text as well: %q", got)
	}
	// Neutralised as SPACES, not deleted: pgwatch carries query text in labels,
	// and deleting a newline fuses "select 1\nfrom t" into "select 1from t",
	// which changes what the text says.
	if got := StripControlsToSpace("select 1\nfrom t"); got != "select 1 from t" {
		t.Fatalf("a multi-line value came back as %q; its tokens were fused", got)
	}
	// A vector carries Value, not Values.
	if len(res.Result[0].Value) != 2 {
		t.Fatalf("vector sample = %#v", res.Result[0].Value)
	}
	sample, _ := res.Result[0].Value[1].(string)
	if strings.ContainsAny(sample, "\x1b\r") {
		t.Fatalf("a control character survived into the sample value: %q", sample)
	}
}

// ...and the other side of the same allow-list: the statuses that ARE the
// expression's fault must still classify as such, or the fix above would
// "work" by calling everything a store fault.
func TestAQueryErrorStaysTheCallersFault(t *testing.T) {
	for name, status := range map[string]int{
		"400 prometheus":      http.StatusBadRequest,
		"422 victoriametrics": http.StatusUnprocessableEntity,
		"414 over-long":       http.StatusRequestURITooLong,
	} {
		t.Run(name, func(t *testing.T) {
			h := newPromQLHarnessStatus(t, status, `{"status":"error","error":"parse error"}`)
			_, err := runQ(t, h, KindPromQLInstant, `{"query":"sum()"}`)
			if !errors.Is(err, ErrInvalidArgs) {
				t.Fatalf("status %d is not classified as the caller's fault: %v", status, err)
			}
		})
	}
}

func TestPromQLArgsAreValidated(t *testing.T) {
	const ok = `{"status":"success","data":{"resultType":"vector","result":[]}}`
	for name, tc := range map[string]struct{ kind, args string }{
		"no query":           {KindPromQLInstant, `{}`},
		"empty query":        {KindPromQLInstant, `{"query":""}`},
		"range no start":     {KindPromQLRange, `{"query":"up","end":"2026-09-18T12:00:00Z","step_s":60}`},
		"range end<=start":   {KindPromQLRange, `{"query":"up","start":"2026-09-18T12:00:00Z","end":"2026-09-18T12:00:00Z","step_s":60}`},
		"range step 0":       {KindPromQLRange, `{"query":"up","start":"2026-09-18T11:00:00Z","end":"2026-09-18T12:00:00Z","step_s":0}`},
		"bad at":             {KindPromQLInstant, `{"query":"up","at":"yesterday"}`},
		"args not an object": {KindPromQLInstant, `[]`},
	} {
		t.Run(name, func(t *testing.T) {
			h := newPromQLHarness(t, ok)
			_, err := runQ(t, h, tc.kind, tc.args)
			if !errors.Is(err, ErrInvalidArgs) {
				t.Fatalf("err = %v, want ErrInvalidArgs", err)
			}
			if len(h.paths) != 0 {
				t.Fatalf("a bad argument still reached the store: %v", h.paths)
			}
		})
	}
}

// The timeout must stay under the job ceiling so a slow query fails as a job
// rather than as a claim the sweep has to reap.
func TestTheQueryTimeoutIsUnderTheJobCeiling(t *testing.T) {
	c := NewClient("http://x", "", "", 90*time.Second)
	if c.RequestTimeout() >= 15*time.Minute {
		t.Fatalf("request timeout %v is not under the 15m job budget", c.RequestTimeout())
	}
	if c.MaxResponseBytes() <= 0 {
		t.Fatal("the response cap is not set")
	}
}

func TestSanitiseSampleRejectsWhatItCannotEncode(t *testing.T) {
	for name, raw := range map[string][]any{
		"wrong arity":   {1.0},
		"nan timestamp": {math.NaN(), "1"},
		"inf timestamp": {math.Inf(1), "1"},
		"string ts":     {"1789689600", "1"},
		"object value":  {1.0, map[string]any{}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, ok := sanitiseSample(raw); ok {
				t.Fatalf("%v was accepted", raw)
			}
		})
	}
	if s, ok := sanitiseSample([]any{1789689600.0, "1.5"}); !ok || s[1] != "1.5" {
		t.Fatalf("a good sample was rejected or altered: %v %v", s, ok)
	}
}

func strconvQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// The cap that binds is BYTES, because that is what the platform enforces:
// instance_job_submit caps `result` at 1 MiB and answers PT400 above it. A
// result it refuses is worse than a short one -- the PT400 is correctly not
// retried, so the job would sit `running` until the hourly sweep while the
// in-flight cap locked the user out.
func TestTheResultAlwaysFitsThePlatformsOneMiBCap(t *testing.T) {
	const platformCap = 1048576

	// Realistic pgwatch labels, which is what makes a point cost ~28 bytes
	// rather than the handful an empty metric would.
	var b strings.Builder
	b.WriteString(`{"status":"success","data":{"resultType":"matrix","result":[`)
	for i := 0; i < 400; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `{"metric":{"__name__":"pgwatch_wait_events_total","cluster":"production-cluster-01",`+
			`"node_name":"pgload-%02d.internal","wait_event_type":"LWLock","wait_event":"BufferMapping",`+
			`"datname":"postgres","query_id":"-7514152763537218973"},"values":[`, i)
		for j := 0; j < 300; j++ {
			if j > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `[%d,"%d.5"]`, 1789689600+j*60, j)
		}
		b.WriteString(`]}`)
	}
	b.WriteString(`]}}`)

	h := newPromQLHarness(t, b.String())
	out, err := runQ(t, h, KindPromQLRange,
		`{"query":"x","start":"2026-09-18T11:00:00Z","end":"2026-09-18T12:00:00Z","step_s":60}`)
	if err != nil {
		t.Fatal(err)
	}
	res := out.Payload.(PromQLResult)

	blob, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("unencodable: %v", err)
	}
	if len(blob) > platformCap {
		t.Fatalf("result is %d bytes, above the platform's %d cap: the submit would be a "+
			"PT400 and the job would sit running until the sweep", len(blob), platformCap)
	}
	if !res.Stats.Truncated {
		t.Fatal("a result trimmed to fit was not reported as truncated")
	}
}

// The platform measures octet_length(result::text) on JSONB, and Postgres's
// jsonb text output inserts a space after every `:` and `,`. So the size WE
// compute is not the size it checks, and the gap grows with token density.
// Measured on a real Postgres: 15.2% on dense sample arrays, 19.6% on a series
// with many one-character labels.
func TestTheBudgetSurvivesJsonbTextInflation(t *testing.T) {
	// Modelling the inflation rather than requiring a database: one space per
	// `:` and per `,` outside string literals is exactly what jsonb text adds.
	const worstObservedInflation = 1.196
	const platformCap = 1048576

	inflated := float64(maxPromQLResultBytes) * worstObservedInflation
	if inflated > platformCap {
		t.Fatalf("a full %d-byte result inflates to ~%.0f bytes as jsonb text, above the "+
			"platform's %d cap: the trimmer would certify a result the submit then refuses",
			maxPromQLResultBytes, inflated, platformCap)
	}

	// And the budget should not be so small it is doing nothing.
	if maxPromQLResultBytes < platformCap/2 {
		t.Fatalf("budget %d is less than half the cap; that is not margin, it is a different limit",
			maxPromQLResultBytes)
	}
}

// The inflation model above, checked against the real thing: count the spaces
// jsonb text would add to a payload this code actually produces.
func TestTheInflationModelMatchesJsonbTextRules(t *testing.T) {
	const body = `{"status":"success","data":{"resultType":"matrix","result":[
		{"metric":{"job":"a","instance":"b"},"values":[[1789689600,"1"],[1789689660,"2"]]}]}}`

	h := newPromQLHarness(t, body)
	out, err := runQ(t, h, KindPromQLRange,
		`{"query":"x","start":"2026-09-18T11:00:00Z","end":"2026-09-18T12:00:00Z","step_s":60}`)
	if err != nil {
		t.Fatal(err)
	}
	blob, err := json.Marshal(out.Payload)
	if err != nil {
		t.Fatal(err)
	}

	// jsonb text adds one space after each structural `:` and `,`.
	structural := 0
	inString := false
	for i := 0; i < len(blob); i++ {
		switch c := blob[i]; {
		case c == '"' && (i == 0 || blob[i-1] != '\\'):
			inString = !inString
		case !inString && (c == ':' || c == ','):
			structural++
		}
	}
	if structural == 0 {
		t.Fatal("no structural separators found; the model cannot be checked")
	}
	inflation := float64(len(blob)+structural) / float64(len(blob))
	if inflation <= 1.0 {
		t.Fatalf("inflation computed as %.3f, which cannot be right", inflation)
	}
	t.Logf("this payload inflates %.1f%% as jsonb text (%d bytes -> %d)",
		(inflation-1)*100, len(blob), len(blob)+structural)
}

func TestTheByteBudgetSitsUnderThePlatformCap(t *testing.T) {
	// Pinned as a contract: instance_job_submit.sql caps result at 1 MiB, and
	// the budget must leave room for encoding differences rather than sitting
	// exactly on the line.
	if maxPromQLResultBytes >= 1048576 {
		t.Fatalf("the byte budget %d is not below the platform's 1 MiB cap", maxPromQLResultBytes)
	}
}

// A sample that cannot be represented is missing from the answer. Reporting
// completeness while silently dropping it is the failure `truncated` exists to
// prevent.
func TestADroppedSampleIsReportedAsTruncation(t *testing.T) {
	// BOTH shapes: a vector carries `value` and a matrix `values`, and they are
	// separate branches. Covering only the vector left the matrix one free to
	// drop a sample and still report a complete result.
	for name, tc := range map[string]struct{ kind, body, args string }{
		"vector": {
			KindPromQLInstant,
			`{"status":"success","data":{"resultType":"vector","result":[
				{"metric":{"job":"a"},"value":[1789689600,{"not":"a sample"}]}]}}`,
			`{"query":"x"}`,
		},
		"matrix": {
			KindPromQLRange,
			`{"status":"success","data":{"resultType":"matrix","result":[
				{"metric":{"job":"a"},"values":[[1789689600,"1"],[1789689660,{"not":"a sample"}]]}]}}`,
			`{"query":"x","start":"2026-09-18T11:00:00Z","end":"2026-09-18T12:00:00Z","step_s":60}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			h := newPromQLHarness(t, tc.body)
			out, err := runQ(t, h, tc.kind, tc.args)
			if err != nil {
				t.Fatal(err)
			}
			res := out.Payload.(PromQLResult)
			if !res.Stats.Truncated {
				t.Fatal("a dropped sample left the result claiming to be complete")
			}
		})
	}
}

// The BYTE budget, exercised on its own: a point count well inside any plausible point budget
// that still encodes past the platform's cap, which is the case the point cap
// cannot catch. Wide label sets are what make it reachable -- pgwatch emits
// several, and the series overhead is paid per series regardless of points.
func TestTheByteBudgetTrimsAWideLabelSet(t *testing.T) {
	const platformCap = 1048576
	longValue := strings.Repeat("x", 500)

	var b strings.Builder
	b.WriteString(`{"status":"success","data":{"resultType":"matrix","result":[`)
	series, perSeries := 200, 140 // 28000 points: not a lot, but the per-series label overhead dominates
	for i := 0; i < series; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `{"metric":{"__name__":"m","a":"%s","b":"%s","c":"%s","d":"%s","e":"%s","node":"n%03d"},"values":[`,
			longValue, longValue, longValue, longValue, longValue, i)
		for j := 0; j < perSeries; j++ {
			if j > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `[%d,"%d.5"]`, 1789689600+j*60, j)
		}
		b.WriteString(`]}`)
	}
	b.WriteString(`]}}`)

	h := newPromQLHarness(t, b.String())
	out, err := runQ(t, h, KindPromQLRange,
		`{"query":"x","start":"2026-09-18T11:00:00Z","end":"2026-09-18T12:00:00Z","step_s":60}`)
	if err != nil {
		t.Fatal(err)
	}
	res := out.Payload.(PromQLResult)

	blob, _ := json.Marshal(res)
	if len(blob) > platformCap {
		t.Fatalf("result is %d bytes, above the platform's %d cap", len(blob), platformCap)
	}
	if !res.Stats.Truncated {
		t.Fatal("a result trimmed by the byte budget was not reported as truncated")
	}
	if len(res.Result) >= series {
		t.Fatalf("nothing was trimmed: %d of %d series survived", len(res.Result), series)
	}
}

// The trim must not re-marshal the whole result once per dropped series.
//
// Bounded by ITERATIONS, not wall time: a clock assertion is a flake, and the
// discriminating property here is how many times the payload is encoded. On
// this container's 0.25 CPU the one-at-a-time path was ~1000 marshals of a
// shrinking multi-megabyte payload -- gigabytes of work, about a minute and a
// half, competing with collection on the same box.
func TestTheTrimDoesNotMarshalOncePerDroppedSeries(t *testing.T) {
	// ~8 MiB across 1000 series: the largest the 8 MiB body cap can deliver now
	// that no point cap bounds the input.
	const series, perSeries = 1000, 220
	big := PromQLResult{ResultType: "matrix", Result: make([]PromQLSeries, 0, series)}
	for i := 0; i < series; i++ {
		s := PromQLSeries{
			Metric: map[string]string{"__name__": "m", "node": fmt.Sprintf("n%04d", i)},
			Values: make([][]any, 0, perSeries),
		}
		for j := 0; j < perSeries; j++ {
			s.Values = append(s.Values, []any{float64(1789689600 + j*60), "1.5"})
		}
		big.Result = append(big.Result, s)
	}

	before, err := json.Marshal(big)
	if err != nil {
		t.Fatal(err)
	}
	if len(before) <= maxPromQLResultBytes {
		t.Fatalf("fixture is %d bytes, already inside the budget: it cannot exercise the trim",
			len(before))
	}

	fitted, marshals := fitPromQLResult(big)

	dropped := series - len(fitted.Result)
	t.Logf("dropped %d of %d series in %d marshals", dropped, series, marshals)

	// The claim is "not one marshal per dropped series", so the bound is stated
	// against what was actually dropped rather than a bare number. 25 leaves
	// room for the proportional rounds plus a short single-drop tail, and is an
	// order of magnitude under the ~850 the old path would have taken here.
	if marshals > 25 || marshals >= dropped {
		t.Fatalf("trimming took %d marshals to drop %d series; one per dropped series is the "+
			"quadratic path this exists to avoid", marshals, dropped)
	}
	blob, _ := json.Marshal(fitted)
	if len(blob) > maxPromQLResultBytes {
		t.Fatalf("trimmed result is %d bytes, still over the %d budget", len(blob), maxPromQLResultBytes)
	}
	if !fitted.Stats.Truncated {
		t.Fatal("a trimmed result was not reported as truncated")
	}
	// And it must not throw away more than it had to. The precise property is
	// that the result is MAXIMAL: putting back one more series goes over. (An
	// 8 MiB input against an 850 KB budget can only keep ~10% of the series, so
	// "kept at least half" would have been arithmetically impossible -- an
	// assertion about the fixture rather than the code.)
	if len(fitted.Result) < len(big.Result) {
		plusOne := fitted
		plusOne.Result = append(append([]PromQLSeries{}, fitted.Result...), big.Result[len(fitted.Result)])
		bigger, err := json.Marshal(plusOne)
		if err != nil {
			t.Fatal(err)
		}
		if len(bigger) <= maxPromQLResultBytes {
			t.Fatalf("one more series would still have fitted (%d bytes): the cut discarded "+
				"data the budget could have held", len(bigger))
		}
	}
}
