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

// --- the sparse-emit contract -------------------------------------------------

// A day of monitoring has 1440 one-minute slots; pgwatch emitted a sample in 100
// of them. The average is over the day, not over the 100 slots that had waiters.
func TestAvgDividesByExpectedSlotsNotBySampleCount(t *testing.T) {
	samples := make([]float64, 100)
	for i := range samples {
		samples[i] = 5
	}
	got := round1(avg(samples, 1440))
	if want := 0.3; got != want {
		t.Fatalf("avg = %v, want %v (dividing by the 100 returned samples gives 5.0, "+
			"a 14x overstatement -- production measured present=501 of expected=4184)", got, want)
	}
}

// query_range can return one boundary point more than the window's slot count.
// The denominator is bumped so avg never exceeds worst1m.
func TestAvgNeverExceedsWorst1mOnBoundaryOverrun(t *testing.T) {
	samples := []float64{4, 4, 4}
	if got, peak := avg(samples, 2), worst1m(samples); got > peak {
		t.Fatalf("avg %v exceeds worst1m %v", got, peak)
	}
}

// Percentiles zero-pad the present samples up to the expected slot count before
// interpolating: two samples in a ten-slot window are the 9th and 10th order
// statistics of [0,0,0,0,0,0,0,0,2,4].
func TestQuantileZeroPadsToExpectedSlots(t *testing.T) {
	samples := []float64{4, 2}
	if got, want := round1(quantilePadded(samples, 10, qP99)), 3.8; got != want {
		t.Fatalf("p99 = %v, want %v", got, want)
	}
	if got, want := round1(quantilePadded(samples, 10, qP999)), 4.0; got != want {
		t.Fatalf("p999 = %v, want %v", got, want)
	}
	// Over the present samples alone the p99 would be 3.98 -- close enough to
	// look right and wrong for every low-activity window.
	if got := round1(quantilePadded(samples, 2, qP99)); got == 3.8 {
		t.Fatal("padding to expectedSlots made no difference; the test proves nothing")
	}
}

// Missing slots are zeros, so they can never raise a maximum.
func TestWorst1mIsUnaffectedBySparseness(t *testing.T) {
	if got := worst1m([]float64{1, 7, 3}); got != 7 {
		t.Fatalf("worst1m = %v, want 7", got)
	}
	if got := worst1m(nil); got != 0 {
		t.Fatalf("worst1m of nothing = %v, want 0", got)
	}
}

// The density gate is where the sparse contract is load-bearing at runtime, and
// it is why the samples must never be zero-filled: with the gaps materialised,
// present would always equal expected, the gate could never fire, and a
// monitoring outage would be published as a confident AAS of 0.
//
// Both halves run the real collection path against the same window. The only
// difference is whether the total series arrives sparse (as the store sends it)
// or zero-filled (as a client that materialised the gaps would build it), and
// that alone flips a skip into a stored zero.
func TestZeroFillingTheGapsWouldDefeatTheDensityGate(t *testing.T) {
	run := func(t *testing.T, totalSeries string) Outcome {
		t.Helper()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			q := r.URL.Query().Get("query")
			if strings.HasPrefix(q, "sum(pgwatch_wait_events_total") {
				w.Write([]byte(totalSeries))
				return
			}
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
		}))
		defer srv.Close()

		args := `{"cluster_name":"c","node_name":"n","vcpus":2,
			"period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-01T00:10:00Z",
			"window_start":"2026-09-01T00:00:00Z"}`
		out, err := Run(context.Background(), NewClient(srv.URL, "", "", time.Second),
			KindAAS, []byte(args), time.Date(2026, 9, 1, 1, 0, 0, 0, time.UTC))
		if err != nil {
			t.Fatal(err)
		}
		return out
	}

	empty := `{"status":"success","data":{"resultType":"matrix","result":[]}}`
	if out := run(t, empty); out.Status != OutcomeSkipped || out.SkipReason != SkipDensity {
		t.Fatalf("a window the store had nothing for came back %q/%q, want skipped/density",
			out.Status, out.SkipReason)
	}

	// The same window, with the ten missing slots materialised as zeros.
	var points []string
	for i := 1; i <= 10; i++ {
		points = append(points, fmt.Sprintf(`[%d,"0"]`, i))
	}
	zeroFilled := `{"status":"success","data":{"resultType":"matrix","result":[
		{"metric":{},"values":[` + strings.Join(points, ",") + `]}]}}`
	if out := run(t, zeroFilled); out.Status != OutcomeOK {
		t.Fatalf("zero-filled input came back %q/%q; this test only means something "+
			"while zero-filling would have produced a stored payload", out.Status, out.SkipReason)
	}
}

// --- the tempfile contract, which runs the other way --------------------------

func TestTempfileDividesByObservedSamples(t *testing.T) {
	r := AggregateTempfile([]float64{1, 2, 30})
	if r.AvgMiBPS != 11.0 {
		t.Fatalf("avg = %v, want 11.0 (dividing by an expected slot count would "+
			"report a gap in monitoring as idle disk)", r.AvgMiBPS)
	}
	if r.WorstMiBPS != 30.0 {
		t.Fatalf("worst = %v, want 30.0", r.WorstMiBPS)
	}
	if r.P99MiBPS != 29.4 {
		t.Fatalf("p99 = %v, want 29.4", r.P99MiBPS)
	}
}

func TestTempfileFiltersNonFiniteAndNegative(t *testing.T) {
	r := AggregateTempfile([]float64{1, math.Inf(1), math.NaN(), -5, 3})
	if r.AvgMiBPS != 2.0 || r.WorstMiBPS != 3.0 {
		t.Fatalf("got avg=%v worst=%v, want 2.0 / 3.0 over the two clean samples",
			r.AvgMiBPS, r.WorstMiBPS)
	}
	if got := AggregateTempfile([]float64{math.NaN()}); got != (TempfileResult{}) {
		t.Fatalf("all-unusable samples = %+v, want the zero result", got)
	}
}

// The AAS path does NOT filter non-finite values -- deliberately the opposite of
// the tempfile path. They are real samples of a real gauge there.
func TestAASSamplesKeepNonFiniteValues(t *testing.T) {
	body := `{"status":"success","data":{"resultType":"matrix","result":[
		{"metric":{"wait_event_type":"IO"},"values":[[1,"1"],[2,"NaN"],[3,"+Inf"]]}]}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	series, err := NewClient(srv.URL, "", "", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600))
	if err != nil {
		t.Fatal(err)
	}
	if len(series) != 1 || len(series[0].Samples) != 3 {
		t.Fatalf("got %d series / %v, want one series of three samples", len(series), series)
	}
	if !math.IsNaN(series[0].Samples[1]) || !math.IsInf(series[0].Samples[2], 1) {
		t.Fatalf("non-finite samples were filtered: %v", series[0].Samples)
	}
}

// --- the queries themselves ---------------------------------------------------

// The wait-event-type regex must reach PromQL with TWO backslashes: the string
// parser unescapes them to one, leaving the regex a literal asterisk. With one
// the store rejects the query outright.
func TestWaitEventTypeRegexKeepsItsDoubleBackslash(t *testing.T) {
	q := SeriesFilter{"c", "n"}.labelMatcher()
	if !strings.Contains(q, `CPU\\*`) {
		t.Fatalf("matcher lost the double backslash: %s", q)
	}
}

func TestEscapeLabelValueEscapesBackslashBeforeQuote(t *testing.T) {
	if got, want := escapeLabelValue(`a"b\c`), `a\"b\\c`; got != want {
		t.Fatalf("escapeLabelValue = %q, want %q", got, want)
	}
}

// The "total" class of the per-queryid ranking reuses the five-type regex, not a
// bare selector: the metric also carries idle wait types, and folding those in
// would diverge from the headline AAS total.
func TestTotalQueryIDClassUsesTheFiveTypeRegex(t *testing.T) {
	var queries []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		queries = append(queries, r.URL.Query().Get("query"))
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	}))
	defer srv.Close()

	_, err := NewClient(srv.URL, "", "", time.Second).
		QueryPerQueryID(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600))
	if err != nil {
		t.Fatal(err)
	}
	if len(queries) != len(queryIDClasses) {
		t.Fatalf("issued %d queries, want one per class (%d)", len(queries), len(queryIDClasses))
	}
	if !strings.Contains(queries[0], `wait_event_type=~"CPU\\*|IO|IPC|Lock|LWLock"`) {
		t.Fatalf("the total class is not filtered by the five types: %s", queries[0])
	}
	if !strings.Contains(queries[len(queries)-1], `wait_event_type="CPU*"`) {
		t.Fatalf("the cpu class lost its exact-type selector: %s", queries[len(queries)-1])
	}
}

// --- slicing and merging ------------------------------------------------------

func TestSplitWindowSlicesAreContiguousAndDisjoint(t *testing.T) {
	start, end := tm(0), tm(25*24*3600) // 25 days = 36001 points at 60s
	parts, err := splitWindow(start, end)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) < 2 {
		t.Fatalf("a 25-day window produced %d slice(s); it exceeds maxPointsPerRange", len(parts))
	}
	if !parts[0].start.Equal(start) {
		t.Fatalf("first slice starts at %v, want %v", parts[0].start, start)
	}
	if last := parts[len(parts)-1]; !last.end.Equal(end) {
		t.Fatalf("last slice ends at %v, want %v", last.end, end)
	}
	step := time.Duration(StepSeconds) * time.Second
	points := int64(0)
	for i, p := range parts {
		points += (p.end.Unix()-p.start.Unix())/StepSeconds + 1
		if pts := (p.end.Unix()-p.start.Unix())/StepSeconds + 1; pts > maxPointsPerRange {
			t.Fatalf("slice %d asks for %d points, over the %d cap", i, pts, maxPointsPerRange)
		}
		if i == 0 {
			continue
		}
		if want := parts[i-1].end.Add(step); !p.start.Equal(want) {
			t.Fatalf("slice %d starts at %v, want exactly one step after the previous end (%v)",
				i, p.start, want)
		}
	}
	if want := (end.Unix()-start.Unix())/StepSeconds + 1; points != want {
		t.Fatalf("slices cover %d points, want %d (a gap or an overlap)", points, want)
	}
}

func TestSplitWindowRefusesAnAbsurdWindow(t *testing.T) {
	if _, err := splitWindow(tm(0), tm(400*24*3600)); !errors.Is(err, ErrWindowTooLong) {
		t.Fatalf("err = %v, want ErrWindowTooLong", err)
	}
}

// Slices are stitched back together by label set, never by position: pgwatch
// sparse-emits, so a series present in one slice can be absent from the next.
func TestQueryRangeMergesSlicesByLabelSet(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The first slice carries both series, the second only the later one.
		if r.URL.Query().Get("start") == "0" {
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
				{"metric":{"wait_event_type":"IO"},"values":[[1,"1"],[2,"2"]]},
				{"metric":{"wait_event_type":"Lock"},"values":[[3,"3"]]}]}}`))
			return
		}
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
			{"metric":{"wait_event_type":"Lock"},"values":[[4,"4"],[5,"5"]]}]}}`))
	}))
	defer srv.Close()

	series, err := NewClient(srv.URL, "", "", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(25*24*3600))
	if err != nil {
		t.Fatal(err)
	}
	byType := map[string][]float64{}
	for _, s := range series {
		byType[s.Type] = s.Samples
	}
	if got := byType["IO"]; len(got) != 2 {
		t.Fatalf("IO samples = %v, want the two from the first slice only", got)
	}
	if got := byType["Lock"]; len(got) != 3 {
		t.Fatalf("Lock samples = %v, want the first slice's one plus the second's two "+
			"(positional merging would have attached them to IO)", got)
	}
}

// --- window clamp and gates ---------------------------------------------------

func TestClampWindowTakesTheLaterStartAndTheEarlierEnd(t *testing.T) {
	req := Request{
		PeriodStart: "2026-09-01T00:00:00Z",
		PeriodEnd:   "2026-09-02T00:00:00Z",
		WindowStart: "2026-09-01T12:00:00Z",
	}
	now := time.Date(2026, 9, 1, 18, 0, 0, 0, time.UTC)
	w, err := clampWindow(req, now)
	if err != nil {
		t.Fatal(err)
	}
	if !w.start.Equal(time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)) {
		t.Fatalf("start = %v, want the retention floor", w.start)
	}
	if !w.end.Equal(now) {
		t.Fatalf("end = %v, want now (the period is still in progress)", w.end)
	}
	if w.expectedSlots != 360 {
		t.Fatalf("expectedSlots = %d, want 360", w.expectedSlots)
	}
}

func TestInvalidArgsArePermanent(t *testing.T) {
	for name, args := range map[string]string{
		"empty object":     `{}`,
		"no node":          `{"cluster_name":"c","period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-02T00:00:00Z"}`,
		"unparseable time": `{"cluster_name":"c","node_name":"n","period_start":"yesterday","period_end":"2026-09-02T00:00:00Z"}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := Run(context.Background(), nil, KindAAS, []byte(args), time.Now())
			if !errors.Is(err, ErrInvalidArgs) {
				t.Fatalf("err = %v, want ErrInvalidArgs", err)
			}
		})
	}
}

func TestUnknownKindIsPermanent(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-02T00:00:00Z"}`
	now := time.Date(2026, 9, 2, 0, 0, 0, 0, time.UTC)
	_, err := Run(context.Background(), nil, "dblab_snapshot", []byte(args), now)
	if !errors.Is(err, ErrUnknownKind) {
		t.Fatalf("err = %v, want ErrUnknownKind", err)
	}
}

// --- enrichment is best-effort ------------------------------------------------

// A failing query-text lookup must never fail a collection that otherwise
// succeeded: the payload goes out without the query field.
func TestQueryTextEnrichmentFailureDoesNotFailTheCollection(t *testing.T) {
	var dropped error
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/query" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		switch q := r.URL.Query().Get("query"); {
		case strings.HasPrefix(q, "topk("):
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
				{"metric":{"query_id":"9","datname":"db"},"values":[[1,"3"]]}]}}`))
		default:
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
				{"metric":{"wait_event_type":"IO"},"values":[[1,"3"]]}]}}`))
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "", "", time.Second)
	client.OnEnrichmentError = func(err error) { dropped = err }

	args := `{"cluster_name":"c","node_name":"n","vcpus":2,
		"period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-01T00:10:00Z",
		"window_start":"2026-09-01T00:00:00Z"}`
	out, err := Run(context.Background(), client, KindAAS, []byte(args),
		time.Date(2026, 9, 1, 1, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatalf("a failed enrichment failed the whole collection: %v", err)
	}
	if out.Status != OutcomeOK {
		t.Fatalf("status = %q, want ok", out.Status)
	}
	if dropped == nil {
		t.Fatal("the enrichment error was swallowed without being reported")
	}
	raw, _ := json.Marshal(out)
	if strings.Contains(string(raw), `"query"`) {
		t.Fatalf("payload carries a query field although enrichment failed: %s", raw)
	}
	if !strings.Contains(string(raw), `"queryid":"9"`) {
		t.Fatalf("the ranking itself was lost: %s", raw)
	}
}

// The ids are gathered in the canonical class order (QueryIDClassKeys), never by
// ranging the class map, so the enrichment PromQL is byte-identical every run.
// Ranging the map instead would still resolve the same texts, but the regex
// alternation order would vary run to run -- which a fixture that pins the query
// exactly turns into a flaky test rather than a caught bug.
func TestCollectQueryIDsIsDeterministicAndCanonical(t *testing.T) {
	// Two classes, each with ids in a different class so the class order is what
	// decides the sequence; ipc/lwlock left empty to prove absent classes are
	// simply skipped rather than reordering anything.
	r := Result{
		TopQueryIDsWorst1m: map[string][]QueryIDEntry{
			"cpu":   {{QueryID: "40"}},
			"total": {{QueryID: "10"}, {QueryID: "20"}},
			"io":    {{QueryID: "30"}},
			"lock":  {{QueryID: "30"}}, // a repeat: first-seen wins, so it drops
		},
	}
	// total, then io, then lock, then cpu -- QueryIDClassKeys order, deduped.
	want := []string{"10", "20", "30", "40"}
	for i := 0; i < 64; i++ { // map iteration is randomized per range; 64 draws makes a wrong order overwhelmingly likely to show
		got := CollectQueryIDs(r)
		if len(got) != len(want) {
			t.Fatalf("run %d: got %v, want %v", i, got, want)
		}
		for j := range want {
			if got[j] != want[j] {
				t.Fatalf("run %d: got %v, want the canonical class order %v", i, got, want)
			}
		}
	}
}

// query_id is carried as a string end to end, so an id above 2^53 survives.
func TestQueryIDsSurviveAsStrings(t *testing.T) {
	const big = "18446744073709551557"
	r := Result{TopQueryIDsWorst1m: map[string][]QueryIDEntry{"total": {{QueryID: big, AAS: 1}}}}
	raw, err := json.Marshal(BuildPayload("n", "c", 1, r))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"queryid":"`+big+`"`) {
		t.Fatalf("query_id did not round-trip: %s", raw)
	}
}

// --- store error classification -----------------------------------------------

func TestStoreErrorsAreRetryableOnlyWhenARetryCouldHelp(t *testing.T) {
	for status, want := range map[int]bool{500: true, 502: true, 401: true, 429: true, 400: false, 422: false} {
		if got := (&UpstreamError{StatusCode: status}).Retryable(); got != want {
			t.Errorf("status %d retryable = %v, want %v", status, got, want)
		}
	}
}

// The request URL carries the PromQL, and the PromQL carries the job's cluster
// and node labels. Following a redirect would hand that URL to whatever host
// Location names.
func TestARedirectNeverForwardsTheQueryToAnotherHost(t *testing.T) {
	var forwarded string
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		forwarded = r.URL.Query().Get("query")
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	}))
	defer elsewhere.Close()

	store := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, elsewhere.URL+r.URL.RequestURI(), http.StatusTemporaryRedirect)
	}))
	defer store.Close()

	_, err := NewClient(store.URL, "", "", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"acme-prod", "n"}, tm(0), tm(600))
	var upstream *UpstreamError
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v, want an UpstreamError", err)
	}
	if upstream.StatusCode != http.StatusTemporaryRedirect {
		t.Fatalf("status = %d, want the redirect passed through as-is", upstream.StatusCode)
	}
	if upstream.Retryable() {
		t.Fatal("a refused redirect is retryable; a misconfigured front end would be dialled thrice")
	}
	if forwarded != "" {
		t.Fatalf("the query was forwarded to the redirect target: %s", forwarded)
	}
}

func tm(sec int64) time.Time { return time.Unix(sec, 0).UTC() }

// --- the rankings -------------------------------------------------------------

// A series that is present but never had a waiter scores zero after padding.
// Emitting it would put "Foo(0.0)" in front of a reader as if it were a top
// consumer.
func TestRankingsDropSeriesThatScoreZero(t *testing.T) {
	perEvent := []EventSeries{
		{Type: TypeIO, Event: "DataFileRead", Samples: []float64{3}},
		{Type: TypeIO, Event: "Quiet", Samples: []float64{0, 0}},
	}
	got := topEvents(perEvent, 10, "worst1m")["io"]
	if len(got) != 1 || got[0] != "DataFileRead(3.0)" {
		t.Fatalf("top events = %v, want only the event with activity", got)
	}

	perQueryID := []QueryIDSeries{
		{Class: "total", QueryID: "1", Datname: "db", Samples: []float64{3}},
		{Class: "total", QueryID: "2", Datname: "db", Samples: []float64{0}},
	}
	entries := topQueryIDs(perQueryID, 10, "worst1m")["total"]
	if len(entries) != 1 || entries[0].QueryID != "1" {
		t.Fatalf("top queryids = %v, want only the id with activity", entries)
	}
}

// Top events are capped at three per type, ranked by the flavor statistic, with
// the event name as a deterministic tie-break.
func TestTopEventsAreCappedAtThreeAndRankedByScore(t *testing.T) {
	perEvent := []EventSeries{
		{Type: TypeLock, Event: "d", Samples: []float64{1}},
		{Type: TypeLock, Event: "a", Samples: []float64{9}},
		{Type: TypeLock, Event: "c", Samples: []float64{5}},
		{Type: TypeLock, Event: "b", Samples: []float64{5}},
	}
	got := topEvents(perEvent, 10, "worst1m")["lock"]
	want := []string{"a(9.0)", "b(5.0)", "c(5.0)"}
	if len(got) != len(want) {
		t.Fatalf("top events = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("top events = %v, want %v", got, want)
		}
	}
}

// query_id ranking is capped at TopQueryIDN per class, ties broken by id then
// datname so two boxes reporting the same window agree.
func TestTopQueryIDsAreCappedAndTieBrokenDeterministically(t *testing.T) {
	var series []QueryIDSeries
	for _, id := range []string{"7", "6", "5", "4", "3", "2", "1"} {
		series = append(series, QueryIDSeries{Class: "io", QueryID: id, Datname: "db", Samples: []float64{4}})
	}
	// Two rows equal on score AND id, differing only on datname: the ranking key
	// is (query_id, datname), so the datname leg of the tie-break is reachable.
	series = append(series,
		QueryIDSeries{Class: "io", QueryID: "0", Datname: "zeta", Samples: []float64{4}},
		QueryIDSeries{Class: "io", QueryID: "0", Datname: "alpha", Samples: []float64{4}})

	entries := topQueryIDs(series, 10, "worst1m")["io"]
	if entries[0].Datname != "alpha" || entries[1].Datname != "zeta" {
		t.Fatalf("equal score and id should order by datname, got %v", entries[:2])
	}
	if len(entries) != TopQueryIDN {
		t.Fatalf("got %d entries, want the TopQueryIDN cap (%d)", len(entries), TopQueryIDN)
	}
	for i, want := range []string{"0", "0", "1", "2", "3"} {
		if entries[i].QueryID != want {
			t.Fatalf("entries = %v, want the equal scores tie-broken by id", entries)
		}
	}
}

// The store requires basic auth on a real box (compose wires VM_AUTH_USERNAME /
// VM_AUTH_PASSWORD into the container), so a break anywhere on this path fails
// every collection with a 401 while nothing local notices.
func TestTheStoreCredentialsAreSent(t *testing.T) {
	var gotUser, gotPass string
	var hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUser, gotPass, hadAuth = r.BasicAuth()
		if !hadAuth || gotUser != "vmauth" || gotPass != "s3cret" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	}))
	defer srv.Close()

	if _, err := NewClient(srv.URL, "vmauth", "s3cret", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600)); err != nil {
		t.Fatalf("the store rejected the request: %v (user=%q sent=%v)", err, gotUser, hadAuth)
	}
}

// Both empty means no header at all: a store without auth configured must not
// receive an empty Authorization.
func TestNoStoreCredentialsMeansNoHeader(t *testing.T) {
	sawHeader := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _, sawHeader = r.BasicAuth()
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	}))
	defer srv.Close()

	if _, err := NewClient(srv.URL, "", "", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600)); err != nil {
		t.Fatal(err)
	}
	if sawHeader {
		t.Fatal("an Authorization header was sent although no credentials were configured")
	}
}

// A window shorter than one step clamps to zero expected slots, which is a
// retention skip. Without the guard it is not a crash -- the statistics bump
// their denominator to len(samples) -- but it queries the store twice to reach
// a "density" skip that says something different from what happened.
func TestASubStepWindowIsARetentionSkip(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("a window with no expected slots must not be queried")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	args := `{"cluster_name":"c","node_name":"n","vcpus":1,
		"period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-01T00:00:30Z",
		"window_start":"2026-09-01T00:00:00Z"}`
	out, err := Run(context.Background(), NewClient(srv.URL, "", "", time.Second),
		KindAAS, []byte(args), time.Date(2026, 9, 1, 1, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Status != OutcomeSkipped || out.SkipReason != SkipRetention {
		t.Fatalf("a 30-second window gave %q/%q, want skipped/retention", out.Status, out.SkipReason)
	}
}

// --- what comes back from the store ------------------------------------------

// The store can answer 200 with a failure, and a body too large has to be
// distinguishable from a malformed one: they mean different things and only one
// of them is worth retrying.
func TestUnusableStoreAnswers(t *testing.T) {
	cases := []struct {
		name      string
		body      string
		wantCode  int
		wantRetry bool
		wantMsg   string
	}{
		{"a non-success status", `{"status":"error","errorType":"bad_data","error":"boom"}`,
			http.StatusBadGateway, true, "non-success"},
		{"a malformed body", `{"status":"success","data":{`,
			http.StatusBadGateway, true, "failed to parse"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Write([]byte(tc.body))
			}))
			defer srv.Close()

			_, err := NewClient(srv.URL, "", "", time.Second).
				QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600))
			var upstream *UpstreamError
			if !errors.As(err, &upstream) {
				t.Fatalf("err = %v, want an UpstreamError", err)
			}
			if upstream.StatusCode != tc.wantCode || upstream.Retryable() != tc.wantRetry {
				t.Fatalf("got %d retryable=%v, want %d retryable=%v",
					upstream.StatusCode, upstream.Retryable(), tc.wantCode, tc.wantRetry)
			}
			if !strings.Contains(upstream.Message, tc.wantMsg) {
				t.Fatalf("message = %q, want it to mention %q", upstream.Message, tc.wantMsg)
			}
		})
	}
}

// A body over the cap is deterministic: the response size tracks series
// cardinality, so the same window overflows every time. Retrying it three times
// spends the store's budget for the same answer.
func TestAnOversizedBodyIsNotRetried(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[`))
		for i := 0; i < 400; i++ {
			w.Write([]byte(`{"metric":{"wait_event":"` + strings.Repeat("x", 1000) + `"},"values":[]},`))
		}
		w.Write([]byte(`]}}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "", "", time.Second)
	client.maxRespBytes = 4096

	_, err := client.QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600))
	var upstream *UpstreamError
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v, want an UpstreamError", err)
	}
	if !strings.Contains(upstream.Message, "size limit") {
		t.Fatalf("message = %q, want it to name the size limit", upstream.Message)
	}
	if upstream.Retryable() {
		t.Fatal("an oversized response is retryable; the same window will overflow every time")
	}
}

// A sample the store sends that does not parse is DROPPED, not recorded as a
// zero: a bogus value counted as an observation inflates the density gate's
// present count and adds a zero to the statistics.
func TestUnparseableSamplesAreDroppedNotZeroed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
			{"metric":{"wait_event_type":"IO"},"values":[[1,"2"],[2,"not a number"],[3],[4,5],[5,"4"]]}]}}`))
	}))
	defer srv.Close()

	series, err := NewClient(srv.URL, "", "", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600))
	if err != nil {
		t.Fatal(err)
	}
	if len(series) != 1 {
		t.Fatalf("got %d series, want 1", len(series))
	}
	if got := series[0].Samples; len(got) != 2 || got[0] != 2 || got[1] != 4 {
		t.Fatalf("samples = %v, want only the two that parsed", got)
	}
}

// The merge key encodes lengths so a separator inside a label value cannot make
// two different label sets look like one -- which would concatenate their
// samples into a single series.
func TestLabelFingerprintCannotCollideThroughAValue(t *testing.T) {
	a := labelFingerprint(map[string]string{"a": "b", "c": "d"})
	b := labelFingerprint(map[string]string{"a=b,c": "d"})
	if a == b {
		t.Fatalf("two different label sets share the key %q", a)
	}
}

// Only one of the pair being set is a plausible half-configured box, and it
// must still authenticate rather than send nothing.
func TestAUsernameWithNoPasswordStillAuthenticates(t *testing.T) {
	var sawHeader bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _, sawHeader = r.BasicAuth()
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	}))
	defer srv.Close()

	if _, err := NewClient(srv.URL, "vmauth", "", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600)); err != nil {
		t.Fatal(err)
	}
	if !sawHeader {
		t.Fatal("no Authorization header was sent although a username is configured")
	}
}

// A kind this build cannot run is refused before the args are parsed and before
// the window is looked at: its args are its own shape, and its window is
// irrelevant.
func TestAnUnknownKindIsRefusedBeforeTheWindowIsEvenLookedAt(t *testing.T) {
	// A window entirely out of retention: without the kind check first, this
	// would be reported as a successful retention skip and the platform would
	// never learn the box cannot run the kind.
	// A window entirely out of retention AND args of another kind's shape: the
	// first would report a successful skip, the second `invalid_args`.
	for name, args := range map[string]string{
		"collection args": `{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
			"period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`,
		"another kind's args": `{"query":"up","at":"2026-09-16T00:00:00Z"}`,
		"no args at all":      `{}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := Run(context.Background(), nil, "promql_instant", []byte(args),
				time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC))
			if !errors.Is(err, ErrUnknownKind) {
				t.Fatalf("err = %v, want ErrUnknownKind", err)
			}
		})
	}
}

// A vector where a matrix was asked for decodes to zero values per series, which
// the density gate reads as "no data": a silent wrong answer, not an error.
func TestAWrongResultTypeIsAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"success","data":{"resultType":"vector","result":[
			{"metric":{"wait_event_type":"IO"},"value":[1,"3"]}]}}`))
	}))
	defer srv.Close()

	_, err := NewClient(srv.URL, "", "", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600))
	if err == nil {
		t.Fatal("a vector answer to a range query was accepted as an empty matrix")
	}
}

// A zero rate is an observation, not an absence: the series is dense, so
// dropping zeros would shrink the denominator and overstate the average.
func TestTempfileKeepsZeroRateSamples(t *testing.T) {
	if got := AggregateTempfile([]float64{0, 0, 4}); got.AvgMiBPS != 1.3 {
		t.Fatalf("avg = %v, want 1.3 over the three observed samples", got.AvgMiBPS)
	}
}

// The production cap, not the one a test reached in and set.
func TestTheClientIsBuiltWithTheProductionResponseCap(t *testing.T) {
	c := NewClient("http://example.invalid", "", "", 7*time.Second)
	if got := c.MaxResponseBytes(); got != maxResponseBytes {
		t.Fatalf("NewClient wired a %d-byte cap, want maxResponseBytes (%d)", got, maxResponseBytes)
	}
	// A non-production timeout, so an accessor that returns a constant instead
	// of the field cannot pass -- which would make this test, and the runner's
	// wiring test, vacuous in turn.
	if got := c.RequestTimeout(); got != 7*time.Second {
		t.Fatalf("RequestTimeout = %v, want the value NewClient was given", got)
	}
	shrunk := NewClient("http://example.invalid", "", "", time.Second)
	shrunk.maxRespBytes = 99
	if got := shrunk.MaxResponseBytes(); got != 99 {
		t.Fatalf("MaxResponseBytes = %d, want the field and not the constant", got)
	}
	if maxPointsPerRange != 30000 {
		t.Fatalf("maxPointsPerRange = %d; it mirrors VictoriaMetrics' default and is "+
			"the constant behind the 422 that lost every monthly row", maxPointsPerRange)
	}
	if maxSlicesPerRange != 8 {
		t.Fatalf("maxSlicesPerRange = %d, want 8 (~166 days)", maxSlicesPerRange)
	}
	if maxResponseBytes != 8<<20 {
		t.Fatalf("maxResponseBytes = %d; the measured ceiling in the README assumes 8 MiB",
			maxResponseBytes)
	}
}

// The merge key encodes the length of the VALUE as well as the name, or a
// separator inside a value folds two distinct series into one.
func TestLabelFingerprintEncodesBothLengths(t *testing.T) {
	pairs := [][2]map[string]string{
		{{"a": "b", "c": "d"}, {"a=b,c": "d"}},
		{{"a": "x,1:b=1:y"}, {"a": "x", "b": "y"}},
		{{"ab": "c"}, {"a": "bc"}},
	}
	for _, p := range pairs {
		if labelFingerprint(p[0]) == labelFingerprint(p[1]) {
			t.Fatalf("%v and %v share the key %q", p[0], p[1], labelFingerprint(p[0]))
		}
	}
}

// Either half of the pair alone still authenticates.
func TestAPasswordWithNoUsernameStillAuthenticates(t *testing.T) {
	var sawHeader bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _, sawHeader = r.BasicAuth()
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	}))
	defer srv.Close()

	if _, err := NewClient(srv.URL, "", "s3cret", time.Second).
		QueryPerType(context.Background(), SeriesFilter{"c", "n"}, tm(0), tm(600)); err != nil {
		t.Fatal(err)
	}
	if !sawHeader {
		t.Fatal("no Authorization header was sent although a password is configured")
	}
}

// The instant lookup's guard, the mirror of the range one.
func TestAMatrixAnswerToAnInstantQueryIsAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
			{"metric":{"queryid":"1"},"values":[[1,"1"]]}]}}`))
	}))
	defer srv.Close()

	_, err := NewClient(srv.URL, "", "", time.Second).
		QueryInfoTexts(context.Background(), []string{"1"}, tm(0), tm(600))
	if err == nil {
		t.Fatal("a matrix answer to an instant query was accepted")
	}
}

// The platform btrims args.node_name before comparing it against the payload's
// results key, so an untrimmed label here makes the two disagree -- and the
// PromQL matcher would carry the whitespace and match no series at all.
func TestLabelsAreTrimmedTheWayThePlatformTrimsThem(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The needles match the fixture's own spacing, or this branch is
		// unreachable and asserts nothing.
		if q := r.URL.Query().Get("query"); strings.Contains(q, `"  n1  "`) ||
			strings.Contains(q, `"  c1  "`) {
			t.Errorf("the selector carries untrimmed labels: %s", q)
		}
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
			{"metric":{},"values":[[1,"1"]]}]}}`))
	}))
	defer srv.Close()

	args := `{"cluster_name":"  c1  ","node_name":"  n1  ","vcpus":1,
		"period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-01T00:10:00Z",
		"window_start":"2026-09-01T00:00:00Z"}`
	out, err := Run(context.Background(), NewClient(srv.URL, "", "", time.Second),
		KindAAS, []byte(args), time.Date(2026, 9, 1, 1, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(out.Payload)
	if !strings.Contains(string(raw), `"n1"`) || strings.Contains(string(raw), `"  n1  "`) {
		t.Fatalf("the results key is not the trimmed node name: %s", raw)
	}
	// cluster_name too: it is echoed into the payload and v1.get_health_matrix
	// reads it as data->>'cluster_name'.
	if !strings.Contains(string(raw), `"cluster_name":"c1"`) {
		t.Fatalf("cluster_name is not trimmed in the payload: %s", raw)
	}
}

// An id ranked only in p999 still has its text resolved: the collection pass
// over the three flavor maps is asymmetric with the enrichment pass, and only
// the enrichment half was covered.
func TestCollectQueryIDsReadsEveryFlavor(t *testing.T) {
	r := Result{
		TopQueryIDsWorst1m: map[string][]QueryIDEntry{"total": {{QueryID: "1"}}},
		TopQueryIDsP99:     map[string][]QueryIDEntry{"total": {{QueryID: "2"}}},
		TopQueryIDsP999:    map[string][]QueryIDEntry{"total": {{QueryID: "3"}}},
	}
	got := CollectQueryIDs(r)
	if len(got) != 3 {
		t.Fatalf("CollectQueryIDs = %v, want one id from each flavor", got)
	}
	for i, want := range []string{"1", "2", "3"} {
		if got[i] != want {
			t.Fatalf("CollectQueryIDs = %v, want worst1m, p99 then p999", got)
		}
	}
}

// vcpus is echoed into the payload and never acted on, so no shape of it
// should cost the job -- but nothing out of range may be FORWARDED either.
//
// Defence in depth: the platform refuses a negative at arming and the producer
// skips <= 0, so no supported path sends one. Should one arrive anyway it
// survives health_matrix_evaluate's `vcpus = 0` guard and turns its AAS
// percentage negative, which reads as a GREEN cell -- so out of range maps to
// 0, that function's explicit unknown, and the cell reads grey instead.
func TestVCPUsIsForwardedOnlyWhenTheValueIsUsable(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want int
	}{
		{`2`, 2},
		{`2.0`, 2},   // to_jsonb() of a numeric
		{`"4"`, 4},   // a text column
		{`2.6`, 3},   // rounded, not truncated
		{`null`, 0},  // absent is fine: 0 is the platform's "unknown"
		{`"n/a"`, 0}, // unparseable is 0, not a dead job
		{`true`, 0},
		{`[]`, 0},
		// Out of range. int(math.Round(1e20)) is an out-of-range float->int
		// conversion, which gave int64-min -- a value the platform's ::int cast
		// cannot even store.
		{`1e20`, 0},
		{`9223372036854775808`, 0},
		{`3e9`, 0}, // over int4, which is what the platform stores
		// Negative is the dangerous one: it passes the platform's `= 0` guard
		// and makes its percentage negative, i.e. a false-green cell.
		{`-5`, 0},
		{`-2.6`, 0},
		{`"-8"`, 0},
		{`2147483647`, 2147483647}, // the ceiling itself still forwards
	} {
		t.Run(tc.raw, func(t *testing.T) {
			args := fmt.Sprintf(`{"cluster_name":"c","node_name":"n","vcpus":%s,`+
				`"period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-01T01:00:00Z"}`, tc.raw)
			req, err := parseRequest([]byte(args))
			if err != nil {
				t.Fatalf("vcpus %s killed the job: %v", tc.raw, err)
			}
			if got := req.vcpus(); got != tc.want {
				t.Fatalf("vcpus(%s) = %d, want %d", tc.raw, got, tc.want)
			}
		})
	}
}

// The catch-all said "not a json object" for anything that failed to decode,
// including a well-formed object whose field TYPES were wrong -- which sends
// whoever debugs it looking at the wrong thing.
func TestADecodeFailureNamesTheDecode(t *testing.T) {
	_, err := parseRequest([]byte(`{"cluster_name":123}`))
	if err == nil {
		t.Fatal("a type mismatch decoded cleanly")
	}
	if !errors.Is(err, ErrInvalidArgs) {
		t.Fatalf("err = %v, want ErrInvalidArgs", err)
	}
	if strings.Contains(err.Error(), "not a json object") {
		t.Fatalf("a valid json object is reported as not being one: %v", err)
	}
}

// A control character survives the trim and then breaks the PromQL selector.
// The store answers 400, which is non-retryable and reported as store_error --
// a bad job arg blamed on the store. It belongs to invalid_args instead.
func TestAControlCharacterInALabelIsABadArgNotAStoreFault(t *testing.T) {
	// BOTH fields, because the check is a per-field loop: pointing its
	// node_name arm at req.ClusterName left every package green when only
	// cluster_name was exercised.
	for _, field := range []string{"cluster_name", "node_name"} {
		for _, bad := range []string{"a\nb", "a\rb", "a\tb", "a\x00b", "a\x7fb"} {
			cluster, node := "c", "n"
			if field == "cluster_name" {
				cluster = bad
			} else {
				node = bad
			}
			// json.Marshal, not %q: Go's quoting emits \x00 and \x7f, which are
			// not valid JSON escapes, so those two cases failed to DECODE
			// before reaching the check and passed for the wrong reason.
			cq, _ := json.Marshal(cluster)
			nq, _ := json.Marshal(node)
			args := fmt.Sprintf(`{"cluster_name":%s,"node_name":%s,`+
				`"period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-01T01:00:00Z"}`,
				cq, nq)

			_, err := parseRequest([]byte(args))
			if !errors.Is(err, ErrInvalidArgs) {
				t.Fatalf("%s %q gave err = %v, want ErrInvalidArgs", field, bad, err)
			}
			// The message names the offending field, which is the whole point
			// of checking them separately.
			if !strings.Contains(err.Error(), field) {
				t.Fatalf("%s %q blamed the wrong field: %v", field, bad, err)
			}
		}
	}
	// A leading/trailing newline is still fine: the trim removes it.
	args := `{"cluster_name":" c\n","node_name":"\tn ",` +
		`"period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-01T01:00:00Z"}`
	req, err := parseRequest([]byte(args))
	if err != nil {
		t.Fatalf("a trimmable label was rejected: %v", err)
	}
	if req.ClusterName != "c" || req.NodeName != "n" {
		t.Fatalf("labels = %q/%q, want c/n", req.ClusterName, req.NodeName)
	}
}
