package collect

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

// The job kinds this process knows how to run. They match the PgQ queue names
// the pull path uses for the same work, so one enqueuer serves both paths.
const (
	KindAAS      = "aas_collect"
	KindTempfile = "tempfile_collect"
)

// The outcomes v1.instance_job_submit takes. Every submit names one: the rpc
// used to infer it from whether `error` was non-null, which made a legitimate
// skip inexpressible -- a skip is neither a payload nor a failure.
const (
	OutcomeOK      = "ok"
	OutcomeSkipped = "skipped"
	OutcomeError   = "error"
)

// Skip reasons. Each means "the collection ran and there is legitimately
// nothing to store", never an error. The three literals are the vocabulary the
// pull path already uses, so the platform stamps *_last_error with the same
// `skipped:<reason>` string whichever route a skip arrived by and an operator
// reads one vocabulary. (The per-reason counters on aas_collect_runs are the
// PgQ consumer's own and are not written on this path.)
const (
	SkipRetention = "retention"
	SkipDensity   = "density"
	SkipNoData    = "no_data"
)

// ErrUnknownKind marks a job kind this build cannot run. It is permanent: an
// older instance must fail such a job rather than retry it forever.
var ErrUnknownKind = errors.New("unknown job kind")

// ErrInvalidArgs marks job args this process cannot act on. Also permanent.
var ErrInvalidArgs = errors.New("invalid job args")

// Request is the subset of the job args a collection needs. The platform also
// sends project_id, pin and granularity; the box does not act on them (the pin
// and the granularity are the platform's own, and a result is applied against
// the row it came from), so they are parsed only for the log line.
type Request struct {
	ClusterName string `json:"cluster_name"`
	NodeName    string `json:"node_name"`
	// RawMessage, not int: vcpus is echoed into the payload and never acted on
	// here, so no shape of it should abort the WHOLE unmarshal -- an aborted
	// unmarshal is ErrInvalidArgs, which is never retried, so the job would die
	// over a field this side only forwards. aas_collect emits a plain int
	// today (jsonb_build_object with ::int, and it skips vcpus <= 0) and
	// tempfile_collect emits no vcpus at all, so this guards against a future
	// producer rather than a shape seen in the wild.
	VCPUs       json.RawMessage `json:"vcpus"`
	Granularity string          `json:"granularity"`
	PeriodStart string          `json:"period_start"`
	PeriodEnd   string          `json:"period_end"`
	WindowStart string          `json:"window_start"`
}

// Outcome is what one job produced.
//
// Payload is submitted BARE -- it is the whole `result` on the wire, with no
// envelope around it. public.instance_job_apply_collection reads `checkId` and
// `results` at the top level and takes the pin, granularity and project from
// the job row, which is the point of that function: a box that could choose the
// pin could destroy hand-ingested history. The pull path's `{status, payload}`
// envelope belongs to its HTTP transport and is unwrapped by its own consumer.
//
// A skip carries no payload, so it cannot be a result at all: it is submitted
// as outcome "skipped" with a skip_reason -- see AsSubmission.
type Outcome struct {
	Status     string
	SkipReason string
	Payload    any
}

// Skipped reports whether this job produced no data to land.
func (o Outcome) Skipped() bool { return o.Status == OutcomeSkipped }

// Submission is what one outcome becomes on the wire. The runner builds its
// request from this and the fixtures assert it, so the two cannot drift: a
// change to the mapping shows up in both.
type Submission struct {
	// Outcome is one of the three literals above. The platform rejects a
	// submit that does not name one.
	Outcome string
	// Result is the BARE payload, set only for OutcomeOK.
	Result any
	// SkipReason is set only for OutcomeSkipped.
	SkipReason string
}

// AsSubmission maps one outcome onto the wire. An error is not produced here --
// the runner owns that, because a job can fail before it ever has an Outcome.
func (o Outcome) AsSubmission() Submission {
	if o.Skipped() {
		return Submission{Outcome: OutcomeSkipped, SkipReason: o.SkipReason}
	}
	return Submission{Outcome: OutcomeOK, Result: o.Payload}
}

// window is the effective, clamped collection window.
type window struct {
	start         time.Time
	end           time.Time
	expectedSlots int
}

// vcpus is what reaches the payload: an int, whatever shape the producer sent.
//
// Lenient about SHAPE, strict about RANGE. The value is only forwarded, so a
// float or a numeric string should not cost the job -- but a value the platform
// cannot use must not be forwarded either.
func (r Request) vcpus() int {
	if len(r.VCPUs) == 0 {
		return 0
	}
	// Bounded to [0, int4] as DEFENCE IN DEPTH, not as the only line: the
	// platform already refuses a negative at arming (aas_register rejects null
	// or < 0; aas_onboard rejects <= 0) and the producer skips <= 0, so no
	// supported path sends one. If one arrives anyway it renders GREEN -- it
	// survives health_matrix_evaluate's `vcpus = 0` guard and makes the
	// percentage negative -- and a false-green cell on a customer's dashboard
	// is worse than a dead job. 0 is that function's explicit unknown. #366.
	var n json.Number
	if err := json.Unmarshal(r.VCPUs, &n); err != nil {
		return 0
	}
	f, err := n.Float64()
	if err != nil || !(f >= 0 && f <= math.MaxInt32) {
		return 0
	}
	return int(math.Round(f))
}

// parseRequest decodes and validates the job args.
func parseRequest(args []byte) (Request, error) {
	var req Request
	if len(args) == 0 {
		return req, fmt.Errorf("%w: empty", ErrInvalidArgs)
	}
	if err := json.Unmarshal(args, &req); err != nil {
		// Not "not a json object": this also fires for a well-formed object
		// whose field TYPES do not match. The detail is dropped rather than
		// wrapped -- classifyFailure maps ErrInvalidArgs to a constant, so a
		// wrapped %v would reach neither a log line nor the platform.
		return req, fmt.Errorf("%w: args did not decode", ErrInvalidArgs)
	}
	// Trimmed, because the platform btrims args.node_name before comparing it
	// against the payload's results key -- untrimmed, the two disagree.
	req.ClusterName = strings.TrimSpace(req.ClusterName)
	req.NodeName = strings.TrimSpace(req.NodeName)
	if req.ClusterName == "" || req.NodeName == "" {
		return req, fmt.Errorf("%w: cluster_name and node_name are required", ErrInvalidArgs)
	}
	// A control character survives the trim above and then breaks the PromQL
	// selector, which the store answers with a 400 -- reported as store_error,
	// "metric store returned 400", blaming the store for a bad job arg. Reject
	// it here, where the class is invalid_args and names the real cause.
	isControl := func(r rune) bool { return r < 0x20 || r == 0x7f }
	for _, f := range []struct {
		name  string
		value string
	}{{"cluster_name", req.ClusterName}, {"node_name", req.NodeName}} {
		if strings.ContainsFunc(f.value, isControl) {
			return req, fmt.Errorf("%w: %s must not contain control characters", ErrInvalidArgs, f.name)
		}
	}
	if req.PeriodStart == "" || req.PeriodEnd == "" {
		return req, fmt.Errorf("%w: period_start and period_end are required", ErrInvalidArgs)
	}
	return req, nil
}

// clampWindow computes the effective window and the expected slot count.
//
// start is the LATER of period_start and window_start, end the EARLIER of
// period_end and now. An expected slot count at or below zero means there is
// nothing to collect and the job is skipped rather than failed.
//
// Both producers currently write window_start as a COPY of period_start
// (aas_collect.sql, tempfile_collect.sql), so the first clamp is a no-op today
// and this skip fires only for a period that has not begun. The arm is kept
// because window_start is the producer's field for a retention floor and is
// where one would be expressed; a period genuinely older than the local store's
// retention currently reaches the density gate instead and is reported as such.
func clampWindow(req Request, now time.Time) (window, error) {
	periodStart, err := time.Parse(time.RFC3339, req.PeriodStart)
	if err != nil {
		return window{}, fmt.Errorf("%w: period_start is not RFC3339", ErrInvalidArgs)
	}
	periodEnd, err := time.Parse(time.RFC3339, req.PeriodEnd)
	if err != nil {
		return window{}, fmt.Errorf("%w: period_end is not RFC3339", ErrInvalidArgs)
	}
	start := periodStart
	if req.WindowStart != "" {
		windowStart, err := time.Parse(time.RFC3339, req.WindowStart)
		if err != nil {
			return window{}, fmt.Errorf("%w: window_start is not RFC3339", ErrInvalidArgs)
		}
		if windowStart.After(start) {
			start = windowStart
		}
	}
	end := periodEnd
	if now.Before(end) {
		end = now
	}
	return window{
		start:         start,
		end:           end,
		expectedSlots: int(end.Sub(start) / (StepSeconds * time.Second)),
	}, nil
}

// Run executes one collection job and returns what to submit.
//
// A returned error means the job did not produce an answer; the caller decides
// whether to retry it by asking the error (see UpstreamError.Retryable).
// ErrInvalidArgs, ErrUnknownKind and ErrWindowTooLong are permanent.
func Run(ctx context.Context, c *Client, kind string, args []byte, now time.Time) (Outcome, error) {
	// The kind is checked FIRST, before the args are even parsed and before the
	// window gate. A kind this build cannot run has to be answered as unknown:
	// its args are its own shape, so parsing them as a collection would report
	// `invalid_args`, and an out-of-retention window would report a successful
	// skip. During a rolling upgrade that adds a kind, the older boxes have to
	// say what is actually wrong.
	if kind != KindAAS && kind != KindTempfile {
		return Outcome{}, fmt.Errorf("%w: %q", ErrUnknownKind, kind)
	}
	req, err := parseRequest(args)
	if err != nil {
		return Outcome{}, err
	}
	w, err := clampWindow(req, now)
	if err != nil {
		return Outcome{}, err
	}
	if w.expectedSlots <= 0 {
		return Outcome{Status: OutcomeSkipped, SkipReason: SkipRetention}, nil
	}

	filter := SeriesFilter{Cluster: req.ClusterName, NodeName: req.NodeName}
	if kind == KindTempfile {
		return runTempfile(ctx, c, req, filter, w)
	}
	return runAAS(ctx, c, req, filter, w)
}

func runAAS(ctx context.Context, c *Client, req Request, filter SeriesFilter, w window) (Outcome, error) {
	// The per-type and total series come first: the density gate decides on them
	// alone, so a skipped window never pays for the per-event and per-class
	// queries (seven more range queries on a store with 0.75 CPU).
	perType, err := c.QueryPerType(ctx, filter, w.start, w.end)
	if err != nil {
		return Outcome{}, err
	}
	totalSeries, err := c.QueryTotalSeries(ctx, filter, w.start, w.end)
	if err != nil {
		return Outcome{}, err
	}

	// Density gate. present is the number of slots the store actually had a
	// total sample in. The clamp is defensive only -- query_range can return
	// expectedSlots+1 boundary points, and clamping down cannot turn a non-zero
	// count into zero, so it does not move the gate below.
	//
	// The gate is present == 0 and nothing more. pgwatch sparse-emits, so a
	// healthy window's present/expected ratio is well below 1.0 -- production
	// measured 501/4184 -- and any ratio floor above zero skips real data (a
	// fixed 0.8 once stopped daily AAS entirely). present == 0 is the one case
	// worth refusing: storing a point built from zero observations would publish
	// a confident AAS of 0 for what is actually a monitoring outage.
	//
	// This is also why the samples are never zero-filled: a client that
	// materialised the missing slots would make present == expected always, the
	// gate could never fire, and every outage would land as AAS 0.
	present := len(totalSeries)
	if present > w.expectedSlots {
		present = w.expectedSlots
	}
	if present == 0 {
		return Outcome{Status: OutcomeSkipped, SkipReason: SkipDensity}, nil
	}

	perEvent, err := c.QueryPerEvent(ctx, filter, w.start, w.end)
	if err != nil {
		return Outcome{}, err
	}
	perQueryID, err := c.QueryPerQueryID(ctx, filter, w.start, w.end)
	if err != nil {
		return Outcome{}, err
	}

	result := Aggregate(AggregateInput{
		ExpectedSlots: w.expectedSlots,
		PerType:       perType,
		TotalSeries:   totalSeries,
		PerEvent:      perEvent,
		PerQueryID:    perQueryID,
	})

	// Best-effort query-text enrichment. It must never fail a collection that
	// otherwise succeeded, so an error here is dropped and the payload goes out
	// without the query field (the reader renders it absent).
	if qids := CollectQueryIDs(result); len(qids) > 0 {
		if texts, terr := c.QueryInfoTexts(ctx, qids, w.start, w.end); terr == nil {
			EnrichQueryTexts(&result, texts)
		} else if c.OnEnrichmentError != nil {
			c.OnEnrichmentError(terr)
		}
	}

	return Outcome{
		Status:  OutcomeOK,
		Payload: BuildPayload(req.NodeName, req.ClusterName, req.vcpus(), result),
	}, nil
}

func runTempfile(ctx context.Context, c *Client, req Request, filter SeriesFilter, w window) (Outcome, error) {
	samples, err := c.QueryTempfileWriteRate(ctx, filter, w.start, w.end)
	if err != nil {
		return Outcome{}, err
	}
	// No density gate here: the rate series is dense, so "no samples at all" is
	// the only unusable case and it has its own reason.
	if len(samples) == 0 {
		return Outcome{Status: OutcomeSkipped, SkipReason: SkipNoData}, nil
	}
	return Outcome{
		Status:  OutcomeOK,
		Payload: BuildTempfilePayload(req.NodeName, req.ClusterName, AggregateTempfile(samples)),
	}, nil
}
