package collect

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// The second workload on the job channel. Unlike a collection, the result is
// read back by a human through the CLI and is never applied to anything: the
// platform stores it as-is and `instance_job_apply_collection` is not reached.
const (
	KindPromQLInstant = "promql_instant"
	KindPromQLRange   = "promql_range"
)

// Only these two subpaths are reachable, and which one is used is decided by
// the job KIND -- never by anything in args. The user supplies one value, the
// expression, and it travels as a single `query` form value; it is never
// interpolated into a larger PromQL string, so escapeLabelValue has no part in
// this path.
const (
	promQLInstantPath = "api/v1/query"
	promQLRangePath   = "api/v1/query_range"
)

// Caps on what one answer may carry.
//
// BYTES is the only real guarantee, because bytes are what the platform
// enforces: instance_job_submit caps `result` at 1 MiB and answers PT400 above
// it, measured as octet_length(result::text) on JSONB.
//
// That is why the budget is 850000 and not 1048576: THE TWO SIDES DO NOT COUNT
// THE SAME BYTES. Go emits compact JSON; Postgres's jsonb text output inserts a
// space after every `:` and `,`, so the platform always sees more than we
// counted, and the gap scales with TOKEN DENSITY rather than size. Measured on
// a real Postgres: 15.2% inflation on dense sample arrays, 19.6% on a series
// carrying many one-character labels. At a 900000 budget that worst case is
// ~1.076 MB and is refused, so a result this trimmer certified as fitting would
// still come back PT400.
//
// 850000 tolerates 23% inflation, costs nothing because every realistic shape
// already fits well inside it, and removes the dependence on a customer's label
// cardinality. Do not raise it back toward 1 MiB without re-measuring the
// jsonb text form, which is the only figure the platform reads.
//
// There is deliberately NO point cap. One was tried and removed: a total-point
// limit is an approximation of the byte budget that gets measured exactly ten
// lines later, and it tightens the wrong axis -- the platform validates
// (end-start)/step_s <= 11000 points PER SERIES with no series bound, so a
// 30000-point total truncated a window the platform had just called legal at
// three series, which is most real queries.
//
// maxPromQLSeries stays, but as a loop bound rather than a policy. Note what it
// does and does not do: it BOUNDS the trim below at 1000 iterations, it does
// not make it linear. The proportional first cut is what does that.
const (
	maxPromQLSeries      = 1000
	maxPromQLResultBytes = 850000
)

// promQLArgs is the whole of what the platform sends. `query` is the only
// user-controlled value; every other field is a bound the platform validated
// before enqueueing.
type promQLArgs struct {
	Query string `json:"query"`
	At    string `json:"at"`
	Start string `json:"start"`
	End   string `json:"end"`
	StepS int    `json:"step_s"`
}

// PromQLResult is what the box submits. It is built field by field from the
// decoded response rather than forwarded, so nothing the store emits reaches
// the platform unexamined.
type PromQLResult struct {
	ResultType string         `json:"resultType"`
	Result     []PromQLSeries `json:"result"`
	Stats      PromQLStats    `json:"stats"`
}

// PromQLSeries carries `value` for a vector and `values` for a matrix, exactly
// as Prometheus shapes them, so the CLI can render either without a second
// vocabulary.
type PromQLSeries struct {
	Metric map[string]string `json:"metric"`
	Value  []any             `json:"value,omitempty"`
	Values [][]any           `json:"values,omitempty"`
}

// PromQLStats is one boolean. Series and point counts were here and are gone:
// both are derivable from the payload at render time, and transmitting them was
// the only reason the trim below had to maintain running totals.
type PromQLStats struct {
	Truncated bool `json:"truncated"`
}

// promQLResponse is the store's own shape. `value` and `values` are decoded
// separately because an instant query returns one and a range query the other.
type promQLResponse struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string `json:"resultType"`
		Result     []struct {
			Metric map[string]string `json:"metric"`
			Value  []any             `json:"value"`
			Values [][]any           `json:"values"`
		} `json:"result"`
	} `json:"data"`
	ErrorType string `json:"errorType"`
	Error     string `json:"error"`
}

// runPromQL answers one query job.
func runPromQL(ctx context.Context, c *Client, kind string, args []byte, now time.Time) (Outcome, error) {
	var a promQLArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return Outcome{}, fmt.Errorf("%w: args did not decode", ErrInvalidArgs)
	}
	if a.Query == "" {
		return Outcome{}, fmt.Errorf("%w: query is required", ErrInvalidArgs)
	}

	params := url.Values{}
	params.Set("query", a.Query)

	subpath := promQLInstantPath
	if kind == KindPromQLRange {
		subpath = promQLRangePath
		start, err := parsePromQLTime(a.Start, "start")
		if err != nil {
			return Outcome{}, err
		}
		end, err := parsePromQLTime(a.End, "end")
		if err != nil {
			return Outcome{}, err
		}
		if !end.After(start) {
			return Outcome{}, fmt.Errorf("%w: end must be after start", ErrInvalidArgs)
		}
		if a.StepS < 1 {
			return Outcome{}, fmt.Errorf("%w: step_s must be at least 1", ErrInvalidArgs)
		}
		params.Set("start", strconv.FormatInt(start.Unix(), 10))
		params.Set("end", strconv.FormatInt(end.Unix(), 10))
		params.Set("step", strconv.Itoa(a.StepS))
	} else if a.At != "" {
		at, err := parsePromQLTime(a.At, "at")
		if err != nil {
			return Outcome{}, err
		}
		params.Set("time", strconv.FormatInt(at.Unix(), 10))
	}

	var resp promQLResponse
	if err := c.getJSON(ctx, subpath, params, &resp); err != nil {
		// A 4xx is the EXPRESSION's fault, not the store's: VictoriaMetrics
		// answers a bad query with 422 and upstream Prometheus with 400. Classed
		// as a bad ARGUMENT so it is not retried three times for the same
		// answer, and carrying the store's own words, which are the whole
		// diagnostic to someone who has just typed it (#378).
		//
		// The test is `!ue.Retryable()`, NOT a hand-written exclusion list. The
		// list drifted the first time it was written: Retryable() names three
		// statuses (>=500, 401, 429) and the list named two, so a store
		// rate-limit became a permanent bad argument and stopped being retried.
		// Asking Retryable() directly cannot disagree with itself.
		//
		// An ALLOW-LIST of the statuses that mean the caller's expression, not an
		// exclusion list of everything else. The exclusion form drifted twice:
		// first it omitted 429, then -- with 403/404 patched in -- it still let
		// 402, 405, 407 and 408 blame the user for a box misconfiguration, and
		// 407 is literally "a proxy in front of the store", which is the reason
		// the previous patch gave for excluding 404.
		//
		// This set is closed and small: the two real stores answer a bad
		// expression with 400 (Prometheus) or 422 (VictoriaMetrics), and 414 is
		// genuinely an over-long expression. Everything else -- including every
		// status nobody has thought of yet -- falls through as a store fault,
		// which is the safe direction: it names the real component and stays
		// retryable where Retryable() says so.
		var ue *UpstreamError
		if errors.As(err, &ue) && isCallerFault(ue.StatusCode) {
			msg := ue.StoreMessage
			if msg == "" {
				msg = ue.Message
			}
			// BOTH sentinels stay in the chain: ErrInvalidArgs for the
			// classification, and the *UpstreamError so StoreMessage can still
			// find the store's words downstream.
			return Outcome{}, fmt.Errorf("%w: %s (%w)", ErrInvalidArgs, msg, ue)
		}
		return Outcome{}, err
	}
	if resp.Status != "success" {
		// Defence in depth: a 200 carrying status:"error" is a shape no
		// Prometheus-compatible store sends, so this is not the syntax-error
		// path -- that one is the 4xx branch above.
		msg := resp.Error
		if msg == "" {
			msg = "query returned a non-success status"
		}
		// Carried on an UpstreamError so StoreMessage can find it downstream --
		// without this the message is built here and then discarded by
		// classifyFailure, exactly as the 4xx one was.
		return Outcome{}, fmt.Errorf("%w: %s (%w)", ErrInvalidArgs, msg,
			&UpstreamError{StatusCode: http.StatusBadGateway, Message: msg, StoreMessage: msg})
	}

	return Outcome{Status: OutcomeOK, Payload: buildPromQLResult(resp)}, nil
}

// StripControlsToSpace is StripControls for TEXT a human will read: the same
// runes are neutralised, but each becomes a space rather than vanishing. A
// multi-line label value -- pgwatch carries query text in labels -- otherwise
// comes back with its tokens fused ("select 1\nfrom t" -> "select 1from t"),
// which changes what the text says rather than sanitising it.
func StripControlsToSpace(s string) string {
	return strings.Map(func(r rune) rune {
		if isControl(r) {
			return ' '
		}
		return r
	}, s)
}

// isCallerFault reports whether a store status means the submitted EXPRESSION
// was wrong, as opposed to the box being pointed at the wrong thing.
func isCallerFault(status int) bool {
	switch status {
	case http.StatusBadRequest, // Prometheus
		http.StatusRequestURITooLong,   // a genuinely over-long expression
		http.StatusUnprocessableEntity: // VictoriaMetrics
		return true
	}
	return false
}

// StripControls removes C0, C1 and DEL, plus LINE/PARAGRAPH SEPARATOR. Applied
// to everything that comes back from the metric store and is later rendered:
// label keys and values, string sample values, and the error message.
//
// U+009B is the single-character CSI -- `ESC [` in 8-bit form -- so dropping ESC
// alone would not be enough. parseRequest REJECTS this same class for
// cluster_name and node_name; here it is stripped, because a metric label is
// not something to fail a whole collection over.
func StripControls(s string) string {
	return strings.Map(func(r rune) rune {
		if isControl(r) {
			return -1
		}
		return r
	}, s)
}

// isControl is the one definition both variants use: C0, DEL, C1, and the two
// Unicode line separators. U+009B is the single-character CSI -- `ESC [` in
// 8-bit form -- so dropping ESC alone would not be enough.
func isControl(r rune) bool {
	return r < 0x20 || r == 0x7f || (r >= 0x80 && r <= 0x9f) || r == 0x2028 || r == 0x2029
}

// StoreMessage returns the metric store's own words about why it refused a
// query, or "" when there were none. It exists so the runner can put them in
// the submitted error: a syntax error is the ONE failure where the store's
// message is the whole diagnostic, and a caller that only has ErrInvalidArgs
// can tell the user nothing more useful than "bad args" -- which points them at
// the platform, where the fault is not (#378).
//
// The text is attacker-influenced: a store echoes the submitted expression back
// inside it, and PromQL string literals take Go-style escapes. Callers MUST
// strip control characters before it reaches a terminal or a stored field.
func StoreMessage(err error) string {
	var ue *UpstreamError
	if errors.As(err, &ue) {
		return ue.StoreMessage
	}
	return ""
}

// buildPromQLResult decodes and RE-ENCODES: every series, sample and label is
// copied field by field, so a malformed or hostile upstream body cannot reach
// the platform by being passed through.
func buildPromQLResult(resp promQLResponse) PromQLResult {
	out := PromQLResult{ResultType: resp.Data.ResultType, Result: []PromQLSeries{}}

	for _, s := range resp.Data.Result {
		if len(out.Result) >= maxPromQLSeries {
			out.Stats.Truncated = true
			break
		}
		series := PromQLSeries{Metric: map[string]string{}}
		for k, v := range s.Metric {
			// The rebuild below guards the SHAPE of a hostile upstream body; it
			// did nothing about the CONTENT, and label values are not ours:
			// pgwatch scrapes them from the customer's Postgres, and
			// application_name is settable by anyone who can connect. The CLI
			// prints them to a terminal with no escaping, so an ESC in a label
			// would execute there (#378).
			//
			// The KEY is DROPPED rather than rewritten. A Prometheus label name
			// is [a-zA-Z_][a-zA-Z0-9_]* and can never legitimately carry a
			// control character, so a key that changes under StripControls is
			// hostile -- and rewriting it can collapse `job` and "job\x01" onto
			// one map entry, silently losing a label, with which one survives
			// decided by Go's randomised map order.
			if clean := StripControls(k); clean == k {
				series.Metric[k] = StripControlsToSpace(v)
			} else {
				out.Stats.Truncated = true
			}
		}

		if len(s.Value) > 0 {
			if sample, ok := sanitiseSample(s.Value); ok {
				series.Value = sample
			} else {
				// A sample we could not represent is missing from the answer.
				// Saying `truncated` is the honest report: claiming a complete
				// result while silently dropping a point is the failure this
				// flag exists to prevent.
				out.Stats.Truncated = true
			}
		}
		for _, raw := range s.Values {
			if sample, ok := sanitiseSample(raw); ok {
				series.Values = append(series.Values, sample)
			} else {
				out.Stats.Truncated = true
			}
		}
		out.Result = append(out.Result, series)
	}

	fitted, _ := fitPromQLResult(out)
	return fitted
}

// fitPromQLResult drops whole series from the end until the encoded result fits
// the platform's cap.
//
// Measured against the real encoding rather than estimated: the estimate that
// matters is the one the platform applies, and a result it refuses is worse
// than a short one -- the submit is a PT400, which is correctly not retried, so
// without this the job would sit `running` until the hourly sweep while the
// in-flight cap locked the user out for that hour.
func fitPromQLResult(out PromQLResult) (PromQLResult, int) {
	marshals := 0
	for len(out.Result) > 0 {
		blob, err := json.Marshal(out)
		marshals++
		if err != nil {
			// Unencodable is a different failure and belongs to the caller;
			// trimming series cannot fix it.
			return out, marshals
		}
		if len(blob) <= maxPromQLResultBytes {
			return out, marshals
		}

		// Cut PROPORTIONALLY first, then refine one series at a time.
		//
		// Dropping one per pass re-marshals the whole result every iteration.
		// Nothing bounds the input but the 8 MiB body cap now that the point cap
		// is gone, so trimming 8 MiB to 850000 across 1000 series was ~1000
		// marshals of a shrinking payload -- gigabytes of marshalling, and
		// roughly a minute and a half of wall clock on this container's 0.25
		// CPU, competing with collection on the same box.
		//
		// The estimate assumes series are of similar size, so it is taken with
		// a margin: overshooting would discard data the budget could have held,
		// and unlike undershooting the refine cannot undo it.
		keep := len(out.Result)
		if est := len(out.Result) * maxPromQLResultBytes / len(blob); est < keep {
			keep = est + est/20 + 2
			if keep >= len(out.Result) {
				keep = len(out.Result) - 1
			}
		} else {
			keep--
		}
		if keep < 0 {
			keep = 0
		}
		out.Result = out.Result[:keep]
		out.Stats.Truncated = true
	}
	return out, marshals
}

// sanitiseSample turns one [timestamp, value] pair into a form encoding/json
// will accept.
//
// Prometheus sends the VALUE as a string ("1.5", "+Inf", "NaN"), which json
// encodes without complaint -- so `1/0` can be answered honestly as "+Inf"
// rather than dropped or zeroed. The trap is the other direction: a float64
// +Inf or NaN reaching json.Marshal makes the WHOLE result unencodable, so the
// job would fail as a transport error instead of returning an answer. That is
// the same trap already fixed for AAS, and a user typing `1/0` reaches it far
// more easily. A store that sends a number instead of a string is therefore
// converted here rather than forwarded.
func sanitiseSample(raw []any) ([]any, bool) {
	if len(raw) != 2 {
		return nil, false
	}
	ts, ok := raw[0].(float64)
	if !ok || math.IsNaN(ts) || math.IsInf(ts, 0) {
		return nil, false
	}
	switch v := raw[1].(type) {
	case string:
		// Same reason as the labels: a sample value arrives as a string and is
		// printed straight to a terminal.
		return []any{ts, StripControlsToSpace(v)}, true
	case float64:
		// A store that sends a JSON number instead of a string. Normalised to a
		// string so the shape is uniform and always encodable. A NON-finite
		// float cannot arrive this way -- JSON has no Inf/NaN literals and a
		// number overflowing float64 fails to decode, taking the whole body
		// with it -- so the finite case is the reachable one and this is belt
		// and braces for any future non-JSON caller.
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return []any{ts, strconv.FormatFloat(v, 'g', -1, 64)}, true
		}
		return []any{ts, strconv.FormatFloat(v, 'f', -1, 64)}, true
	default:
		return nil, false
	}
}

func parsePromQLTime(s, field string) (time.Time, error) {
	if s == "" {
		return time.Time{}, fmt.Errorf("%w: %s is required", ErrInvalidArgs, field)
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}, fmt.Errorf("%w: %s is not RFC3339", ErrInvalidArgs, field)
	}
	return t, nil
}
