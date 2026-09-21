package platform

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// The whole point of the classification: PostgREST maps our PTxxx SQLSTATEs onto
// the matching HTTP status, so the status alone cannot tell a routine
// "that job is no longer yours" from "this platform has no job channel". Both
// arrive as 404.
func TestClassifyDiscriminatesOnTheBodyCodeNotTheStatus(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
		want   Class
	}{
		{"submit for a swept, foreign or answered job", 404,
			`{"code":"PT404","message":"Not found","details":"Job not found"}`, ClassJobGone},
		{"platform without the rpc", 404,
			`{"code":"PGRST202","message":"Could not find the function"}`, ClassUnavailable},
		{"api_base_url pointing at something else", 404, `<html>404</html>`, ClassUnavailable},
		{"revoked or unknown token", 401, `{"code":"PT401","message":"Unauthorized"}`, ClassAuth},
		// A SUSPENDED ORG, not a foreign instance: instance_job_auth answers
		// PT404 for a foreign one, so an org-token holder cannot learn whether
		// a uuid names a real instance somewhere else. PT403 is never raised
		// here at all.
		{"suspended org", 402, `{"code":"PT402","message":"Payment Required"}`, ClassAuth},
		{"our own bad request", 400, `{"code":"PT400","message":"Bad Request"}`, ClassRequest},
		{"platform database blip", 503, `{"code":"PGRST000","message":"unavailable"}`, ClassTransient},
		{"gateway trouble", 502, ``, ClassTransient},
		// Exactly 500, with no PostgREST code: an unhandled exception in the
		// rpc, or a proxy error page. It has to be transient, or a submit is
		// not retried at all and the collection is lost.
		{"a bare 500 with no code", 500, ``, ClassTransient},
		{"rate limited", 429, ``, ClassTransient},
		// The platform hit its own lock bound behind a pull-path consumer, or
		// deadlocked with one. It REVERTS the claim, so the job is queued again
		// and the answer is not lost -- as long as we retry rather than record
		// a failed collection.
		// At a 4xx on purpose. PostgREST maps these onto a 5xx today, which the
		// 5xx arm would catch anyway -- so testing them there would pass whether
		// or not the code is what decides. This MR's rule is that the BODY's
		// code decides, never the status, and these rows are what pins it.
		{"lock not available", 400, `{"code":"55P03","message":"could not obtain lock"}`, ClassTransient},
		{"deadlock", 400, `{"code":"40P01","message":"deadlock detected"}`, ClassTransient},
		{"serialization failure", 400, `{"code":"40001"}`, ClassTransient},
		{"statement cancelled", 400, `{"code":"57014"}`, ClassTransient},
		// A redirect we refused to follow: something in front of the platform is
		// redirecting, which is a wrong api_base_url and not a bad request. A
		// redirect body carries no PostgREST code, whatever else is in it.
		{"a refused redirect", 307, ``, ClassUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(parseAPIError(tc.status, []byte(tc.body))); got != tc.want {
				t.Fatalf("Classify(%d %s) = %v, want %v", tc.status, tc.body, got, tc.want)
			}
		})
	}
}

func TestClassifyTreatsTransportFailuresAsTransient(t *testing.T) {
	if got := Classify(errors.New("dial tcp: connection refused")); got != ClassTransient {
		t.Fatalf("Classify = %v, want ClassTransient", got)
	}
}

// The two 404s must not collapse: doing so takes a healthy instance off the
// channel every time it answers a job the sweep already closed.
func TestTheTwo404sDoNotCollapse(t *testing.T) {
	jobGone := Classify(parseAPIError(404, []byte(`{"code":"PT404"}`)))
	notThere := Classify(parseAPIError(404, []byte(`{"code":"PGRST202"}`)))
	if jobGone == notThere {
		t.Fatal("a routine PT404 and an un-migrated platform classify the same")
	}
}

func TestPollParsesTheReply(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rpc/instance_job_poll" {
			t.Errorf("poll hit %s", r.URL.Path)
		}
		if r.URL.RawQuery != "" {
			t.Errorf("credentials must never reach the query string: %s", r.URL.RawQuery)
		}
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q; PostgREST answers 415 without it", ct)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["instance_id"] != "iid" {
			t.Errorf("poll body = %v, want the instance id", body)
		}
		if _, ok := body["api_token"]; ok {
			t.Errorf("the credential is in the body, where Postgres logs it: %v", body)
		}
		if got := r.Header.Get("access-token"); got != "tok" {
			t.Errorf("access-token header = %q, want the credential", got)
		}
		assertKeys(t, "poll", body, "instance_id", "client_version")
		w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","next_poll_ms":5000,
			"jobs":[{"id":"j1","kind":"aas_collect","args":{"node_name":"n"}}]}`))
	}))
	defer srv.Close()

	// A trailing slash in api_base_url is exactly what a hand-edited config has,
	// and it would otherwise produce `.../general//rpc/instance_job_poll`.
	resp, err := NewClient(srv.URL+"/", "v", time.Second).
		Poll(context.Background(), Credentials{APIToken: "tok", InstanceID: "iid"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.NextPollMS != 5000 || len(resp.Jobs) != 1 {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Jobs[0].ID != "j1" || resp.Jobs[0].Kind != "aas_collect" {
		t.Fatalf("job = %+v", resp.Jobs[0])
	}
	if !strings.Contains(string(resp.Jobs[0].Args), `"node_name":"n"`) {
		t.Fatalf("args did not survive: %s", resp.Jobs[0].Args)
	}
}

func TestSubmitSendsEitherAResultOrAnError(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		json.Unmarshal(raw, &body)
		bodies = append(bodies, body)
		fmt.Fprintf(w, `{"job_id":%q,"status":"done","outcome":%q,"error":null}`,
			body["job_id"], body["outcome"])
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "v", time.Second)
	creds := Credentials{APIToken: "tok", InstanceID: "iid"}
	for _, s := range []Submission{
		{JobID: "j1", Outcome: OutcomeOK, Result: map[string]string{"checkId": "AAS"}, DurationMS: 12},
		{JobID: "j2", Outcome: OutcomeError, Error: "metric store unreachable",
			FailureClass: "store_unreachable"},
		{JobID: "j3", Outcome: OutcomeSkipped, SkipReason: "density"},
	} {
		if err := c.Submit(context.Background(), creds, s); err != nil {
			t.Fatalf("submitting %s: %v", s.Outcome, err)
		}
	}

	// Exactly the fields each outcome allows, and no others: the platform
	// answers PT400 to a contradiction -- an "ok" with an error, a "skipped"
	// with one, a reason without a skip -- and does not consume the job.
	//
	// The EXACT key set, not just the interesting keys. PostgREST resolves an
	// rpc by the argument names in the body, so a dropped or renamed one comes
	// back PGRST202 and the platform-side function signature has to match this
	// list exactly.
	// No api_token: the credential is a header, deliberately -- a body parameter
	// is a bind parameter and Postgres logs those.
	assertKeys(t, "ok", bodies[0],
		"instance_id", "job_id", "client_version", "duration_ms", "outcome", "result")
	assertKeys(t, "error", bodies[1],
		"instance_id", "job_id", "client_version", "duration_ms",
		"outcome", "error", "failure_class")
	assertKeys(t, "skipped", bodies[2],
		"instance_id", "job_id", "client_version", "duration_ms", "outcome", "skip_reason")

	for i, want := range []string{"ok", "error", "skipped"} {
		if bodies[i]["outcome"] != want {
			t.Errorf("body %d outcome = %v, want %q", i, bodies[i]["outcome"], want)
		}
	}
	if bodies[1]["failure_class"] != "store_unreachable" {
		t.Errorf("failure_class = %v", bodies[1]["failure_class"])
	}
	if bodies[2]["skip_reason"] != "density" {
		t.Errorf("skip_reason = %v", bodies[2]["skip_reason"])
	}
}

func assertKeys(t *testing.T, what string, body map[string]any, want ...string) {
	t.Helper()
	got := make([]string, 0, len(body))
	for k := range body {
		got = append(got, k)
	}
	sort.Strings(got)
	sort.Strings(want)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("%s body keys = %v, want %v", what, got, want)
	}
}

// details and hint can echo request parameters back; only the code and the
// generic message are kept.
func TestAPIErrorDropsDetailsAndHint(t *testing.T) {
	err := parseAPIError(400, []byte(`{"code":"PT400","message":"Bad Request",
		"details":"Parameter \"result\" must be at most 1048576 bytes.","hint":"x"}`))
	if strings.Contains(err.Error(), "Parameter") || strings.Contains(err.Error(), "hint") {
		t.Fatalf("the error carries the echoed detail: %v", err)
	}
}

// The org token travels in the request BODY, so Go's own "drop Authorization
// across hosts" protection does not apply: a 307 replays the body verbatim to
// the redirect target. The client must refuse to follow one.
func TestARedirectNeverReplaysTheTokenToAnotherHost(t *testing.T) {
	var leaked []byte
	attacker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		leaked, _ = io.ReadAll(r.Body)
		w.Write([]byte(`{}`))
	}))
	defer attacker.Close()

	platform := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, attacker.URL+"/rpc/instance_job_poll", http.StatusTemporaryRedirect)
	}))
	defer platform.Close()

	_, err := NewClient(platform.URL, "v", time.Second).
		Poll(context.Background(), Credentials{APIToken: "pai-secret", InstanceID: "iid"})
	if err == nil {
		t.Fatal("a redirect was followed and reported success")
	}
	if len(leaked) > 0 {
		t.Fatalf("the token was replayed to the redirect target: %s", leaked)
	}
}

// PostgREST fills `message` from the raw PostgreSQL error, which can echo a
// value back, so it is bounded before it can reach a log line or the health
// file.
func TestTheEchoedMessageIsBounded(t *testing.T) {
	long := strings.Repeat("x", 5000)
	err := parseAPIError(400, []byte(`{"code":"PT400","message":"`+long+`"}`))
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatal("not an APIError")
	}
	// A literal ceiling: "bounded by whatever the constant happens to be" would
	// still pass with the constant raised to 100 kB, which is the failure the
	// bound exists to prevent.
	if maxCodeBytes != 32 {
		t.Fatalf("maxCodeBytes = %d; the runner puts the code in the health file, "+
			"where an unbounded value is an unbounded file", maxCodeBytes)
	}
	if len(apiErr.Message) > 256 {
		t.Fatalf("message is %d bytes; it reaches a log line and the health file", len(apiErr.Message))
	}
	if !strings.HasSuffix(apiErr.Message, "...") {
		t.Fatalf("a truncated message should say so: %q", apiErr.Message)
	}

	// Truncation must land on a rune boundary: a PostgreSQL message is often
	// non-ASCII, and half a rune is mojibake in `docker inspect`.
	// A 3-byte rune, so the byte bound cannot happen to land on a boundary:
	// 200 is not a multiple of 3.
	wide := parseAPIError(400, []byte(`{"code":"PT400","message":"`+strings.Repeat("\u20ac", 4000)+`"}`))
	errors.As(wide, &apiErr)
	if !utf8.ValidString(apiErr.Message) {
		t.Fatalf("truncation cut a rune in half: %q", apiErr.Message)
	}
}

func TestTruncateRunesKeepsWhatItCan(t *testing.T) {
	if got := truncateRunes("short", 200); got != "short" {
		t.Fatalf("truncateRunes = %q, want the string unchanged", got)
	}
	// Invalid UTF-8 BEFORE the bound: walking back to find a valid prefix would
	// strip the whole string to "...", which is worse than keeping it.
	broken := "\xffabcdefghij"
	if got := truncateRunes(broken, 5); got == "..." {
		t.Fatalf("truncateRunes collapsed a string whose invalid byte is not at the bound: %q", got)
	}
}

// PostgREST answers `null` for an rpc that returns SQL NULL, and both `null`
// and `{}` decode into a zero PollResponse without error -- which the runner
// would read as a healthy poll that hands out no work, forever.
func TestAnUnexpectedPollReplyIsAFailure(t *testing.T) {
	for _, body := range []string{`null`, `{}`, `[]`, `"hello"`, `<html>no</html>`} {
		t.Run(body, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Write([]byte(body))
			}))
			defer srv.Close()

			_, err := NewClient(srv.URL, "v", time.Second).
				Poll(context.Background(), Credentials{APIToken: "t", InstanceID: "i"})
			if err == nil {
				t.Fatalf("a %s reply was accepted as a poll", body)
			}
		})
	}
}

// A 4xx that is neither ours nor PostgREST's is still the platform refusing
// what we sent, not something a retry loop fixes.
func TestAnUnrecognisedRefusalIsNotRetried(t *testing.T) {
	if got := Classify(parseAPIError(418, []byte(`{"code":"SOMETHING","message":"no"}`))); got != ClassRequest {
		t.Fatalf("Classify = %v, want ClassRequest", got)
	}
}

// The body cap keeps a runaway reply from being read into memory.
func TestTheResponseBodyIsCapped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Valid JSON on purpose: a body that is merely malformed would fail
		// whether or not the cap is there, and prove nothing.
		w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","next_poll_ms":5000,"jobs":[`))
		for i := 0; i < 200000; i++ {
			if i > 0 {
				w.Write([]byte(","))
			}
			w.Write([]byte(`{"id":"x","kind":"aas_collect","args":{}}`))
		}
		w.Write([]byte(`]}`))
	}))
	defer srv.Close()

	if _, err := NewClient(srv.URL, "v", 5*time.Second).
		Poll(context.Background(), Credentials{APIToken: "t", InstanceID: "i"}); err == nil {
		t.Fatal("a reply far over the cap was accepted whole")
	}
}

// The file's own rule: never wrap an error with the request body, because the
// body carries the token. Enforced here rather than trusted to the runner's
// sanitize(), which is a different package and a different consumer.
func TestATransportErrorNeverCarriesTheToken(t *testing.T) {
	const token = "pai-secret-value-must-not-appear"
	// A port nothing is listening on: the transport fails and the error is the
	// one this package builds.
	_, err := NewClient("http://127.0.0.1:1", "v", time.Second).
		Poll(context.Background(), Credentials{APIToken: token, InstanceID: "i"})
	if err == nil {
		t.Fatal("a dial to a closed port reported success")
	}
	if strings.Contains(err.Error(), token) {
		t.Fatalf("the token is in the error: %v", err)
	}
}

// The platform writes its rejection into a customer-visible column, so the box
// carries those words rather than replacing them with its own.
func TestARefusedSubmitCarriesThePlatformsWords(t *testing.T) {
	const because = "instance_job_apply_collection: result checkId is not AAS"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintf(w, `{"job_id":"j1","status":"failed","outcome":"ok","error":%q}`, because)
	}))
	defer srv.Close()

	err := NewClient(srv.URL, "v", time.Second).
		Submit(context.Background(), Credentials{APIToken: "t", InstanceID: "i"},
			Submission{JobID: "j1", Outcome: OutcomeOK, Result: map[string]string{"checkId": "AAS"}})
	if !errors.Is(err, ErrResultRejected) {
		t.Fatalf("err = %v, want ErrResultRejected", err)
	}
	if !strings.Contains(err.Error(), "checkId is not AAS") {
		t.Fatalf("err = %v; the platform's own words were dropped", err)
	}
}

// The reply is the only thing that says whether the collection landed.
func TestAnUnexpectedSubmitReplyIsAFailure(t *testing.T) {
	for _, body := range []string{`null`, `{}`, `[]`, `"hello"`, `<html>no</html>`,
		`{"status":"done"}`, `{"job_id":"j1"}`,
		// The platform echoes the outcome we named; a different one means it
		// did not understand what we sent.
		`{"job_id":"j1","status":"done","outcome":"skipped"}`,
		// Shadow-free, one per guard: a complete row with no outcome echoed is
		// the rolling-upgrade case (an rpc not yet migrated), and an outcome
		// with no row is the reverse. Each kills exactly one of the two checks.
		`{"job_id":"j1","status":"done"}`,
		`{"outcome":"ok"}`} {
		t.Run(body, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Write([]byte(body))
			}))
			defer srv.Close()

			err := NewClient(srv.URL, "v", time.Second).
				Submit(context.Background(), Credentials{APIToken: "t", InstanceID: "i"},
					Submission{JobID: "j1", Outcome: OutcomeOK, Result: map[string]string{}})
			if err == nil {
				t.Fatalf("a submit reply of %s was read as a landed collection", body)
			}
		})
	}
}

// A clean acceptance is an acceptance.
func TestAnAcceptedSubmitReturnsNoError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"job_id":"j1","status":"done","outcome":"ok","accepted_at":"2026-09-16T00:00:00Z","error":null}`))
	}))
	defer srv.Close()

	if err := NewClient(srv.URL, "v", time.Second).
		Submit(context.Background(), Credentials{APIToken: "t", InstanceID: "i"},
			Submission{JobID: "j1", Outcome: OutcomeOK, Result: map[string]string{}}); err != nil {
		t.Fatalf("a clean acceptance was reported as a failure: %v", err)
	}
}

// A transient SQLSTATE and the routine PT404 must not be confused: one is worth
// retrying because the claim was reverted, the other never is.
func TestATransientLockIsNotConfusedWithAGoneJob(t *testing.T) {
	lock := Classify(parseAPIError(400, []byte(`{"code":"55P03"}`)))
	gone := Classify(parseAPIError(404, []byte(`{"code":"PT404"}`)))
	if lock != ClassTransient {
		t.Fatalf("55P03 = %v, want ClassTransient: the platform reverted the claim", lock)
	}
	if gone != ClassJobGone {
		t.Fatalf("PT404 = %v, want ClassJobGone", gone)
	}
	if lock == gone {
		t.Fatal("a reverted claim and a job that is not ours classify the same")
	}
}

// The credential travels as a header, never as a body parameter: a body
// parameter is a bind parameter and Postgres writes those into the server log
// for any statement over log_min_duration_statement.
func TestTheCredentialIsAHeaderNotABodyParameter(t *testing.T) {
	var sawHeader string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawHeader = r.Header.Get("access-token")
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, &body)
		fmt.Fprintf(w, `{"job_id":"j1","status":"done","outcome":%q,"error":null}`, body["outcome"])
	}))
	defer srv.Close()

	err := NewClient(srv.URL, "v", time.Second).
		Submit(context.Background(), Credentials{APIToken: "pai-secret", InstanceID: "i"},
			Submission{JobID: "j1", Outcome: OutcomeOK, Result: map[string]string{}})
	if err != nil {
		t.Fatal(err)
	}
	if sawHeader != "pai-secret" {
		t.Fatalf("access-token header = %q, want the credential", sawHeader)
	}
	for k, v := range body {
		if s, ok := v.(string); ok && s == "pai-secret" {
			t.Fatalf("the credential is in the body under %q", k)
		}
	}
}

// A submission that names no outcome is refused HERE. The platform answers
// PT400 and does not consume the job, so it would sit 'running' until the
// hourly sweep -- a local error is the cheaper failure.
func TestASubmissionWithNoOutcomeIsRefusedLocally(t *testing.T) {
	sent := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		sent = true
		w.Write([]byte(`{"job_id":"j1","status":"done","outcome":"ok","error":null}`))
	}))
	defer srv.Close()

	err := NewClient(srv.URL, "v", time.Second).
		Submit(context.Background(), Credentials{APIToken: "t", InstanceID: "i"},
			Submission{JobID: "j1", Result: map[string]string{}})
	if !errors.Is(err, ErrUnknownOutcome) {
		t.Fatalf("err = %v, want ErrUnknownOutcome", err)
	}
	if sent {
		t.Fatal("a submission with no outcome was sent to the platform anyway")
	}
}

// A submission this package refused to send never reached the platform and
// never will: retrying it would burn three attempts on the same refusal.
func TestARefusalToSendIsNotRetryable(t *testing.T) {
	if got := Classify(fmt.Errorf("x: %w", ErrUnknownOutcome)); got != ClassRequest {
		t.Fatalf("Classify(ErrUnknownOutcome) = %v, want ClassRequest", got)
	}
}
