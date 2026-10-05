package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const testVerifyToken = "dblab-verify-must-not-appear"

// dblabHarness wires a Runner to a fake platform and a fake DBLab engine
// through the real config file, so the test exercises the production wiring
// (platform-all#805).
type dblabHarness struct {
	runner *Runner
	logs   *strings.Builder
	// mu guards every counter below. The DBLab arm runs a claimed batch on a
	// worker pool (platform-all#816), so the platform and engine handlers are
	// entered concurrently -- without this `submits` loses entries outright and
	// the race detector fires on the handler-versus-handler writes.
	//
	// It does NOT fire on a TEST-goroutine read after tick() returns: the HTTP
	// round trip and wg.Wait() give it a happens-before edge. So going through
	// the accessors below is a discipline with no CI enforcement, and the next
	// assertion moved to BEFORE tick returns is the one that breaks silently.
	mu      sync.Mutex
	polls   int
	submits []map[string]any
	// engineCalls counts requests that reached the engine, which is how a test
	// tells a retried read from a write that was answered once.
	engineCalls int
	engine      func(h *dblabHarness, w http.ResponseWriter, r *http.Request)
}

// snapshotSubmits copies the answers recorded so far. Callers read the copy, so
// a worker still finishing cannot mutate the slice under an assertion.
func (h *dblabHarness) snapshotSubmits() []map[string]any {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]map[string]any, len(h.submits))
	copy(out, h.submits)
	return out
}

func (h *dblabHarness) engineCallCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.engineCalls
}

// healthStamp reads the verdict the loop has published, the way the sequential
// arm's TestHealthIsRefreshedBetweenJobs reads it: from the file, so the
// assertion is about what a healthcheck would actually see.
func (h *dblabHarness) healthStamp(t *testing.T) struct {
	UpdatedAt   time.Time `json:"updated_at"`
	NextCheckBy time.Time `json:"next_check_by"`
} {
	t.Helper()
	var state struct {
		UpdatedAt   time.Time `json:"updated_at"`
		NextCheckBy time.Time `json:"next_check_by"`
	}
	raw, err := os.ReadFile(h.runner.healthPath)
	if err != nil {
		t.Fatalf("health file unreadable: %v", err)
	}
	if err := json.Unmarshal(raw, &state); err != nil {
		t.Fatalf("health file unparseable: %v", err)
	}
	return state
}

func (h *dblabHarness) pollCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.polls
}

func newDBLabHarness(t *testing.T, jobs string) *dblabHarness {
	t.Helper()
	h := &dblabHarness{logs: &strings.Builder{}}

	platformSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "dblab_job_poll"):
			h.mu.Lock()
			h.polls++
			n := h.polls
			h.mu.Unlock()
			if n > 1 {
				w.Write([]byte(`{"server_time":"2026-09-25T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			w.Write([]byte(`{"server_time":"2026-09-25T00:00:00Z","jobs":` + jobs + `,"next_poll_ms":5000}`))
		case strings.HasSuffix(r.URL.Path, "dblab_job_submit"):
			raw, _ := io.ReadAll(r.Body)
			r.Body = io.NopCloser(bytes.NewReader(raw))
			var body map[string]any
			json.Unmarshal(raw, &body)
			h.mu.Lock()
			h.submits = append(h.submits, body)
			h.mu.Unlock()
			w.Write(echoDBLabSubmit(body))
		default:
			// The monitoring rpcs must never be called by a DBLab box: a wrong
			// turn here would have the box poll for another target's work with
			// its own credential.
			t.Errorf("a DBLab box called %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(platformSrv.Close)

	engineSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		h.engineCalls++
		h.mu.Unlock()
		if h.engine != nil {
			h.engine(h, w, r)
			return
		}
		w.Write([]byte(`{"pools":[{"fileSystem":{"used":4096}}]}`))
	}))
	t.Cleanup(engineSrv.Close)

	dir := t.TempDir()
	configPath := filepath.Join(dir, ".pgwatch-config")
	content := "dblab_token=" + testToken + "\napi_base_url=" + platformSrv.URL +
		"\ndblab_url=" + engineSrv.URL +
		"\ndblab_verify_token=" + testVerifyToken + "\n"
	if err := os.WriteFile(configPath, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", configPath)
	// A value on the machine running the tests must not decide which channel the
	// box serves -- and a closed loopback port rather than blank, so a
	// precedence regression fails here instead of dialling production.
	t.Setenv("PGAI_INSTANCE_ID", "")
	t.Setenv("PGAI_MONITORING_INSTANCE_ID", "")
	t.Setenv("PGAI_DBLAB_TOKEN", "")
	t.Setenv("PGAI_DBLAB_URL", "http://127.0.0.1:1")
	t.Setenv("PGAI_DBLAB_VERIFY_TOKEN", "")
	t.Setenv("PGAI_API_BASE_URL", "http://127.0.0.1:1")
	t.Setenv("PROMETHEUS_URL", "http://127.0.0.1:1")
	t.Setenv("VM_AUTH_USERNAME", "")
	t.Setenv("VM_AUTH_PASSWORD", "")

	h.runner = New(filepath.Join(dir, "health"), "test")
	h.runner.sleep = func(ctx context.Context, _ time.Duration) bool { return ctx.Err() == nil }

	log.SetOutput(h.logs)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	return h
}

// echoDBLabSubmit answers a submit the way v1.dblab_job_submit does, captured
// from the rpc on a real database rather than written from the header:
//
//	{"error": null, "job_id": "<the job just answered>", "status": "done",
//	 "outcome": "ok", "accepted_at": "2026-09-26T12:29:14.703881+00:00"}
//
// job_id is the row's, not a constant, because that is what the rpc does. The
// client does not compare it today, so this fidelity buys no assertion by
// itself -- the batch tests assert on what the box SENT. It is here so a reply
// fixture cannot quietly become a shape the platform never produces.
func echoDBLabSubmit(body map[string]any) []byte {
	outcome, _ := body["outcome"].(string)
	status := "done"
	if outcome == "error" {
		status = "failed"
	}
	jobID, _ := body["job_id"].(string)
	return []byte(fmt.Sprintf(
		`{"error":null,"job_id":%q,"status":%q,"outcome":%q,"accepted_at":"2026-09-26T12:29:14.703881+00:00"}`,
		jobID, status, outcome))
}

func dblabJob(action, method string) string {
	return `[{"id":"j1","kind":"dblab_call","args":{"method":"` + method +
		`","action":"` + action + `","purpose":"data_usage"}}]`
}

// The whole loop on a DBLab box: poll the DBLab rpc, call the local engine, post
// the engine's reply back.
func TestTickRunsADBLabCallAndSubmitsTheEngineReply(t *testing.T) {
	var gotToken, gotPath string
	h := newDBLabHarness(t, dblabJob("/status", "GET"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		gotToken = r.Header.Get("Verification-Token")
		gotPath = r.URL.Path
		w.Write([]byte(`{"pools":[{"fileSystem":{"used":4096}}]}`))
	}

	h.runner.tick(context.Background())

	if gotPath != "/status" {
		t.Fatalf("the engine was called at %q, want /status", gotPath)
	}
	if gotToken != testVerifyToken {
		t.Fatalf("the engine call carried no verification token")
	}
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if h.submits[0]["outcome"] != "ok" {
		t.Fatalf("outcome = %v, want ok: %v", h.submits[0]["outcome"], h.submits[0])
	}
	// The engine's object, submitted BARE: the platform stores it verbatim and
	// public.data_usage_collect reads `fileSystem` out of it.
	result, ok := h.submits[0]["result"].(map[string]any)
	if !ok {
		t.Fatalf("result = %#v, want the engine's object", h.submits[0]["result"])
	}
	if _, ok := result["pools"]; !ok {
		t.Fatalf("result = %v, want the engine's reply relayed", result)
	}
	// THE BOX NAMES NO INSTANCE, on either call. The platform derives the engine
	// from the credential, and PostgREST resolves an rpc by its body keys -- so an
	// extra key would not be ignored, it would match no function and 404.
	for _, k := range []string{"dblab_instance_id", "instance_id"} {
		if _, ok := h.submits[0][k]; ok {
			t.Fatalf("submit carried %q: %v", k, h.submits[0])
		}
	}
}

// A POST, a PATCH or a DELETE against the engine is NOT idempotent. A
// `POST /clone` whose answer never came back may well have created the clone, so
// re-sending it would create a second one on the customer's disk to recover from
// a timeout.
func TestAFailedWriteIsAnsweredRatherThanRepeated(t *testing.T) {
	h := newDBLabHarness(t, dblabJob("/clone", "POST"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"message":"engine is busy"}`))
	}

	h.runner.tick(context.Background())

	if h.engineCalls != 1 {
		t.Fatalf("the engine was called %d times for a write, want exactly 1", h.engineCalls)
	}
	if len(h.submits) != 1 || h.submits[0]["outcome"] != "error" {
		t.Fatalf("submits = %v, want one answered failure", h.submits)
	}
}

// A CALL THAT OVERRUNS NAMES THE ENGINE CALL, NOT A COLLECTION.
//
// The text goes to the platform in `error` and is what the person waiting on
// `pgai dblab clone create` reads. classifyFailure's deadline arm was written
// for the monitoring channel, so this box answered a timed-out clone create
// with "collection exceeded the local time budget" -- naming work a DBLab box
// never does, and pointing whoever is debugging it at the metric store.
//
// The CLASS is deliberately unchanged: the platform stores and aggregates on
// failure_class, and a timeout is a timeout whichever channel produced it.
//
// Literals on both sides. The engine call count is here only to show the
// overrun was not somehow answered without a call -- it does NOT pin the write
// rule: measured, `dblabWriteAttempts = 3` leaves this test green, because a
// budget that has already expired takes the `ctx.Err() != nil` exit before any
// retry. TestAFailedWriteIsAnsweredRatherThanRepeated is what pins that, and
// it does go red on the same mutation.
func TestATimedOutEngineCallIsNotReportedAsACollection(t *testing.T) {
	h := newDBLabHarness(t, dblabJob("/clone", "POST"))

	// Released from a cleanup as well as at the end: httptest's Close waits for
	// outstanding requests, so a t.Fatalf below with the handler still parked
	// would hang the package instead of failing one test.
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseEngine := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseEngine)

	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		<-release
		w.Write([]byte(`{"id":"c1"}`))
	}
	// Short enough that the budget, not the test, decides when the call ends.
	h.runner.budget = 50 * time.Millisecond

	h.runner.tick(context.Background())
	releaseEngine()

	subs := h.snapshotSubmits()
	if len(subs) != 1 {
		t.Fatalf("submits = %v, want the overrun answered exactly once", subs)
	}
	if got := subs[0]["error"]; got != "the engine call exceeded the local time budget" {
		t.Fatalf("error = %q, want the ENGINE CALL named -- a DBLab box runs no collections", got)
	}
	if got := subs[0]["failure_class"]; got != "timeout" {
		t.Fatalf("failure_class = %q, want %q: the class is the platform's contract and does not move", got, "timeout")
	}
	if h.engineCallCount() != 1 {
		t.Fatalf("the engine was called %d times for a write that overran, want exactly 1", h.engineCallCount())
	}
}

// A PATCH is a WRITE too (#393), and the ladder is chosen on the METHOD -- so
// the only thing keeping `PATCH /clone/{id}` off the read ladder is that it is
// not a GET. A 500 here would be retried three times for a read; the engine may
// already have applied the protection change before the answer was lost, so it
// is attempted once and answered as failed, exactly like a POST.
func TestAFailedPatchIsAnsweredRatherThanRepeated(t *testing.T) {
	h := newDBLabHarness(t,
		`[{"id":"j1","kind":"dblab_call","args":{"method":"PATCH","action":"/clone/c1","data":{"protected":true},"purpose":"api_call"}}]`)
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"message":"engine is busy"}`))
	}

	h.runner.tick(context.Background())

	if h.engineCalls != 1 {
		t.Fatalf("the engine was called %d times for a PATCH, want exactly 1", h.engineCalls)
	}
	if len(h.submits) != 1 || h.submits[0]["outcome"] != "error" {
		t.Fatalf("submits = %v, want one answered failure", h.submits)
	}
}

// ...while a read is safe to repeat, so a momentary engine failure does not cost
// the job.
func TestAFailedReadIsRetriedAndCanStillSucceed(t *testing.T) {
	h := newDBLabHarness(t, dblabJob("/status", "GET"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		if h.engineCalls < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Write([]byte(`{"pools":[]}`))
	}

	h.runner.tick(context.Background())

	if h.engineCalls < 2 {
		t.Fatalf("the engine was called %d times for a read, want a retry", h.engineCalls)
	}
	if len(h.submits) != 1 || h.submits[0]["outcome"] != "ok" {
		t.Fatalf("submits = %v, want the retry to have landed", h.submits)
	}
}

// A 4xx is the call being wrong, so retrying it would be wrong again -- and the
// engine's own words name what was wrong, which is the whole diagnostic.
func TestAnEngineRefusalIsReportedOnceWithItsOwnMessage(t *testing.T) {
	h := newDBLabHarness(t, dblabJob("/clone/secret-clone-name", "GET"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"message":"clone not found"}`))
	}

	h.runner.tick(context.Background())

	if h.engineCalls != 1 {
		t.Fatalf("a 404 was retried %d times", h.engineCalls)
	}
	if len(h.submits) != 1 {
		t.Fatalf("submits = %v, want one", h.submits)
	}
	// The STATUS rides in the class (#402). Without it the Console cannot tell a
	// deleted clone's 404 from a broken engine's 500 -- both arrive as
	// `status: failed` inside an HTTP 200 -- and spun forever on the 404.
	if h.submits[0]["failure_class"] != "engine_error_404" {
		t.Fatalf("failure_class = %v, want engine_error_404", h.submits[0]["failure_class"])
	}
	errText, _ := h.submits[0]["error"].(string)
	if !strings.Contains(errText, "clone not found") {
		t.Fatalf("error = %q, want the engine's message", errText)
	}
}

// The args are the PLATFORM's, so a shape this build cannot act on is permanent:
// retrying it three times would burn the job budget to fail identically.
func TestBadArgsFailImmediately(t *testing.T) {
	h := newDBLabHarness(t, `[{"id":"j1","kind":"dblab_call","args":{"method":"GET","action":"//evil.example.com/x"}}]`)

	h.runner.tick(context.Background())

	if h.engineCalls != 0 {
		t.Fatalf("the engine was called %d times for args that never parsed", h.engineCalls)
	}
	if len(h.submits) != 1 || h.submits[0]["failure_class"] != "invalid_args" {
		t.Fatalf("submits = %v, want one invalid_args failure", h.submits)
	}
}

// Neither the org token nor the engine's verification token may reach a log
// line, and neither may the job's action -- it carries clone and branch names.
func TestNoSecretOrActionReachesTheLog(t *testing.T) {
	h := newDBLabHarness(t, dblabJob("/clone/secret-clone-name", "GET"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}

	h.runner.tick(context.Background())

	logs := h.logs.String()
	for _, secret := range []string{testToken, testVerifyToken, "secret-clone-name"} {
		if strings.Contains(logs, secret) {
			t.Fatalf("the log carries %q:\n%s", secret, logs)
		}
	}
}

// platform-all#815, end to end on the box: a DELETE the engine answered with a 200 and no
// body is submitted as a SUCCESS whose result is null.
//
// Before this, the box answered `outcome: "error"` with
// "the engine returned an empty reply", and since a DELETE is never retried the
// console showed a red error over a clone that had already been destroyed.
func TestAnEmptyEngineReplyIsSubmittedAsASuccessfulNull(t *testing.T) {
	h := newDBLabHarness(t, dblabJob("/clone/abc", "DELETE"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		// Exactly what engine routes.go does for destroyClone: it returns
		// without calling api.Write* at all.
	}

	h.runner.tick(context.Background())

	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if h.submits[0]["outcome"] != "ok" {
		t.Fatalf("outcome = %v, want ok: %v", h.submits[0]["outcome"], h.submits[0])
	}
	// A JSON null, which PostgREST v9.0.1 binds to a SQL NULL on
	// v1.dblab_job_submit's jsonb parameter -- measured on a rig, because
	// public.data_usage_collect's `result is not null` gate turns on it, and
	// because the console reads the pull path's null for the same call.
	if raw, ok := h.submits[0]["result"]; !ok || raw != nil {
		t.Fatalf("result = %#v (present=%v), want a null", raw, ok)
	}
	if _, ok := h.submits[0]["error"]; ok {
		t.Fatalf("an empty reply was reported as a failure: %v", h.submits[0])
	}
}

// And the other half of platform-all#815 through the same loop: a YAML body reaches the
// platform as a storable envelope naming its content type, instead of the
// failure the rig measured ("the engine reply is not JSON").
func TestAYamlEngineReplyIsSubmittedAsAnEnvelope(t *testing.T) {
	const body = "server:\n    verificationToken: \"****\"\n    port: 2345\n"
	h := newDBLabHarness(t, dblabJob("/admin/config.yaml", "GET"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
		w.Write([]byte(body))
	}

	h.runner.tick(context.Background())

	if len(h.submits) != 1 || h.submits[0]["outcome"] != "ok" {
		t.Fatalf("submits = %v, want one ok", h.submits)
	}
	env, ok := h.submits[0]["result"].(map[string]any)
	if !ok {
		t.Fatalf("result = %#v, want an object", h.submits[0]["result"])
	}
	carried, ok := env["pgai_body"].(map[string]any)
	if !ok {
		t.Fatalf("result = %v, want a pgai_body envelope", env)
	}
	if carried["body"] != body {
		t.Errorf("body = %#v, want %#v", carried["body"], body)
	}
	if carried["content_type"] != "application/yaml; charset=utf-8" {
		t.Errorf("content_type = %#v", carried["content_type"])
	}
	if carried["encoding"] != "text" {
		t.Errorf("encoding = %#v, want text", carried["encoding"])
	}
}

// A body that cannot be read is still the ENGINE's failure. The read error
// carries no sentinel unless client.go gives it one, and classifyFailure's
// default arm then names the metric store -- a component a DBLab box does not
// have, so the operator is pointed at something that is not there.
func TestATruncatedEngineBodyIsReportedAgainstTheEngine(t *testing.T) {
	h := newDBLabHarness(t, dblabJob("/status", "GET"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		// Promise more than is sent, then abort: the client's ReadAll fails
		// mid-body, which is what a broken chunked encoding looks like here.
		w.Header().Set("Content-Length", "4096")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("{\"pool"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		panic(http.ErrAbortHandler)
	}

	h.runner.tick(context.Background())

	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if h.submits[0]["outcome"] != "error" {
		t.Fatalf("outcome = %v, want error: %v", h.submits[0]["outcome"], h.submits[0])
	}
	if got := h.submits[0]["failure_class"]; got != "engine_unreachable" {
		t.Errorf("failure_class = %v, want engine_unreachable", got)
	}
	if got, _ := h.submits[0]["error"].(string); strings.Contains(got, "metric store") {
		t.Errorf("error = %q, which names a component a DBLab box does not have", got)
	}
}

// The last reply-shape failure this channel has left, and both lines that
// handle it were touched here. An oversize reply is PERMANENT for this
// payload: a retry re-reads the same body to fail the same way, and the
// default classification arm names the metric store -- which a DBLab box does
// not have, the same misnomer the truncated-body case above exists to prevent.
func TestAnOversizeEngineReplyIsNotRetriedAndNamesTheEngine(t *testing.T) {
	h := newDBLabHarness(t, dblabJob("/metrics", "GET"))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		// Control bytes: valid UTF-8 and not NUL, so the text arm takes them
		// and each becomes six of \u00XX escape -- far past the submit cap
		// while the raw body stays well under the read cap.
		w.Write(bytes.Repeat([]byte("\x01"), 1<<18))
	}

	h.runner.tick(context.Background())

	// A GET is retried three times when a retry could help. This one cannot.
	if h.engineCalls != 1 {
		t.Errorf("the engine was called %d times, want 1", h.engineCalls)
	}
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if got := h.submits[0]["failure_class"]; got != "oversize_reply" {
		t.Errorf("failure_class = %v, want oversize_reply", got)
	}
	if got, _ := h.submits[0]["error"].(string); strings.Contains(got, "metric store") {
		t.Errorf("error = %q, which names a component a DBLab box does not have", got)
	}
}
