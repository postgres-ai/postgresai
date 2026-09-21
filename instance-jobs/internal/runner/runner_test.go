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
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"errors"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/collect"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/config"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/platform"
)

// The worst case wall clock for one job and one answer, spelled out here rather
// than read from submitBudget. healthDeadline is DEFINED as
// `next + budget + submitBudget + slack`, so an assertion phrased in terms of
// submitBudget cancels it off both sides and holds for any value including
// zero -- it checks an identity rather than the composition. These are the
// arithmetic, done once, by hand:
//
//	collection  15m0s   the job budget
//	8 attempts   4m0s   8 x the 30s per-request timeout
//	7 backoffs   4m40s  10s + 20s + ... + 70s
//
// They are a FLOOR, and TestTheSubmitBudgetOutlastsAContendedBatch fails when
// the constants move away from them: re-derive these by hand when it does,
// rather than widening them until it passes.
const (
	worstCaseJob    = 15 * time.Minute
	worstCaseSubmit = 8*time.Minute + 40*time.Second
)

// outcomeOf reads back the outcome a submit named, so a fake platform can echo
// it the way the real one does.
func outcomeOf(r *http.Request) string {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return ""
	}
	r.Body = io.NopCloser(bytes.NewReader(raw))
	var body struct {
		Outcome string `json:"outcome"`
	}
	json.Unmarshal(raw, &body)
	return body.Outcome
}

// echoOutcome is the ordinary accepted reply.
func echoOutcome(r *http.Request) []byte {
	return []byte(fmt.Sprintf(`{"job_id":"j1","status":"done","outcome":%q,"error":null}`,
		outcomeOf(r)))
}

// testToken is distinctive on purpose: a log line that merely says "token" must
// not fail the leak test, but the value itself must never appear.
const testToken = "pai-secret-value-must-not-appear"

// harness wires a Runner to a fake platform and a fake metric store through the
// real config file, so the test exercises the production wiring.
type harness struct {
	runner     *Runner
	healthPath string
	logs       *strings.Builder
	polls      int
	submits    []map[string]any
	// storeStatus lets a test flip the fake store from failing to working
	// without rebuilding the harness.
	storeStatus int
	// submitReply overrides the body the fake platform answers a submit with.
	submitReply string
	// submitHandler overrides the submit answer entirely.
	submitHandler func(http.ResponseWriter, *http.Request)
}

func newHarness(t *testing.T, configured bool, handle func(h *harness, w http.ResponseWriter, r *http.Request)) *harness {
	t.Helper()
	h := &harness{logs: &strings.Builder{}}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			h.polls++
		}
		if strings.HasSuffix(r.URL.Path, "instance_job_submit") {
			raw, _ := io.ReadAll(r.Body)
			// Put it back: the per-test handler reads it too, to echo the
			// outcome the way the real platform does.
			r.Body = io.NopCloser(bytes.NewReader(raw))
			var body map[string]any
			json.Unmarshal(raw, &body)
			h.submits = append(h.submits, body)
		}
		handle(h, w, r)
	}))
	t.Cleanup(srv.Close)

	dir := t.TempDir()
	configPath := filepath.Join(dir, ".pgwatch-config")
	content := "api_key=" + testToken + "\napi_base_url=" + srv.URL + "\n"
	if configured {
		content += "instance_id=11111111-1111-1111-1111-111111111111\n"
	}
	if err := os.WriteFile(configPath, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", configPath)
	t.Setenv("PROMETHEUS_URL", srv.URL)
	// These decide whether the instance counts as configured, and .env.example
	// tells developers to set them, so a value on the machine running the tests
	// must not decide the outcome.
	t.Setenv("PGAI_INSTANCE_ID", "")
	t.Setenv("PGAI_MONITORING_INSTANCE_ID", "")
	// A closed loopback port, not blank: if the file ever stopped winning over
	// the environment, these tests would otherwise dial the PRODUCTION platform
	// with a fake token. Here a precedence regression fails locally and fast.
	t.Setenv("PGAI_API_BASE_URL", "http://127.0.0.1:1")
	t.Setenv("VM_AUTH_USERNAME", "")
	t.Setenv("VM_AUTH_PASSWORD", "")

	h.healthPath = filepath.Join(dir, "health")
	h.runner = New(h.healthPath, "test")
	// Never actually sleep inside a test.
	h.runner.sleep = func(ctx context.Context, _ time.Duration) bool { return ctx.Err() == nil }

	log.SetOutput(h.logs)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	return h
}

// health reads the verdict on the runner's own clock, so a test driving a fake
// one is not measured against the wall.
func (h *harness) health(t *testing.T) error {
	t.Helper()
	return CheckHealth(h.healthPath, h.runner.now())
}

// --- the loop -----------------------------------------------------------------

// One job, run and answered, with the platform's own interval honoured.
func TestTickRunsAJobAndSubmitsTheAnswer(t *testing.T) {
	// The window sits entirely before its retention floor, so the job resolves
	// without the store being touched at all.
	args := `{"cluster_name":"c","node_name":"n","vcpus":2,
		"period_start":"2020-01-01T00:00:00Z","period_end":"2020-01-02T00:00:00Z",
		"window_start":"2026-01-01T00:00:00Z"}`
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args)
			return
		}
		w.Write(echoOutcome(r))
	})

	clock := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	h.runner.now = func() time.Time {
		clock = clock.Add(250 * time.Millisecond)
		return clock
	}
	// Duration comes off the monotonic seam, not the wall clock, so it needs
	// its own stub -- a real sub-millisecond job would report 0 and make the
	// "not hard-wired to 0" assertion below meaningless.
	var elapsedFrom time.Time
	h.runner.elapsed = func(t time.Time) time.Duration {
		elapsedFrom = t
		return 250 * time.Millisecond
	}

	wait := h.runner.tick(context.Background())
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	// A skip names itself as one: it is neither a payload nor a failure, and
	// the platform rejects a submit that does not say which of the three it is.
	if _, ok := h.submits[0]["result"]; ok {
		t.Errorf("a skip was submitted as a result: %v", h.submits[0])
	}
	if _, ok := h.submits[0]["error"]; ok {
		t.Errorf("a skip was submitted as an error: %v", h.submits[0])
	}
	if h.submits[0]["outcome"] != "skipped" {
		t.Fatalf("outcome = %v, want skipped", h.submits[0]["outcome"])
	}
	if h.submits[0]["skip_reason"] != "retention" {
		t.Fatalf("skip_reason = %v, want retention", h.submits[0]["skip_reason"])
	}
	// A value, not just the key: duration_ms hard-wired to 0 would look the same.
	if d, ok := h.submits[0]["duration_ms"].(float64); !ok || d <= 0 {
		t.Errorf("duration_ms = %v, want the measured duration", h.submits[0]["duration_ms"])
	}
	// And it must be measured from a MONOTONIC reading. r.now() is
	// time.Now().UTC(), and .UTC() strips the monotonic part, so a backward NTP
	// step would make the duration negative.
	//
	// `==`, not Equal(): Round(0) preserves the INSTANT and drops only the
	// monotonic reading, so Equal() is true for every time.Time and would make
	// this vacuous. Struct comparison is the only thing that sees the reading.
	if elapsedFrom.IsZero() {
		t.Fatal("the duration seam was never called")
	}
	if elapsedFrom == elapsedFrom.Round(0) {
		t.Errorf("duration measured from a wall-clock reading (%v): an NTP step "+
			"would make it negative", elapsedFrom)
	}
	// 5000ms +/- 20%.
	if wait < 4*time.Second || wait > 6*time.Second {
		t.Fatalf("wait = %v, want the platform's 5s interval with at most 20%% jitter", wait)
	}
	if err := h.health(t); err != nil {
		t.Fatalf("health after a good tick: %v", err)
	}
}

// Jobs are answered one at a time, each submitted before the next is started:
// the platform stamps started_at at pickup and sweeps a job still running after
// an hour, so a batch answered late has its tail refused.
func TestJobsAreAnsweredOneAtATime(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
		"period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`
	var order []string
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			order = append(order, "poll")
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s},
				{"id":"j2","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args, args)
			return
		}
		order = append(order, "submit")
		w.Write(echoOutcome(r))
	})

	h.runner.tick(context.Background())
	want := []string{"poll", "submit", "submit"}
	if len(order) != len(want) {
		t.Fatalf("call order = %v, want %v", order, want)
	}
	if h.submits[0]["job_id"] != "j1" || h.submits[1]["job_id"] != "j2" {
		t.Fatalf("answers arrived out of order: %v", h.submits)
	}
}

// --- health -------------------------------------------------------------------

// An instance whose profile was enabled but whose id was never written must not
// look green: that is exactly the dead channel `mon health` has to surface. It
// still idles instead of crash-looping, and says so at most once an hour.
func TestUnconfiguredInstanceIdlesUnhealthilyAndLogsOncePerHour(t *testing.T) {
	h := newHarness(t, false, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		t.Error("an unconfigured instance must not call the platform")
		w.WriteHeader(http.StatusInternalServerError)
	})

	for i := 0; i < 3; i++ {
		if wait := h.runner.tick(context.Background()); wait < 8*time.Minute || wait > 12*time.Minute {
			t.Fatalf("idle wait = %v, want about the default poll interval", wait)
		}
	}
	if h.polls != 0 {
		t.Fatalf("polled %d times while unconfigured", h.polls)
	}
	if err := h.health(t); err == nil {
		t.Fatal("an idling, unprovisioned instance reported healthy")
	} else if !strings.Contains(err.Error(), "instance_id") {
		t.Fatalf("health reason does not name what is missing: %v", err)
	}
	if got := strings.Count(h.logs.String(), "idle:"); got != 1 {
		t.Fatalf("logged the idle line %d times in three ticks, want 1", got)
	}
}

// A rejected credential is not fixed by polling harder.
func TestRejectedCredentialBacksOffAndFlipsHealth(t *testing.T) {
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"code":"PT401","message":"Unauthorized"}`))
	})

	// Literals, not the constant: a loop written in terms of unhealthyAfter
	// cannot notice unhealthyAfter changing, and at the default interval each
	// extra failure is another ten minutes before an operator sees red.
	const flipsAt = 3
	// longBackoff and defaultPollInterval are both ten minutes by specification
	// (plan §4: "a genuine missing-function response backs off ten minutes"),
	// so no runtime assertion can tell the refusal paths from the ordinary one
	// -- swapping the constants is an equivalent mutant. What CAN be pinned is
	// the value itself, so shortening the poll interval cannot silently drag
	// the refusal paths down with it.
	if longBackoff != 10*time.Minute {
		t.Fatalf("longBackoff = %v, want the ten minutes the plan specifies", longBackoff)
	}
	if got := h.runner.pollError(&platform.APIError{StatusCode: 401, Code: "PT401"}); got != longBackoff {
		t.Fatalf("a rejected credential backs off %v, want longBackoff", got)
	}
	h.runner.consecutiveFailures = 0
	for i := 1; i <= flipsAt; i++ {
		wait := h.runner.tick(context.Background())
		if wait < 8*time.Minute {
			t.Fatalf("wait after a rejected credential = %v, want the long backoff", wait)
		}
		err := h.health(t)
		if i < flipsAt && err != nil {
			t.Fatalf("health flipped after %d failure(s), want %d (unhealthyAfter is %d): %v",
				i, flipsAt, unhealthyAfter, err)
		}
		if i == flipsAt && err == nil {
			t.Fatalf("health still green after %d consecutive failed polls (unhealthyAfter is %d)",
				i, unhealthyAfter)
		}
	}
}

func TestOneGoodPollClearsTheFailureCount(t *testing.T) {
	fail := true
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		if fail {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
	})
	h.runner.tick(context.Background())
	h.runner.tick(context.Background())
	fail = false
	h.runner.tick(context.Background())
	if h.runner.consecutiveFailures != 0 {
		t.Fatalf("consecutiveFailures = %d after a good poll", h.runner.consecutiveFailures)
	}
	if err := h.health(t); err != nil {
		t.Fatalf("health after recovery: %v", err)
	}
}

// A PT404 on submit is routine -- swept, foreign, answered or expired -- and
// must not be treated as an outage.
func TestJobGoneOnSubmitIsRoutine(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
		"period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args)
			return
		}
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"code":"PT404","message":"Not found","details":"Job not found"}`))
	})

	// Routine, repeatedly: a stuck sweep closes jobs in batches, so several in a
	// row is an ordinary condition and must not flip the container.
	for i := 1; i <= unhealthyAfter; i++ {
		h.polls = 0
		h.runner.tick(context.Background())
		if err := h.health(t); err != nil {
			t.Fatalf("PT404 number %d took the instance off the channel: %v", i, err)
		}
	}
	if h.runner.consecutiveFailures != 0 {
		t.Fatalf("consecutiveFailures = %d after a routine PT404", h.runner.consecutiveFailures)
	}
	if h.runner.consecutiveJobFailures != 0 {
		t.Fatalf("consecutiveJobFailures = %d after a routine PT404",
			h.runner.consecutiveJobFailures)
	}
	if !strings.Contains(h.logs.String(), "no longer open") {
		t.Fatalf("logs do not record the routine case: %s", h.logs.String())
	}
}

func TestHealthCheckFailsWhenTheLoopStopsWriting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "health")
	now := time.Now().UTC()
	if err := writeHealth(path, true, "", now, now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if err := CheckHealth(path, now); err != nil {
		t.Fatalf("fresh health: %v", err)
	}
	if err := CheckHealth(path, now.Add(2*time.Minute)); err == nil {
		t.Fatal("a wedged loop kept reporting healthy past its own deadline")
	}
	if err := CheckHealth(filepath.Join(t.TempDir(), "absent"), now); err == nil {
		t.Fatal("a missing health file reported healthy")
	}
}

// The service runs read_only with a tmpfs mounted only at /tmp, so the default
// health path must live there: written anywhere else every write fails and the
// healthcheck reads a missing file forever, reporting a working box dead.
func TestDefaultHealthPathIsOnTheWritableTmpfs(t *testing.T) {
	// The compose side of the same contract, so a change to either file is
	// caught rather than only the Go constant.
	compose, err := os.ReadFile(filepath.Join("..", "..", "..", "docker-compose.yml"))
	if err != nil {
		t.Fatalf("compose file not readable: %v", err)
	}
	if !strings.Contains(string(compose), "- /tmp:") {
		t.Fatal("docker-compose.yml no longer mounts a tmpfs at /tmp")
	}
	if !strings.HasPrefix(DefaultHealthPath, "/tmp/") {
		t.Fatalf("DefaultHealthPath = %q, want it under /tmp -- the only mount the "+
			"read_only container can write", DefaultHealthPath)
	}
}

// --- intervals ----------------------------------------------------------------

func TestClampIntervalBoundsWhateverThePlatformAsksFor(t *testing.T) {
	cases := map[time.Duration]time.Duration{
		0:                defaultPollInterval,
		-1 * time.Second: defaultPollInterval,
		time.Millisecond: minPollInterval,
		30 * time.Second: 30 * time.Second,
		24 * time.Hour:   maxPollInterval,
		90 * time.Minute: maxPollInterval,
	}
	for in, want := range cases {
		if got := clampInterval(in); got != want {
			t.Errorf("clampInterval(%v) = %v, want %v", in, got, want)
		}
	}
}

func TestJitterStaysInsideTheClamp(t *testing.T) {
	r := New(filepath.Join(t.TempDir(), "health"), "test")
	for i := 0; i < 500; i++ {
		got := r.jittered(5 * time.Second)
		if got < 4*time.Second || got > 6*time.Second {
			t.Fatalf("jittered(5s) = %v, outside +/-20%%", got)
		}
		if got := r.jittered(maxPollInterval); got > maxPollInterval {
			t.Fatalf("jitter carried the interval past the clamp: %v", got)
		}
	}
}

// --- what reaches a log line or the platform ----------------------------------

// A transport error carries the request URL, and that URL carries the PromQL
// built from the job's labels. Neither the platform nor the log gets it.
func TestFailuresNeverCarryTheQueryURL(t *testing.T) {
	leaky := &url.Error{
		Op:  "Get",
		URL: "http://sink-prometheus:9090/api/v1/query_range?query=sum(pgwatch_wait_events_total%7Bcluster%3D%22acme-prod%22%7D)",
		Err: fmt.Errorf("dial tcp: connection refused"),
	}
	text, class := describeFailure(leaky)
	if strings.Contains(text, "acme-prod") || strings.Contains(text, "pgwatch_wait_events_total") {
		t.Fatalf("the submitted error carries the query: %q", text)
	}
	if class != "store_unreachable" {
		t.Fatalf("failure_class = %q", class)
	}
	if got := sanitize(leaky).Error(); strings.Contains(got, "acme-prod") || strings.Contains(got, "query=") {
		t.Fatalf("the logged error carries the query: %q", got)
	}
	if !strings.Contains(sanitize(leaky).Error(), "connection refused") {
		t.Fatalf("sanitizing threw away the useful part: %q", sanitize(leaky).Error())
	}
}

// The submitted error text must fit the platform's 512-byte cap, or the submit
// itself is rejected with PT400 and the answer is lost.
func TestSubmittedErrorTextFitsThePlatformCap(t *testing.T) {
	for _, err := range []error{
		fmt.Errorf("%w: %s", errUnencodableResult, strings.Repeat("x", 4000)),
		fmt.Errorf("boom: %s", strings.Repeat("y", 4000)),
		&collect.UpstreamError{StatusCode: 500, Message: strings.Repeat("z", 4000)},
		// The input that makes this assertion BITE. Every case above lands on a
		// constant classifyFailure arm of at most 41 bytes, so the cap could
		// never trip and `describeFailure` was free to drop truncate entirely
		// with the suite green. The invalid_args arm now forwards up to
		// maxErrorBodyBytes of store text, so it is the live one -- and it is
		// multibyte, because slicing mid-rune at 512 is where a naive cap
		// produces invalid UTF-8 and the platform answers 22021. A THREE-byte
		// rune, deliberately: the prefix is 28 bytes, 512-28 = 484, and 484 is
		// even -- so with a 2-byte rune the naive cut lands on a boundary and
		// the UTF-8 assertion below would be vacuous. 484 is not divisible by 3.
		fmt.Errorf("%w: %s (%w)", collect.ErrInvalidArgs, "parse error",
			&collect.UpstreamError{
				StatusCode:   422,
				Message:      "metric store returned status 422",
				StoreMessage: strings.Repeat("\u65e5", 2000),
			}),
	} {
		// The literal, not errorMaxBytes: the platform's cap is 512 bytes and a
		// submit over it comes back PT400, losing the answer.
		text, _ := describeFailure(err)
		if len(text) > 512 {
			t.Fatalf("error text is %d bytes, over the platform's 512 (errorMaxBytes is %d)",
				len(text), errorMaxBytes)
		}
		if !utf8.ValidString(text) {
			t.Fatalf("error text is not valid UTF-8, which the platform rejects with 22021: %q", text)
		}
	}
	// The literal, the way longBackoff and maxResponseBytes are pinned. This
	// used to read "every classifyFailure arm is a short canned string, so an
	// assertion phrased in terms of errorMaxBytes can never trip" -- no longer
	// true: the invalid_args arm forwards the store's message, which is why the
	// multibyte case above was added.
	if errorMaxBytes != 512 {
		t.Fatalf("errorMaxBytes = %d, want the platform's 512-byte cap", errorMaxBytes)
	}
	if got := truncate(strings.Repeat("q", 4000), errorMaxBytes); len(got) != 512 {
		t.Fatalf("truncate left %d bytes, want 512", len(got))
	}
}

// Nothing the process logs may carry the token.
func TestTheTokenNeverReachesALogLine(t *testing.T) {
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	})
	h.runner.tick(context.Background())
	if strings.Contains(h.logs.String(), testToken) {
		t.Fatalf("the token appears in the logs: %s", h.logs.String())
	}
}

// --- the store-retry path -----------------------------------------------------

// collectJobArgs is a window in the past whose retention floor is the period
// start, so the job actually reaches the store instead of resolving as a
// retention skip the way the loop tests above do.
const collectJobArgs = `{"cluster_name":"c","node_name":"n","vcpus":2,
	"period_start":"2026-09-01T00:00:00Z","period_end":"2026-09-01T00:10:00Z",
	"window_start":"2026-09-01T00:00:00Z"}`

const emptyMatrix = `{"status":"success","data":{"resultType":"matrix","result":[]}}`

// storeHarness serves one collection job and lets the test decide what the
// store answers on each request.
// platform builds the client and credentials the runner would build for this
// harness, so a test can drive an unexported method directly instead of going
// the long way round through tick().
func (h *harness) platform(t *testing.T) (*platform.Client, platform.Credentials) {
	t.Helper()
	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("loading the harness config: %v", err)
	}
	return platform.NewClient(cfg.APIBaseURL, "test", platformTimeout),
		platform.Credentials{APIToken: cfg.APIToken, InstanceID: cfg.InstanceID}
}

func storeHarness(t *testing.T, store func(n int, w http.ResponseWriter, r *http.Request)) (*harness, *int) {
	t.Helper()
	storeRequests := 0
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/api/v1/"):
			storeRequests++
			store(storeRequests, w, r)
		case strings.HasSuffix(r.URL.Path, "instance_job_poll"):
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`,
				collectJobArgs)
		default:
			if h.submitHandler != nil {
				h.submitHandler(w, r)
				return
			}
			if h.submitReply != "" {
				w.Write([]byte(h.submitReply))
				return
			}
			w.Write(echoOutcome(r))
		}
	})
	return h, &storeRequests
}

// A momentary store failure must not lose the job: the collection re-runs.
func TestAStoreBlipIsRetriedWithinTheJob(t *testing.T) {
	h, requests := storeHarness(t, func(n int, w http.ResponseWriter, _ *http.Request) {
		if n <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Write([]byte(emptyMatrix))
	})

	h.runner.tick(context.Background())
	if *requests != 4 {
		t.Fatalf("store requests = %d, want 4 (two failed attempts, then per-type + total)", *requests)
	}
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if h.submits[0]["skip_reason"] != "density" {
		t.Fatalf("skip_reason = %v, want the answer from the attempt that succeeded",
			h.submits[0]["skip_reason"])
	}
}

// A store that never comes back is retried a bounded number of times and then
// answered as failed, so the job does not sit 'running' until the sweep.
func TestAStoreThatStaysDownGivesUpAfterTheBoundedAttempts(t *testing.T) {
	h, requests := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})

	h.runner.tick(context.Background())
	// The literal, not storeAttempts: an expectation written as the constant
	// under test cannot notice the constant changing.
	if *requests != 3 {
		t.Fatalf("store requests = %d, want 3 (storeAttempts is %d)", *requests, storeAttempts)
	}
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if _, ok := h.submits[0]["result"]; ok {
		t.Errorf("a failed job carried a result: %v", h.submits[0])
	}
	if h.submits[0]["failure_class"] != "store_error" {
		t.Errorf("failure_class = %v, want store_error", h.submits[0]["failure_class"])
	}
}

// A 4xx from the store is our own bad query, not a blip: retrying it would only
// spend the store's budget on the same rejection.
func TestANonRetryableStoreErrorIsNotRetried(t *testing.T) {
	h, requests := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})

	h.runner.tick(context.Background())
	if *requests != 1 {
		t.Fatalf("store requests = %d, want 1", *requests)
	}
	if h.submits[0]["failure_class"] != "store_error" {
		t.Errorf("failure_class = %v, want store_error", h.submits[0]["failure_class"])
	}
}

// Non-finite values reach the AAS metrics unfiltered (that is the contract), and
// encoding/json refuses them. The job must be answered as failed rather than
// left looking like a transport failure, which would keep it 'running' until the
// platform's hourly sweep.
func TestANonFiniteSampleIsAnsweredAsAFailedJob(t *testing.T) {
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("query")
		switch {
		case strings.HasPrefix(q, "sum by (wait_event_type)("):
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
				{"metric":{"wait_event_type":"IO"},"values":[[1,"+Inf"]]}]}}`))
		case strings.HasPrefix(q, "sum(pgwatch_wait_events_total"):
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
				{"metric":{},"values":[[1,"1"]]}]}}`))
		default:
			w.Write([]byte(emptyMatrix))
		}
	})

	h.runner.tick(context.Background())
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if _, ok := h.submits[0]["result"]; ok {
		t.Errorf("an unencodable result was submitted anyway: %v", h.submits[0])
	}
	if h.submits[0]["failure_class"] != "unencodable_result" {
		t.Fatalf("failure_class = %v, want unencodable_result", h.submits[0]["failure_class"])
	}
}

// The work is already done and the job stays 'running' platform-side until the
// hourly sweep, so a reset on the way back must not throw the collection away.
func TestATransientSubmitFailureIsRetried(t *testing.T) {
	submitAttemptsSeen := 0
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`,
				`{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
				  "period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`)
			return
		}
		submitAttemptsSeen++
		if submitAttemptsSeen == 1 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.Write(echoOutcome(r))
	})

	h.runner.tick(context.Background())
	if submitAttemptsSeen != 2 {
		t.Fatalf("submit attempts = %d, want 2 (one 502, then the retry)", submitAttemptsSeen)
	}
	if err := h.health(t); err != nil {
		t.Fatalf("a retried submit flipped health: %v", err)
	}
}

// A PT404 means the job is not ours to answer any more; retrying it would only
// collect the same refusal.
func TestAJobGoneSubmitIsNotRetried(t *testing.T) {
	submitAttemptsSeen := 0
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`,
				`{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
				  "period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`)
			return
		}
		submitAttemptsSeen++
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"code":"PT404","message":"Not found"}`))
	})

	h.runner.tick(context.Background())
	if submitAttemptsSeen != 1 {
		t.Fatalf("submit attempts = %d, want 1", submitAttemptsSeen)
	}
	// Not just "it stopped": it stopped on the ROUTINE branch. Without the
	// early return the loop also stops, but the instance logs a failure for
	// something that is not one.
	logs := h.logs.String()
	if !strings.Contains(logs, "no longer open") {
		t.Fatalf("the routine case was not recorded as routine: %s", logs)
	}
	if strings.Contains(logs, "submitting job j1 failed") {
		t.Fatalf("a routine PT404 was logged as a failure: %s", logs)
	}
}

// The health file's deadline allows one job budget, so a tick running several
// jobs has to stamp it as it goes or a working loop looks wedged.
func TestHealthIsRefreshedBetweenJobs(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
		"period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`
	var stampsAtSubmit []time.Time
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s},
				{"id":"j2","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args, args)
			return
		}
		// Read the deadline the loop has published at the moment each answer
		// arrives; it must have moved between the two jobs.
		raw, err := os.ReadFile(h.healthPath)
		if err == nil {
			var state struct {
				UpdatedAt time.Time `json:"updated_at"`
			}
			if json.Unmarshal(raw, &state) == nil {
				stampsAtSubmit = append(stampsAtSubmit, state.UpdatedAt)
			}
		}
		w.Write(echoOutcome(r))
	})

	// A fake clock that advances a minute per reading, so the assertion is exact
	// rather than "strictly greater by however long a localhost round-trip took".
	clock := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	h.runner.now = func() time.Time {
		clock = clock.Add(time.Minute)
		return clock
	}

	h.runner.tick(context.Background())
	if len(stampsAtSubmit) != 2 {
		t.Fatalf("saw %d health stamps, want one per job", len(stampsAtSubmit))
	}
	if !stampsAtSubmit[1].After(stampsAtSubmit[0]) {
		t.Fatalf("health was not re-stamped between jobs: %v then %v",
			stampsAtSubmit[0], stampsAtSubmit[1])
	}
}

// One hung request must not be able to eat the job budget: an attempt aborts at
// its first hang, so what has to hold is that three such aborts plus their
// backoff still fit. (An attempt that does NOT hang issues many requests; this
// is about the hanging one, which is the case the timeout exists for.)
//
// The arithmetic is only half of it. What actually had the bug was the wiring,
// so assert the timeout the production constructor really hands to the store
// client -- a correct constant nothing uses would otherwise pass.
func TestOneHungStoreRequestCannotEatTheJobBudget(t *testing.T) {
	if storeRequestTimeout >= jobBudget {
		t.Fatalf("storeRequestTimeout %v is not under jobBudget %v", storeRequestTimeout, jobBudget)
	}
	spent := storeRequestTimeout*storeAttempts + storeBackoff*time.Duration(storeAttempts)
	if spent >= jobBudget {
		t.Fatalf("%d hung attempts of %v plus backoff is %v, which does not fit the %v budget",
			storeAttempts, storeRequestTimeout, spent, jobBudget)
	}

	built := New(filepath.Join(t.TempDir(), "health"), "test").
		storeClient(config.Config{StoreURL: "http://example.invalid"})
	if got := built.RequestTimeout(); got != storeRequestTimeout {
		t.Fatalf("the runner wired the store client with %v, want storeRequestTimeout (%v)",
			got, storeRequestTimeout)
	}
}

// The health file is best-effort: a write that cannot land is logged, never
// fatal, because a loop that is otherwise working must not be taken down by its
// own reporting.
func TestAFailedHealthWriteIsReportedButNotFatal(t *testing.T) {
	missingDir := filepath.Join(t.TempDir(), "absent", "health")
	now := time.Now()
	if err := writeHealth(missingDir, true, "", now, now.Add(time.Minute)); err == nil {
		t.Fatal("writeHealth returned no error for an unwritable path")
	}

	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
	})
	h.runner.healthPath = missingDir
	if wait := h.runner.tick(context.Background()); wait <= 0 {
		t.Fatalf("the tick did not complete after a failed health write (wait %v)", wait)
	}
	if !strings.Contains(h.logs.String(), "could not write the health file") {
		t.Fatalf("the failed write was not reported: %s", h.logs.String())
	}
}

// ClassUnavailable and ClassAuth keep the platform's detail in the LOCAL log,
// even though the health file gets only our own reason. Dropping it leaves an
// operator with a one-line verdict and no status to act on.
func TestAnUnavailablePlatformStillLogsWhatCameBack(t *testing.T) {
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://elsewhere.example.com/rpc/instance_job_poll",
			http.StatusTemporaryRedirect)
	})

	h.runner.tick(context.Background())

	logs := h.logs.String()
	if !strings.Contains(logs, "redirected") {
		t.Fatalf("the log does not name the redirect: %s", logs)
	}
	if !strings.Contains(logs, "307") {
		t.Fatalf("the log carries no detail of what came back: %s", logs)
	}
	// ...while the health file still carries only our own words.
	raw, err := os.ReadFile(h.healthPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "elsewhere.example.com") {
		t.Fatalf("the health file carries the redirect target: %s", raw)
	}
}

// The health file's reason is read by an operator through `docker inspect`, and
// Docker keeps it in .State.Health.Log. It must carry only text this process
// wrote: PostgREST fills `message` from the raw PostgreSQL error, which can echo
// a request value back. The detail belongs in the local log, not there.
func TestTheHealthReasonNeverCarriesThePlatformMessage(t *testing.T) {
	const echoed = "invalid input syntax for type uuid: \"00000000-0000-0000-0000-00000000dead\""
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, `{"code":"PT400","message":%q}`, echoed)
	})

	h.runner.tick(context.Background())

	raw, err := os.ReadFile(h.healthPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "00000000-0000-0000-0000-00000000dead") {
		t.Fatalf("the health file echoes the platform's message: %s", raw)
	}
	// The code is ours to enumerate, so it is welcome there.
	if !strings.Contains(string(raw), "PT400") {
		t.Fatalf("the health file does not say which code came back: %s", raw)
	}

	// The code arrives from the other side like everything else, so it is
	// bounded: an unbounded one is an unbounded health file, in a 16 MiB tmpfs.
	long := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, `{"code":%q,"message":"x"}`, strings.Repeat("A", 5000))
	})
	long.runner.tick(context.Background())
	bloated, err := os.ReadFile(long.healthPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(bloated) > 512 {
		t.Fatalf("the health file is %d bytes; the platform's code is not bounded", len(bloated))
	}
}

// Run seeds the health file before it polls anything, which is why the
// container's HEALTHCHECK can have a short start period: a probe firing in the
// first seconds must read a verdict, not a missing file.
func TestRunSeedsTheHealthFileBeforeItPolls(t *testing.T) {
	polled := make(chan struct{}, 1)
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		select {
		case polled <- struct{}{}:
		default:
		}
		w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
	})

	// Stop at the first sleep, i.e. after one tick.
	ctx, cancel := context.WithCancel(context.Background())
	h.runner.sleep = func(context.Context, time.Duration) bool {
		cancel()
		return false
	}
	// The file must exist before the first poll is answered, so check it from
	// inside the request rather than afterwards.
	seededBeforePoll := false
	inner := h.runner.platformClient
	h.runner.platformClient = func(baseURL string) *platform.Client {
		if _, err := os.Stat(h.healthPath); err == nil {
			seededBeforePoll = true
		}
		return inner(baseURL)
	}

	if err := h.runner.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("Run: %v", err)
	}
	if len(polled) == 0 {
		t.Fatal("Run never polled")
	}
	if !seededBeforePoll {
		t.Fatal("Run polled before it had written any health verdict")
	}
}

// The health deadline has to cover the submit that follows a job, retries and
// all, or a working loop is called wedged whenever one has to retry.
func TestTheHealthDeadlineCoversAJobAndItsSubmit(t *testing.T) {
	// What has to hold is the SLACK over the sleep, not the total: at the
	// minimum interval the sleep contributes a second and the budget must still
	// cover a full job plus a submit that uses every retry.
	r := New(filepath.Join(t.TempDir(), "health"), "test")
	// It TRACKS r.budget, not the constant: asserting only that the deadline
	// covers the budget would pass with the constant, which is larger. Two
	// budgets, and the deadlines have to differ by exactly the difference.
	r.budget = 42 * time.Second
	short := r.healthDeadline(time.Minute)
	r.budget = 142 * time.Second
	long := r.healthDeadline(time.Minute)
	if long-short != 100*time.Second {
		t.Fatalf("healthDeadline moved by %v when the budget moved by 100s; it is not "+
			"reading the budget", long-short)
	}
	r.budget = worstCaseJob
	// TWO submits, not one: a refused answer is followed by the terminal close,
	// which is a separate submission on the same retry ladder, so a transient
	// failure on it burns a second whole budget. Budgeting for one leaves the
	// deadline 6m40s short of what a single job can legitimately take, and a
	// working box then reports dead through the container HEALTHCHECK (#378).
	work := worstCaseJob + 2*worstCaseSubmit
	// A minute of margin over the worst case, not merely "more than". Budgeting
	// for the job alone leaves 21 seconds, which is the kind of number that
	// turns into a false alarm the first time anything gets slower.
	const margin = time.Minute
	for _, next := range []time.Duration{minPollInterval, defaultPollInterval, maxPollInterval} {
		if slack := r.healthDeadline(next) - next; slack < work+margin {
			t.Fatalf("healthDeadline(%v) leaves %v over the sleep; a job and its submit can take "+
				"%v, so that is under the %v margin", next, slack, work, margin)
		}
	}
}

// writeHealth renames into place so a healthcheck can never read a half-written
// file, and leaves no debris behind when it does.
func TestWriteHealthLeavesNoTemporaryFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "health")
	now := time.Now().UTC()
	if err := writeHealth(path, true, "", now, now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "health" {
		var names []string
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("directory holds %v, want only the health file", names)
	}

	// Rename, not write-in-place: replacing the file needs permission on the
	// DIRECTORY, so a read-only health file left behind by anything still gets
	// replaced -- and a reader can never catch a half-written one.
	if os.Geteuid() == 0 {
		// CI runs as root in golang:1.24, where DAC is bypassed and an
		// in-place write would succeed too. Say so rather than pass quietly.
		t.Skip("running as root: the read-only half of this test cannot discriminate")
	}
	if err := os.Chmod(path, 0o400); err != nil {
		t.Fatal(err)
	}
	if err := writeHealth(path, false, "still writable", now, now.Add(time.Minute)); err != nil {
		t.Fatalf("a read-only health file was not replaced: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "still writable") {
		t.Fatalf("the health file was not replaced: %s", raw)
	}
}

// The platform stores and aggregates on failure_class, so it is a contract in
// the same sense the rpc argument names are. Literals on both sides.
func TestEveryFailureIsClassified(t *testing.T) {
	cases := []struct {
		err   error
		text  string
		class string
	}{
		{fmt.Errorf("%w: bad", collect.ErrInvalidArgs), "job args could not be used", "invalid_args"},
		{fmt.Errorf("%w: %q", collect.ErrUnknownKind, "x"),
			"this instance does not know this job kind", "unknown_kind"},
		{collect.ErrWindowTooLong, "collection window is too long", "window_too_long"},
		{context.DeadlineExceeded, "collection exceeded the local time budget", "timeout"},
		{context.Canceled, "collection was cancelled", "cancelled"},
		{fmt.Errorf("%w: json", errUnencodableResult),
			"the collected result could not be encoded", "unencodable_result"},
		{&collect.UpstreamError{StatusCode: 503}, "metric store returned 503", "store_error"},
		{fmt.Errorf("dial tcp: refused"), "metric store unreachable", "store_unreachable"},
		// Its own class on purpose: without it a panic fell through to
		// store_unreachable and reported a metric-store outage on a box whose
		// store was fine.
		//
		// On failure_class being open rather than enumerated: read at
		// platform-all `feature/agent-channel`, instance_job_submit length-caps
		// it at 64 and requires `error`, and enumerates only `outcome`. Named
		// ref on purpose -- the file does not exist on `main` yet, so this is
		// what the branch does, not settled platform behaviour. If that RPC ever
		// enumerates the class, a panicking job's submit is REJECTED and the job
		// sits `running` until the sweep, which is worse than what it replaced.
		{fmt.Errorf("%w (kind %q)", errCollectionPanicked, "aas_collect"),
			"collection failed unexpectedly", "panic"},
	}
	for _, tc := range cases {
		t.Run(tc.class, func(t *testing.T) {
			text, class := describeFailure(tc.err)
			if text != tc.text || class != tc.class {
				t.Fatalf("describeFailure = (%q, %q), want (%q, %q)", text, class, tc.text, tc.class)
			}
		})
	}
}

// The jitter exists to stop a fleet provisioned together from polling in
// lockstep, so "inside the bounds" is not enough: it has to actually vary, in
// both directions.
func TestTheJitterActuallyVaries(t *testing.T) {
	r := New(filepath.Join(t.TempDir(), "health"), "test")
	const nominal = 5 * time.Second
	var below, above int
	seen := map[time.Duration]bool{}
	for i := 0; i < 500; i++ {
		got := r.jittered(nominal)
		seen[got] = true
		switch {
		case got < nominal:
			below++
		case got > nominal:
			above++
		}
	}
	if below == 0 || above == 0 {
		t.Fatalf("jitter is one-sided: %d below, %d above", below, above)
	}
	if len(seen) < 10 {
		t.Fatalf("jitter produced %d distinct values in 500 draws", len(seen))
	}
}

// One idle line an hour, pinned from below as well as above.
func TestTheIdleLineReturnsAfterAnHour(t *testing.T) {
	h := newHarness(t, false, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		t.Error("an unconfigured instance must not call the platform")
		w.WriteHeader(http.StatusInternalServerError)
	})
	clock := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	h.runner.now = func() time.Time { return clock }

	h.runner.tick(context.Background())
	// Advance well inside the hour, so a narrowed interval is caught too: with
	// the clock frozen both ticks are at the same instant and any positive
	// threshold would pass.
	clock = clock.Add(30 * time.Minute)
	h.runner.tick(context.Background())
	if got := strings.Count(h.logs.String(), "idle:"); got != 1 {
		t.Fatalf("logged %d idle lines inside the hour, want 1", got)
	}
	// The literal hour, not the constant: advancing by idleLogInterval would
	// still pass with the interval widened to a day.
	clock = clock.Add(time.Hour + time.Second)
	h.runner.tick(context.Background())
	if got := strings.Count(h.logs.String(), "idle:"); got != 2 {
		t.Fatalf("logged %d idle lines after the hour elapsed, want 2", got)
	}
}

// A poll that keeps succeeding while every job fails is a dead channel with a
// healthy handshake.
func TestJobsFailingInARowFlipHealth(t *testing.T) {
	var h *harness
	h, _ = storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		if h.storeStatus == http.StatusOK {
			w.Write([]byte(emptyMatrix))
			return
		}
		w.WriteHeader(http.StatusBadRequest) // non-retryable, one attempt per job
	})

	// Pinned from BELOW as well: a threshold written in terms of the constant
	// cannot notice the constant moving, and each extra failure is another
	// interval before an operator sees red.
	const flipsAt = 3
	for i := 1; i <= flipsAt; i++ {
		h.polls = 0 // hand out a job on every tick
		h.runner.tick(context.Background())
		if i < flipsAt {
			if err := h.health(t); err != nil {
				t.Fatalf("health flipped after %d failed job(s), want %d: %v", i, flipsAt, err)
			}
		}
	}
	if err := h.health(t); err == nil {
		t.Fatal("three jobs failed in a row and the container still reports healthy")
	} else if !strings.Contains(err.Error(), "jobs in a row failed") {
		t.Fatalf("health reason does not name the cause: %v", err)
	}

	// And it recovers: one job that works clears the run.
	h.storeStatus = http.StatusOK
	h.polls = 0
	h.runner.tick(context.Background())
	if err := h.health(t); err != nil {
		t.Fatalf("a job that succeeded did not clear the failed run: %v", err)
	}
}

// An unreadable credential file is permanent, not a transient poll failure: it
// must not report healthy while three polls go by.
func TestAnUnreadableConfigIsUnhealthyAtOnce(t *testing.T) {
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		t.Error("the platform must not be called with an unreadable config")
		w.WriteHeader(http.StatusInternalServerError)
	})
	// A directory where the file should be: unreadable as a file, and
	// reproducible without depending on the test's uid.
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", t.TempDir())

	h.runner.tick(context.Background())
	if err := h.health(t); err == nil {
		t.Fatal("an unreadable config reported healthy")
	} else if !strings.Contains(err.Error(), "config unreadable") {
		t.Fatalf("health reason = %v, want the config", err)
	}
}

// The temp file is removed on the failure paths too: a health path that is a
// directory fails at the rename, once per tick and once per job, and would
// otherwise leak a .health-* into a 16 MiB tmpfs every time.
func TestAFailedRenameLeavesNoDebris(t *testing.T) {
	dir := t.TempDir()
	blocked := filepath.Join(dir, "health")
	if err := os.Mkdir(blocked, 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	for i := 0; i < 3; i++ {
		if err := writeHealth(blocked, true, "", now, now.Add(time.Minute)); err == nil {
			t.Fatal("writing over a directory reported success")
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		var names []string
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("three failed writes left %v behind", names)
	}
}

// --- what actually goes on the wire ------------------------------------------

// The platform reads checkId and results at the TOP LEVEL of `result` and takes
// the pin from the job row, so the payload is submitted bare. An envelope would
// be recorded apply_rejected, in a column every member of the org can see.
func TestASuccessfulCollectionSubmitsTheBarePayload(t *testing.T) {
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, r *http.Request) {
		switch q := r.URL.Query().Get("query"); {
		case strings.HasPrefix(q, "sum(pgwatch_wait_events_total"),
			strings.HasPrefix(q, "sum by (wait_event_type)("):
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
				{"metric":{"wait_event_type":"IO"},"values":[[1,"3"]]}]}}`))
		default:
			w.Write([]byte(emptyMatrix))
		}
	})

	h.runner.tick(context.Background())
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	result, ok := h.submits[0]["result"].(map[string]any)
	if !ok {
		t.Fatalf("no result was submitted: %v", h.submits[0])
	}
	if result["checkId"] != "AAS" {
		t.Fatalf("checkId is not at the top level of result: %v", result)
	}
	if _, ok := result["results"].(map[string]any); !ok {
		t.Fatalf("results is not at the top level of result: %v", result)
	}
	if _, wrapped := result["payload"]; wrapped {
		t.Fatalf("the payload is wrapped in an envelope: %v", result)
	}
	if _, ok := h.submits[0]["error"]; ok {
		t.Fatalf("a successful collection carried an error: %v", h.submits[0])
	}
	if h.submits[0]["outcome"] != "ok" {
		t.Fatalf("outcome = %v, want ok", h.submits[0]["outcome"])
	}
}

// The platform RECORDS a refused payload on the job row instead of raising, so
// a box that does not read the reply believes a collection landed when it did
// not -- and reports healthy while every one of them is rejected.
func TestARefusedResultCountsAsAJobFailure(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
		"period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`
	submits := 0
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args)
			return
		}
		submits++
		fmt.Fprintf(w, `{"job_id":"j1","status":"failed","outcome":%q,
			"accepted_at":"2026-09-16T00:00:00Z",
			"error":"instance_job_apply_collection: result checkId is not AAS"}`, outcomeOf(r))
	})

	for i := 1; i <= 3; i++ {
		h.polls = 0
		h.runner.tick(context.Background())
	}
	// Two submits per job: the refused answer, then the terminal error that
	// closes it. Without the second the job would sit `running` until the
	// hourly sweep.
	if submits != 6 {
		t.Fatalf("submitted %d times, want 6 (three jobs, each answered then closed)", submits)
	}
	if !strings.Contains(h.logs.String(), "was not accepted") {
		t.Fatalf("a refused result was not recorded as one: %s", h.logs.String())
	}
	if err := h.health(t); err == nil {
		t.Fatal("three refused results and the container still reports healthy")
	}
}

// A refused payload is the platform's verdict on the bytes, so re-sending them
// would only be refused again.
func TestARefusedResultIsNotRetried(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
		"period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`
	submits := 0
	var outcomes []string
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args)
			return
		}
		submits++
		outcomes = append(outcomes, outcomeOf(r))
		fmt.Fprintf(w, `{"job_id":"j1","status":"failed","outcome":%q,"error":"refused"}`, outcomeOf(r))
	})

	h.runner.tick(context.Background())
	// The refused ANSWER is sent once and never repeated. The second submit is
	// a different submission -- outcome=error -- which is what closes the job
	// instead of leaving it `running` for the hourly sweep.
	want := []string{"skipped", "error"}
	if len(outcomes) != len(want) || outcomes[0] != want[0] || outcomes[1] != want[1] {
		t.Fatalf("submitted outcomes %v, want %v", outcomes, want)
	}
	if submits != 2 {
		t.Fatalf("submitted %d times, want 2 (the answer, then the close)", submits)
	}
}

// Oversize is the case the terminal close exists for, and it is NOT the shape
// the close was keyed on: instance_job_submit raises PT400 for a result it will
// not take, which rolls the whole call back, so there is no reply body to carry
// a rejection and the job is left 'running' for the hourly sweep. Keying on the
// in-body sentinel alone made the close unreachable for the louder half.
func TestAnAnswerRefusedWithPT400IsClosedAsFailed(t *testing.T) {
	var outcomes []string
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, r *http.Request) {
		// A real payload, so the refused answer is an `ok` carrying a result --
		// the only kind of answer an oversize refusal can be about.
		if strings.HasPrefix(r.URL.Query().Get("query"), "sum(pgwatch_wait_events_total") {
			w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[
				{"metric":{"wait_event_type":"IO"},"values":[[1,"3"]]}]}}`))
			return
		}
		w.Write([]byte(emptyMatrix))
	})
	h.submitHandler = func(w http.ResponseWriter, r *http.Request) {
		outcome := outcomeOf(r)
		outcomes = append(outcomes, outcome)
		if outcome == "ok" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"code":"PT400","message":"result exceeds the maximum size"}`))
			return
		}
		w.Write(echoOutcome(r))
	}

	h.runner.tick(context.Background())

	// The refused answer is sent ONCE -- a PT400 is a verdict, not a blip --
	// and is followed by the close, which is a different submission.
	want := []string{"ok", "error"}
	if len(outcomes) != len(want) || outcomes[0] != want[0] || outcomes[1] != want[1] {
		t.Fatalf("submitted outcomes %v, want %v (the answer, then the close)", outcomes, want)
	}
	// The assertion that the fixture is discriminating: an `ok` that carried no
	// result would be refused for a different reason and prove nothing here.
	if _, ok := h.submits[0]["result"].(map[string]any); !ok {
		t.Fatalf("the refused answer carried no result: %v", h.submits[0])
	}
	// Legible, not a bare transport failure: this string is what the customer
	// sees on the job row.
	if got := h.submits[1]["failure_class"]; got != "result_rejected" {
		t.Fatalf("the close named failure_class %v, want result_rejected", got)
	}
}

// The close is one extra submit, not a loop: an answer that already says
// `error` cannot be improved by saying it again, and a PT400 on the close
// itself must end the job rather than start another round.
func TestARefusedErrorAnswerIsNotClosedAgain(t *testing.T) {
	submits := 0
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		// Nothing the box can answer with: the collection fails, so the answer
		// itself is an `error`.
		w.WriteHeader(http.StatusNotImplemented)
	})
	h.submitHandler = func(w http.ResponseWriter, r *http.Request) {
		submits++
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"code":"PT400","message":"result exceeds the maximum size"}`))
	}

	h.runner.tick(context.Background())

	if submits != 1 {
		t.Fatalf("submitted %d times, want 1 (a refused error answer is not re-closed)", submits)
	}
	if got := h.submits[0]["outcome"]; got != "error" {
		t.Fatalf("outcome = %v, want error; the fixture did not produce a failed collection", got)
	}
}

// The store's message has to reach the SUBMITTED error, not merely the error
// value inside package collect.
//
// This is the assertion the collect-level tests structurally cannot make: they
// stop at `err.Error()`, one layer below `classifyFailure`, which is where a
// constant string replaced the message and threw the whole fix away. `pgai
// promql` prints the submitted `error` verbatim, so this is the text the user
// actually sees when they fat-finger an expression (#378).
func TestTheStoresMessageReachesTheSubmittedError(t *testing.T) {
	const storeMsg = `cannot parse "totl(up)": unsupported function "totl"`
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/api/v1/"):
			w.WriteHeader(http.StatusUnprocessableEntity)
			fmt.Fprintf(w, `{"status":"error","errorType":"422","error":%q}`, storeMsg)
		case strings.HasSuffix(r.URL.Path, "instance_job_poll"):
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"promql_instant","args":{"query":"totl(up)"}}],"next_poll_ms":5000}`))
		default:
			w.Write(echoOutcome(r))
		}
	})

	h.runner.tick(context.Background())

	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if got := h.submits[0]["failure_class"]; got != "invalid_args" {
		t.Fatalf("failure_class = %v, want invalid_args", got)
	}
	got, _ := h.submits[0]["error"].(string)
	if !strings.Contains(got, "unsupported function") {
		t.Fatalf("the store's message never reached the wire: error = %q -- the user is told the platform's args are bad when the store told us exactly what was wrong", got)
	}
}

// ...and it must arrive with the control characters removed. The store echoes
// the submitted expression back inside its message, PromQL string literals take
// Go-style escapes, and the CLI prints the field with no escaping of its own --
// so an ESC in an expression would execute in whoever's terminal reads the job
// back. Same hole round 9 found in cli/lib/promql.ts, on a second path.
func TestTheStoresMessageIsStrippedOfControlCharacters(t *testing.T) {
	const storeMsg = "cannot parse \"\x1b[2J\x1b[31mRED\r\n\": bad"
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/api/v1/"):
			w.WriteHeader(http.StatusUnprocessableEntity)
			b, _ := json.Marshal(map[string]string{"status": "error", "error": storeMsg})
			w.Write(b)
		case strings.HasSuffix(r.URL.Path, "instance_job_poll"):
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"promql_instant","args":{"query":"up"}}],"next_poll_ms":5000}`))
		default:
			w.Write(echoOutcome(r))
		}
	})

	h.runner.tick(context.Background())

	got, _ := h.submits[0]["error"].(string)
	if strings.ContainsAny(got, "\x00\x1b\r\n") {
		t.Fatalf("a control character reached the submitted error: %q", got)
	}
	if !strings.Contains(got, "RED") {
		t.Fatalf("stripping removed the message itself: %q", got)
	}
}

// refusedByPlatform's ErrUnknownOutcome exclusion, which was a mutation
// survivor: disabling it left the whole runner suite green.
//
// The guard is load-bearing. Classify(ErrUnknownOutcome) is ClassRequest
// (pinned by platform/client_test.go), so without the early return an UNSENT
// submission -- our own bug forming an outcome the wire does not accept, which
// never left the process -- would read as "the platform refused it" and trigger
// the terminal error-close. That closes a job as failed on a verdict the
// platform never gave (#378).
func TestAnUnsentSubmissionIsNotAPlatformVerdict(t *testing.T) {
	if refusedByPlatform(fmt.Errorf("forming the submission: %w", platform.ErrUnknownOutcome)) {
		t.Fatal("an unsent submission read as a platform refusal, so runJob would close the job as failed on a verdict that was never given")
	}
	// CONTROL: the two errors that ARE verdicts must still be recognised, or the
	// assertion above would also pass with the whole function returning false.
	if !refusedByPlatform(fmt.Errorf("x: %w", platform.ErrResultRejected)) {
		t.Fatal("a rejected result is no longer a platform refusal")
	}
}

// shutdownSubmitBudget has to sit between two numbers that belong to other
// systems, and each earlier version got one of them wrong.
//
// Above the PLATFORM's lock_timeout: v1.instance_job_submit does
// `set_config('lock_timeout','5s')` and waits that out for the monitoring_instances
// row, which the pull path's consumer can hold for a whole PgQ batch. A deadline
// under 5s cannot reach either outcome, so it makes delivery strictly WORSE in
// the one contention case the retry ladder was sized against. (4s did this.)
//
// Below DOCKER's stop grace: docker-compose sets no stop_grace_period for this
// service, so the default 10s applies. This is the WHOLE shutdown budget, not
// one submit's — runJob can send the answer and then a terminal close, and 8s
// EACH was 16s against a 10s grace.
//
// Both foreign literals on purpose, not the constants they mirror: the point is
// to fail here when someone changes ours (#378).
func TestTheShutdownBudgetFitsBetweenTheLockTimeoutAndTheStopGrace(t *testing.T) {
	const platformSubmitLockTimeout = 5 * time.Second
	const dockerStopGrace = 10 * time.Second

	if shutdownSubmitBudget <= platformSubmitLockTimeout {
		t.Fatalf("shutdownSubmitBudget %v is at or under the platform's %v lock_timeout, so a contended submit is aborted client-side before the server can answer at all",
			shutdownSubmitBudget, platformSubmitLockTimeout)
	}
	if shutdownSubmitBudget >= dockerStopGrace {
		t.Fatalf("shutdownSubmitBudget %v is at or over docker's %v stop grace, so the process is SIGKILLed mid-submit",
			shutdownSubmitBudget, dockerStopGrace)
	}
}

// ...and it is ONE budget for the pair, not one each. Two detached submits must
// not be able to outlast the grace between them.
func TestTwoDetachedSubmitsShareOneBudget(t *testing.T) {
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	})
	client, creds := h.platform(t)
	base := time.Date(2026, 9, 21, 12, 0, 0, 0, time.UTC)
	// The clock ADVANCES between the two submits. With a frozen clock
	// `now().Add(budget)` is the same instant every time, so re-deriving the
	// deadline per submit would be indistinguishable from sharing it and this
	// test would pass against the very mutation it exists to catch.
	elapsed := time.Duration(0)
	h.runner.now = func() time.Time { return base.Add(elapsed) }

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	for i := 0; i < 2; i++ {
		_ = h.runner.submit(ctx, client, creds,
			platform.Submission{JobID: "j1", Outcome: platform.OutcomeOK, Result: map[string]any{}})
		elapsed += 3 * time.Second
	}
	if got := h.runner.shutdownDeadline.Sub(base); got != shutdownSubmitBudget {
		t.Fatalf("after two detached submits the deadline is %v past the first, want %v -- each took its own budget, so the pair can outlast the stop grace",
			got, shutdownSubmitBudget)
	}
}

// The submit retries exist for one specific thing, and the budget has to
// outlast it: instance_job_submit waits out its own lock_timeout for the
// monitoring_instances row, which the pull path's consumer can hold for a whole
// PgQ batch. A budget shorter than a batch is a retry loop that cannot win the
// race it was written for.
func TestTheSubmitBudgetOutlastsAContendedBatch(t *testing.T) {
	// The platform's bound, not ours: instance_job_submit sets lock_timeout to
	// 5s, so a contended attempt costs that before it comes back 55P03.
	const platformLockTimeout = 5 * time.Second
	// What the loop actually covers when every attempt is contended: the waits
	// between attempts plus the lock wait each attempt burns.
	covered := submitBackoff*(submitAttempts*(submitAttempts-1)/2) +
		submitAttempts*platformLockTimeout
	// Sized against a realistic batch -- a handful of events, each one outbound
	// call -- not against a pathological one, which nothing local can outlast.
	const realisticBatch = 5 * time.Minute
	if covered < realisticBatch {
		t.Fatalf("the submit retries give up after %v; the row can be held for %v",
			covered, realisticBatch)
	}
	// Drift alarm for the hand-derived worstCaseSubmit the healthDeadline tests
	// assert against. They cannot read submitBudget without going vacuous, so
	// this is where the two are tied together.
	if submitBudget != worstCaseSubmit {
		t.Fatalf("submitBudget is %v, worstCaseSubmit says %v; re-derive the latter by hand",
			submitBudget, worstCaseSubmit)
	}
}

// A fleet being drained -- the platform flag turned off -- hands out no work.
// A box must not stay red on a run of failures it can no longer retry.
func TestAPollWithNoWorkClearsAFailedRun(t *testing.T) {
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
	})
	h.runner.consecutiveJobFailures = unhealthyAfter

	h.runner.tick(context.Background())
	if err := h.health(t); err != nil {
		t.Fatalf("a drained box stayed unhealthy on a run it cannot retry: %v", err)
	}
}

// A box already judged dead must not flash green for the duration of every
// later job: the stamp before a job refreshes the deadline, not the verdict.
func TestTheVerdictSurvivesTheNextJob(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
		"period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`
	var duringJob error
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args)
			return
		}
		// Read the published verdict at the moment the job's answer arrives,
		// i.e. after the pre-job stamp.
		duringJob = CheckHealth(h.healthPath, h.runner.now())
		fmt.Fprintf(w, `{"job_id":"j1","status":"failed","outcome":%q,"error":"refused"}`, outcomeOf(r))
	})

	for i := 1; i <= 4; i++ {
		h.polls = 0
		h.runner.tick(context.Background())
	}
	if duringJob == nil {
		t.Fatal("the health file reported healthy while a job ran on an already-dead channel")
	}
}

// The job context is the collection budget: a job that overruns it is answered
// as a timeout rather than left to sit, and a budget wired to anything shorter
// would cancel every real multi-query collection.
func TestAJobThatOverrunsTheBudgetIsAnsweredAsATimeout(t *testing.T) {
	// The handler blocks on the job's context OR a wall-clock backstop: r.budget
	// is read once when the job context is made, so a test cannot shorten it
	// later -- without the backstop a budget that is never applied would HANG
	// the package until the go test timeout instead of failing here.
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(3 * time.Second):
			w.Write([]byte(emptyMatrix))
		}
	})
	h.runner.budget = 50 * time.Millisecond
	// Real sleeps inside the job, so the hung request is actually cut off.
	h.runner.sleep = func(ctx context.Context, d time.Duration) bool { return ctx.Err() == nil }

	started := time.Now()
	h.runner.tick(context.Background())
	elapsed := time.Since(started)
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	// EXACTLY timeout, and quickly: without the job's own budget the request
	// falls back to the 90s per-request timeout and still lands on a plausible
	// class, so accepting an alternative makes the test blind to no budget.
	if got := h.submits[0]["failure_class"]; got != "timeout" {
		t.Fatalf("failure_class = %v, want timeout", got)
	}
	if elapsed > 5*time.Second {
		t.Fatalf("the job took %v; the per-job budget did not cut it off", elapsed)
	}
	if New(filepath.Join(t.TempDir(), "h"), "test").budget != jobBudget {
		t.Fatal("the production runner does not use jobBudget as its per-job ceiling")
	}
}

// The platform's stuck sweep fails a job still 'running' after an hour, so the
// local ceiling has to stay under it or a box is failed while it is working.
func TestTheJobBudgetStaysUnderThePlatformSweep(t *testing.T) {
	const platformStuckAfter = time.Hour
	if jobBudget >= platformStuckAfter {
		t.Fatalf("jobBudget %v is not under the platform's %v stuck sweep",
			jobBudget, platformStuckAfter)
	}
	// One claim, one job: the platform's constraint is
	// claim_limit x per_job_ceiling < stuck_after.
	//
	// TWO submit budgets. The answer is one submission and the terminal close
	// after a refusal is another, on the same ladder -- so the real ceiling is
	// jobBudget + 2*submitBudget = 32m20s. Pinning one submit passes at
	// submitAttempts=14 (51m) while the true ceiling is 1h27m, past the sweep.
	if jobBudget+2*submitBudget >= platformStuckAfter {
		t.Fatalf("a job, its answer and the terminal close can take %v, at or over the %v sweep",
			jobBudget+2*submitBudget, platformStuckAfter)
	}
}

// The deadline written to the file has to be the computed one, not the sleep:
// a correct helper nothing uses looks identical from the helper's own test.
func TestTheWrittenDeadlineIsTheComputedOne(t *testing.T) {
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
	})
	h.runner.tick(context.Background())

	raw, err := os.ReadFile(h.healthPath)
	if err != nil {
		t.Fatal(err)
	}
	var state struct {
		NextCheckBy time.Time `json:"next_check_by"`
		UpdatedAt   time.Time `json:"updated_at"`
	}
	if err := json.Unmarshal(raw, &state); err != nil {
		t.Fatal(err)
	}
	if slack := state.NextCheckBy.Sub(state.UpdatedAt); slack < worstCaseJob+worstCaseSubmit {
		t.Fatalf("the file allows %v before the next stamp, which does not cover a job "+
			"and its submit", slack)
	}
}

// The two things ClassUnavailable covers are told apart in the reason, because
// they call for different actions: a wrong api_base_url, or something in front
// of the platform redirecting.
func TestAMissingRPCAndARedirectReadDifferently(t *testing.T) {
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"code":"PGRST202","message":"Could not find the function"}`))
	})
	h.runner.tick(context.Background())
	if !strings.Contains(h.logs.String(), "no job channel at this api_base_url") {
		t.Fatalf("a missing rpc did not say so: %s", h.logs.String())
	}
	if strings.Contains(h.logs.String(), "redirected") {
		t.Fatalf("a missing rpc was reported as a redirect: %s", h.logs.String())
	}
}

// Which store failures are worth another attempt, asserted directly: the retry
// tests above drive a 500 or a 400, which exercises UpstreamError.Retryable and
// never this function's own arms.
func TestWhichStoreFailuresAreRetried(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"invalid args", collect.ErrInvalidArgs, false},
		{"unknown kind", collect.ErrUnknownKind, false},
		{"window too long", collect.ErrWindowTooLong, false},
		{"wrapped permanent error", fmt.Errorf("x: %w", collect.ErrInvalidArgs), false},
		{"store 503", &collect.UpstreamError{StatusCode: 503}, true},
		{"store 401", &collect.UpstreamError{StatusCode: 401}, true},
		{"store 400", &collect.UpstreamError{StatusCode: 400}, false},
		{"oversized body", &collect.UpstreamError{StatusCode: 413}, false},
		// A container restart on our own compose network is the commonest blip.
		{"connection refused", fmt.Errorf("dial tcp: connection refused"), true},
		// A panic is deterministic: retrying it panics again, three times, and
		// burns the job budget doing it.
		{"panic", fmt.Errorf("%w (kind %q)", errCollectionPanicked, "aas_collect"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := retryableStoreError(tc.err); got != tc.want {
				t.Fatalf("retryableStoreError = %v, want %v", got, tc.want)
			}
		})
	}
}

// A kind this build cannot run reaches the platform as such, without touching
// the store and without a retry.
func TestAnUnknownKindIsAnsweredWithoutTouchingTheStore(t *testing.T) {
	storeRequests := 0
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/api/v1/"):
			storeRequests++
			w.Write([]byte(emptyMatrix))
		case strings.HasSuffix(r.URL.Path, "instance_job_poll"):
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			// A real future kind with its OWN args, not collection-shaped.
			// promql_labels is a discovery endpoint, explicitly out of scope in
			// the PromQL contract -- it needs a `match[]` parameter -- so it is
			// still a kind this build cannot run even though the two promql_*
			// query kinds now dispatch.
			w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1",
				"kind":"promql_labels","args":{"match":"up"}}],
				"next_poll_ms":5000}`))
		default:
			w.Write(echoOutcome(r))
		}
	})

	h.runner.tick(context.Background())
	if storeRequests != 0 {
		t.Fatalf("a kind this build cannot run made %d store requests", storeRequests)
	}
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	if got := h.submits[0]["failure_class"]; got != "unknown_kind" {
		t.Fatalf("failure_class = %v, want unknown_kind (args of another kind must not "+
			"be read as bad collection args)", got)
	}
}

// The submit retry is bounded: an unbounded loop would stop the box polling and
// stop it re-stamping health, without ever reporting anything.
func TestTheSubmitRetryIsBounded(t *testing.T) {
	attempts := 0
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`,
				`{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
				  "period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`)
			return
		}
		attempts++
		if attempts > submitAttempts+4 {
			// An unbounded retry loop must FAIL this test, not hang it: with the
			// sleep stubbed out it would spin forever and take the whole package
			// down on the go test timeout instead of saying what is wrong.
			w.WriteHeader(http.StatusBadRequest) // non-transient, so the loop exits
			return
		}
		w.WriteHeader(http.StatusBadGateway)
	})

	h.runner.tick(context.Background())
	// The literal, not submitAttempts.
	if attempts != 8 {
		t.Fatalf("submit attempts = %d, want 8 (submitAttempts is %d)", attempts, submitAttempts)
	}
}

// The reply is the only thing that says whether the collection landed, so a
// reply that says nothing is not a success.
func TestAnUnexpectedSubmitReplyIsNotASuccess(t *testing.T) {
	for _, body := range []string{`null`, `{}`, `{"status":"done"}`, `{"job_id":"j1"}`} {
		t.Run(body, func(t *testing.T) {
			h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
				w.Write([]byte(emptyMatrix))
			})
			// storeHarness answers submits from its default arm; override it.
			h.submitReply = body
			// The same escape hatch TestTheSubmitRetryIsBounded has: without a
			// bound on the retry loop this test would SPIN rather than fail,
			// and the package would die on the go test timeout with a
			// goroutine dump instead of a named assertion.
			attempts := 0
			h.submitHandler = func(w http.ResponseWriter, _ *http.Request) {
				attempts++
				if attempts > 10 {
					w.WriteHeader(http.StatusBadRequest) // non-transient: the loop exits
					return
				}
				w.Write([]byte(body))
			}

			h.runner.tick(context.Background())
			if h.runner.consecutiveJobFailures == 0 {
				t.Fatalf("a submit reply of %s was counted as a landed collection", body)
			}
		})
	}
}

// Backoffs have to be non-zero, or three attempts against a struggling
// 0.75-CPU store arrive as fast as the network allows.
func TestTheRetriesActuallyWait(t *testing.T) {
	if storeBackoff <= 0 || submitBackoff <= 0 {
		t.Fatalf("storeBackoff=%v submitBackoff=%v; a retry with no pause is a hot loop",
			storeBackoff, submitBackoff)
	}
}

// The platform reverts the claim on 55P03/40P01, so the job is queued again and
// the answer is not lost -- as long as the box retries rather than recording a
// failed collection.
func TestARevertedClaimIsRetriedNotRecordedAsAFailure(t *testing.T) {
	attempts := 0
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(emptyMatrix))
	})
	h.submitHandler = func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"code":"55P03","message":"could not obtain lock"}`))
			return
		}
		w.Write(echoOutcome(r))
	}

	h.runner.tick(context.Background())
	if attempts != 2 {
		t.Fatalf("submit attempts = %d, want 2 (the lock, then the retry)", attempts)
	}
	if h.runner.consecutiveJobFailures != 0 {
		t.Fatalf("a reverted claim was recorded as a failed collection (count %d)",
			h.runner.consecutiveJobFailures)
	}
}

// The retries escalate. Three spaced attempts against a store already under
// memory pressure becoming three near-immediate ones would otherwise look
// identical to this suite.
func TestTheBackoffEscalates(t *testing.T) {
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	var waited []time.Duration
	h.runner.sleep = func(ctx context.Context, d time.Duration) bool {
		waited = append(waited, d)
		return ctx.Err() == nil
	}

	h.runner.tick(context.Background())
	if len(waited) < 2 {
		t.Fatalf("the job slept %d times between %d attempts", len(waited), storeAttempts)
	}
	if waited[0] != storeBackoff || waited[1] != 2*storeBackoff {
		t.Fatalf("store backoffs = %v, want %v then %v", waited[:2], storeBackoff, 2*storeBackoff)
	}
}

// The submit loop escalates too.
func TestTheSubmitBackoffEscalates(t *testing.T) {
	attempts := 0
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(emptyMatrix))
	})
	h.submitHandler = func(w http.ResponseWriter, _ *http.Request) {
		attempts++
		w.WriteHeader(http.StatusBadGateway)
	}
	var waited []time.Duration
	h.runner.sleep = func(ctx context.Context, d time.Duration) bool {
		waited = append(waited, d)
		return ctx.Err() == nil
	}

	h.runner.tick(context.Background())
	if attempts != 8 {
		t.Fatalf("submit attempts = %d, want 8 (submitAttempts is %d)", attempts, submitAttempts)
	}
	if len(waited) < 2 || waited[0] != submitBackoff || waited[1] != 2*submitBackoff {
		t.Fatalf("submit backoffs = %v, want %v then %v", waited, submitBackoff, 2*submitBackoff)
	}
	// The TAIL too, not just the first two: a cap quietly added to the backoff
	// would leave the head identical and cut the coverage the budget is sized
	// for, which is what TestTheSubmitBudgetOutlastsAContendedBatch computes.
	if last, want := waited[len(waited)-1], time.Duration(submitAttempts-1)*submitBackoff; last != want {
		t.Fatalf("the last submit backoff was %v, want %v (backoffs: %v)", last, want, waited)
	}
}

// A two-job batch stops at a cancelled context rather than starting the next.
func TestACancelledTickStopsBetweenJobs(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","period_start":"2020-01-01T00:00:00Z",
		"period_end":"2020-01-02T00:00:00Z","window_start":"2026-01-01T00:00:00Z"}`
	ctx, cancel := context.WithCancel(context.Background())
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[
				{"id":"j1","kind":"aas_collect","args":%s},
				{"id":"j2","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args, args)
			return
		}
		cancel() // the first answer arrives, then shutdown
		w.Write(echoOutcome(r))
	})

	h.runner.tick(ctx)
	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers after the context was cancelled, want 1", len(h.submits))
	}
	// Not STARTED, not merely not-submitted: a cancelled job still logs, so a
	// loop that kept going would name j2 here.
	if strings.Contains(h.logs.String(), "job j2") {
		t.Fatalf("the second job was started after the context was cancelled: %s",
			h.logs.String())
	}
}

// SIGTERM DURING a job must still deliver the answer the box already has.
//
// TestACancelledTickStopsBetweenJobs covers cancel BETWEEN jobs; this is the
// other half. The work is finished, so dropping it is pure loss -- and worse
// than loss on the query channel: instance_query_enqueue's in-flight cap is 1
// and its PT409 says "Nothing returns its id, so wait for it", so an ordinary
// `docker compose up -d` during a query locks the user out of that instance
// until the platform's hourly sweep (#378).
//
// The answer goes out on a context that survives the cancellation. One attempt,
// not the retry ladder: the container's stop grace is 10s.
func TestAnAnswerSurvivesCancellationMidJob(t *testing.T) {
	// Driven through r.submit directly, with an ALREADY-CANCELLED context. Two
	// earlier shapes of this test went through tick() and both passed for the
	// wrong reason: cancelling from the store handler also cancels the per-job
	// context, so the collection aborted and what "survived" was an error
	// report; cancelling from the submit handler lands after the first attempt
	// is already in flight, so the detached branch is never entered at all.
	// A mutation destroying the answer on the detached path survived both.
	//
	// The claim is narrow and worth stating exactly: given work that is already
	// finished and a dead context, submit still delivers THAT submission,
	// unaltered. Everything about how the context died is someone else's test.
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	})
	client, creds := h.platform(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	want := platform.Submission{
		JobID:      "j1",
		DurationMS: 7,
		Outcome:    platform.OutcomeOK,
		Result:     map[string]any{"checkId": "AAS", "results": map[string]any{}},
	}
	if err := h.runner.submit(ctx, client, creds, want); err != nil {
		t.Fatalf("submit on a dead context returned %v; the finished answer was dropped and the job stays running until the platform's hourly sweep", err)
	}

	if len(h.submits) != 1 {
		t.Fatalf("submitted %d answers, want 1", len(h.submits))
	}
	// The CONTENT, not the count: an error report also arrives as one submit.
	if got := h.submits[0]["outcome"]; got != platform.OutcomeOK {
		t.Fatalf("outcome = %v, want ok -- the answer was replaced on the way out: %v", got, h.submits[0])
	}
	if h.submits[0]["result"] == nil {
		t.Fatalf("the answer arrived with no result: %v", h.submits[0])
	}
}

// ...and the same claim end to end, because the unit test above pins r.submit
// and nothing else: runJob is free to stop calling submit on a dead context and
// that test stays green. Which is the regression the whole fix exists to
// prevent, so it needs an assertion on the live path.
//
// The reachable production shape: SIGTERM lands mid-collection, the per-job
// context dies with it, execute returns context.Canceled, runJob builds an
// ERROR submission -- and that submission still has to go out, or the job sits
// `running` until the platform's hourly sweep with the in-flight cap held.
func TestACancelledJobIsStillClosedOutThroughTick(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		cancel()
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	})

	h.runner.tick(ctx)

	if len(h.submits) != 1 {
		t.Fatalf("submitted %d after cancellation mid-collection, want 1 -- the job is left running for the hourly sweep and holds the in-flight cap: %s",
			len(h.submits), h.logs.String())
	}
	if got := h.submits[0]["failure_class"]; got != "cancelled" {
		t.Fatalf("failure_class = %v, want cancelled: %v", got, h.submits[0])
	}
}

// A detached submit that FAILS must still report the failure, or runJob cannot
// count it and refusedByPlatform never sees it. Swallowing the error here was a
// mutation survivor.
func TestAFailedDetachedSubmitIsReported(t *testing.T) {
	h, _ := storeHarness(t, func(_ int, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[]}}`))
	})
	h.submitHandler = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"code":"PT500","message":"nope"}`))
	}
	client, creds := h.platform(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := h.runner.submit(ctx, client, creds,
		platform.Submission{JobID: "j1", Outcome: platform.OutcomeOK, Result: map[string]any{}})
	if err == nil {
		t.Fatal("a failed detached submit reported success, so the job counts as answered when nothing was recorded")
	}
}

// A panic in a collection must come back as an error, not take the loop down.
// The container restarts unless-stopped and the platform re-queues a swept job,
// so an unguarded panic would crash-loop the box and starve every other job.
// The stack is the ONLY trace a recovered panic leaves -- the error text is a
// canned literal and the panic value is deliberately dropped -- so both halves
// matter: it has to be logged, and the panic value must not be in it.
func TestAPanicIsLoggedWithoutItsValue(t *testing.T) {
	var buf bytes.Buffer
	prev := log.Writer()
	log.SetOutput(&buf)
	defer log.SetOutput(prev)

	now := time.Now().UTC()
	args := fmt.Sprintf(`{"cluster_name":"c","node_name":"n","vcpus":2,`+
		`"period_start":%q,"period_end":%q}`,
		now.Add(-time.Hour).Format(time.RFC3339), now.Format(time.RFC3339))
	_, err := runCollectSafely(context.Background(), nil,
		platform.Job{ID: "job-1", Kind: "aas_collect", Args: []byte(args)}, now)
	if err == nil {
		t.Fatal("expected an error")
	}

	logged := buf.String()
	if !strings.Contains(logged, "panicked; recovered") {
		t.Fatalf("the panic left no trace in the log: %q", logged)
	}
	if !strings.Contains(logged, "job-1") || !strings.Contains(logged, "aas_collect") {
		t.Fatalf("the log does not identify the job: %q", logged)
	}
	// The panic VALUE can carry a store response or a label. A Go stack shows
	// scalar argument words but never string or slice CONTENTS, so the check is
	// that the runtime's own description of the panic is not in the line.
	if strings.Contains(logged, "invalid memory address") ||
		strings.Contains(logged, "nil pointer dereference") {
		t.Fatalf("the panic value reached the log: %q", logged)
	}
}

func TestACollectionPanicBecomesAnError(t *testing.T) {
	// A nil store client is the cheapest real panic: collect.Run dereferences
	// it. What matters is that the panic does not escape, whatever its source.
	now := time.Now().UTC()
	args := fmt.Sprintf(`{"cluster_name":"c","node_name":"n","vcpus":2,`+
		`"period_start":%q,"period_end":%q}`,
		now.Add(-time.Hour).Format(time.RFC3339), now.Format(time.RFC3339))

	outcome, err := runCollectSafely(context.Background(), nil,
		platform.Job{ID: "job-1", Kind: "aas_collect", Args: []byte(args)}, now)

	if err == nil {
		t.Fatal("a panicking collection returned no error")
	}
	if !errors.Is(err, errCollectionPanicked) {
		t.Fatalf("error is not the panic sentinel: %v", err)
	}
	// The panic VALUE can carry a store response or a label, so it must not
	// reach the message. (The stack goes to the local log, which carries no
	// data, and is what makes a panic findable at all.)
	if strings.Contains(err.Error(), "nil pointer") || strings.Contains(err.Error(), "runtime") {
		t.Fatalf("panic detail leaked into the error: %v", err)
	}
	if outcome.Payload != nil {
		t.Fatalf("a panicking collection returned a payload: %+v", outcome.Payload)
	}

	// A panic is deterministic, so retrying it just panics again and burns the
	// job budget. Without the sentinel it fell through to "assume transient".
	if retryableStoreError(err) {
		t.Fatal("a panic is being retried as if it were transient")
	}

	// And it must not be reported as a metric-store outage: the store is fine.
	text, class := describeFailure(err)
	if class != "panic" {
		t.Fatalf("failure class = %q, want \"panic\" (store_unreachable was the old, wrong answer)", class)
	}
	if strings.Contains(text, "metric store") {
		t.Fatalf("a panic is described as a store problem: %q", text)
	}
}

// truncate caps a BYTE length, so a naive slice can split a multi-byte rune and
// produce invalid UTF-8 -- which Postgres rejects with 22021, dropping the
// job's answer instead of shortening it.
//
// TestTruncateKeepsTheValidTextAroundABadByte below is what a first attempt got
// wrong: it walked the cut point back while the WHOLE prefix was invalid, so a
// single bad byte early in the string discarded everything after it, down to ""
// for a leading one. An empty `error` alongside a non-null `failure_class` is a
// PT400 on the platform side -- not retried, does not consume the job -- so the
// "fix" stranded the very job it was meant to describe.
func TestTruncateDoesNotSplitARune(t *testing.T) {
	// "é" is two bytes, so a cap of 5 lands mid-rune on "aaaaé".
	const s = "aaaaé"
	got := truncate(s, 5)
	if !utf8.ValidString(got) {
		t.Fatalf("truncate produced invalid UTF-8: %q", got)
	}
	if got != "aaaa" {
		t.Fatalf("truncate = %q, want %q", got, "aaaa")
	}
	if truncate(s, 6) != s {
		t.Fatalf("a string that fits was altered")
	}
}

func TestTruncateCleansEvenWhenTheInputFits(t *testing.T) {
	// The early return used to be on length alone, so a SHORT invalid string
	// went back unexamined. Postgres rejects any invalid byte, not just a split
	// rune, so validity is the requirement at every length.
	//
	// No caller sends such a string today -- classifyFailure returns constant
	// ASCII -- so this pins a guard, not a live path.
	for _, in := range []string{"\xff", "abc\xffdef", "\xc3", "err: \xe2(\xa1"} {
		got := truncate(in, errorMaxBytes)
		if !utf8.ValidString(got) {
			t.Fatalf("truncate(%q) = %q, which is not valid UTF-8", in, got)
		}
	}
	// A clean short string is still returned untouched.
	if got := truncate("plain ascii", errorMaxBytes); got != "plain ascii" {
		t.Fatalf("a valid string was altered: %q", got)
	}
}

func TestTruncateKeepsTheValidTextAroundABadByte(t *testing.T) {
	valid := strings.Repeat("a", 600)
	cases := []struct {
		name      string
		in        string
		want      int  // bytes that must survive, at least
		wantEmpty bool // ...or nothing at all, exactly
	}{
		{"bad byte early", "\xff" + valid, 500, false},
		{"bad byte in the middle", strings.Repeat("a", 20) + "\xff" + valid, 500, false},
		// Nothing can survive, and the assertion is exact rather than a floor:
		// a replacement character instead of "" would keep bytes here.
		{"entirely invalid", strings.Repeat("\xff", 600), 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := truncate(tc.in, errorMaxBytes)
			if !utf8.ValidString(got) {
				t.Fatalf("truncate produced invalid UTF-8: %q", got)
			}
			if tc.wantEmpty && len(got) != 0 {
				t.Fatalf("truncate kept %d bytes of nothing but bad bytes: %q", len(got), got)
			}
			if len(got) < tc.want {
				t.Fatalf("truncate kept %d bytes, want at least %d: the diagnostic "+
					"was thrown away rather than cleaned", len(got), tc.want)
			}
		})
	}
}

// A backward clock step must not produce a negative duration_ms. The wall clock
// can step (NTP on a freshly booted VM); the monotonic seam cannot, but the
// clamp is what makes that guarantee independent of the seam's source.
func TestABackwardClockStepDoesNotSubmitANegativeDuration(t *testing.T) {
	args := `{"cluster_name":"c","node_name":"n","vcpus":2,
		"period_start":"2020-01-01T00:00:00Z","period_end":"2020-01-02T00:00:00Z",
		"window_start":"2026-01-01T00:00:00Z"}`
	h := newHarness(t, true, func(h *harness, w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "instance_job_poll") {
			if h.polls > 1 {
				w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			fmt.Fprintf(w, `{"server_time":"2026-09-16T00:00:00Z","jobs":[{"id":"j1","kind":"aas_collect","args":%s}],"next_poll_ms":5000}`, args)
			return
		}
		w.Write(echoOutcome(r))
	})

	// The step: elapsed comes back negative, as a wall-clock subtraction would
	// across an NTP correction.
	h.runner.elapsed = func(time.Time) time.Duration { return -3 * time.Second }

	h.runner.tick(context.Background())
	if len(h.submits) != 1 {
		t.Fatalf("submits = %d, want 1", len(h.submits))
	}
	d, ok := h.submits[0]["duration_ms"].(float64)
	if !ok {
		t.Fatalf("duration_ms = %v, want a number", h.submits[0]["duration_ms"])
	}
	if d < 0 {
		t.Fatalf("duration_ms = %v: a negative duration reaches the platform, "+
			"which can reject the submit and strand the job", d)
	}
}
