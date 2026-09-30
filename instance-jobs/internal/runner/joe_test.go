package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// The signing secret the golden signatures below were computed with, and a value
// that must never reach a submit or a log.
const testJoeSecret = "signing-secret"

// GOLDEN SIGNATURES, computed OUTSIDE Go (Python's hmac, openssl and pgcrypto
// evaluating public.payload_signature_get). Asserted here as well as in
// internal/joe so the WIRING is covered: that the runner hands the client the
// secret from the config file rather than an empty string, which a test
// recomputing the HMAC from whatever the client holds could not notice.
const (
	joeSigEmptyBody = "v0=735036e8dc2ae05e2e29b2b37a7acbe8467f011709b6e935661133b25bcd4786"
	joeSigCommand   = "v0=159874d3b1e1b09a4e6356f113fe0afe6855fe8cecbf573eac5622869811f87a"
	joeCommandBody  = `{"channel_id":"C1","command_id":"7"}`
)

// joeCall is one request that reached the stand-in Joe.
type joeCall struct {
	Method    string
	Path      string
	Body      string
	Signature string
}

// joeHarness wires a Runner to a fake platform and a fake Joe through the real
// config file, so the test exercises the production wiring (#398).
type joeHarness struct {
	runner *Runner
	logs   *strings.Builder

	// mu guards every field below: the Joe arm runs a claimed batch on a worker
	// pool, so the platform and Joe handlers are entered concurrently.
	mu      sync.Mutex
	polls   int
	submits []map[string]any
	calls   []joeCall
	joe     func(h *joeHarness, w http.ResponseWriter, r *http.Request, c joeCall)
}

func (h *joeHarness) snapshotSubmits() []map[string]any {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]map[string]any, len(h.submits))
	copy(out, h.submits)
	return out
}

func (h *joeHarness) snapshotCalls() []joeCall {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]joeCall, len(h.calls))
	copy(out, h.calls)
	return out
}

// callsTo counts the requests that reached one path, which is how a test tells a
// retried read from a write that was answered once.
func (h *joeHarness) callsTo(path string) int {
	n := 0
	for _, c := range h.snapshotCalls() {
		if c.Path == path {
			n++
		}
	}
	return n
}

func newJoeHarness(t *testing.T, jobs string) *joeHarness {
	t.Helper()
	h := &joeHarness{logs: &strings.Builder{}}

	platformSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "joe_job_poll"):
			h.mu.Lock()
			h.polls++
			n := h.polls
			h.mu.Unlock()
			if n > 1 {
				_, _ = w.Write([]byte(`{"server_time":"2026-09-30T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
				return
			}
			_, _ = w.Write([]byte(`{"server_time":"2026-09-30T00:00:00Z","jobs":` + jobs + `,"next_poll_ms":5000}`))
		case strings.HasSuffix(r.URL.Path, "joe_job_submit"):
			raw, _ := io.ReadAll(r.Body)
			r.Body = io.NopCloser(bytes.NewReader(raw))
			var body map[string]any
			_ = json.Unmarshal(raw, &body)
			h.mu.Lock()
			h.submits = append(h.submits, body)
			h.mu.Unlock()
			_, _ = w.Write(echoJoeSubmit(body))
		default:
			// The other channels' rpcs must never be called by a Joe box: a wrong
			// turn here would have the box poll for another target's work with its
			// own credential.
			t.Errorf("a Joe box called %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(platformSrv.Close)

	joeSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c := joeCall{
			Method:    r.Method,
			Path:      r.URL.Path,
			Body:      string(raw),
			Signature: r.Header.Get("Verification-Signature"),
		}
		h.mu.Lock()
		h.calls = append(h.calls, c)
		handler := h.joe
		h.mu.Unlock()
		if handler != nil {
			handler(h, w, r, c)
			return
		}
		if c.Path == "/webui/channels" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`[{"channel_id":"C1"}]`))
			return
		}
		// What Joe's command handler actually does: hand the message to a goroutine
		// and return, writing no status and no body.
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(joeSrv.Close)

	dir := t.TempDir()
	configPath := filepath.Join(dir, ".pgwatch-config")
	content := "joe_token=" + testToken + "\napi_base_url=" + platformSrv.URL +
		"\njoe_url=" + joeSrv.URL +
		"\njoe_verify_token=" + testJoeSecret + "\n"
	if err := os.WriteFile(configPath, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", configPath)
	// A value on the machine running the tests must not decide which channel the
	// box serves -- and a closed loopback port rather than blank, so a precedence
	// regression fails here instead of dialling production.
	t.Setenv("PGAI_INSTANCE_ID", "")
	t.Setenv("PGAI_MONITORING_INSTANCE_ID", "")
	t.Setenv("PGAI_DBLAB_TOKEN", "")
	t.Setenv("PGAI_DBLAB_URL", "http://127.0.0.1:1")
	t.Setenv("PGAI_DBLAB_VERIFY_TOKEN", "")
	t.Setenv("PGAI_JOE_TOKEN", "")
	t.Setenv("PGAI_JOE_URL", "http://127.0.0.1:1")
	t.Setenv("PGAI_JOE_VERIFY_TOKEN", "")
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

// echoJoeSubmit answers a submit the way the rpc does.
func echoJoeSubmit(body map[string]any) []byte {
	outcome, _ := body["outcome"].(string)
	status := "done"
	if outcome == "error" {
		status = "failed"
	}
	jobID, _ := body["job_id"].(string)
	out := map[string]any{
		"error": nil, "job_id": jobID, "status": status, "outcome": outcome,
		"accepted_at": "2026-09-30T12:29:14.703881+00:00",
	}
	raw, _ := json.Marshal(out)
	return raw
}

// joeCommandJob is the shape the platform enqueues for a command: the payload it
// can author, and resolve_channel for the one field it cannot.
func joeCommandJob(id string) string {
	return `{"id":"` + id + `","kind":"joe_call","args":{"method":"POST",` +
		`"action":"/webui/command","resolve_channel":true,"purpose":"joe_call",` +
		`"data":{"command_id":"7"}}}`
}

// THE WHOLE LOOP ON A JOE BOX, and the point of the slice: ONE job does the
// channel lookup and the command POST, so the inversion does not pay the poll
// interval twice for what v1.joe_command_run does in one round trip today.
func TestTickResolvesTheChannelAndPostsTheCommandInOneJob(t *testing.T) {
	h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
	h.runner.tick(context.Background())

	calls := h.snapshotCalls()
	if len(calls) != 2 {
		t.Fatalf("Joe saw %d calls, want 2 (the lookup and the command): %+v", len(calls), calls)
	}
	// ORDER IS THE CONTRACT: the channel has to be known before the body can be
	// signed, so the lookup cannot follow the command.
	if calls[0].Method != "GET" || calls[0].Path != "/webui/channels" {
		t.Errorf("first call = %s %s, want GET /webui/channels", calls[0].Method, calls[0].Path)
	}
	if calls[1].Method != "POST" || calls[1].Path != "/webui/command" {
		t.Errorf("second call = %s %s, want POST /webui/command", calls[1].Method, calls[1].Path)
	}
	// The lookup is a bodyless GET signed over nothing; the golden value also
	// proves the configured secret reached the client.
	if calls[0].Body != "" || calls[0].Signature != joeSigEmptyBody {
		t.Errorf("lookup: body %q sig %s, want empty and %s", calls[0].Body, calls[0].Signature, joeSigEmptyBody)
	}
	// The command carries the RESOLVED channel, and is signed over exactly those
	// bytes -- which is why the platform could not have signed it.
	if calls[1].Body != joeCommandBody {
		t.Errorf("command body = %s, want %s", calls[1].Body, joeCommandBody)
	}
	if calls[1].Signature != joeSigCommand {
		t.Errorf("command signature = %s, want %s", calls[1].Signature, joeSigCommand)
	}

	subs := h.snapshotSubmits()
	if len(subs) != 1 {
		t.Fatalf("submitted %d answers, want 1: %+v", len(subs), subs)
	}
	if subs[0]["outcome"] != "ok" {
		t.Errorf("outcome = %v, want ok: %+v", subs[0]["outcome"], subs[0])
	}
	// Joe's handler writes NOTHING, so there is no reply to store -- and a
	// resolve_channel job records the channel this box chose instead. It is the
	// only record anywhere of where the command went.
	result, _ := subs[0]["result"].(map[string]any)
	if result == nil || result["channel_id"] != "C1" {
		t.Errorf("result = %#v, want the resolved channel recorded", subs[0]["result"])
	}
	// One poll: the tick did both calls. Two would mean the lookup had been split
	// into its own job and the latency doubled.
	if h.polls != 1 {
		t.Errorf("polled %d times in one tick, want 1", h.polls)
	}
}

// THE ID JOE ADVERTISED REACHES JOE UNCHANGED, asserted THROUGH THE RUNNER and not
// only inside internal/joe. Joe registers its processors under the channelID
// exactly as configured and looks one up with a plain map read, so a trim anywhere
// between the lookup and the POST names a channel it does not have -- a 400 on a
// write that joeWriteAttempts never retries. internal/joe pins the trim inside
// Channels; this pins the two steps after it, which is where the same one line
// would otherwise go unnoticed.
func TestThePaddedChannelIdReachesJoeUnchanged(t *testing.T) {
	const padded = " C1\t"
	h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, c joeCall) {
		if c.Path == "/webui/channels" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`[{"channel_id":" C1\t"}]`))
			return
		}
		w.WriteHeader(http.StatusOK)
	}
	h.runner.tick(context.Background())

	calls := h.snapshotCalls()
	if len(calls) != 2 {
		t.Fatalf("Joe saw %d calls, want 2: %+v", len(calls), calls)
	}
	const wantBody = `{"channel_id":" C1\t","command_id":"7"}`
	if calls[1].Body != wantBody {
		t.Fatalf("posted\n  %s\nwant\n  %s", calls[1].Body, wantBody)
	}
	subs := h.snapshotSubmits()
	if len(subs) != 1 || subs[0]["outcome"] != "ok" {
		t.Fatalf("submitted %+v, want one ok answer", subs)
	}
	// The stored record names the channel the command actually went to, so it has
	// to carry the same bytes.
	result, _ := subs[0]["result"].(map[string]any)
	if result == nil || result["channel_id"] != padded {
		t.Fatalf("result = %#v, want channel_id %q", subs[0]["result"], padded)
	}
}

// A WRITE IS ATTEMPTED ONCE. Joe answers 200 before it has done anything, so a
// POST whose reply was lost may have been accepted; re-sending would run the
// command twice on the customer's clone.
func TestAJoeWriteIsAttemptedOnceAndAnsweredAsFailed(t *testing.T) {
	h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, c joeCall) {
		if c.Path == "/webui/channels" {
			_, _ = w.Write([]byte(`[{"channel_id":"C1"}]`))
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	}
	h.runner.tick(context.Background())

	if n := h.callsTo("/webui/command"); n != 1 {
		t.Fatalf("the command was sent %d times, want exactly 1 -- a write is never re-sent", n)
	}
	subs := h.snapshotSubmits()
	if len(subs) != 1 || subs[0]["outcome"] != "error" {
		t.Fatalf("submitted %+v, want one error answer", subs)
	}
	if subs[0]["failure_class"] != "joe_error" {
		t.Errorf("failure_class = %v, want joe_error", subs[0]["failure_class"])
	}
	if msg, _ := subs[0]["error"].(string); !strings.Contains(msg, "500") {
		t.Errorf("error = %q, want it to name the status", msg)
	}
}

// THE LOOKUP IS RETRIED EVEN THOUGH THE JOB IS A WRITE, and this is the
// distinction that makes the Joe arm its own: a failed lookup means the command
// was never sent, so repeating it cannot duplicate anything. Failing the whole
// command on a blipped config read would be stricter than the write rule asks.
func TestAFailedChannelLookupIsRetriedBeforeAWrite(t *testing.T) {
	var lookups int
	h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, c joeCall) {
		if c.Path == "/webui/channels" {
			lookups++
			if lookups < 3 {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			_, _ = w.Write([]byte(`[{"channel_id":"C1"}]`))
			return
		}
		w.WriteHeader(http.StatusOK)
	}
	h.runner.tick(context.Background())

	if lookups != 3 {
		t.Fatalf("the lookup was tried %d times, want 3", lookups)
	}
	// And the command is still sent exactly ONCE: retrying the read must not
	// multiply the write.
	if n := h.callsTo("/webui/command"); n != 1 {
		t.Fatalf("the command was sent %d times, want exactly 1", n)
	}
	subs := h.snapshotSubmits()
	if len(subs) != 1 || subs[0]["outcome"] != "ok" {
		t.Fatalf("submitted %+v, want one ok answer", subs)
	}
}

// A lookup that never recovers fails the job WITHOUT sending the command: there
// is no channel to address it to.
func TestALookupThatNeverRecoversNeverSendsTheCommand(t *testing.T) {
	h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, _ joeCall) {
		w.WriteHeader(http.StatusInternalServerError)
	}
	h.runner.tick(context.Background())

	if n := h.callsTo("/webui/command"); n != 0 {
		t.Fatalf("the command was sent %d times with no channel resolved, want 0", n)
	}
	subs := h.snapshotSubmits()
	if len(subs) != 1 || subs[0]["outcome"] != "error" {
		t.Fatalf("submitted %+v, want one error answer", subs)
	}
}

// A Joe serving no channels is a SKIP, not a failure: nothing was delivered and
// nothing at Joe failed, so 'error' would report a fault that did not happen. It
// is still its OWN config rather than a blip, so it is asked once, not three times.
func TestAJoeWithNoChannelsIsSkippedNotFailed(t *testing.T) {
	h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, c joeCall) {
		if c.Path == "/webui/channels" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`[]`))
			return
		}
		w.WriteHeader(http.StatusOK)
	}
	h.runner.tick(context.Background())

	if n := h.callsTo("/webui/channels"); n != 1 {
		t.Fatalf("the lookup was tried %d times, want 1 -- an empty channel list is permanent", n)
	}
	if n := h.callsTo("/webui/command"); n != 0 {
		t.Fatalf("the command was sent %d times, want 0", n)
	}
	subs := h.snapshotSubmits()
	if len(subs) != 1 {
		t.Fatalf("submitted %+v, want one answer", subs)
	}
	if subs[0]["outcome"] != "skipped" {
		t.Errorf("outcome = %v, want skipped", subs[0]["outcome"])
	}
	if subs[0]["skip_reason"] != "no_channels" {
		t.Errorf("skip_reason = %v, want no_channels", subs[0]["skip_reason"])
	}
	// v1.joe_job_submit answers PT400 to any contradiction, so a skip must carry
	// NEITHER an error nor a failure_class -- and would not consume the job.
	for _, key := range []string{"error", "failure_class", "result"} {
		if v, ok := subs[0][key]; ok && v != nil {
			t.Errorf("a skip carries %s = %#v; the rpc refuses the contradiction", key, v)
		}
	}
}

// ONLY the no-channels case is a skip. A lookup that fails because Joe is broken
// or refuses the signature IS a fault, and reporting it as a skip would hide a
// misconfigured box behind a `done` job.
func TestALookupFailureThatIsNotNoChannelsStaysAnError(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
	}{
		{"joe is broken", http.StatusInternalServerError},
		{"signature refused", http.StatusForbidden},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
			h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, _ joeCall) {
				w.WriteHeader(tc.status)
			}
			h.runner.tick(context.Background())

			subs := h.snapshotSubmits()
			if len(subs) != 1 {
				t.Fatalf("submitted %+v, want one answer", subs)
			}
			if subs[0]["outcome"] != "error" {
				t.Errorf("outcome = %v, want error -- this is a fault, not a skip", subs[0]["outcome"])
			}
			if _, ok := subs[0]["skip_reason"]; ok && subs[0]["skip_reason"] != nil {
				t.Errorf("a fault carries skip_reason = %#v", subs[0]["skip_reason"])
			}
		})
	}
}

// A 200 THAT IS NOT A CHANNEL LIST IS A FAULT TOO, and this is the one the skip
// arm is easiest to widen into by accident: the body arrives with a 200, so nothing
// in the transport says anything is wrong. Folding it into no_channels would submit
// a command that never left the box as `skipped`, let v1.joe_job_submit close the
// job `done`, and -- because runJob counts a skip as a success -- leave the box
// reporting healthy while every command vanished.
func TestABodyThatIsNotAChannelListIsAnErrorNotASkip(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{
		{"an html error page from something else on joe's port", `<html><body>Bad Gateway</body></html>`},
		{"a json error envelope", `{"message":"internal error","code":500}`},
		{"a renamed shape", `{"items":[{"channel_id":"C1"}]}`},
		// A generic REST list on Joe's port: an ARRAY, so the top-level guard passes
		// and only the per-element one catches it. This was still a skip after the
		// first fix, and the box stayed green through five consecutive faulty ticks.
		{"an array of objects that are not channels", `[{"id":"C1","name":"prod"}]`},
		{"an array of empty objects", `[{}]`},
		{"an empty 200", ``},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
			h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, c joeCall) {
				if c.Path == "/webui/channels" {
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte(tc.body))
					return
				}
				w.WriteHeader(http.StatusOK)
			}
			h.runner.tick(context.Background())

			if n := h.callsTo("/webui/command"); n != 0 {
				t.Fatalf("the command was sent %d times, want 0 -- no channel was resolved", n)
			}
			subs := h.snapshotSubmits()
			if len(subs) != 1 {
				t.Fatalf("submitted %+v, want one answer", subs)
			}
			if subs[0]["outcome"] != "error" {
				t.Errorf("outcome = %v, want error -- a skip would close a lost command as done",
					subs[0]["outcome"])
			}
			if v, ok := subs[0]["skip_reason"]; ok && v != nil {
				t.Errorf("skip_reason = %#v; this is a fault, not a box that serves no channel", v)
			}
			// The class, not just the outcome: 'no_channels' here would be the same
			// mistake one layer down, and it is what an operator reads.
			if subs[0]["failure_class"] != "bad_channel_list" {
				t.Errorf("failure_class = %v, want bad_channel_list", subs[0]["failure_class"])
			}
			// Named as Joe's. The default arm's fallback is the metric store, which a
			// Joe box does not have.
			if msg, _ := subs[0]["error"].(string); !strings.Contains(msg, "joe") ||
				strings.Contains(msg, "metric store") {
				t.Errorf("error = %q, want it to name joe and not the metric store", msg)
			}
			// Retried, unlike an empty list: this is not Joe's own config answering,
			// and a read costs one local GET to repeat.
			if n := h.callsTo("/webui/channels"); n != joeChannelAttempts {
				t.Errorf("the lookup was tried %d times, want %d", n, joeChannelAttempts)
			}
		})
	}
}

// Every OTHER action still stores Joe's reply, or NULL when there is none. The
// channel record is scoped to resolve_channel, so it cannot displace a real reply.
func TestAJobWithoutResolveChannelStoresNoChannelRecord(t *testing.T) {
	h := newJoeHarness(t,
		`[{"id":"j1","kind":"joe_call","args":{"method":"POST","action":"/webui/command",`+
			`"data":{"channel_id":"named-by-the-platform"}}}]`)
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, _ joeCall) {
		w.WriteHeader(http.StatusOK) // Joe's real command handler: no body.
	}
	h.runner.tick(context.Background())

	// No lookup at all, and no channel record: the platform named the channel.
	if n := h.callsTo("/webui/channels"); n != 0 {
		t.Errorf("looked the channel up %d times without being asked to", n)
	}
	subs := h.snapshotSubmits()
	if len(subs) != 1 || subs[0]["outcome"] != "ok" {
		t.Fatalf("submitted %+v, want one ok answer", subs)
	}
	if v, ok := subs[0]["result"]; ok && v != nil {
		t.Errorf("result = %#v, want null -- Joe wrote no body and nothing was resolved", v)
	}
}

// The platform does NOT strip nulls through `data` (unlike dblab_call_precheck),
// because the body is a signed payload. So a null must survive the substitution
// into the bytes that get signed -- dropping it would sign a body the platform
// never authored.
func TestANullInsideTheDataSurvivesIntoTheSignedBody(t *testing.T) {
	h := newJoeHarness(t,
		`[{"id":"j1","kind":"joe_call","args":{"method":"POST","action":"/webui/command",`+
			`"resolve_channel":true,"data":{"command_id":"7","comment":null}}}]`)
	h.runner.tick(context.Background())

	calls := h.snapshotCalls()
	if len(calls) != 2 {
		t.Fatalf("Joe saw %d calls, want 2: %+v", len(calls), calls)
	}
	const want = `{"channel_id":"C1","command_id":"7","comment":null}`
	if calls[1].Body != want {
		t.Fatalf("sent body\n  %s\nwant\n  %s", calls[1].Body, want)
	}
}

// A GET job IS retried: the read rule is unchanged, and a job whose action is the
// channel list goes through the same path as any other call.
func TestAJoeReadIsRetried(t *testing.T) {
	var tries int
	h := newJoeHarness(t,
		`[{"id":"j1","kind":"joe_call","args":{"method":"GET","action":"/webui/channels"}}]`)
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, _ joeCall) {
		tries++
		if tries < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"channel_id":"C1"}]`))
	}
	h.runner.tick(context.Background())

	if tries != 3 {
		t.Fatalf("the read was tried %d times, want 3", tries)
	}
	subs := h.snapshotSubmits()
	if len(subs) != 1 || subs[0]["outcome"] != "ok" {
		t.Fatalf("submitted %+v, want one ok answer", subs)
	}
	// The reply is relayed verbatim, so the caller gets what Joe said.
	result, _ := json.Marshal(subs[0]["result"])
	if !strings.Contains(string(result), "C1") {
		t.Errorf("result = %s, want Joe's own reply", result)
	}
}

// A 403 is how Joe refuses a SIGNATURE, so it means this box's joe_verify_token
// and Joe's signingSecret disagree. Not retried: three more attempts restate it.
func TestAJoeSignatureRefusalIsNotRetriedPastTheSecondScheme(t *testing.T) {
	h := newJoeHarness(t,
		`[{"id":"j1","kind":"joe_call","args":{"method":"GET","action":"/webui/channels"}}]`)
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, _ joeCall) {
		w.WriteHeader(http.StatusForbidden)
	}
	h.runner.tick(context.Background())

	// Two: the two signature schemes for a bodyless GET, and no ladder beyond.
	if n := h.callsTo("/webui/channels"); n != 2 {
		t.Fatalf("Joe was called %d times, want 2 (one per signature scheme)", n)
	}
	subs := h.snapshotSubmits()
	if len(subs) != 1 || subs[0]["failure_class"] != "joe_error" {
		t.Fatalf("submitted %+v, want one joe_error answer", subs)
	}
}

// THE SIGNING SECRET IS NEVER REPORTED. It is the HMAC key, it is in no header
// value, and neither a submitted error nor a log line may carry it.
func TestTheJoeSecretNeverLeavesTheBox(t *testing.T) {
	h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, _ joeCall) {
		w.WriteHeader(http.StatusForbidden)
	}
	h.runner.tick(context.Background())

	raw, _ := json.Marshal(h.snapshotSubmits())
	for _, secret := range []string{testJoeSecret, testToken} {
		if bytes.Contains(raw, []byte(secret)) {
			t.Errorf("a submitted answer carries a credential: %s", raw)
		}
		if strings.Contains(h.logs.String(), secret) {
			t.Errorf("a log line carries a credential: %s", h.logs.String())
		}
	}
}

// Joe box, Joe rpcs. The harness fails any other path, so this pins that the
// credential picked the channel end to end.
func TestAJoeBoxPollsOnlyTheJoeRpcs(t *testing.T) {
	h := newJoeHarness(t, `[]`)
	h.runner.tick(context.Background())
	if h.polls != 1 {
		t.Fatalf("polled %d times, want 1", h.polls)
	}
}

// A BATCH RUNS ON A BOUNDED POOL, and the bound must EQUAL the platform's claim
// limit -- see joeConcurrency. Five, spelled out by hand: a test written in terms
// of joeConcurrency cannot notice joeConcurrency changing.
func TestAJoeBatchRunsAtMostThePoolAtOnce(t *testing.T) {
	const pool = 5
	const batch = pool + 1

	jobs := make([]string, 0, batch)
	for i := 1; i <= batch; i++ {
		jobs = append(jobs, joeCommandJob("j"+string(rune('0'+i))))
	}

	arrivals := make(chan struct{}, batch)
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseBatch := func() { releaseOnce.Do(func() { close(release) }) }
	h := newJoeHarness(t, "["+strings.Join(jobs, ",")+"]")
	// RELEASED ON EVERY EXIT, or a t.Fatalf below leaves the blocked handler
	// holding httptest.Server.Close and the whole package times out INSTEAD of
	// reporting the assertion. Measured with joeConcurrency at 1: the run ended
	// `panic: test timed out` with no --- FAIL line, the "only %d commands were in
	// flight" message never printed, and every other test in internal/runner lost.
	//
	// A defer, NOT t.Cleanup: newJoeHarness registers the servers' Close as
	// cleanups first, and cleanups run LIFO, so a cleanup added here would run
	// AFTER them -- which is the deadlock, one step later.
	defer releaseBatch()
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, c joeCall) {
		if c.Path == "/webui/channels" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`[{"channel_id":"C1"}]`))
			return
		}
		// The command blocks, so every started job is in flight at once.
		arrivals <- struct{}{}
		<-release
		w.WriteHeader(http.StatusOK)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		h.runner.tick(context.Background())
	}()

	// FIVE reach Joe while none has answered. A smaller pool never gets here:
	// what is missing is the fifth, not the sixth.
	for i := 0; i < pool; i++ {
		select {
		case <-arrivals:
		case <-time.After(20 * time.Second):
			t.Fatalf("only %d commands were in flight, want %d", i, pool)
		}
	}
	// ...and the SIXTH does not join them. Unbounded, or any pool at or above the
	// batch size, arrives here immediately.
	select {
	case <-arrivals:
		t.Fatalf("%d commands were in flight at once, want at most %d -- the pool is not "+
			"bounding the batch", pool+1, pool)
	case <-time.After(500 * time.Millisecond):
	}

	releaseBatch()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("the tick did not finish after the batch was released")
	}

	// The surplus WAITED rather than being dropped: every job is still answered.
	ids := make([]string, 0, batch)
	for _, s := range h.snapshotSubmits() {
		id, _ := s["job_id"].(string)
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if len(ids) != batch {
		t.Fatalf("submitted %v, want all %d answered -- the queued job must still be answered",
			ids, batch)
	}
}

// THE CHANNEL RECORD DISPLACES NOTHING. It is stored only because Joe's command
// handler writes no body; a resolve_channel job that does get a reply must store
// that reply, not the channel. Without this, the record could quietly overwrite a
// real answer and the channel would be the only thing anyone ever saw.
func TestTheChannelRecordNeverDisplacesARealReply(t *testing.T) {
	const reply = `{"accepted":true}`
	h := newJoeHarness(t, "["+joeCommandJob("j1")+"]")
	h.joe = func(h *joeHarness, w http.ResponseWriter, _ *http.Request, c joeCall) {
		if c.Path == "/webui/channels" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`[{"channel_id":"C1"}]`))
			return
		}
		// A Joe that DOES answer the command POST with a body.
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(reply))
	}
	h.runner.tick(context.Background())

	subs := h.snapshotSubmits()
	if len(subs) != 1 || subs[0]["outcome"] != "ok" {
		t.Fatalf("submitted %+v, want one ok answer", subs)
	}
	result, _ := subs[0]["result"].(map[string]any)
	if result == nil || result["accepted"] != true {
		t.Fatalf("result = %#v, want Joe's own reply %s", subs[0]["result"], reply)
	}
	if _, ok := result["channel_id"]; ok {
		t.Errorf("the channel record displaced Joe's reply: %#v", result)
	}
}

// EVERY FAILURE A JOE BOX CAN REPORT IS NAMED AS JOE'S, and this is the test that
// pins the mechanism the shared reply envelope made necessary.
//
// internal/reply gave both channels ONE oversize sentinel
// (dblab.ErrOversizeReply == joe.ErrOversizeReply == reply.ErrOversize), so the
// only thing telling a Joe reply from an engine reply is executeJoe's errJoeCall
// wrap plus the order of classifyFailure's arms. Both were previously unasserted:
// deleting the whole `defer`, or moving the engine's oversize arm above the Joe
// ones, left the suite green -- and a Joe box then reported "the ENGINE reply is
// too large", "COLLECTION exceeded the local time budget", or "METRIC STORE
// unreachable", each naming a component the box does not have. The platform shows
// that text to whoever ran the command.
//
// So each row asserts the CLASS and the WORDS. The class alone is not enough: the
// engine's oversize arm answers the same `oversize_reply`, and the collection
// timeout arm the same `timeout`.
func TestEveryJoeFailureIsReportedAsJoesOwn(t *testing.T) {
	// Over the 64 KiB the channel lookup reads, which is the cheapest way to reach
	// the shared ceiling.
	oversize := `[{"channel_id":"` + strings.Repeat("x", 64<<10) + `"}]`

	for _, tc := range []struct {
		name string
		jobs string
		// budget shrinks the per-job ceiling so a hanging Joe is observable.
		budget time.Duration
		joe    func(h *joeHarness, w http.ResponseWriter, r *http.Request, c joeCall)
		class  string
		// wantIn must appear in the submitted error; notWantIn must not. notWantIn
		// is the half that catches a lost errJoeCall or a reordered arm.
		wantIn    string
		notWantIn string
		// joeCalls is how many requests should have reached Joe at all.
		joeCalls int
	}{
		{
			name: "a reply over the shared ceiling",
			jobs: "[" + joeCommandJob("j1") + "]",
			joe: func(_ *joeHarness, w http.ResponseWriter, _ *http.Request, _ joeCall) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(oversize))
			},
			class:     "oversize_reply",
			wantIn:    "joe",
			notWantIn: "engine",
			// Once: the same call returns the same unusable answer, so the ceiling
			// belongs in retryableJoeError's permanent set.
			joeCalls: 1,
		},
		{
			name: "joe drops the connection",
			jobs: "[" + joeCommandJob("j1") + "]",
			joe: func(_ *joeHarness, w http.ResponseWriter, _ *http.Request, _ joeCall) {
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					return
				}
				_ = conn.Close()
			},
			class:     "joe_unreachable",
			wantIn:    "joe unreachable",
			notWantIn: "metric store",
			// A transport failure against a service on this very box is retried.
			joeCalls: joeChannelAttempts,
		},
		{
			name: "a write that outlives the job budget",
			jobs: `[{"id":"j1","kind":"joe_call","args":{"method":"POST",` +
				`"action":"/webui/command","data":{"command_id":"7"}}}]`,
			budget: 50 * time.Millisecond,
			joe: func(_ *joeHarness, w http.ResponseWriter, r *http.Request, _ joeCall) {
				<-r.Context().Done()
			},
			class:     "timeout",
			wantIn:    "joe call",
			notWantIn: "collection",
			joeCalls:  1,
		},
		{
			name: "args this box cannot act on",
			jobs: `[{"id":"j1","kind":"joe_call","args":{"method":"DELETE",` +
				`"action":"/webui/command"}}]`,
			class: "invalid_args",
			// Refused before any request is built, so there is nothing about Joe to
			// say -- but it must not read as the metric store either.
			notWantIn: "metric store",
			joeCalls:  0,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newJoeHarness(t, tc.jobs)
			if tc.budget != 0 {
				h.runner.budget = tc.budget
			}
			if tc.joe != nil {
				h.joe = tc.joe
			}
			h.runner.tick(context.Background())

			subs := h.snapshotSubmits()
			if len(subs) != 1 {
				t.Fatalf("submitted %+v, want one answer", subs)
			}
			if subs[0]["outcome"] != "error" {
				t.Fatalf("outcome = %v, want error", subs[0]["outcome"])
			}
			if subs[0]["failure_class"] != tc.class {
				t.Errorf("failure_class = %v, want %s", subs[0]["failure_class"], tc.class)
			}
			msg, _ := subs[0]["error"].(string)
			if tc.wantIn != "" && !strings.Contains(msg, tc.wantIn) {
				t.Errorf("error = %q, want it to contain %q", msg, tc.wantIn)
			}
			if tc.notWantIn != "" && strings.Contains(msg, tc.notWantIn) {
				t.Errorf("error = %q names %q, which a Joe box does not have", msg, tc.notWantIn)
			}
			if n := len(h.snapshotCalls()); n != tc.joeCalls {
				t.Errorf("%d requests reached Joe, want %d", n, tc.joeCalls)
			}
		})
	}
}
