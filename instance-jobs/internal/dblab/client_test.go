package dblab

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const testVerifyToken = "dblab-verify-must-not-appear"

func newTestClient(t *testing.T, h http.HandlerFunc) *Client {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return NewClient(srv.URL, testVerifyToken, 5*time.Second)
}

func mustParse(t *testing.T, args string) Request {
	t.Helper()
	req, err := Parse([]byte(args))
	if err != nil {
		t.Fatalf("Parse(%s) = %v", args, err)
	}
	return req
}

// --- args -------------------------------------------------------------------

// THE GUARD THIS PACKAGE EXISTS FOR. The platform sends a path and this box
// supplies the host; a value that carries its own host would send the engine's
// verification token somewhere else. The platform rejects these too -- agreeing
// in one place only would put the guard on the far side of a channel this box
// does not control.
func TestParseRefusesAnythingThatIsNotAnEnginePath(t *testing.T) {
	for _, action := range []string{
		"//evil.example.com/x",
		"https://evil.example.com/x",
		"http:/x",
		"status",
		"",
		"/../../etc/passwd",
		"/clone/../../admin",
		// PERCENT-ENCODED traversal, and this is the case that makes the
		// box-side guard load-bearing rather than duplicated: the platform's
		// check is a literal `like '%..%'` on the raw string and passes all
		// three, while url.Parse decodes them into u.Path and resolvePath sees
		// the "..". Measured, not assumed: "/a/%2e%2e/b", "/a/..%2fb" and
		// "/a/%2E%2E/b" all parse to Path "/a/../b".
		"/a/%2e%2e/b",
		"/a/%2E%2E/b",
		"/a/..%2fb",
	} {
		args := `{"method":"GET","action":` + mustJSON(t, action) + `}`
		if _, err := Parse([]byte(args)); !errors.Is(err, ErrInvalidArgs) {
			t.Errorf("Parse(action=%q) = %v, want ErrInvalidArgs", action, err)
		}
	}
}

func TestParseRefusesAMethodTheEngineDoesNotTake(t *testing.T) {
	// PATCH is deliberately NOT here (#393): it is the verb the Console's clone
	// deletion-protection control sends, and the set stays closed around exactly
	// the four the engine answers.
	for _, method := range []string{"TRACE", "PUT", "CONNECT", "HEAD", "OPTIONS", ""} {
		args := `{"method":` + mustJSON(t, method) + `,"action":"/status"}`
		if _, err := Parse([]byte(args)); !errors.Is(err, ErrInvalidArgs) {
			t.Errorf("Parse(method=%q) = %v, want ErrInvalidArgs", method, err)
		}
	}
}

// Literal verbs rather than allowedMethods: a test that reads the map under test
// would pass against any map, including the one that is missing PATCH.
func TestParseAcceptsTheEngineVerbsAndNormalisesTheCase(t *testing.T) {
	for _, method := range []string{"get", "Post", "DELETE", "patch"} {
		req := mustParse(t, `{"method":"`+method+`","action":"/status"}`)
		if req.Method != strings.ToUpper(method) {
			t.Errorf("method %q parsed as %q", method, req.Method)
		}
	}
}

// The Console's "Deletion protection" control, the platform's only PATCH
// caller: a body-carrying write against one clone. Parse must keep the body --
// dropped, it reaches the engine bodyless, api.ReadJSON refuses an empty body
// and the toggle fails 400. A body of `null` or `{}` is the worse shape: the
// engine reads it as every field zero, CLEARS protection and answers 200.
func TestParseKeepsThePatchBodyForCloneProtection(t *testing.T) {
	req := mustParse(t,
		`{"method":"patch","action":"/clone/my-clone","data":{"protected":true,"protectionDurationMinutes":60},"purpose":"api_call"}`)

	if req.Method != "PATCH" {
		t.Errorf("method = %q, want PATCH", req.Method)
	}
	if req.Action != "/clone/my-clone" {
		t.Errorf("action = %q", req.Action)
	}
	var body map[string]any
	if err := json.Unmarshal(req.Data, &body); err != nil {
		t.Fatalf("data did not survive Parse: %v", err)
	}
	if body["protected"] != true {
		t.Errorf("data = %s, want the protection payload", req.Data)
	}
}

// `purpose` is the platform's own discriminator and this side must ignore it
// rather than fail on a key it does not know: the platform is free to add more.
func TestParseIgnoresPlatformOnlyKeys(t *testing.T) {
	req := mustParse(t, `{"method":"GET","action":"/status","purpose":"data_usage","future":1}`)
	if req.Action != "/status" {
		t.Errorf("action = %q", req.Action)
	}
}

// --- the call ---------------------------------------------------------------

func TestDoSendsTheVerificationTokenAndRelaysTheReply(t *testing.T) {
	var gotToken, gotPath, gotMethod, gotBody string
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotToken = r.Header.Get("Verification-Token")
		gotPath = r.URL.Path
		gotMethod = r.Method
		buf := make([]byte, 64)
		n, _ := r.Body.Read(buf)
		gotBody = string(buf[:n])
		w.Write([]byte(`{"pools":[{"fileSystem":{"used":7}}]}`))
	})

	payload, err := c.Do(context.Background(),
		mustParse(t, `{"method":"POST","action":"/clone","data":{"id":"x"}}`))
	if err != nil {
		t.Fatalf("Do = %v", err)
	}
	if gotToken != testVerifyToken {
		t.Errorf("Verification-Token = %q", gotToken)
	}
	if gotMethod != http.MethodPost || gotPath != "/clone" {
		t.Errorf("request was %s %s, want POST /clone", gotMethod, gotPath)
	}
	if !strings.Contains(gotBody, `"id":"x"`) {
		t.Errorf("body = %q, want the job's data", gotBody)
	}
	// Relayed VERBATIM: this box does not know what any endpoint returns, and
	// the platform stores the answer as it arrived.
	var back map[string]any
	if err := json.Unmarshal(payload, &back); err != nil {
		t.Fatalf("payload is not the engine's JSON: %v", err)
	}
	if _, ok := back["pools"]; !ok {
		t.Errorf("payload = %s, want the engine's own object", payload)
	}
}

// The clone-protection PATCH as it actually goes out (#393). The body and the
// content type are built generically, so what this pins is that a PATCH reaches
// the engine at all and reaches it whole -- verb, path and payload.
func TestDoSendsThePatchBodyToTheEngine(t *testing.T) {
	var gotMethod, gotPath, gotType string
	var gotBody []byte
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotType = r.Header.Get("Content-Type")
		gotBody, _ = io.ReadAll(r.Body)
		w.Write([]byte(`{"id":"my-clone","protected":true}`))
	})

	payload, err := c.Do(context.Background(),
		mustParse(t, `{"method":"patch","action":"/clone/my-clone","data":{"protected":true,"protectionDurationMinutes":60}}`))
	if err != nil {
		t.Fatalf("Do = %v", err)
	}
	if gotMethod != "PATCH" || gotPath != "/clone/my-clone" {
		t.Errorf("request was %s %s, want PATCH /clone/my-clone", gotMethod, gotPath)
	}
	if gotType != "application/json" {
		t.Errorf("Content-Type = %q", gotType)
	}
	var sent map[string]any
	if err := json.Unmarshal(gotBody, &sent); err != nil {
		t.Fatalf("the engine got %q, which is not the job's body: %v", gotBody, err)
	}
	if sent["protected"] != true || sent["protectionDurationMinutes"] != float64(60) {
		t.Errorf("the engine got %s, want the whole protection payload", gotBody)
	}
	if !strings.Contains(string(payload), `"protected":true`) {
		t.Errorf("payload = %s, want the engine's reply relayed", payload)
	}
}

func TestDoReportsANonSuccessAsAnEngineError(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"message":"clone not found"}`))
	})

	_, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/status"}`))
	var engineErr *EngineError
	if !errors.As(err, &engineErr) {
		t.Fatalf("Do = %v, want *EngineError", err)
	}
	if engineErr.StatusCode != http.StatusBadRequest {
		t.Errorf("StatusCode = %d", engineErr.StatusCode)
	}
	// Carried, because the engine is ours and its words name what was wrong.
	if engineErr.Message != "clone not found" {
		t.Errorf("Message = %q", engineErr.Message)
	}
	// A 4xx is the platform's call being wrong; re-sending it would be wrong
	// again. A 5xx and a 401 are momentary and are retried.
	if engineErr.Retryable() {
		t.Error("a 400 is retryable")
	}
	for _, code := range []int{500, 502, 401, 429} {
		if !(&EngineError{StatusCode: code}).Retryable() {
			t.Errorf("%d is not retryable", code)
		}
	}
}

// A 200 that is not JSON is still an ANSWER, and since platform-all#815 it is
// carried rather than refused -- see reply_test.go, which pins each shape
// against a reply captured off a real engine. What this keeps is the one thing that must
// still never happen: whatever the body was, the payload handed to the runner
// is valid JSON, because the platform stores it in a jsonb column.
func TestDoAlwaysProducesSomethingThePlatformCanStore(t *testing.T) {
	cases := []struct {
		name, body string
		wantNil    bool
	}{
		{"empty", "", true},
		{"whitespace", "   \n", true},
		{"html", "<html>hello</html>", false},
		{"truncated json", `{"a":`, false},
		{"yaml", "key: val\n", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tc.body))
			})
			payload, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/status"}`))
			if err != nil {
				t.Fatalf("Do = %v", err)
			}
			// Named rather than skipped by a nil guard: without this the two
			// empty cases assert nothing beyond err == nil.
			if tc.wantNil {
				if payload != nil {
					t.Fatalf("payload = %s, want nil", payload)
				}
				return
			}
			if !json.Valid(payload) {
				t.Fatalf("payload is not storable JSON: %q", payload)
			}
		})
	}
}

// The read cap's REFUSE side, at exactly one byte over -- truncating instead
// would submit a valid-looking partial answer.
//
// ONE byte, not ten. The body used to be the cap plus ten, which pins that a
// large reply is refused but not WHERE: mutating the guard to `len(raw) >
// maxResponseBytes+1`, so a reply one byte over is accepted, left this whole
// package green. TestDoRelaysAJsonReplyExactlyAtTheReadCap is the accept side,
// and the two one byte apart are what pin the boundary between them.
//
// 1048577 is written out by hand. Phrased as maxResponseBytes+1 the test reads
// the constant under test and passes against any value of it.
func TestDoRefusesAnOversizeReplyRatherThanTruncatingIt(t *testing.T) {
	const oneOverTheCap = 1048577 // maxResponseBytes + 1, spelled out by hand.
	// `{"pad":""}` is ten bytes of envelope, so the padding is the rest.
	body := `{"pad":"` + strings.Repeat("x", oneOverTheCap-10) + `"}`
	if len(body) != oneOverTheCap {
		t.Fatalf("fixture is %d bytes, want exactly %d", len(body), oneOverTheCap)
	}

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	})
	if _, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/status"}`)); !errors.Is(err, ErrOversizeReply) {
		t.Fatalf("Do(%d bytes) = %v, want ErrOversizeReply one byte over the cap", oneOverTheCap, err)
	}
}

// Go replays a custom header verbatim across hosts, so following a redirect
// would hand the engine's verification token to whatever sent it.
func TestDoRefusesToFollowARedirect(t *testing.T) {
	var elsewhereGotToken string
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		elsewhereGotToken = r.Header.Get("Verification-Token")
		w.Write([]byte(`{"stolen":true}`))
	}))
	t.Cleanup(elsewhere.Close)

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, elsewhere.URL+"/status", http.StatusTemporaryRedirect)
	})

	_, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/status"}`))
	var engineErr *EngineError
	if !errors.As(err, &engineErr) || engineErr.StatusCode != http.StatusTemporaryRedirect {
		t.Fatalf("Do = %v, want the 307 reported rather than followed", err)
	}
	if elsewhereGotToken != "" {
		t.Fatalf("the redirect target received the verification token %q", elsewhereGotToken)
	}
}

// *url.Error's text carries the request URL, and the URL carries the job's
// ACTION -- a clone id, a branch name, a snapshot name. That is what must not
// survive into an error the runner logs and reports.
func TestDoDropsTheRequestUrlFromATransportError(t *testing.T) {
	c := NewClient("http://127.0.0.1:1", testVerifyToken, time.Second)
	_, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/clone/secret-clone-name"}`))
	if !errors.Is(err, ErrEngineUnreachable) {
		t.Fatalf("Do = %v, want ErrEngineUnreachable", err)
	}
	if strings.Contains(err.Error(), "secret-clone-name") {
		t.Fatalf("the error carries the job's action: %v", err)
	}
	// ...and it still says what happened, or the sentinel is all a diagnosis has.
	if !strings.Contains(err.Error(), "refused") {
		t.Fatalf("the error lost the root cause: %v", err)
	}
}

// The chain survives the wrap: the runner classifies a deadline as a timeout,
// which is a different remedy from an engine that is not listening.
func TestDoKeepsADeadlineDistinguishable(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := c.Do(ctx, mustParse(t, `{"method":"GET","action":"/status"}`))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Do = %v, want a chain carrying context.DeadlineExceeded", err)
	}
}

func mustJSON(t *testing.T, v string) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
