package platform

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// call is one recorded request: which rpc, which credential, what body.
type call struct {
	RPC    string
	Header string
	Body   map[string]any
}

// recordCalls stands in for the platform and records what each rpc was asked.
func recordCalls(t *testing.T, reply string) (*Client, *[]call) {
	t.Helper()
	var calls []call
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		json.Unmarshal(raw, &body)
		parts := strings.Split(r.URL.Path, "/")
		calls = append(calls, call{
			RPC:    parts[len(parts)-1],
			Header: r.Header.Get("access-token"),
			Body:   body,
		})
		w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)
	return NewClient(srv.URL, "test", 5*time.Second), &calls
}

// The three channels are separate SURFACES on a shared table (platform-all#805,
// #398). Which rpc a box calls -- and whether it names an instance at all -- is
// decided entirely by which credential its config carries.
func TestTheCredentialPicksTheChannel(t *testing.T) {
	cases := []struct {
		name       string
		creds      Credentials
		wantPoll   string
		wantSubmit string
		wantHeader string
		// wantInstanceID is "" when the body must carry NO instance key.
		wantInstanceID string
	}{
		{
			name:           "monitoring",
			creds:          Credentials{APIToken: "org-tok", InstanceID: "11111111-1111-1111-1111-111111111111"},
			wantPoll:       "instance_job_poll",
			wantSubmit:     "instance_job_submit",
			wantHeader:     "org-tok",
			wantInstanceID: "11111111-1111-1111-1111-111111111111",
		},
		{
			name:       "dblab",
			creds:      Credentials{DBLabToken: "engine-tok"},
			wantPoll:   "dblab_job_poll",
			wantSubmit: "dblab_job_submit",
			// The ENGINE's own token, not an org one -- a DBLab box holds no org
			// token at all.
			wantHeader: "engine-tok",
		},
		{
			name:       "joe",
			creds:      Credentials{JoeToken: "joe-tok"},
			wantPoll:   "joe_job_poll",
			wantSubmit: "joe_job_submit",
			// Joe's own token, for the same reason, and no instance id either.
			wantHeader: "joe-tok",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, calls := recordCalls(t,
				`{"server_time":"2026-09-25T00:00:00Z","jobs":[],"next_poll_ms":5000}`)
			if _, err := c.Poll(context.Background(), tc.creds); err != nil {
				t.Fatalf("Poll = %v", err)
			}
			assertCall(t, (*calls)[0], tc.wantPoll, tc.wantHeader, tc.wantInstanceID)

			c2, calls2 := recordCalls(t,
				`{"job_id":"j1","status":"done","outcome":"ok","error":null}`)
			err := c2.Submit(context.Background(), tc.creds,
				Submission{JobID: "j1", Outcome: OutcomeOK, Result: map[string]any{"a": 1}})
			if err != nil {
				t.Fatalf("Submit = %v", err)
			}
			assertCall(t, (*calls2)[0], tc.wantSubmit, tc.wantHeader, tc.wantInstanceID)
		})
	}
}

func assertCall(t *testing.T, got call, wantRPC, wantHeader, wantInstanceID string) {
	t.Helper()
	if got.RPC != wantRPC {
		t.Fatalf("rpc = %q, want %q", got.RPC, wantRPC)
	}
	if got.Header != wantHeader {
		t.Fatalf("access-token = %q, want %q", got.Header, wantHeader)
	}
	// No channel may carry another's key, and only monitoring may carry an
	// instance id: PostgREST resolves an rpc by its body keys, so a stray one
	// matches no function and the call 404s rather than being ignored.
	for _, key := range []string{"dblab_instance_id", "joe_instance_id"} {
		if _, ok := got.Body[key]; ok {
			t.Fatalf("body carries %s, which no rpc takes: %v", key, got.Body)
		}
	}
	if wantInstanceID == "" {
		if _, ok := got.Body["instance_id"]; ok {
			t.Fatalf("a box-credentialled channel named an instance: %v", got.Body)
		}
		return
	}
	if got.Body["instance_id"] != wantInstanceID {
		t.Fatalf("instance_id = %#v, want %q", got.Body["instance_id"], wantInstanceID)
	}
}

// The reply shape is shared, so everything the monitoring channel's guards catch
// -- a `null` body, a missing job_id, an echoed outcome that does not match --
// must catch it on the DBLab channel too. They are the only thing that says
// whether the answer landed.
func TestTheDBLabRepliesAreGuardedTheSameWay(t *testing.T) {
	creds := Credentials{DBLabToken: "engine-tok"}

	c, _ := recordCalls(t, `null`)
	if _, err := c.Poll(context.Background(), creds); err == nil {
		t.Fatal("a null poll reply was accepted")
	}

	c, _ = recordCalls(t, `{"job_id":"j1","status":"done","outcome":"skipped","error":null}`)
	err := c.Submit(context.Background(), creds, Submission{JobID: "j1", Outcome: OutcomeOK, Result: 1})
	if err == nil {
		t.Fatal("an outcome that came back different was accepted")
	}
}
