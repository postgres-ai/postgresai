package platform

import (
	"context"
	"testing"
)

// The reply shape is shared across all three channels, so every guard the
// monitoring channel has -- a `null` body, a missing job_id, an echoed outcome
// that does not match -- must catch it on the Joe channel too. They are the only
// thing that says whether the answer landed.
func TestTheJoeRepliesAreGuardedTheSameWay(t *testing.T) {
	creds := Credentials{JoeToken: "joe-tok"}

	c, _ := recordCalls(t, `null`)
	if _, err := c.Poll(context.Background(), creds); err == nil {
		t.Fatal("a null poll reply was accepted")
	}

	c, _ = recordCalls(t, `{"jobs":[],"next_poll_ms":5000}`)
	if _, err := c.Poll(context.Background(), creds); err == nil {
		t.Fatal("a poll reply with no server_time was accepted")
	}

	c, _ = recordCalls(t, `{"job_id":"j1","status":"done","outcome":"skipped","error":null}`)
	err := c.Submit(context.Background(), creds, Submission{JobID: "j1", Outcome: OutcomeOK, Result: 1})
	if err == nil {
		t.Fatal("an outcome that came back different was accepted")
	}

	c, _ = recordCalls(t, `{"job_id":"j1","status":"done","outcome":"ok","error":"too large"}`)
	err = c.Submit(context.Background(), creds, Submission{JobID: "j1", Outcome: OutcomeOK, Result: 1})
	if err == nil {
		t.Fatal("a result the platform refused was recorded as landed")
	}
}

// A Joe box holds NO org token, so the credential in the header must be Joe's own
// even when both fields are somehow populated -- the config layer refuses that
// combination, and this is the half of the guarantee that lives here.
func TestTheJoeTokenIsTheCredentialSent(t *testing.T) {
	c, calls := recordCalls(t, `{"server_time":"2026-09-30T00:00:00Z","jobs":[],"next_poll_ms":5000}`)
	creds := Credentials{APIToken: "org-tok", JoeToken: "joe-tok"}
	if _, err := c.Poll(context.Background(), creds); err != nil {
		t.Fatal(err)
	}
	if got := (*calls)[0].Header; got != "joe-tok" {
		t.Fatalf("access-token = %q, want the Joe token", got)
	}
	if (*calls)[0].RPC != "joe_job_poll" {
		t.Fatalf("rpc = %q, want joe_job_poll", (*calls)[0].RPC)
	}
}
