// Package platform speaks to the PostgresAI platform's job RPCs over PostgREST.
//
// Authentication is the org API token the instance already holds in
// .pgwatch-config, plus the instance id. The token goes in the `access-token`
// HEADER, not the body: a body parameter becomes a bind parameter, and Postgres
// writes bind parameters into the server log for any statement over
// log_min_duration_statement -- which on the fleet's highest-frequency rpc is a
// plaintext credential in the logs. public.api_token_check reads the header
// when the parameter is null, and both rpcs are granted to pai_api_anonymous,
// so a header-only call authenticates. The token never reaches argv, a URL, or
// a log line.
package platform

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

// maxResponseBytes caps the response body we read from the platform.
const maxResponseBytes = 4 << 20

// maxMessageBytes bounds the platform message we carry into an error. PostgREST
// fills `message` from the raw PostgreSQL error, which can echo a value back.
const maxMessageBytes = 200

// maxCodeBytes bounds the code. It is supposed to be a PTxxx or PGRSTxxx, but
// it arrives from the other side like everything else, and the runner puts it
// in the health file -- where an unbounded value is an unbounded file.
const maxCodeBytes = 32

// Job is one unit of work handed out by the poll.
type Job struct {
	ID   string          `json:"id"`
	Kind string          `json:"kind"`
	Args json.RawMessage `json:"args"`
}

// PollResponse is v1.instance_job_poll's reply.
type PollResponse struct {
	ServerTime string `json:"server_time"`
	Jobs       []Job  `json:"jobs"`
	NextPollMS int    `json:"next_poll_ms"`
}

// ErrUnexpectedReply marks a 200 whose body is not the reply we asked for.
// PostgREST answers `null` for an rpc that returns SQL NULL, and both `null`
// and `{}` decode into a zero PollResponse without error -- which the runner
// would read as a healthy poll that happened to hand out no work, forever.
var ErrUnexpectedReply = errors.New("platform reply is not the expected shape")

// APIError is an error answer from the platform. Code is the body's `code`
// field -- a PostgREST error code (PGRSTxxx) or one of our own PTxxx SQLSTATEs.
//
// Callers MUST discriminate on Code, never on StatusCode: PostgREST maps a
// PTxxx onto the matching HTTP status, so a 404 from submit is the routine
// "that job is no longer yours" and a 404 from a platform that has not been
// migrated is PGRST202. Treating every 404 alike takes a healthy instance off
// the channel.
type APIError struct {
	StatusCode int
	Code       string
	Message    string
}

// truncateRunes cuts on a rune boundary: a PostgreSQL message is often
// non-ASCII, and half a rune is mojibake in the log and the health file.
func truncateRunes(s string, max int) string {
	if len(s) <= max {
		return s
	}
	// Walk back at most one rune's worth: further back means the string was
	// already invalid before the bound, and silently emptying it would be worse
	// than keeping what is there.
	cut := s[:max]
	for i := 0; i < utf8.UTFMax-1 && !utf8.ValidString(cut); i++ {
		cut = cut[:len(cut)-1]
	}
	return cut + "..."
}

func (e *APIError) Error() string {
	return fmt.Sprintf("platform returned %d (code %s): %s", e.StatusCode, e.Code, e.Message)
}

// Class is how the runner should react to a failed call.
type Class int

const (
	// ClassTransient: network trouble, a 5xx or a rate limit. Retry.
	ClassTransient Class = iota
	// ClassJobGone: PT404 from submit -- unknown, foreign, already-answered or
	// expired. Routine: drop the job and carry on polling.
	ClassJobGone
	// ClassAuth: the credential was rejected. Back off; polling harder will not
	// mint a new token.
	ClassAuth
	// ClassRequest: the platform refused what we sent. A bug on this side; do
	// not retry the same call.
	ClassRequest
	// ClassUnavailable: the RPC is not there -- an un-migrated platform or a
	// wrong api_base_url. Back off hard.
	ClassUnavailable
)

// Classify maps an error onto the runner's reaction.
func Classify(err error) Class {
	// A submission this package refused to send never reached the platform and
	// never will: retrying it would burn three attempts on the same refusal.
	if errors.Is(err, ErrUnknownOutcome) {
		return ClassRequest
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		// Network, DNS, TLS, timeout: nothing was decided platform-side.
		return ClassTransient
	}
	switch {
	case transientSQLStates[apiErr.Code]:
		// Checked FIRST: these arrive with whatever status PostgREST maps them
		// to, and the claim has already been reverted platform-side.
		return ClassTransient
	case apiErr.Code == "PT404":
		return ClassJobGone
	case apiErr.Code == "PT401" || apiErr.Code == "PT402":
		// PT402 is a suspended org. PT403 is deliberately NOT raised by this
		// channel: instance_job_auth answers PT404 for a foreign instance so an
		// org-token holder cannot learn whether a uuid names a real instance
		// somewhere else.
		return ClassAuth
	case apiErr.Code == "PT400":
		return ClassRequest
	case apiErr.StatusCode >= 500 || apiErr.StatusCode == http.StatusTooManyRequests:
		// Before the PGRST arm on purpose: PGRST000 ("could not connect to the
		// database") arrives as a 503 and is a blip, not a platform without the
		// channel. Parking for ten minutes on a database restart would be wrong.
		return ClassTransient
	case strings.HasPrefix(apiErr.Code, "PGRST"):
		// PGRST202 (no such function) and its neighbours mean this platform does
		// not have the channel; PGRST301 and friends are credential-shaped but
		// equally not something a retry loop fixes.
		return ClassUnavailable
	case apiErr.Code == "" && apiErr.StatusCode == http.StatusNotFound:
		// No error body at all: api_base_url points at something that is not
		// this platform's PostgREST.
		return ClassUnavailable
	case apiErr.StatusCode >= 300 && apiErr.StatusCode < 400:
		// A redirect we refused to follow (see NewClient). Something in front of
		// the platform is redirecting us, which is a wrong api_base_url, not a
		// bad request -- and a redirect body carries no PostgREST code, so
		// there is nothing else to discriminate on.
		return ClassUnavailable
	default:
		return ClassRequest
	}
}

// Credentials are read fresh on every call, so a .pgwatch-config written after
// the container started is picked up without a restart.
type Credentials struct {
	APIToken   string
	InstanceID string
}

// The outcome literals the rpc takes, and the only three it accepts.
const (
	OutcomeOK      = "ok"
	OutcomeSkipped = "skipped"
	OutcomeError   = "error"
)

// transientSQLStates are the PostgreSQL conditions the platform re-raises
// rather than recording against the box: it hit its own lock bound behind a
// pull-path consumer, or deadlocked with one. Both REVERT the claim, so the job
// is queued again and the answer is not lost -- as long as we retry rather than
// record a failure.
var transientSQLStates = map[string]bool{
	"55P03": true, // lock_not_available
	"40P01": true, // deadlock_detected
	"40001": true, // serialization_failure
	"57014": true, // query_canceled
}

// Client calls the platform's job RPCs.
type Client struct {
	httpClient    *http.Client
	baseURL       string
	clientVersion string
}

// NewClient builds a platform client. baseURL is the API root that already
// carries the reporter's RPCs, e.g. https://postgres.ai/api/general.
func NewClient(baseURL, clientVersion string, timeout time.Duration) *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: timeout,
			// Go drops only `Authorization` on a cross-host redirect; a custom
			// header like ours is replayed verbatim, so a 307 would hand the
			// org token to the redirect target. A PostgREST rpc never
			// legitimately redirects, so refuse to follow one.
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		baseURL:       strings.TrimRight(baseURL, "/"),
		clientVersion: clientVersion,
	}
}

// Poll asks for work.
func (c *Client) Poll(ctx context.Context, creds Credentials) (*PollResponse, error) {
	body := map[string]any{
		"instance_id":    creds.InstanceID,
		"client_version": c.clientVersion,
	}
	var out PollResponse
	if err := c.call(ctx, "instance_job_poll", creds, body, &out); err != nil {
		return nil, err
	}
	// server_time is unconditional in the rpc's json_build_object, so its
	// absence means we did not get the reply we asked for.
	if out.ServerTime == "" {
		return nil, fmt.Errorf("%w: no server_time", ErrUnexpectedReply)
	}
	return &out, nil
}

// Submission is one answered job. Outcome is REQUIRED: the rpc used to infer it
// from whether Error was set, which left a legitimate skip with no way to say so.
type Submission struct {
	JobID string
	// Outcome is "ok", "skipped" or "error". The three are mutually exclusive
	// with the fields below, and the platform answers PT400 to a contradiction
	// WITHOUT consuming the job, so a malformed submit is our bug to fix rather
	// than a lost collection.
	Outcome string
	// Result is the bare payload; only with Outcome "ok".
	Result any
	// SkipReason is "retention", "density" or "no_data"; only with "skipped".
	SkipReason string
	// Error and FailureClass; only with "error".
	Error        string
	FailureClass string
	DurationMS   int
}

// SubmitResult is the rpc's reply. Error is always present and is NULL unless
// the platform REFUSED the payload -- a rejected result is recorded on the job
// row, not raised, so a caller that ignores this believes a collection landed
// when it did not. Status is the row's lifecycle (done/failed); Outcome is what
// we told the platform happened, echoed back.
type SubmitResult struct {
	JobID      string `json:"job_id"`
	Status     string `json:"status"`
	Outcome    string `json:"outcome"`
	AcceptedAt string `json:"accepted_at"`
	Error      string `json:"error"`
}

// ErrResultRejected marks a submit the platform accepted as a request and
// refused as a payload.
var ErrResultRejected = errors.New("platform refused the result")

// ErrUnknownOutcome marks a submission this package refuses to send.
var ErrUnknownOutcome = errors.New("submission names no known outcome")

// Submit posts the answer to one job. A non-nil error means the answer did not
// land: either the call failed, or the platform refused the payload.
func (c *Client) Submit(ctx context.Context, creds Credentials, s Submission) error {
	body := map[string]any{
		"instance_id":    creds.InstanceID,
		"job_id":         s.JobID,
		"client_version": c.clientVersion,
		"duration_ms":    s.DurationMS,
		"outcome":        s.Outcome,
	}
	// Exactly the fields the named outcome allows: the platform rejects any
	// other combination, and rejects it without consuming the job.
	switch s.Outcome {
	case OutcomeOK:
		body["result"] = s.Result
	case OutcomeSkipped:
		body["skip_reason"] = s.SkipReason
	case OutcomeError:
		body["error"] = s.Error
		body["failure_class"] = s.FailureClass
	default:
		// Not sent at all. The platform answers PT400 to an unknown outcome and
		// does NOT consume the job, so it would sit 'running' until the hourly
		// sweep -- a local error is the cheaper failure.
		return fmt.Errorf("%w: outcome %q", ErrUnknownOutcome, s.Outcome)
	}
	var out SubmitResult
	if err := c.call(ctx, "instance_job_submit", creds, body, &out); err != nil {
		return err
	}
	// Same guard as the poll, and for the same reason: `null` and `{}` decode
	// into a zero SubmitResult without error, and this reply is the ONLY thing
	// that says whether the collection landed. job_id and status are both
	// unconditional in the rpc's json_build_object.
	if out.JobID == "" || out.Status == "" {
		return fmt.Errorf("%w: no job_id/status", ErrUnexpectedReply)
	}
	// The platform echoes the outcome we named. A different one means it did
	// not understand what we sent, which is not something to record as landed.
	if out.Outcome != s.Outcome {
		return fmt.Errorf("%w: outcome %q came back as %q", ErrUnexpectedReply, s.Outcome, out.Outcome)
	}
	if out.Error != "" {
		// The platform's own words, and it writes them into a customer-visible
		// column, so they are safe to carry.
		return fmt.Errorf("%w: %s", ErrResultRejected, out.Error)
	}
	return nil
}

// call posts one RPC. out may be nil when the reply is not needed.
func (c *Client) call(ctx context.Context, rpc string, creds Credentials, body map[string]any, out any) error {
	encoded, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/rpc/"+rpc, bytes.NewReader(encoded))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	// The credential, deliberately not a body parameter -- see the package
	// comment. api_token_check falls back to this header when the rpc's
	// api_token argument is null.
	req.Header.Set("access-token", creds.APIToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Never wrap with the request body or the headers: one of them carries
		// the token.
		return fmt.Errorf("%s request failed: %w", rpc, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return fmt.Errorf("%s response read failed: %w", rpc, err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return parseAPIError(resp.StatusCode, raw)
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("%s: %w", rpc, ErrUnexpectedReply)
	}
	return nil
}

// parseAPIError reads PostgREST's error envelope. `details` and `hint` are
// deliberately not carried: they can echo request parameters back.
func parseAPIError(status int, raw []byte) error {
	var envelope struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	// A non-JSON body (a proxy's HTML error page) leaves Code empty, which
	// Classify reads as "this is not our PostgREST".
	_ = json.Unmarshal(raw, &envelope)
	return &APIError{
		StatusCode: status,
		Code:       truncateRunes(envelope.Code, maxCodeBytes),
		Message:    truncateRunes(envelope.Message, maxMessageBytes),
	}
}
