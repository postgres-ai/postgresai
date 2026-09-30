// Package dblab runs one platform-authored API call against the DBLab engine on
// this box and hands the engine's reply back verbatim.
//
// The engine sits on the local network, addressed by a URL this box configures
// for itself. The platform never supplies a host: the job carries a METHOD, an
// absolute PATH and an optional body, and the path is resolved against the local
// base URL here. That split is the whole SSRF story -- see resolve.
//
// Nothing in this package logs. The action and the body are the platform's, but
// the engine's reply is customer data and a package that cannot log cannot leak
// it; failures are returned as errors for the runner to classify.
package dblab

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/reply"
)

// KindCall is the platform's job kind for an engine API call.
const KindCall = "dblab_call"

// maxResponseBytes caps the engine reply we read. A DBLab /status is a few KiB;
// the platform's own submit cap is 1 MiB and refuses anything above it, so
// reading more than this could only ever produce a payload the platform would
// reject after we had spent the memory on it. It is a READ bound and, since
// #392, no longer a sufficient condition for submittability -- see
// reply.MaxResultBytes, which is the one the platform actually checks.
const maxResponseBytes = 1 << 20

// maxErrorBodyBytes bounds the non-2xx read, which is a different question: no
// engine error message is near the success cap, and reading the full cap on
// every failure would cost heap in a small container for nothing.
const maxErrorBodyBytes = 8 << 10

// ErrInvalidArgs marks job args this box cannot act on. Permanent: the platform
// authored them, so retrying cannot make them different.
var ErrInvalidArgs = errors.New("invalid dblab_call args")

// Request is what the platform sends. `purpose` is the platform's own
// discriminator (public.data_usage_collect uses it to find its own jobs) and is
// deliberately not read here: this side executes the call and nothing else.
type Request struct {
	Method string          `json:"method"`
	Action string          `json:"action"`
	Data   json.RawMessage `json:"data"`
}

// EngineError is a non-2xx answer from the local engine.
type EngineError struct {
	StatusCode int
	// Message is the engine's own `message` field when it sent JSON. The engine
	// is ours and its errors name what was wrong with the call, which is the
	// whole diagnostic to whoever made it -- so it is carried rather than
	// replaced with a constant. It is still untrusted text: the runner bounds
	// and sanitises anything it forwards.
	Message string
}

func (e *EngineError) Error() string {
	return fmt.Sprintf("dblab engine returned %d", e.StatusCode)
}

// Retryable reports whether re-running the same call could plausibly succeed.
// A 401 is retryable here for the same reason the metric store's is: the engine
// is ours on this box, so a rejected verification token means the token and the
// engine's own config are momentarily out of step (a restart mid-rotation), not
// a revoked credential. A 4xx otherwise is the platform's call being wrong, and
// re-sending it would be wrong again.
func (e *EngineError) Retryable() bool {
	return e.StatusCode >= 500 || e.StatusCode == http.StatusUnauthorized ||
		e.StatusCode == http.StatusTooManyRequests
}

// Client calls the DBLab engine on this box.
type Client struct {
	httpClient  *http.Client
	baseURL     string
	verifyToken string
}

// NewClient builds an engine client. baseURL is the engine's address on this
// box; verifyToken is the engine's shared verification token.
func NewClient(baseURL, verifyToken string, timeout time.Duration) *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: timeout,
			// Refused for the same reason the platform client refuses one: a
			// custom header is replayed verbatim across hosts by Go's client, so
			// a redirect would hand the engine's verification token to whatever
			// sent it. An engine endpoint never legitimately redirects.
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		baseURL:     strings.TrimRight(baseURL, "/"),
		verifyToken: verifyToken,
	}
}

// Methods the engine understands, and the only ones this box will build a
// request for. The platform validates the same set; repeating it here is not
// redundancy but the refusal to let a future platform bug turn into an
// arbitrary verb against a local service.
//
// PATCH is here for the Console's clone deletion protection (#393), but the
// ACTION is the caller's: the engine also routes PATCH to /snapshot/{id} and
// /branch/{name}, where a `deleteAt` body schedules a deletion. That is ungated
// on the pull path and the job channel alike, deliberately, until
// platform-all#829 decides whether the gate keys on the effect. A WRITE either
// way: the runner picks the ladder on the method, and a non-GET gets one attempt.
var allowedMethods = map[string]bool{
	http.MethodGet:    true,
	http.MethodPost:   true,
	http.MethodPatch:  true,
	http.MethodDelete: true,
}

// Parse validates the job args. Separate from Do so a malformed job fails
// before any request is built, and so the failure is ErrInvalidArgs -- which the
// runner never retries.
func Parse(args []byte) (Request, error) {
	var req Request
	if err := json.Unmarshal(args, &req); err != nil {
		return Request{}, fmt.Errorf("%w: %v", ErrInvalidArgs, err)
	}
	req.Method = strings.ToUpper(strings.TrimSpace(req.Method))
	if !allowedMethods[req.Method] {
		return Request{}, fmt.Errorf("%w: method %q", ErrInvalidArgs, req.Method)
	}
	if _, err := resolvePath(req.Action); err != nil {
		return Request{}, err
	}
	return req, nil
}

// resolvePath turns the platform's action into a path, refusing anything that
// is not one.
//
// THE PLATFORM NEVER SUPPLIES A HOST, and this is where that is enforced rather
// than assumed. url.Parse of "//evil.example.com/x" yields a URL with a HOST and
// no scheme, so resolving it against the base would send the engine's
// verification token to evil.example.com -- the platform rejects that shape too,
// and agreeing about it in one place only would mean the guard lives on the far
// side of a channel this box does not control.
func resolvePath(action string) (string, error) {
	if !strings.HasPrefix(action, "/") || strings.HasPrefix(action, "//") {
		return "", fmt.Errorf("%w: action %q is not an absolute engine path", ErrInvalidArgs, action)
	}
	u, err := url.Parse(action)
	if err != nil {
		return "", fmt.Errorf("%w: action %q: %v", ErrInvalidArgs, action, err)
	}
	if u.Scheme != "" || u.Host != "" || u.Opaque != "" {
		return "", fmt.Errorf("%w: action %q carries a host", ErrInvalidArgs, action)
	}
	// This is the ONLY thing that stops a traversal: Do concatenates the base
	// and the path, which collapses nothing. The action is returned as it
	// arrived, so a percent-encoded ".." that url.Parse decoded into u.Path is
	// refused here rather than being re-encoded and passed on.
	if strings.Contains(u.Path, "..") {
		return "", fmt.Errorf("%w: action %q traverses", ErrInvalidArgs, action)
	}
	return action, nil
}

// Do runs one call and returns the engine's reply as something the platform can
// store in its jsonb `result` column. A json.RawMessage rather than a decoded
// shape: this box does not know what any endpoint returns and must not.
//
// Three shapes come back, decided by the REPLY and never by the path (platform-all#815) --
// a channel that knew which endpoints answer in what would be wrong again the
// next time the engine grew one: a JSON body verbatim, never wrapped; anything
// else in a reply.Encode envelope; and (nil, nil) for an empty 200, which is
// an answer rather than a failure. Each branch below says why.
func (c *Client) Do(ctx context.Context, req Request) (json.RawMessage, error) {
	path, err := resolvePath(req.Action)
	if err != nil {
		return nil, err
	}

	var body io.Reader
	if len(req.Data) > 0 {
		body = bytes.NewReader(req.Data)
	}

	httpReq, err := http.NewRequestWithContext(ctx, req.Method, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("Content-Type", "application/json")
	// The engine's shared secret. Never in the URL, never in an error: the
	// wrapping below drops *url.Error's request URL for the same reason.
	httpReq.Header.Set("Verification-Token", c.verifyToken)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		// Never wrap with the request or the headers: one of them carries the
		// verification token.
		var urlErr *url.Error
		if errors.As(err, &urlErr) {
			// rootCause, not urlErr.Err: *url.Error's own text carries the request
			// URL and *net.OpError's carries the dialled address, while the root is
			// the part that says what happened ("connection refused"). Two %w so
			// the chain survives -- the runner classifies a deadline before it
			// classifies an unreachable engine.
			return nil, fmt.Errorf("%w: %s: %w", ErrEngineUnreachable, urlErr.Op, rootCause(urlErr.Err))
		}
		return nil, fmt.Errorf("%w: %w", ErrEngineUnreachable, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var envelope struct {
			Message string `json:"message"`
		}
		_ = json.NewDecoder(io.LimitReader(resp.Body, maxErrorBodyBytes)).Decode(&envelope)
		return nil, &EngineError{StatusCode: resp.StatusCode, Message: envelope.Message}
	}

	// LimitReader plus one byte, so a reply AT the cap is told apart from one
	// over it: silently truncating would submit a valid-looking partial answer.
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		// Named, or classifyFailure's default arm reports a truncated body as
		// "metric store unreachable" -- a component a DBLab box does not have.
		return nil, fmt.Errorf("%w: response read failed: %w", ErrEngineUnreachable, err)
	}
	if len(raw) > maxResponseBytes {
		return nil, fmt.Errorf("%w: engine reply exceeds %d bytes", ErrOversizeReply, maxResponseBytes)
	}
	// Empty, verbatim or enveloped -- the SHARED decision (internal/reply), made
	// on the reply and never on the path, so Joe's channel answers the same shape
	// for the same body. An empty 200 is an ANSWER: the concern that used to fail
	// it belongs to public.data_usage_collect, which gates on
	// `result is not null` and is therefore blind to a null answer anyway.
	return reply.Carry(resp.Header.Get("Content-Type"), raw)
}

// rootCause unwraps to the innermost error.
func rootCause(err error) error {
	for {
		next := errors.Unwrap(err)
		if next == nil {
			return err
		}
		err = next
	}
}

// The one way a 200 can still fail to be an answer: it does not fit. Permanent
// for this attempt's payload, so the runner reports it rather than retrying.
//
// It IS reply.ErrOversize rather than a sentinel of its own, so the read bound
// here and the envelope's ceiling there are one condition for every caller that
// matches on it -- and Joe's channel refuses an oversize reply as the same thing.
//
// platform-all#815 removed the other two. A non-JSON body is now carried and an
// empty one is an answer, so ErrNonJSONReply and ErrEmptyReply no longer have a
// producer; leaving the sentinels behind would claim a behaviour that is gone.
var ErrOversizeReply = reply.ErrOversize

// ErrEngineUnreachable marks a transport failure against the local engine. It
// is a sentinel rather than a bare wrap because the runner's fallback
// classification is "metric store unreachable", which on a DBLab box would name
// a component that is not there.
//
// The wrap deliberately drops *url.Error's request URL: it carries the engine's
// address and the job's action, and this error is reported to the platform.
var ErrEngineUnreachable = errors.New("dblab engine unreachable")
