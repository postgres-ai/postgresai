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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"
)

// KindCall is the platform's job kind for an engine API call.
const KindCall = "dblab_call"

// maxResponseBytes caps the engine reply we read. A DBLab /status is a few KiB;
// the platform's own submit cap is 1 MiB and refuses anything above it, so
// reading more than this could only ever produce a payload the platform would
// reject after we had spent the memory on it. It is a READ bound and, since
// #392, no longer a sufficient condition for submittability -- see
// maxResultBytes below, which is the one the platform actually checks.
const maxResponseBytes = 1 << 20

// maxResultBytes is that platform cap, stated here because since platform-all#815 it has to
// be checked against the ENVELOPE rather than the raw body: JSON-escaping text
// and base64-ing binary both inflate, so a reply that passed maxResponseBytes
// can still be too large to submit. v1.dblab_job_submit measures
// octet_length(result::text) against the same number.
const maxResultBytes = 1 << 20

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
// else in an encodeRawBody envelope; and (nil, nil) for an empty 200, which is
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
	// Nothing to carry, and nothing wrong. The concern that used to fail this --
	// "the engine said nothing" recorded as a successful reading -- belongs to
	// public.data_usage_collect, and that consumer already gates on
	// `result is not null`, so a null answer is structurally invisible to it.
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	// json.Valid is Go's grammar, not what a jsonb column will hold. The same
	// two classes the envelope arm refuses have to be refused here, or the fix
	// covers every endpoint except the ones the console uses: invalid UTF-8
	// (22021) and a \u0000 escape (22P05). Both take the envelope instead.
	if json.Valid(raw) && utf8.Valid(raw) && !containsNulEscape(raw) {
		return json.RawMessage(raw), nil
	}
	return encodeRawBody(resp.Header.Get("Content-Type"), raw)
}

const (
	// encodingText is a body a jsonb string can hold verbatim.
	encodingText = "text"
	// encodingBase64 is one it cannot. Carrying it is what makes this a fix for
	// the class rather than for config.yaml: an endpoint that answers with a
	// gzip or an image round-trips too.
	encodingBase64 = "base64"
)

// contentTypeMaxBytes bounds the relayed header. The engine is ours, but this
// value reaches a browser, and every other piece of text this box forwards is
// bounded.
const contentTypeMaxBytes = 256

// rawEnvelope is a reply that is not JSON, carried so it can round-trip through
// a jsonb column. It is SELF-DESCRIBING on purpose: the alternative the issue
// weighed -- wrapping the body in a bare json string -- is cheaper but leaves
// the caller needing to know which endpoints are wrapped, which is exactly the
// per-endpoint knowledge that drifts.
//
// One reserved key, so a caller's test for "is this an envelope?" is one
// lookup, and a distinctive one, so no engine object is KNOWN to use it.
// Nothing enforces that: a reply shaped exactly like the envelope passes
// through and the console would unwrap it. Enforcing it would mean wrapping a
// JSON body for containing a string, which is the one thing this must not do.
type rawEnvelope struct {
	Body rawBody `json:"pgai_body"`
}

type rawBody struct {
	ContentType string `json:"content_type"`
	Encoding    string `json:"encoding"`
	Body        string `json:"body"`
}

// encodeRawBody wraps a non-JSON reply.
func encodeRawBody(contentType string, raw []byte) (json.RawMessage, error) {
	// Cleaned BEFORE it is bounded, the same order runner.truncate uses: without
	// any cleaning a header of invalid bytes was carried as 768, because Go's
	// encoder turns each into a 3-byte U+FFFD; and cleaning after the cut would
	// spend the 256 on bytes that are then dropped. The NUL goes too -- it is
	// valid UTF-8, so ToValidUTF8 would leave the one escape the column refuses
	// in the one field that is not the body. Unreachable over the wire today
	// (net/textproto rejects a control byte in a header), so this is the belt.
	cleanType := strings.ReplaceAll(strings.ToValidUTF8(contentType, ""), "\x00", "")
	env := rawEnvelope{Body: rawBody{ContentType: truncateUTF8(cleanType, contentTypeMaxBytes)}}

	// TEXT only for what a jsonb string can actually hold. utf8.Valid rules out
	// the silent U+FFFD substitution Go's encoder would otherwise make, and the
	// NUL check rules out \u0000 -- the one escape Postgres refuses outright
	// ("unsupported Unicode escape sequence"), which would turn a reply that is
	// merely unreadable into a submit the platform rejects.
	if utf8.Valid(raw) && !bytes.ContainsRune(raw, 0) {
		env.Body.Encoding = encodingText
		env.Body.Body = string(raw)
	} else {
		env.Body.Encoding = encodingBase64
		env.Body.Body = base64.StdEncoding.EncodeToString(raw)
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	// Go escapes <, > and & by default, six bytes for one. jsonb's ::text
	// renders them back as themselves, so leaving the escaping on would measure
	// this payload against the platform's cap in a currency the platform does
	// not use -- and could refuse a body that would have fit.
	enc.SetEscapeHTML(false)
	if err := enc.Encode(env); err != nil {
		return nil, fmt.Errorf("dblab reply could not be encoded: %w", err)
	}
	out := bytes.TrimRight(buf.Bytes(), "\n")

	// Against the cap MINUS what Postgres's renderer adds, because the platform
	// measures octet_length(result::text) and jsonb's ::text puts a space after
	// every separator. Measuring Go's compact bytes alone accepts an envelope
	// the submit then refuses with a PT400.
	limit := maxResultBytes - jsonbSeparatorBytes
	if len(out) > limit {
		// Reported in the platform's currency, not the local budget: what it
		// measures is the STORED length, and what it accepts is maxResultBytes.
		// "at most": Go escapes U+2028/U+2029 where jsonb writes them raw, so
		// this over-counts by 3 per occurrence. Over-refusal, never the reverse.
		return nil, fmt.Errorf("%w: carrying the %d-byte reply stores as at most %d bytes, over the %d the platform accepts",
			ErrOversizeReply, len(raw), len(out)+jsonbSeparatorBytes, maxResultBytes)
	}
	return json.RawMessage(out), nil
}

// jsonbSeparatorBytes is what Postgres adds to THIS envelope when it renders
// the stored jsonb back as text: one space after each of its four colons and
// two commas. Derived from the shape above, so it changes if a field does.
const jsonbSeparatorBytes = 6

// containsNulEscape reports whether a JSON body carries a \u0000 Postgres
// refuses ("unsupported Unicode escape sequence"). The substring is a PREFILTER
// only: `\\u0000` is a backslash followed by five characters and stores fine, so
// wrapping on the text alone would wrap a reply that had to pass through.
//
// The decision is made on the TOKEN STREAM rather than a decoded document,
// because unmarshalling loses the answer two ways -- a map keeps only the last
// of a duplicate key, and a number no float64 holds fails the whole decode
// while Postgres stores it in numeric and still refuses the escape. Both were
// measured relaying a NUL the column then rejected. Go resolves the escape into
// a real NUL in the token and leaves the literal alone, which is the whole
// reason this reads tokens instead of bytes.
func containsNulEscape(raw []byte) bool {
	if !bytes.Contains(raw, []byte("u0000")) {
		return false
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	// UseNumber, or a big exponent ends the scan early on a body that json.Valid
	// already accepted.
	dec.UseNumber()
	for {
		tok, err := dec.Token()
		if err != nil {
			return false
		}
		if str, ok := tok.(string); ok && strings.ContainsRune(str, 0) {
			return true
		}
	}
}

// truncateUTF8 cuts s to at most n bytes without splitting a rune.
func truncateUTF8(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
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
// platform-all#815 removed the other two. A non-JSON body is now carried (encodeRawBody)
// and an empty one is an answer, so ErrNonJSONReply and ErrEmptyReply no longer
// have a producer; leaving the sentinels behind would claim a behaviour that
// is gone.
var ErrOversizeReply = errors.New("dblab engine reply is too large")

// ErrEngineUnreachable marks a transport failure against the local engine. It
// is a sentinel rather than a bare wrap because the runner's fallback
// classification is "metric store unreachable", which on a DBLab box would name
// a component that is not there.
//
// The wrap deliberately drops *url.Error's request URL: it carries the engine's
// address and the job's action, and this error is reported to the platform.
var ErrEngineUnreachable = errors.New("dblab engine unreachable")
