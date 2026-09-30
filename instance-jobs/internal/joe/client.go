// Package joe runs one platform-authored API call against the Joe on this box
// and hands Joe's reply back verbatim.
//
// Joe sits on the local network, addressed by a URL this box configures for
// itself. The platform never supplies a host: the job carries a METHOD, an
// absolute PATH and an optional body, and the path is resolved against the local
// base URL here -- the same split, and the same SSRF story, as internal/dblab's.
//
// TWO THINGS ARE JOE'S OWN. Every call is SIGNED rather than carrying a bearer
// header (see sign), and the signature covers the bytes actually transmitted, so
// only the side that builds the request can produce one. And the CHANNEL is a
// property of this box: the platform cannot name it, so Channels resolves it here
// and the caller puts it in the body before signing (#398).
//
// Nothing in this package logs. The action and the body are the platform's, but
// Joe's reply is customer data and a package that cannot log cannot leak it;
// failures are returned as errors for the runner to classify.
package joe

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
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

// KindCall is the platform's job kind for a Joe API call.
const KindCall = "joe_call"

// channelsAction is the path Channels asks. Named because it is also the value
// the platform puts in a job's `action` when it wants the list itself, and the
// two must not drift.
const channelsAction = "/webui/channels"

// maxResponseBytes caps the reply we read. Joe's are small -- a channel list is
// bytes and a command POST answers with nothing at all -- but the bound is the
// platform's: its submit cap is 1 MiB and refuses anything above it, so reading
// more could only produce a payload the platform would reject after we had spent
// the memory on it. It is a READ bound, not a sufficient condition for
// submittability -- reply.MaxResultBytes is the one the platform checks.
const maxResponseBytes = 1 << 20

// maxErrorBodyBytes bounds the non-2xx read. Joe's failures carry no envelope
// and often no body at all, so this is small on purpose: reading the full cap on
// every failure would cost heap in a small container for nothing.
const maxErrorBodyBytes = 8 << 10

// ErrInvalidArgs marks job args this box cannot act on. Permanent: the platform
// authored them, so retrying cannot make them different.
var ErrInvalidArgs = errors.New("invalid joe_call args")

// ErrNoChannels marks a Joe that ANSWERED A CHANNEL LIST and served none: a
// configured webui workspace whose `channels` list is empty, which Joe reports as
// a 200 with `[]` (joe/pkg/connection/webui/assistant.go). It is a state of the
// box's own config, not a blip, so it is permanent -- asking three times says the
// same thing -- and it is the ONE lookup outcome the runner turns into a skip.
//
// A Joe with an empty channelMapping is NOT this: it refuses to start
// (joe/cmd/joe/main.go), so it answers nothing and the lookup fails as
// ErrJoeUnreachable. A Joe with no webui communication type at all never registers
// the route, so the mux answers 404 -- a *JoeError, and an error rather than a
// skip, because a box that no longer serves webui is a fault.
var ErrNoChannels = errors.New("joe advertises no channels")

// ErrBadChannelList marks a 200 that is not a channel list at all -- an HTML
// error page from a proxy, a JSON error envelope, something else listening on
// Joe's port, or a shape a future Joe renamed.
//
// SEPARATE FROM ErrNoChannels, and that separation is the point: these bodies say
// nothing about how many channels Joe serves, so folding them in would submit a
// lost command as `skipped` and let the platform mark the job done with the box
// still reporting healthy. It is a fault, so it is reported as one.
//
// Retryable, unlike ErrNoChannels: this is not Joe's own config answering, and the
// lookup is a read that costs one local GET to repeat.
var ErrBadChannelList = errors.New("joe did not answer with a channel list")

// ErrJoeUnreachable marks a transport failure against the local Joe. A sentinel
// rather than a bare wrap because the runner's fallback classification is
// "metric store unreachable", which on a Joe box names a component that is not
// there.
//
// The wrap deliberately drops *url.Error's request URL: it carries Joe's address
// and the job's action, and this error is reported to the platform.
var ErrJoeUnreachable = errors.New("joe unreachable")

// ErrOversizeReply is the shared ceiling (internal/reply): the one way a 200 can
// still fail to be an answer is that it does not fit.
var ErrOversizeReply = reply.ErrOversize

// Request is what the platform sends. `purpose` is the platform's own
// discriminator and is deliberately not read here: this side executes the call
// and nothing else.
type Request struct {
	Method string          `json:"method"`
	Action string          `json:"action"`
	Data   json.RawMessage `json:"data"`
	// ResolveChannel asks this box to look its channel up and put it in the body
	// before signing. The platform CANNOT do this: the channel lives in Joe's own
	// config, and a pre-flight lookup from the platform is the round trip the
	// inversion exists to remove. Explicit rather than inferred from the action,
	// so nothing depends on this side recognising a path by name.
	ResolveChannel bool `json:"resolve_channel"`
}

// JoeError is a non-2xx answer from the local Joe.
type JoeError struct {
	StatusCode int
	// Message is whatever Joe said, and Joe usually says nothing: the verifier's
	// 403 and the command handler's 400 write a status and no body, and only
	// http.Error writes text. Carried when there is any, because Joe is ours and
	// its words name what was wrong with the call. Still untrusted text: the
	// runner bounds and sanitises anything it forwards.
	Message string
}

func (e *JoeError) Error() string {
	return fmt.Sprintf("joe returned %d", e.StatusCode)
}

// Retryable reports whether re-running the same call could plausibly succeed.
//
// A 403 is NOT retryable, and that is the one worth stating: it is how Joe
// refuses a signature, so it means this box's joe_verify_token and Joe's own
// signingSecret disagree -- a configuration fault that three more attempts
// restate. The signature FALLBACK below is not a retry of the same call; it is a
// second scheme, tried once.
//
// A 401 is retryable for the same reason the metric store's is: Joe is ours on
// this box, so a momentary rejection means it is restarting mid-rotation.
func (e *JoeError) Retryable() bool {
	return e.StatusCode >= 500 || e.StatusCode == http.StatusUnauthorized ||
		e.StatusCode == http.StatusTooManyRequests
}

// Client calls the Joe on this box.
type Client struct {
	httpClient *http.Client
	baseURL    string
	secret     []byte
}

// NewClient builds a Joe client. baseURL is Joe's address on this box;
// signingSecret is the HMAC key -- the box's joe_communication_signing_secret,
// which is Joe's own `signingSecret`.
func NewClient(baseURL, signingSecret string, timeout time.Duration) *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: timeout,
			// Refused for the same reason the platform client refuses one: Go
			// replays a custom header verbatim across hosts, so a redirect would
			// hand the signature -- and on a GET with no body, a signature that is
			// a constant function of the secret -- to whatever sent it. A Joe
			// endpoint never legitimately redirects.
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		baseURL: strings.TrimRight(baseURL, "/"),
		secret:  []byte(signingSecret),
	}
}

// The verbs Joe's webui surface understands, and the only ones this box will
// build a request for: /webui/channels is a GET, /webui/command and /webui/verify
// are POSTs, and there is nothing else. NARROWER THAN THE ENGINE'S on purpose --
// a PATCH or a DELETE against Joe has no meaning, so accepting one would only let
// a future platform bug become an arbitrary verb against a local service. The
// platform validates the same set; this is the half of that agreement which lives
// on the side of the channel that actually makes the request.
var allowedMethods = map[string]bool{
	http.MethodGet:  true,
	http.MethodPost: true,
}

// Parse validates the job args. Separate from Do so a malformed job fails before
// any request is built, and so the failure is ErrInvalidArgs -- which the runner
// never retries.
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
	// ONE NOTION OF EMPTY. `data: null` and an absent `data` are the same intent --
	// no payload -- but json.RawMessage keeps the four bytes `null`, so downstream
	// `len(body) > 0` read them as "this request has a body": the bodyless-GET
	// fallback in call was silently lost and `null` went on the wire as the payload,
	// while isJSONObjectOrEmpty and withChannel treated the very same input as
	// empty. Normalised here, once, rather than at each of the three places that
	// ask.
	//
	// A null INSIDE the data still survives untouched -- the platform does not strip
	// those and the body is a signed payload, so dropping one would sign a body the
	// platform never authored.
	if len(bytes.TrimSpace(req.Data)) == 0 || bytes.Equal(bytes.TrimSpace(req.Data), []byte("null")) {
		req.Data = nil
	}
	// Refused before any request is built rather than at substitution time: the
	// channel goes into a JSON OBJECT, so args that ask for the substitution and
	// carry something else are a job this box can never run.
	if req.ResolveChannel && !isJSONObjectOrEmpty(req.Data) {
		return Request{}, fmt.Errorf("%w: resolve_channel needs a json object body", ErrInvalidArgs)
	}
	return req, nil
}

// resolvePath turns the platform's action into a path, refusing anything that is
// not one. The DBLab arm refuses the identical shapes for the identical reason
// (internal/dblab.resolvePath); the two are kept apart because the sentinel and
// the verb set are each channel's own.
//
// THE PLATFORM NEVER SUPPLIES A HOST, and this is where that is enforced rather
// than assumed. url.Parse of "//evil.example.com/x" yields a URL with a HOST and
// no scheme, so resolving it against the base would send Joe's signature to
// evil.example.com.
func resolvePath(action string) (string, error) {
	if !strings.HasPrefix(action, "/") || strings.HasPrefix(action, "//") {
		return "", fmt.Errorf("%w: action %q is not an absolute joe path", ErrInvalidArgs, action)
	}
	u, err := url.Parse(action)
	if err != nil {
		return "", fmt.Errorf("%w: action %q: %v", ErrInvalidArgs, action, err)
	}
	if u.Scheme != "" || u.Host != "" || u.Opaque != "" {
		return "", fmt.Errorf("%w: action %q carries a host", ErrInvalidArgs, action)
	}
	// This is the ONLY thing that stops a traversal: the request below
	// concatenates the base and the path, which collapses nothing. The action is
	// used as it arrived, so a percent-encoded ".." that url.Parse decoded into
	// u.Path is refused here rather than being re-encoded and passed on.
	if strings.Contains(u.Path, "..") {
		return "", fmt.Errorf("%w: action %q traverses", ErrInvalidArgs, action)
	}
	return action, nil
}

// channelsFrom reads the elements of a list, requiring each to CARRY channel_id.
//
// THE SAME REASON THE WRAPPER NEEDS ITS KEY, one level down: decoding straight into
// []channel lets encoding/json ignore unknown fields, so a list of ANY objects --
// a generic REST collection on Joe's port, an element key a future Joe renamed, `[{}]`,
// `[null]` -- yields entries with a blank ID, which then reads as "Joe serves no
// channels" and submits a lost command as a skip with the box still green. An
// element that HAS the key and holds nothing usable is a different thing, and stays
// the skip.
func channelsFrom(raws []json.RawMessage) ([]channel, error) {
	list := make([]channel, 0, len(raws))
	for i, raw := range raws {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return nil, fmt.Errorf("%w: entry %d is not a channel: %v", ErrBadChannelList, i, err)
		}
		// A json `null` element unmarshals into a NIL map without erroring, so the
		// lookup below is what refuses it rather than the decode.
		if _, ok := fields[channelIDKey]; !ok {
			return nil, fmt.Errorf("%w: entry %d has no `%s`", ErrBadChannelList, i, channelIDKey)
		}
		var ch channel
		if err := json.Unmarshal(raw, &ch); err != nil {
			return nil, fmt.Errorf("%w: entry %d has a `%s` that is not a string: %v",
				ErrBadChannelList, i, channelIDKey, err)
		}
		list = append(list, ch)
	}
	return list, nil
}

// The signature scheme, and it is Joe's rather than ours: its verifier computes
// HMAC-SHA256 over bodyPrefix followed by the request body, hex-encodes it, and
// compares against the header with signaturePrefix stripped
// (joe/pkg/connection/webui/verifier.go).
const (
	signatureHeader  = "Verification-Signature"
	signaturePrefix  = "v0="
	bodyPrefix       = "v0:"
	emptyObjectBody  = "{}"
	channelIDKey     = "channel_id"
	maxChannelsBytes = 64 << 10
)

// sign returns the header value for a body.
//
// IT COVERS THE BYTES ACTUALLY TRANSMITTED, which is the whole reason the
// platform cannot pre-compute one and hand it over in the job: Joe reads
// r.Body and MACs what it read, so substituting the channel into the body
// changes the signature, and only the side that assembles the final bytes can
// produce it. It is also why the platform's own signature is taken over
// `payload::text` -- jsonb's rendering is what pg_http then puts on the wire.
func (c *Client) sign(body []byte) string {
	mac := hmac.New(sha256.New, c.secret)
	mac.Write([]byte(bodyPrefix))
	mac.Write(body)
	return signaturePrefix + hex.EncodeToString(mac.Sum(nil))
}

// channel is one entry of Joe's channel list.
type channel struct {
	ID string `json:"channel_id"`
}

// Channels asks the local Joe which channels it serves, in the order Joe
// advertises them.
//
// A READ, and the caller retries it: a failed lookup means the command was never
// sent, so repeating it cannot duplicate anything -- unlike the write that
// follows. Keeping the two apart is why this is a method rather than a step
// hidden inside Do.
func (c *Client) Channels(ctx context.Context) ([]string, error) {
	raw, err := c.channelsBody(ctx)
	if err != nil {
		return nil, err
	}
	list, err := parseChannelList(raw)
	if err != nil {
		return nil, err
	}
	// THE RAW ID GOES BACK ON THE WIRE, and only the usability test trims. Joe keys
	// msgProcessors on the id exactly as configured and getProcessingService is a
	// plain map lookup, so a padded id posted trimmed is a channel Joe does not
	// have -- a 400 on a write, which is never retried, so the command is lost on a
	// box where the pull path works (v1.joe_command_run takes `reply #>>
	// '{0,channel_id}'` raw too).
	ids := make([]string, 0, len(list))
	for _, ch := range list {
		if strings.TrimSpace(ch.ID) != "" {
			ids = append(ids, ch.ID)
		}
	}
	// Joe answered a list and it holds nothing usable: the ONLY lookup outcome that
	// is a skip rather than a fault.
	if len(ids) == 0 {
		return nil, ErrNoChannels
	}
	return ids, nil
}

// parseChannelList reads Joe's channel list, and refuses a body that is not one.
//
// Joe answers a BARE ARRAY -- [{"channel_id": "..."}] -- and the `channels`
// wrapper is accepted for the same reason v1.joe_command_run coalesces both: one
// of the two is what some version answers, and guessing wrong means no channel at
// all.
//
// IT DISPATCHES ON THE FIRST BYTE, and unmarshalling into an anonymous struct is
// exactly what it must not do: encoding/json ignores unknown fields, so EVERY json
// object decodes into a `channels` wrapper with a nil slice -- which then reads as
// "Joe serves no channels" and submits a lost command as a skip. The `channels`
// key has to be PRESENT for this to be that shape.
func parseChannelList(raw []byte) ([]channel, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		// reply.Carry answers (nil, nil) for an empty 200, which is a legitimate
		// answer on the command route and not one here.
		return nil, fmt.Errorf("%w: the reply is empty", ErrBadChannelList)
	}
	switch trimmed[0] {
	case '[':
		var raws []json.RawMessage
		if err := json.Unmarshal(trimmed, &raws); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrBadChannelList, err)
		}
		return channelsFrom(raws)
	case '{':
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(trimmed, &fields); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrBadChannelList, err)
		}
		inner, ok := fields["channels"]
		if !ok {
			return nil, fmt.Errorf("%w: the reply is a json object with no `channels`", ErrBadChannelList)
		}
		var raws []json.RawMessage
		if err := json.Unmarshal(inner, &raws); err != nil {
			return nil, fmt.Errorf("%w: `channels` is not a list: %v", ErrBadChannelList, err)
		}
		return channelsFrom(raws)
	default:
		return nil, fmt.Errorf("%w: the reply is neither a list nor an object", ErrBadChannelList)
	}
}

// channelsBody performs the signed GET.
func (c *Client) channelsBody(ctx context.Context) (json.RawMessage, error) {
	return c.call(ctx, http.MethodGet, channelsAction, nil, maxChannelsBytes)
}

// call sends one request, retrying ONCE on a 403 with the second signature
// scheme when there is a second one to try.
//
// WHY TWO SCHEMES. Joe MACs the bytes it received, and a GET's body is the thing
// HTTP implementations disagree about: the platform signs `v0:{}` and hands
// pg_http a `{}` body, pg_http sends a bodyless GET, and Joe -- having read
// nothing -- computes `v0:` and answers 403. That is why v1.joe_command_run
// retries, and this keeps the behaviour. This side sends no body and signs `v0:`
// FIRST, because that is the pair Joe actually verifies in production today; the
// `{}` pair is the belt, for a Joe or a proxy that materialises one.
//
// It applies to ANY bodyless GET rather than only to the channel lookup: a job
// whose action IS /webui/channels goes through Do, and having the belt on one
// route and not the other would be a difference nobody could predict from the
// outside. A request WITH a body has one true rendering and gets one attempt --
// and never a second one for a write, which must not be re-sent.
func (c *Client) call(ctx context.Context, method, path string, body []byte, readCap int64) (json.RawMessage, error) {
	raw, err := c.send(ctx, method, path, body, readCap)
	if method != http.MethodGet || len(body) > 0 {
		return raw, err
	}
	var joeErr *JoeError
	if errors.As(err, &joeErr) && joeErr.StatusCode == http.StatusForbidden {
		return c.send(ctx, method, path, []byte(emptyObjectBody), readCap)
	}
	return raw, err
}

// Do runs one call and returns Joe's reply as something the platform can store in
// its jsonb `result` column. A json.RawMessage rather than a decoded shape: this
// box does not know what any endpoint returns and must not.
//
// channelID is the value Channels resolved, and is used only when the args asked
// for it. The substitution happens BEFORE signing, necessarily.
func (c *Client) Do(ctx context.Context, req Request, channelID string) (json.RawMessage, error) {
	path, err := resolvePath(req.Action)
	if err != nil {
		return nil, err
	}

	body := []byte(req.Data)
	if req.ResolveChannel {
		if body, err = withChannel(body, channelID); err != nil {
			return nil, err
		}
	}
	return c.call(ctx, req.Method, path, body, maxResponseBytes)
}

// withChannel puts the resolved channel in the body.
//
// It OVERWRITES any value already there. A BELT rather than a precedence rule:
// public.joe_call_precheck answers PT400 to a body naming a channel alongside
// resolve_channel, so that combination is refused at enqueue and never arrives.
// If one ever did, the channel is a property of this box -- v1.joe_command_run
// takes Joe's FIRST advertised channel for the same reason -- so the value we
// just looked up is the right one.
func withChannel(data []byte, channelID string) ([]byte, error) {
	if channelID == "" {
		// Reached only if a caller asked for the substitution and passed nothing,
		// which is this process's bug rather than the platform's -- but an empty
		// channel_id reaches Joe as a 400 with no body, so it is named here.
		return nil, fmt.Errorf("%w: no channel was resolved", ErrInvalidArgs)
	}
	fields := map[string]json.RawMessage{}
	if len(bytes.TrimSpace(data)) > 0 && !bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		if err := json.Unmarshal(data, &fields); err != nil {
			return nil, fmt.Errorf("%w: body is not a json object: %v", ErrInvalidArgs, err)
		}
	}
	id, err := json.Marshal(channelID)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidArgs, err)
	}
	fields[channelIDKey] = id
	// Go sorts a map's keys, so the bytes are deterministic -- which matters only
	// for reading a test failure, never for the signature: that is computed over
	// whatever these bytes turn out to be.
	out, err := json.Marshal(fields)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidArgs, err)
	}
	return out, nil
}

// isJSONObjectOrEmpty reports whether data can take a channel_id.
func isJSONObjectOrEmpty(data []byte) bool {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return true
	}
	var probe map[string]json.RawMessage
	return json.Unmarshal(trimmed, &probe) == nil
}

// send performs one signed request and reads the reply.
func (c *Client) send(ctx context.Context, method, path string, body []byte, readCap int64) (json.RawMessage, error) {
	var reader io.Reader
	if len(body) > 0 {
		reader = bytes.NewReader(body)
	}
	httpReq, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("Content-Type", "application/json")
	// Signed over `body` -- the exact bytes bytes.NewReader will transmit. Never
	// in the URL and never in an error: the wrapping below drops *url.Error's
	// request URL for the same reason.
	httpReq.Header.Set(signatureHeader, c.sign(body))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		// Never wrap with the request or the headers: one of them carries the
		// signature.
		var urlErr *url.Error
		if errors.As(err, &urlErr) {
			// rootCause, not urlErr.Err: *url.Error's own text carries the request
			// URL and *net.OpError's carries the dialled address, while the root is
			// the part that says what happened ("connection refused"). Two %w so the
			// chain survives -- the runner classifies a deadline before it
			// classifies an unreachable Joe.
			return nil, fmt.Errorf("%w: %s: %w", ErrJoeUnreachable, urlErr.Op, rootCause(urlErr.Err))
		}
		return nil, fmt.Errorf("%w: %w", ErrJoeUnreachable, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &JoeError{StatusCode: resp.StatusCode, Message: errorMessage(resp.Body)}
	}

	// LimitReader plus one byte, so a reply AT the cap is told apart from one over
	// it: silently truncating would submit a valid-looking partial answer.
	raw, err := io.ReadAll(io.LimitReader(resp.Body, readCap+1))
	if err != nil {
		// Named, or classifyFailure's default arm reports a truncated body as
		// "metric store unreachable" -- a component a Joe box does not have.
		return nil, fmt.Errorf("%w: response read failed: %w", ErrJoeUnreachable, err)
	}
	if int64(len(raw)) > readCap {
		return nil, fmt.Errorf("%w: joe reply exceeds %d bytes", ErrOversizeReply, readCap)
	}
	// Empty, verbatim or enveloped -- the SHARED decision (internal/reply), so the
	// same body answers the same shape on either channel. AN EMPTY 200 IS THE
	// NORMAL ANSWER HERE, not an edge case: Joe's command handler hands the
	// message to a goroutine and returns without writing anything, so a
	// successful command legitimately carries no result at all.
	return reply.Carry(resp.Header.Get("Content-Type"), raw)
}

// errorMessage reads what Joe said about a failure. Joe has no error envelope --
// the verifier's 403 and the command handler's 400 write a status and nothing
// else, and only http.Error writes text -- so the JSON `message` of the engine's
// shape is tried first and the bounded raw text is the fallback.
func errorMessage(body io.Reader) string {
	raw, err := io.ReadAll(io.LimitReader(body, maxErrorBodyBytes))
	if err != nil || len(bytes.TrimSpace(raw)) == 0 {
		return ""
	}
	var envelope struct {
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &envelope) == nil && envelope.Message != "" {
		return envelope.Message
	}
	return strings.TrimSpace(string(raw))
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
