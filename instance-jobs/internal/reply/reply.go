// Package reply carries a local service's HTTP reply into something the platform
// can store in its jsonb `result` column, and hands it back verbatim whenever
// that is possible.
//
// BOTH channels that relay a reply go through Carry -- the DBLab engine's
// (platform-all#815) and Joe's (#398) -- so the same endpoint answers the same
// shape whichever route reached it. That is the whole reason this is a package
// rather than a helper in each: the rules below are Postgres's and subtle, and a
// second copy of them is how the two routes come to disagree.
//
// Nothing here logs. A relayed body is customer data.
package reply

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

// MaxResultBytes is the platform's own cap on a submitted result, and since
// platform-all#815 it has to be checked against the ENVELOPE rather than the raw
// body: JSON-escaping text and base64-ing binary both inflate, so a body that
// passed a read bound can still be too large to submit. v1.dblab_job_submit and
// v1.joe_job_submit both measure octet_length(result::text) against the same
// number.
const MaxResultBytes = 1 << 20

// ErrOversize marks a reply that cannot be submitted because it does not fit.
// Permanent for this attempt's payload: the same call would produce the same
// unusable answer, so a caller reports it rather than retrying.
var ErrOversize = errors.New("reply is too large to submit")

// Carry decides what to hand the platform, and decides it on the REPLY rather
// than on the path. A channel that knew which endpoints answer in what would be
// wrong again the next time the service grew one.
//
// Three shapes: (nil, nil) for an empty body, which is an ANSWER and not a
// failure; a JSON body verbatim, never wrapped; and anything else in a
// self-describing envelope. json.Valid is Go's grammar and not what a jsonb
// column will hold, so the two classes Postgres refuses outright -- invalid
// UTF-8 (22021) and a \u0000 escape (22P05) -- take the envelope too.
func Carry(contentType string, raw []byte) (json.RawMessage, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	if json.Valid(raw) && utf8.Valid(raw) && !containsNulEscape(raw) {
		return json.RawMessage(raw), nil
	}
	return Encode(contentType, raw)
}

const (
	// encodingText is a body a jsonb string can hold verbatim.
	encodingText = "text"
	// encodingBase64 is one it cannot. Carrying it is what makes this a fix for
	// the class rather than for one endpoint: a gzip or an image round-trips too.
	encodingBase64 = "base64"
)

// contentTypeMaxBytes bounds the relayed header. The service is ours, but this
// value reaches a browser, and every other piece of text we forward is bounded.
const contentTypeMaxBytes = 256

// rawEnvelope is a reply that is not JSON, carried so it can round-trip through
// a jsonb column. It is SELF-DESCRIBING on purpose: wrapping the body in a bare
// json string is cheaper but leaves the caller needing to know which endpoints
// are wrapped, which is exactly the per-endpoint knowledge that drifts.
//
// One reserved key, so a caller's test for "is this an envelope?" is one lookup,
// and a distinctive one, so no reply is KNOWN to use it. Nothing enforces that: a
// reply shaped exactly like the envelope passes through and the console would
// unwrap it. Enforcing it would mean wrapping a JSON body for containing a
// string, which is the one thing this must not do.
type rawEnvelope struct {
	Body rawBody `json:"pgai_body"`
}

type rawBody struct {
	ContentType string `json:"content_type"`
	Encoding    string `json:"encoding"`
	Body        string `json:"body"`
}

// Encode wraps a non-JSON reply. Exported so the ceiling below can be pinned
// from both sides directly rather than only through a whole HTTP round trip.
func Encode(contentType string, raw []byte) (json.RawMessage, error) {
	// Cleaned BEFORE it is bounded: without any cleaning a header of invalid
	// bytes was carried as 768, because Go's encoder turns each into a 3-byte
	// U+FFFD; and cleaning after the cut would spend the 256 on bytes that are
	// then dropped. The NUL goes too -- it is valid UTF-8, so ToValidUTF8 would
	// leave the one escape the column refuses in the one field that is not the
	// body. Unreachable over the wire today (net/textproto rejects a control byte
	// in a header), so this is the belt.
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
	// Go escapes <, > and & by default, six bytes for one. jsonb's ::text renders
	// them back as themselves, so leaving the escaping on would measure this
	// payload against the platform's cap in a currency the platform does not use
	// -- and could refuse a body that would have fit.
	enc.SetEscapeHTML(false)
	if err := enc.Encode(env); err != nil {
		return nil, fmt.Errorf("reply could not be encoded: %w", err)
	}
	out := bytes.TrimRight(buf.Bytes(), "\n")

	// Against the cap MINUS what Postgres's renderer adds, because the platform
	// measures octet_length(result::text) and jsonb's ::text puts a space after
	// every separator. Measuring Go's compact bytes alone accepts an envelope the
	// submit then refuses with a PT400.
	limit := MaxResultBytes - jsonbSeparatorBytes
	if len(out) > limit {
		// Reported in the platform's currency, not the local budget: what it
		// measures is the STORED length, and what it accepts is MaxResultBytes.
		// "at most": Go escapes U+2028/U+2029 where jsonb writes them raw, so this
		// over-counts by 3 per occurrence. Over-refusal, never the reverse.
		return nil, fmt.Errorf("%w: carrying the %d-byte reply stores as at most %d bytes, over the %d the platform accepts",
			ErrOversize, len(raw), len(out)+jsonbSeparatorBytes, MaxResultBytes)
	}
	return json.RawMessage(out), nil
}

// jsonbSeparatorBytes is what Postgres adds to THIS envelope when it renders the
// stored jsonb back as text: one space after each of its four colons and two
// commas. Derived from the shape above, so it changes if a field does.
const jsonbSeparatorBytes = 6

// containsNulEscape reports whether a JSON body carries a \u0000 Postgres refuses
// ("unsupported Unicode escape sequence"). The substring is a PREFILTER only:
// `\\u0000` is a backslash followed by five characters and stores fine, so
// wrapping on the text alone would wrap a reply that had to pass through.
//
// The decision is made on the TOKEN STREAM rather than a decoded document,
// because unmarshalling loses the answer two ways -- a map keeps only the last of
// a duplicate key, and a number no float64 holds fails the whole decode while
// Postgres stores it in numeric and still refuses the escape. Both were measured
// relaying a NUL the column then rejected. Go resolves the escape into a real NUL
// in the token and leaves the literal alone, which is the whole reason this reads
// tokens instead of bytes.
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
