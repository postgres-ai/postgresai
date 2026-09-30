package dblab

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/reply"
	"unicode/utf8"
)

// Literals, not the constants under test: a suite that imports the constant it
// is asserting compiles only against the new code, and a red run that does not
// build proves less than one that runs and fails.
const (
	wantBodyKey        = "pgai_body"
	wantContentTypeMax = 256
	submitCapBytes     = 1 << 20 // v1.dblab_job_submit's cap on result::text
	readCapBytes       = 1 << 20 // what Do reads before calling a reply oversize
)

// The replies in testdata/ came off a real CE 4.2.0 engine (see its README).
// A YAML string written here would only prove that the code handles the string
// its author imagined; issue platform-all#815 was found because the real one is 15 KiB of
// comments, masked secrets and non-ASCII-free text.
func realReply(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	if len(raw) == 0 {
		t.Fatalf("%s is empty -- the capture did not make it into the repo", name)
	}
	return raw
}

// decodeRaw unwraps the envelope and returns the body as the bytes the engine
// sent. It fails the test if what came back is not an envelope at all.
//
// It returns the ENCODING too, and that is load-bearing: both arms decode to
// the same bytes, so a round-trip assertion alone passes whether the body went
// to base64 or to text -- and the whole point of the base64 arm is that text
// would have been refused by the platform.
func decodeRaw(t *testing.T, payload json.RawMessage) (contentType, encoding string, body []byte) {
	t.Helper()
	var env struct {
		Body *struct {
			ContentType string `json:"content_type"`
			Encoding    string `json:"encoding"`
			Body        string `json:"body"`
		} `json:"pgai_body"`
	}
	if err := json.Unmarshal(payload, &env); err != nil {
		t.Fatalf("payload is not JSON: %v", err)
	}
	if env.Body == nil {
		t.Fatalf("payload carries no %q: %s", wantBodyKey, payload)
	}
	switch env.Body.Encoding {
	case "text":
		return env.Body.ContentType, "text", []byte(env.Body.Body)
	case "base64":
		decoded, err := base64.StdEncoding.DecodeString(env.Body.Body)
		if err != nil {
			t.Fatalf("body is not base64: %v", err)
		}
		return env.Body.ContentType, "base64", decoded
	default:
		t.Fatalf("unknown encoding %q", env.Body.Encoding)
		return "", "", nil
	}
}

// THE BUG (platform-all#815). Measured on a rig as
// `/admin/config.yaml | failed | the engine reply is not JSON`: the platform
// stores a result in a jsonb column, so a YAML body had nowhere to go and every
// console page load that reached the Configuration surface failed.
func TestDoCarriesTheEnginesRealConfigYaml(t *testing.T) {
	want := realReply(t, "admin_config.yaml.golden")
	const wantType = "application/yaml; charset=utf-8"

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", wantType)
		w.Write(want)
	})

	payload, err := c.Do(context.Background(),
		mustParse(t, `{"method":"GET","action":"/admin/config.yaml"}`))
	if err != nil {
		t.Fatalf("Do = %v, want the engine's YAML carried", err)
	}

	gotType, gotEnc, got := decodeRaw(t, payload)
	if !bytes.Equal(got, want) {
		t.Errorf("body round-tripped to %d bytes, want the engine's %d", len(got), len(want))
	}
	// Plain UTF-8 YAML: a jsonb string holds it verbatim, so base64 here would
	// be a caller decoding for nothing.
	if gotEnc != "text" {
		t.Errorf("encoding = %q, want text", gotEnc)
	}
	// The content type is the half that makes the envelope self-describing: a
	// caller decides what the bytes are without knowing which endpoint it asked.
	if gotType != wantType {
		t.Errorf("content_type = %q, want %q", gotType, wantType)
	}
}

// /admin/config.yaml is NOT the only one. /metrics answers in the Prometheus
// text exposition format, and an endpoint added next year will answer in
// something else again -- so the envelope carries a content type rather than
// the channel learning a list of paths.
func TestDoCarriesTheEnginesRealMetrics(t *testing.T) {
	want := realReply(t, "metrics.txt.golden")
	const wantType = "text/plain; version=0.0.4; charset=utf-8; escaping=underscores"

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", wantType)
		w.Write(want)
	})

	payload, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/metrics"}`))
	if err != nil {
		t.Fatalf("Do = %v", err)
	}
	gotType, gotEnc, got := decodeRaw(t, payload)
	if !bytes.Equal(got, want) {
		t.Errorf("body round-tripped to %d bytes, want %d", len(got), len(want))
	}
	if gotEnc != "text" {
		t.Errorf("encoding = %q, want text", gotEnc)
	}
	if gotType != wantType {
		t.Errorf("content_type = %q, want %q", gotType, wantType)
	}
}

// THE OTHER HALF OF platform-all#815, and the worse one. DELETE /clone/{id} and
// POST /clone/{id}/reset return a 200 with NO BODY (engine routes.go: neither
// handler calls api.Write*). Measured on the rig against a real clone:
//
//	DELETE /clone/ui-proof-clone
//	  HTTP/1.1 200 OK
//	  Content-Length: 0
//	  body bytes: 0          <- and NO Content-Type header at all
//
// The pull path relays that as a null and the console reads it as success.
// Refusing it here made an inverted instance report a red error over a clone
// that was already gone -- on a verb that is never retried, so the user could
// not even tell whether it had happened.
//
// AN EMPTY BODY IS NOT A NON-JSON BODY; it is an ABSENT one, and the two are
// answered differently on purpose. Conflating them is how `null` came to mean
// "success" here in the first place: an envelope around "" would be a value
// where the engine sent none.
func TestDoTreatsAnEmpty200AsTheAnswerItIs(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"no body at all", ""},
		{"whitespace only", "   \n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tc.body))
			})
			payload, err := c.Do(context.Background(),
				mustParse(t, `{"method":"DELETE","action":"/clone/abc"}`))
			if err != nil {
				t.Fatalf("Do = %v, want an empty answer", err)
			}
			// nil, not an envelope around "": the platform's `result` stays SQL
			// NULL, which is exactly what the pull path's null reaches the
			// console as, and what public.data_usage_collect's
			// `result is not null` gate already skips.
			if payload != nil {
				t.Errorf("payload = %s, want nil", payload)
			}
		})
	}
}

// THE PAIR THAT SETTLES THE DESIGN. /admin/config and /admin/config.yaml are
// siblings on the same engine, differing only in the suffix, and one is JSON
// while the other is not -- so whether a reply is wrapped is decided by the
// REPLY and never by the path. Both bodies here are real captures off the same
// engine in the same session.
func TestTheSiblingConfigEndpointsAreToldApartByTheirReply(t *testing.T) {
	asJSON := realReply(t, "admin_config.json.golden")
	asYAML := realReply(t, "admin_config.yaml.golden")

	serve := func(contentType string, body []byte) json.RawMessage {
		t.Helper()
		c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", contentType)
			w.Write(body)
		})
		payload, err := c.Do(context.Background(),
			mustParse(t, `{"method":"GET","action":"/admin/config"}`))
		if err != nil {
			t.Fatalf("Do = %v", err)
		}
		return payload
	}

	// The JSON sibling: through untouched, byte for byte, no envelope.
	plain := serve("application/json; charset=utf-8", asJSON)
	if !bytes.Equal(plain, asJSON) {
		t.Errorf("the JSON config was not relayed verbatim")
	}
	if bytes.Contains(plain, []byte(wantBodyKey)) {
		t.Errorf("the JSON config was wrapped: %s", plain[:80])
	}

	// The YAML sibling, at a path that only DIFFERS BY SUFFIX: wrapped.
	wrapped := serve("application/yaml; charset=utf-8", asYAML)
	gotType, _, got := decodeRaw(t, wrapped)
	if !bytes.Equal(got, asYAML) {
		t.Errorf("the YAML config round-tripped to %d bytes, want %d", len(got), len(asYAML))
	}
	if gotType != "application/yaml; charset=utf-8" {
		t.Errorf("content_type = %q", gotType)
	}
}

// The constraint the fix is not allowed to break: a JSON reply is passed
// through untouched, byte for byte, and is never wrapped. Anything else would
// change every endpoint that works today in order to fix one that does not.
func TestDoLeavesAJsonReplyExactlyAsItArrived(t *testing.T) {
	// Deliberately awkward: odd key order, odd spacing, a unicode escape, an
	// HTML-significant character. A re-encode would disturb all four.
	want := []byte("{\"z\":1,\n  \"a\" : [null,true,\"<&>\",\"\\u00e9\"] }")

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Write(want)
	})

	payload, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/status"}`))
	if err != nil {
		t.Fatalf("Do = %v", err)
	}
	if !bytes.Equal(payload, want) {
		t.Errorf("payload = %s, want the engine's own bytes %s", payload, want)
	}
	if bytes.Contains(payload, []byte(wantBodyKey)) {
		t.Errorf("a JSON reply was wrapped: %s", payload)
	}
}

// A jsonb string cannot hold every byte sequence, and the two it cannot hold
// are not exotic: Go's encoder silently substitutes U+FFFD for invalid UTF-8,
// and Postgres rejects \u0000 outright ("unsupported Unicode escape sequence"),
// which would turn an unreadable reply into a refused submit. Both go to
// base64, and both must come back byte for byte.
func TestDoBase64sABodyAJsonbStringCannotHold(t *testing.T) {
	cases := []struct {
		name string
		body []byte
	}{
		{"invalid utf-8", []byte{0x1f, 0x8b, 0x08, 0xff, 0xfe, 'a'}},
		{"embedded NUL", []byte("ok\x00then")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if utf8.Valid(tc.body) && !bytes.ContainsRune(tc.body, 0) {
				t.Fatalf("fixture %q is not actually unholdable", tc.body)
			}
			c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/octet-stream")
				w.Write(tc.body)
			})
			payload, err := c.Do(context.Background(),
				mustParse(t, `{"method":"GET","action":"/observation/download"}`))
			if err != nil {
				t.Fatalf("Do = %v", err)
			}
			if !json.Valid(payload) {
				t.Fatalf("payload is not valid JSON: %q", payload)
			}
			_, gotEnc, got := decodeRaw(t, payload)
			if !bytes.Equal(got, tc.body) {
				t.Errorf("round-tripped to %q, want %q", got, tc.body)
			}
			// THE assertion. Both arms decode to the same bytes, so without
			// this the text arm passes too -- and a text-encoded \u0000 is
			// the refused submit this case exists to prevent.
			if gotEnc != "base64" {
				t.Errorf("encoding = %q, want base64", gotEnc)
			}
		})
	}
}

// A broken JSON reply is carried too, and that is the point of choosing a
// self-describing envelope over a per-endpoint rule: the channel never has to
// know which paths answer in what. The caller gets the bytes and can see what
// the engine actually said.
func TestDoCarriesATruncatedJsonReplyRatherThanRefusingIt(t *testing.T) {
	const body = `{"a":`
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(body))
	})
	payload, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/status"}`))
	if err != nil {
		t.Fatalf("Do = %v", err)
	}
	if _, _, got := decodeRaw(t, payload); string(got) != body {
		t.Errorf("body = %q, want %q", got, body)
	}
}

// The bound, and it is on the ENVELOPE rather than the raw body: escaping a
// text body and base64ing a binary one both inflate, so a reply that passed the
// read cap can still be too large for the platform to accept. Finding that out
// here is what stops the box submitting a payload the platform will refuse.
func TestDoRefusesAnEnvelopeTooLargeToSubmit(t *testing.T) {
	// Well under the READ cap, three quarters of it, and unholdable as text --
	// so base64 pushes the envelope over the submit cap while the raw body
	// never comes near it.
	body := append([]byte{0xff}, bytes.Repeat([]byte("\x00"), submitCapBytes*3/4)...)
	if len(body) > readCapBytes {
		t.Fatalf("fixture is %d bytes, which the read cap refuses first", len(body))
	}
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	})
	_, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/metrics"}`))
	if !errors.Is(err, ErrOversizeReply) {
		t.Fatalf("Do = %v, want ErrOversizeReply", err)
	}
}

// A hostile or broken engine sets the header; it is relayed to a browser, so it
// is bounded here like any other relayed text.
func TestDoBoundsTheContentTypeItCarries(t *testing.T) {
	cases := []struct{ name, pad, wantSuffix string }{
		{name: "ascii", pad: strings.Repeat("x", 4096)},
		// Sized so the 256-byte cut lands INSIDE a rune: "text/plain; pad=" is
		// 16 bytes, +238 = 254, and the euro sign occupies 254..256. Cutting on
		// the byte leaves a half rune, Go's encoder substitutes U+FFFD, and the
		// console is shown a content type the engine never sent. Padding with
		// ASCII never reaches that branch at all.
		{name: "multibyte", pad: strings.Repeat("a", 238) + strings.Repeat("\u20ac", 50)},
		// obs-text: net/textproto lets bytes >= 0x80 through, and Go's encoder
		// then turns each invalid one into a 3-byte U+FFFD -- so a header cut
		// to 256 bytes can still be carried as 768.
		{name: "invalid utf-8", pad: strings.Repeat("\xff", 300)},
		// Cleaning happens BEFORE the cut, so the 256 bytes are spent on header
		// text rather than on bytes that are about to be dropped: all 200 a's
		// survive here, where cutting first would keep 140 of them.
		{
			name:       "invalid bytes ahead of real text",
			pad:        strings.Repeat("\xff", 100) + strings.Repeat("a", 200),
			wantSuffix: strings.Repeat("a", 200),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/plain; pad="+tc.pad)
				w.Write([]byte("not json"))
			})
			payload, err := c.Do(context.Background(),
				mustParse(t, `{"method":"GET","action":"/metrics"}`))
			if err != nil {
				t.Fatalf("Do = %v", err)
			}
			gotType, _, _ := decodeRaw(t, payload)
			if len(gotType) > wantContentTypeMax {
				t.Errorf("content_type is %d bytes, want at most %d", len(gotType), wantContentTypeMax)
			}
			if !utf8.ValidString(gotType) {
				t.Errorf("content_type is not valid UTF-8: %q", gotType)
			}
			// The header held no U+FFFD, so one in the payload was made here.
			if strings.ContainsRune(gotType, utf8.RuneError) {
				t.Errorf("content_type was cut mid-rune: %q", gotType)
			}
			if tc.wantSuffix != "" && !strings.HasSuffix(gotType, tc.wantSuffix) {
				t.Errorf("content_type is %d bytes and does not end in the %d bytes of real header that fit",
					len(gotType), len(tc.wantSuffix))
			}
		})
	}
}

// jsonbWouldRefuse reports what Postgres refuses about a value bound to a jsonb
// column: bytes that are not UTF-8 (22021), or a string holding a NUL (22P05,
// "unsupported Unicode escape sequence").
//
// This RESTATES the rule the client applies; it does not independently verify
// it, and a comment here once claimed otherwise. A Go oracle that decided jsonb
// acceptance for itself would be the model of Postgres this branch declined to
// build. The independent check is the platform half's SQL test, which runs
// against a real Postgres.
func jsonbWouldRefuse(t *testing.T, payload []byte) bool {
	t.Helper()
	if !utf8.Valid(payload) {
		return true
	}
	// The token stream, not a decoded document: unmarshalling drops a shadowed
	// duplicate key and gives up on a number no float64 holds, and a NUL hiding
	// in either is still a NUL to Postgres.
	dec := json.NewDecoder(bytes.NewReader(payload))
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

// A body can be structurally valid JSON and STILL be a value the column
// refuses, so the rule the envelope arm applies has to be applied here too --
// otherwise the fix covers every endpoint except the ones the console uses.
// Measured against a real Postgres, not inferred:
//
//	'{"a":"caf\xe9"}'::jsonb   -> 22021 invalid byte sequence for encoding UTF8
//	'{"a":"x\u0000y"}'::jsonb  -> 22P05 unsupported Unicode escape sequence
//
// json.Valid sees neither: its scanner accepts any byte >= 0x20 inside a
// string, and a \u0000 ESCAPE is six perfectly ordinary ASCII characters.
//
// The encodings differ on purpose. Invalid UTF-8 has to be base64'd; the NUL
// escape does not, because wrapping already neutralises it -- inside the
// envelope's `body` string the backslash is itself escaped, so what reaches
// the column is the literal text and not an escape.
func TestDoDoesNotPassThroughJsonTheColumnWouldRefuse(t *testing.T) {
	cases := []struct {
		name    string
		body    []byte
		wantEnc string
	}{
		{"invalid utf-8", []byte("{\"name\":\"caf\xe9\"}"), "base64"},
		{"nul escape", []byte(`{"name":"a\u0000b"}`), "text"},
		// Unmarshalling into a map keeps only the last value, so a walk over
		// the decoded document never sees this one.
		{"nul under a shadowed duplicate key", []byte(`{"a":"x\u0000y","a":"z"}`), "text"},
		// And unmarshalling gives up entirely here ("cannot unmarshal number
		// 1e400 into Go value of type float64"), so a walk never runs at all --
		// while Postgres holds 1e400 in numeric and still refuses the escape.
		{"nul beside a number no float64 holds", []byte(`{"n":1e400,"a":"x\u0000y"}`), "text"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !json.Valid(tc.body) {
				t.Fatalf("fixture is not valid JSON, so it never reaches the pass-through arm")
			}
			if !jsonbWouldRefuse(t, tc.body) {
				t.Fatalf("fixture is storable as it stands, so it proves nothing")
			}

			c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.Write(tc.body)
			})
			payload, err := c.Do(context.Background(),
				mustParse(t, `{"method":"GET","action":"/status"}`))
			if err != nil {
				t.Fatalf("Do = %v", err)
			}

			// THE assertion: whatever arm it took, the platform can store it.
			if jsonbWouldRefuse(t, payload) {
				t.Fatalf("payload is still a value the column refuses: %q", payload)
			}
			_, gotEnc, got := decodeRaw(t, payload)
			if gotEnc != tc.wantEnc {
				t.Errorf("encoding = %q, want %q", gotEnc, tc.wantEnc)
			}
			if !bytes.Equal(got, tc.body) {
				t.Errorf("round-tripped to %q, want %q", got, tc.body)
			}
		})
	}
}

// The other side of that guard, and the constraint it must not break: a body
// that merely LOOKS like the escape is storable and has to go through
// untouched. `\\u0000` is a backslash followed by five characters, which is
// why the substring test is a prefilter and the decision is made on the
// decoded value.
func TestDoStillPassesThroughAnEscapedBackslashThatLooksLikeNul(t *testing.T) {
	body := []byte(`{"name":"a\\u0000b"}`)
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	})
	payload, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/status"}`))
	if err != nil {
		t.Fatalf("Do = %v", err)
	}
	if !bytes.Equal(payload, body) {
		t.Errorf("payload = %s, want the engine's own bytes %s", payload, body)
	}
	if bytes.Contains(payload, []byte(wantBodyKey)) {
		t.Errorf("a storable JSON reply was wrapped: %s", payload)
	}
}

// The ceiling, pinned from BOTH sides. Only the refusal was asserted before, so
// a cap wrong by 16x still passed: nothing named the largest envelope that must
// be ACCEPTED.
//
// It is the cap minus 6 because the platform measures
// octet_length(result::text), and jsonb's ::text renderer puts a space after
// each of this envelope's four colons and two commas.
func TestTheEnvelopeCeilingIsTheOneThePlatformMeasures(t *testing.T) {
	const ct = "text/plain"
	const pgSeparators = 6

	// The scaffold is measured, not counted, so a field added to the envelope
	// moves this test rather than quietly loosening it.
	probe, err := reply.Encode(ct, []byte("x"))
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	scaffold := len(probe) - 1

	// The separator count is DERIVED, not counted by hand on both sides. A
	// field added to the envelope brings a colon and a comma with it, and two
	// hand-written 6s would agree with each other while the cap quietly went
	// two bytes too generous. No value in the probe holds a ':' or a ','.
	if got := strings.Count(string(probe), ":") + strings.Count(string(probe), ","); got != pgSeparators {
		t.Fatalf("the envelope has %d separators, so ::text adds %d bytes, not %d", got, got, pgSeparators)
	}

	// ASCII, so one raw byte is one encoded byte and the arithmetic is exact.
	largest := submitCapBytes - pgSeparators - scaffold

	payload, err := reply.Encode(ct, bytes.Repeat([]byte("a"), largest))
	if err != nil {
		t.Fatalf("a %d-byte body was refused: %v", largest, err)
	}
	if got := len(payload) + pgSeparators; got > submitCapBytes {
		t.Errorf("the platform would measure %d bytes, over the %d it accepts", got, submitCapBytes)
	}
	if _, err := reply.Encode(ct, bytes.Repeat([]byte("a"), largest+1)); !errors.Is(err, ErrOversizeReply) {
		t.Errorf("one byte more = %v, want ErrOversizeReply", err)
	}
}

// The text arm inflates too, and harder than base64 can: a control byte is six
// bytes of \u00XX escape. A body at a quarter of the cap is already over it,
// which is why the bound is on the envelope and not on what was read.
func TestDoRefusesATextBodyEscapingInflatesPastTheCap(t *testing.T) {
	// \u0001 is valid UTF-8 and is not NUL, so it takes the TEXT arm.
	body := bytes.Repeat([]byte("\x01"), submitCapBytes/4)
	if len(body) > readCapBytes {
		t.Fatalf("fixture is %d bytes, which the read cap refuses first", len(body))
	}
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	})
	_, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/metrics"}`))
	if !errors.Is(err, ErrOversizeReply) {
		t.Fatalf("Do = %v, want ErrOversizeReply", err)
	}
}

// The envelope is written with HTML escaping OFF. jsonb's ::text renders
// < back as "<", so leaving it on would measure this payload against the
// platform's cap in a currency the platform does not use -- six bytes for one.
func TestTheEnvelopeDoesNotHtmlEscapeTheBodyItCarries(t *testing.T) {
	const body = "a: <b> & <c>\n"
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/yaml")
		w.Write([]byte(body))
	})
	payload, err := c.Do(context.Background(),
		mustParse(t, `{"method":"GET","action":"/admin/config.yaml"}`))
	if err != nil {
		t.Fatalf("Do = %v", err)
	}
	if !bytes.Contains(payload, []byte("<b> & <c>")) {
		t.Errorf("the carried body was HTML-escaped: %s", payload)
	}
}

// The read cap's ACCEPT side. Only the refusal was pinned, so `>` -> `>=`
// survived -- and a reply of exactly the cap is a reply, not an overrun. It is
// JSON and ASCII on purpose: it passes through unwrapped, so the envelope cap
// plays no part and this measures the read cap alone.
func TestDoRelaysAJsonReplyExactlyAtTheReadCap(t *testing.T) {
	body := append([]byte(`["`), bytes.Repeat([]byte("a"), readCapBytes-4)...)
	body = append(body, '"', ']')
	if len(body) != readCapBytes {
		t.Fatalf("fixture is %d bytes, want exactly %d", len(body), readCapBytes)
	}

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	})
	payload, err := c.Do(context.Background(), mustParse(t, `{"method":"GET","action":"/status"}`))
	if err != nil {
		t.Fatalf("Do = %v, want the reply relayed", err)
	}
	if !bytes.Equal(payload, body) {
		t.Fatalf("payload is %d bytes, want the engine's %d", len(payload), len(body))
	}
}

// The envelope arm is reachable only through Do today, and Go's net/textproto
// refuses a response whose header holds a control byte ("malformed MIME header
// line") -- so this cannot arrive over the wire. It is cleaned anyway: a NUL is
// valid UTF-8, so ToValidUTF8 alone would leave the one escape this whole
// branch exists to keep out of the column, in the one field that is not the
// body.
func TestTheCarriedContentTypeHoldsNoNul(t *testing.T) {
	payload, err := reply.Encode("text/plain\x00x", []byte("not json"))
	if err != nil {
		t.Fatalf("reply.Encode = %v", err)
	}
	if bytes.Contains(payload, []byte(`\u0000`)) {
		t.Errorf("payload carries a \\u0000 the column refuses: %s", payload)
	}
	if jsonbWouldRefuse(t, payload) {
		t.Errorf("payload is a value the column refuses: %s", payload)
	}
}
