package joe

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

// The secret every vector below is computed with.
const testSecret = "signing-secret"

// GOLDEN SIGNATURES, computed OUTSIDE Go and hardcoded.
//
// Recomputing the HMAC in the test with Go's own crypto/hmac would assert that
// Sign agrees with itself and would pass just as happily if the prefix, the
// ordering or the digest were wrong. These three were produced independently by
// Python's hmac, openssl dgst and -- the one that matters -- PostgreSQL's
// pgcrypto, evaluating what the platform puts on the wire today:
//
//	select 'v0=' || encode(hmac('v0:' || <body>, 'signing-secret', 'sha256'), 'hex')
//
// sigEmptyBody is also exactly public.payload_signature_get's empty-body fallback
// (v1.joe_command_run's 403 retry) and sigEmptyObject exactly its '{}' form.
const (
	sigEmptyBody    = "v0=735036e8dc2ae05e2e29b2b37a7acbe8467f011709b6e935661133b25bcd4786"
	sigEmptyObject  = "v0=93dc09a4f3e86e506ad74ca670cb669dc00ca8d52ecc4fdd1eda9e90cd7d3384"
	sigCommandBody  = "v0=f7c79eb9730c7507e7e5ee464f691df41837c13d9d3febf5e64bae3343ecb622"
	commandBodySent = `{"channel_id":"C123","command_id":"7","session_id":"webui-i3","text":"explain select 1","timestamp":"2026-09-30 10:00:00","user_id":"a_b_1"}`
)

type seen struct {
	method    string
	path      string
	body      string
	signature string
}

// newServer starts a stand-in Joe and returns a client pointed at it, plus the
// requests it received in order.
func newServer(t *testing.T, handler func(w http.ResponseWriter, r *http.Request, got seen)) (*Client, *[]seen) {
	t.Helper()
	var requests []seen
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		got := seen{
			method:    r.Method,
			path:      r.URL.Path,
			body:      string(raw),
			signature: r.Header.Get("Verification-Signature"),
		}
		requests = append(requests, got)
		handler(w, r, got)
	}))
	t.Cleanup(srv.Close)
	return NewClient(srv.URL, testSecret, 5*time.Second), &requests
}

// Joe MACs "v0:" followed by the bytes it read, so the signature is a function of
// the transmitted body and nothing else.
func TestSignMatchesThePlatformsOwnSignature(t *testing.T) {
	c := NewClient("http://127.0.0.1:2400", testSecret, time.Second)

	if got := c.sign(nil); got != sigEmptyBody {
		t.Errorf("sign(nil) = %s, want %s", got, sigEmptyBody)
	}
	if got := c.sign([]byte("{}")); got != sigEmptyObject {
		t.Errorf("sign({}) = %s, want %s", got, sigEmptyObject)
	}
	if got := c.sign([]byte(commandBodySent)); got != sigCommandBody {
		t.Errorf("sign(command) = %s, want %s", got, sigCommandBody)
	}
	// The two schemes are DIFFERENT values, which is the whole reason the 403
	// fallback exists: an empty body and a '{}' body are not interchangeable.
	if c.sign(nil) == c.sign([]byte("{}")) {
		t.Error("the empty-body and {} signatures are equal, so the fallback is a no-op")
	}
}

// A different secret must produce a different signature -- otherwise the test
// above would pass against a Sign that ignored the key entirely.
func TestSignDependsOnTheSecret(t *testing.T) {
	other := NewClient("http://127.0.0.1:2400", "another-secret", time.Second)
	if got := other.sign(nil); got == sigEmptyBody {
		t.Errorf("a different secret produced the same signature %s", got)
	}
}

// Joe answers a bare array. The `channels` wrapper is accepted too, because
// v1.joe_command_run coalesces both and guessing wrong means no channel at all.
func TestChannelsReadsBothShapesJoeMightAnswer(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{
		{"bare array", `[{"channel_id":"first"},{"channel_id":"second"}]`},
		{"wrapped", `{"channels":[{"channel_id":"first"},{"channel_id":"second"}]}`},
		// FIELDS ALONGSIDE THE KEY ARE TOLERATED, and that has to be asserted rather
		// than left to encoding/json's default: channelsFrom requires channel_id to
		// be PRESENT, and tightening that one step further -- to a decoder that
		// refuses unknown fields -- would refuse every real Joe the day
		// config.Channel grows an exported field, since Joe keeps its other two out
		// of the reply with `json:"-"` and a new one would default to being emitted.
		{"an array with fields we do not read", `[{"channel_id":"first","name":"prod"},` +
			`{"dblab_server":"dev","channel_id":"second"}]`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			})
			ids, err := c.Channels(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			// Joe's ORDER is kept: v1.joe_command_run takes the first advertised
			// channel, so which one is first is part of the contract.
			if len(ids) != 2 || ids[0] != "first" || ids[1] != "second" {
				t.Fatalf("Channels() = %v, want [first second]", ids)
			}
		})
	}
}

// The channel list is a bodyless GET signed over the EMPTY body, because that is
// the pair Joe actually verifies today: the platform hands pg_http a '{}' body,
// pg_http sends none, and Joe MACs what it read.
func TestChannelsSendsABodylessGetSignedOverNothing(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		_, _ = w.Write([]byte(`[{"channel_id":"only"}]`))
	})
	if _, err := c.Channels(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(*requests) != 1 {
		t.Fatalf("sent %d requests, want 1", len(*requests))
	}
	got := (*requests)[0]
	if got.method != "GET" || got.path != "/webui/channels" {
		t.Errorf("sent %s %s, want GET /webui/channels", got.method, got.path)
	}
	if got.body != "" {
		t.Errorf("sent body %q, want none", got.body)
	}
	if got.signature != sigEmptyBody {
		t.Errorf("signature = %s, want the empty-body one %s", got.signature, sigEmptyBody)
	}
}

// THE 403 FALLBACK. Joe verifies the bytes it received, and a GET's body is what
// HTTP implementations disagree about, so a refusal is retried ONCE with the
// second scheme -- the behaviour v1.joe_command_run:122-151 carries today.
func TestChannelsRetriesA403WithTheOtherSignatureScheme(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, got seen) {
		if got.body == "" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		_, _ = w.Write([]byte(`[{"channel_id":"reached-on-the-second-scheme"}]`))
	})
	ids, err := c.Channels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != "reached-on-the-second-scheme" {
		t.Fatalf("Channels() = %v", ids)
	}
	if len(*requests) != 2 {
		t.Fatalf("sent %d requests, want 2 (one per scheme)", len(*requests))
	}
	first, second := (*requests)[0], (*requests)[1]
	if first.body != "" || first.signature != sigEmptyBody {
		t.Errorf("first attempt: body %q sig %s, want empty body and %s", first.body, first.signature, sigEmptyBody)
	}
	if second.body != "{}" || second.signature != sigEmptyObject {
		t.Errorf("second attempt: body %q sig %s, want {} and %s", second.body, second.signature, sigEmptyObject)
	}
}

// ONLY a 403. The fallback exists because Joe refuses a signature it computed over
// different bytes than we signed, and nothing else it answers means that -- so a
// 400, a 404 or a 5xx gets ONE attempt. Widened to any *JoeError the belt doubles
// every failing request, which the runner's read ladder then multiplies by three.
//
// TestChannelsRetriesA403WithTheOtherSignatureScheme is the positive control: the
// same setup answering 403 does send two.
func TestOnlyA403GetsTheSecondSignatureScheme(t *testing.T) {
	for _, status := range []int{
		http.StatusBadRequest,
		http.StatusUnauthorized,
		http.StatusNotFound,
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusServiceUnavailable,
	} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
				w.WriteHeader(status)
			})
			if _, err := c.Channels(context.Background()); err == nil {
				t.Fatalf("Channels() succeeded against a %d", status)
			}
			if len(*requests) != 1 {
				t.Fatalf("sent %d requests against a %d, want exactly 1 -- the belt is for a "+
					"refused signature, not for any failure", len(*requests), status)
			}
		})
	}
}

// THE ID GOES BACK ON THE WIRE EXACTLY AS JOE ADVERTISED IT. Joe keys
// msgProcessors on the id as configured and looks it up with a plain map read, and
// pkg/config expands ${VAR} in channelID verbatim -- so relaying a trimmed copy
// posts a channel Joe does not have, which is a 400 on a write that is never
// retried. The pull path takes it raw too.
func TestTheAdvertisedChannelIdIsRelayedByteForByte(t *testing.T) {
	const padded = " pgload\t"
	c, requests := newServer(t, func(w http.ResponseWriter, r *http.Request, _ seen) {
		if r.URL.Path == channelsAction {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`[{"channel_id":"` + " pgload\\t" + `"}]`))
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	ids, err := c.Channels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != padded {
		t.Fatalf("Channels() = %q, want the raw %q", ids, padded)
	}
	req, err := Parse([]byte(`{"method":"POST","action":"/webui/command","resolve_channel":true,"data":{}}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Do(context.Background(), req, ids[0]); err != nil {
		t.Fatal(err)
	}
	const want = `{"channel_id":" pgload\t"}`
	if got := (*requests)[1].body; got != want {
		t.Fatalf("posted\n  %s\nwant\n  %s", got, want)
	}
}

// `data: null` IS `data` ABSENT, and Parse normalises the two so that stays true
// downstream. json.RawMessage keeps the four bytes `null`, so before this the same
// input was empty to withChannel and non-empty to call: the bodyless-GET belt was
// silently lost and `null` was transmitted as the payload.
func TestANullDataIsTheSameAsNoData(t *testing.T) {
	for _, args := range []string{
		`{"method":"GET","action":"/webui/channels"}`,
		`{"method":"GET","action":"/webui/channels","data":null}`,
	} {
		t.Run(args, func(t *testing.T) {
			c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
				w.WriteHeader(http.StatusForbidden)
			})
			req, err := Parse([]byte(args))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Do(context.Background(), req, ""); err == nil {
				t.Fatal("Do() succeeded against a 403")
			}
			// Both schemes tried: this is a bodyless GET either way.
			if len(*requests) != 2 {
				t.Fatalf("sent %d requests, want 2 -- `data: null` carries no body", len(*requests))
			}
			if (*requests)[0].body != "" || (*requests)[0].signature != sigEmptyBody {
				t.Errorf("first attempt: body %q sig %s, want an empty body signed %s",
					(*requests)[0].body, (*requests)[0].signature, sigEmptyBody)
			}
			if (*requests)[1].body != "{}" || (*requests)[1].signature != sigEmptyObject {
				t.Errorf("second attempt: body %q sig %s, want {} signed %s",
					(*requests)[1].body, (*requests)[1].signature, sigEmptyObject)
			}
		})
	}
}

// A REQUEST WITH A BODY HAS ONE TRUE RENDERING, so it gets one attempt -- and that
// is a property of the BODY, not of the verb. A GET carrying one is the half the
// POST test cannot reach: re-sending it would replace the platform's own payload
// with `{}` and sign that instead.
func TestA403OnAGetWithABodyIsNotRetried(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		w.WriteHeader(http.StatusForbidden)
	})
	req, err := Parse([]byte(`{"method":"GET","action":"/webui/channels","data":{"q":"1"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Do(context.Background(), req, ""); err == nil {
		t.Fatal("Do() succeeded against a 403")
	}
	if len(*requests) != 1 {
		t.Fatalf("sent %d requests, want exactly 1", len(*requests))
	}
	if (*requests)[0].body != `{"q":"1"}` {
		t.Errorf("body = %q, want the platform's own payload", (*requests)[0].body)
	}
}

// Exactly TWO schemes. A 403 that survives both is a signing secret that
// disagrees with Joe's, which more attempts restate rather than fix.
func TestChannelsStopsAfterTheSecondScheme(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		w.WriteHeader(http.StatusForbidden)
	})
	_, err := c.Channels(context.Background())
	var joeErr *JoeError
	if !errors.As(err, &joeErr) || joeErr.StatusCode != 403 {
		t.Fatalf("Channels() error = %v, want a 403 JoeError", err)
	}
	if joeErr.Retryable() {
		t.Error("a 403 reads as retryable; a wrong signing secret is not fixed by retrying")
	}
	if len(*requests) != 2 {
		t.Fatalf("sent %d requests, want exactly 2", len(*requests))
	}
}

// A Joe that ANSWERED A LIST and serves nothing usable. Permanent: it is the box's
// own config, and the only lookup outcome the runner turns into a skip.
func TestChannelsReportsABoxThatServesNone(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{
		{"empty array", `[]`},
		{"no channel ids", `[{"channel_id":""}]`},
		{"blank channel ids", `[{"channel_id":"  "}]`},
		{"an empty wrapper", `{"channels":[]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(tc.body))
			})
			_, err := c.Channels(context.Background())
			if !errors.Is(err, ErrNoChannels) {
				t.Fatalf("Channels() error = %v, want ErrNoChannels", err)
			}
			if errors.Is(err, ErrBadChannelList) {
				t.Error("an empty list reads as a bad list; the skip would become an error")
			}
		})
	}
}

// A 200 THAT IS NOT A CHANNEL LIST IS A FAULT, NOT A SKIP, and this is the
// assertion that pins the difference. Every one of these bodies says nothing about
// how many channels Joe serves, so reporting them as ErrNoChannels would submit a
// lost command as `skipped`, let the platform close the job `done`, and leave the
// box green -- runJob counts a skip as a success.
//
// The `{...}` rows are the ones that need the guard: unmarshalling into an
// anonymous struct would decode EVERY json object into a `channels` wrapper with a
// nil slice, because encoding/json ignores unknown fields.
//
// TestChannelsReadsBothShapesJoeMightAnswer is the positive control -- the two
// shapes that ARE a channel list resolve, so these refusals are not a parser that
// refuses everything.
func TestChannelsRefusesABodyThatIsNotAChannelList(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{
		{"an html error page from something else on the port", `<html><body>Bad Gateway</body></html>`},
		{"a json error envelope", `{"message":"internal error","code":500}`},
		{"a json object with no channels key", `{"unexpected":true}`},
		{"a renamed shape", `{"items":[{"channel_id":"pgload"}]}`},
		{"a channels key that is not a list", `{"channels":{"channel_id":"pgload"}}`},
		// THE SAME MISTAKE ONE LEVEL DOWN. Decoding straight into []channel lets
		// encoding/json ignore unknown fields, so a list of ANY objects would yield
		// blank ids and read as "Joe serves none".
		{"an array of objects with a renamed key", `[{"id":"pgload","name":"prod"}]`},
		{"an array of objects with a camel-cased key", `[{"channelId":"pgload"}]`},
		{"an array of empty objects", `[{}]`},
		{"an array of nulls", `[null]`},
		{"an array of scalars", `[1,2,3]`},
		{"a wrapper holding objects with a renamed key", `{"channels":[{"id":"pgload"}]}`},
		{"a channel_id that is not a string", `[{"channel_id":42}]`},
		{"an empty 200", ``},
		{"a bare json string", `"nope"`},
		{"json null", `null`},
		{"a json number", `42`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(tc.body))
			})
			_, err := c.Channels(context.Background())
			if !errors.Is(err, ErrBadChannelList) {
				t.Fatalf("Channels() error = %v, want ErrBadChannelList", err)
			}
			if errors.Is(err, ErrNoChannels) {
				t.Error("a body that is not a channel list reads as 'joe serves none'; " +
					"the runner would submit the lost command as skipped")
			}
		})
	}
}

// THE SUBSTITUTION AND THE SIGNATURE ARE ONE STEP. The channel goes into the body
// and the signature is taken over the result, so the platform could not have
// pre-computed it -- which is the reason the lookup happens on this box at all.
func TestDoPutsTheResolvedChannelInTheBodyItSigns(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		w.WriteHeader(http.StatusOK)
	})
	req, err := Parse([]byte(`{"method":"POST","action":"/webui/command","resolve_channel":true,` +
		`"data":{"text":"explain select 1","command_id":"7","session_id":"webui-i3",` +
		`"user_id":"a_b_1","timestamp":"2026-09-30 10:00:00"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Do(context.Background(), req, "C123"); err != nil {
		t.Fatal(err)
	}
	got := (*requests)[0]
	// The exact bytes, then the signature over exactly those bytes. Both literal:
	// asserting only the signature would pass against a body that had lost a
	// field, and asserting only the body would pass against an unsigned request.
	if got.body != commandBodySent {
		t.Errorf("sent body\n  %s\nwant\n  %s", got.body, commandBodySent)
	}
	if got.signature != sigCommandBody {
		t.Errorf("signature = %s, want %s", got.signature, sigCommandBody)
	}
}

// The resolved value WINS. v1.joe_command_run takes Joe's first advertised
// channel, so a value arriving beside resolve_channel is at best a duplicate of
// what we just looked up and at worst stale.
func TestDoOverwritesAChannelTheArgsAlreadyCarried(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {})
	req, err := Parse([]byte(`{"method":"POST","action":"/webui/command","resolve_channel":true,` +
		`"data":{"channel_id":"stale-from-the-platform"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Do(context.Background(), req, "resolved-here"); err != nil {
		t.Fatal(err)
	}
	if body := (*requests)[0].body; body != `{"channel_id":"resolved-here"}` {
		t.Errorf("sent %s, want the resolved channel", body)
	}
}

// Without the flag the body goes through untouched: an action that already knows
// its channel, or any other Joe endpoint, is not rewritten.
func TestDoLeavesTheBodyAloneWithoutTheFlag(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {})
	req, err := Parse([]byte(`{"method":"POST","action":"/webui/verify","data":{"challenge":"abc"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Do(context.Background(), req, "unused"); err != nil {
		t.Fatal(err)
	}
	if body := (*requests)[0].body; body != `{"challenge":"abc"}` {
		t.Errorf("sent %s, want the args' body verbatim", body)
	}
}

// AN EMPTY 200 IS THE NORMAL ANSWER to a command, not an edge case: Joe's handler
// hands the message to a goroutine and returns without writing anything. It is an
// answer, so it must not read as a failure.
func TestDoTreatsAnEmpty200AsTheAnswerItIs(t *testing.T) {
	c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		w.WriteHeader(http.StatusOK)
	})
	req, err := Parse([]byte(`{"method":"POST","action":"/webui/command","data":{"channel_id":"c"}}`))
	if err != nil {
		t.Fatal(err)
	}
	payload, err := c.Do(context.Background(), req, "")
	if err != nil {
		t.Fatalf("an empty 200 failed: %v", err)
	}
	if payload != nil {
		t.Errorf("payload = %s, want nil so the platform stores a SQL NULL", payload)
	}
}

// Joe's verbs and no others. NARROWER than the engine's four: a PATCH or DELETE
// against Joe has no meaning, so accepting one would only let a platform bug
// become an arbitrary verb against a local service.
func TestParseTakesOnlyTheVerbsJoeServes(t *testing.T) {
	for _, method := range []string{"GET", "POST", "get", " post "} {
		args := `{"method":"` + method + `","action":"/webui/channels"}`
		if _, err := Parse([]byte(args)); err != nil {
			t.Errorf("Parse(%s) = %v, want accepted", method, err)
		}
	}
	for _, method := range []string{"PATCH", "DELETE", "PUT", "HEAD", "", "TRACE"} {
		args := `{"method":"` + method + `","action":"/webui/channels"}`
		if _, err := Parse([]byte(args)); !errors.Is(err, ErrInvalidArgs) {
			t.Errorf("Parse(%q) = %v, want ErrInvalidArgs", method, err)
		}
	}
}

// THE PLATFORM NEVER SUPPLIES A HOST, enforced here rather than assumed: the
// action is resolved against Joe's address, so a protocol-relative value would
// send the signature to a host of the caller's choosing.
func TestParseRefusesAnythingThatIsNotAJoePath(t *testing.T) {
	for _, action := range []string{
		"//evil.example.com/x",
		"http://evil.example.com/x",
		"webui/channels",
		"",
		"/webui/../../etc/passwd",
		"/webui/%2e%2e/secret",
		"mailto:x@example.com",
	} {
		args := `{"method":"GET","action":"` + action + `"}`
		if _, err := Parse([]byte(args)); !errors.Is(err, ErrInvalidArgs) {
			t.Errorf("Parse(action %q) = %v, want ErrInvalidArgs", action, err)
		}
	}
	if _, err := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`)); err != nil {
		t.Errorf("a plain path was refused: %v", err)
	}
}

// A body the channel cannot go into is refused before any request is built: the
// job can never run, so it must not reach the wire as one attempt.
func TestParseRefusesResolveChannelOnABodyThatIsNotAnObject(t *testing.T) {
	for _, data := range []string{`[1,2]`, `"text"`, `7`, `true`} {
		args := `{"method":"POST","action":"/webui/command","resolve_channel":true,"data":` + data + `}`
		if _, err := Parse([]byte(args)); !errors.Is(err, ErrInvalidArgs) {
			t.Errorf("Parse(data %s) = %v, want ErrInvalidArgs", data, err)
		}
	}
	// An absent or null body is fine: the channel is then the only field.
	for _, args := range []string{
		`{"method":"POST","action":"/webui/command","resolve_channel":true}`,
		`{"method":"POST","action":"/webui/command","resolve_channel":true,"data":null}`,
	} {
		if _, err := Parse([]byte(args)); err != nil {
			t.Errorf("Parse(%s) = %v, want accepted", args, err)
		}
	}
}

// The platform's own keys are not this side's business: it executes the call.
func TestParseIgnoresPlatformOnlyKeys(t *testing.T) {
	req, err := Parse([]byte(`{"method":"GET","action":"/webui/channels","purpose":"joe_call"}`))
	if err != nil {
		t.Fatal(err)
	}
	if req.Method != "GET" || req.Action != "/webui/channels" {
		t.Fatalf("req = %+v", req)
	}
}

// A non-JSON reply takes the SHARED envelope, so the same body answers the same
// shape whichever channel reached it (platform-all#815).
func TestDoCarriesANonJsonReplyInTheSharedEnvelope(t *testing.T) {
	c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("not json at all"))
	})
	req, _ := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`))
	payload, err := c.Do(context.Background(), req, "")
	if err != nil {
		t.Fatal(err)
	}
	var env struct {
		Body struct {
			ContentType string `json:"content_type"`
			Encoding    string `json:"encoding"`
			Body        string `json:"body"`
		} `json:"pgai_body"`
	}
	if err := json.Unmarshal(payload, &env); err != nil {
		t.Fatalf("payload %s is not the envelope: %v", payload, err)
	}
	if env.Body.Encoding != "text" || env.Body.Body != "not json at all" {
		t.Errorf("envelope = %+v", env.Body)
	}
}

// A JSON reply is relayed verbatim, never wrapped.
func TestDoLeavesAJsonReplyExactlyAsItArrived(t *testing.T) {
	const body = `{"status":"OK","n":1}`
	c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})
	req, _ := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`))
	payload, err := c.Do(context.Background(), req, "")
	if err != nil {
		t.Fatal(err)
	}
	if string(payload) != body {
		t.Errorf("payload = %s, want %s", payload, body)
	}
}

// Joe writes a status and usually no body, so the class has to come from the
// status. Which of them a retry could clear is the runner's decision, made on
// this.
func TestJoeErrorRetryability(t *testing.T) {
	for status, wantRetryable := range map[int]bool{
		400: false, // the command handler's own refusal
		401: true,  // a restart mid-rotation on this very box
		403: false, // the signing secret disagrees with Joe's
		404: false, // no such endpoint, or no channels configured
		429: true,
		500: true,
		503: true,
	} {
		err := &JoeError{StatusCode: status}
		if got := err.Retryable(); got != wantRetryable {
			t.Errorf("JoeError{%d}.Retryable() = %v, want %v", status, got, wantRetryable)
		}
	}
}

// Joe has no error envelope, so plain text is carried; a JSON `message` is read
// when there is one.
func TestDoCarriesWhateverJoeSaidAboutAFailure(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{"plain text", "message processor for \"x\" channel not found\n", `message processor for "x" channel not found`},
		{"json envelope", `{"message":"nope"}`, "nope"},
		{"no body at all", "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(tc.body))
			})
			req, _ := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`))
			_, err := c.Do(context.Background(), req, "")
			var joeErr *JoeError
			if !errors.As(err, &joeErr) {
				t.Fatalf("error = %v, want a JoeError", err)
			}
			if joeErr.Message != tc.want {
				t.Errorf("Message = %q, want %q", joeErr.Message, tc.want)
			}
		})
	}
}

// Go replays a custom header verbatim across hosts, so following a redirect would
// hand the signature to whatever sent it.
func TestDoRefusesToFollowARedirect(t *testing.T) {
	var leaked bool
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Verification-Signature") != "" {
			leaked = true
		}
	}))
	t.Cleanup(elsewhere.Close)

	c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		http.Redirect(w, &http.Request{}, elsewhere.URL+"/webui/channels", http.StatusTemporaryRedirect)
	})
	req, _ := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`))
	_, err := c.Do(context.Background(), req, "")
	if err == nil {
		t.Fatal("a redirect was followed and reported as success")
	}
	if leaked {
		t.Error("the signature reached the redirect target")
	}
}

// The error is reported to the platform, and *url.Error's text carries Joe's
// address and the job's action.
func TestDoDropsTheRequestUrlFromATransportError(t *testing.T) {
	// A port nothing listens on, so the dial fails rather than the request.
	c := NewClient("http://127.0.0.1:1", testSecret, time.Second)
	req, _ := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`))
	_, err := c.Do(context.Background(), req, "")
	if !errors.Is(err, ErrJoeUnreachable) {
		t.Fatalf("error = %v, want ErrJoeUnreachable", err)
	}
	if strings.Contains(err.Error(), "127.0.0.1:1") || strings.Contains(err.Error(), "/webui/channels") {
		t.Errorf("error carries the address or the action: %v", err)
	}
}

// A reply over the read bound is refused rather than truncated: a truncated body
// would be submitted as a valid-looking partial answer.
func TestDoRefusesAnOversizeReplyRatherThanTruncatingIt(t *testing.T) {
	c, _ := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		w.Header().Set("Content-Type", "application/json")
		// One byte past the read cap. The literal, not the constant: a test written
		// in terms of maxResponseBytes cannot notice maxResponseBytes changing.
		_, _ = w.Write([]byte(`"` + strings.Repeat("a", 1<<20) + `"`))
	})
	req, _ := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`))
	if _, err := c.Do(context.Background(), req, ""); !errors.Is(err, ErrOversizeReply) {
		t.Fatalf("error = %v, want ErrOversizeReply", err)
	}
}

// A cancelled context stays distinguishable through the unreachable wrap: the
// runner classifies a deadline before it classifies an unreachable Joe.
func TestDoKeepsADeadlineDistinguishable(t *testing.T) {
	c, _ := newServer(t, func(w http.ResponseWriter, r *http.Request, _ seen) {
		<-r.Context().Done()
	})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	req, _ := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`))
	_, err := c.Do(ctx, req, "")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want it to carry context.DeadlineExceeded", err)
	}
}

// The belt reaches a job whose action IS the channel list, so the two routes to
// /webui/channels behave identically.
func TestAGetThroughDoGetsTheSameSignatureFallback(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, got seen) {
		if got.body == "" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"channel_id":"c"}]`))
	})
	req, _ := Parse([]byte(`{"method":"GET","action":"/webui/channels"}`))
	if _, err := c.Do(context.Background(), req, ""); err != nil {
		t.Fatal(err)
	}
	if len(*requests) != 2 {
		t.Fatalf("sent %d requests, want 2 (one per scheme)", len(*requests))
	}
}

// A WRITE IS SENT ONCE. A POST has one true rendering, so a 403 is not a scheme
// to retry -- and re-sending a command that may have been accepted would run it
// twice on the customer's clone.
func TestA403OnAWriteIsNotRetried(t *testing.T) {
	c, requests := newServer(t, func(w http.ResponseWriter, _ *http.Request, _ seen) {
		w.WriteHeader(http.StatusForbidden)
	})
	req, _ := Parse([]byte(`{"method":"POST","action":"/webui/command","resolve_channel":true}`))
	if _, err := c.Do(context.Background(), req, "C1"); err == nil {
		t.Fatal("a 403 was reported as success")
	}
	if len(*requests) != 1 {
		t.Fatalf("sent %d requests, want exactly 1 -- a write is never re-sent", len(*requests))
	}
}
