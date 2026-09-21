package collect

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"testing"
	"time"
)

// A fixture is (recorded store responses + the request params that produced
// them) -> what the box would SUBMIT. The responses are keyed by the EXACT
// query text, so the fixture pins the PromQL, the label escaping, the request
// window, the arithmetic and the wire shape in one artifact. A query the
// fixture does not carry fails the test, and a recorded response the code never
// asks for fails it too: the set of queries is pinned in both directions.
//
// `expected` is the SUBMISSION, not the internal outcome: every submit names its
// `outcome`, a payload goes bare under `result`, and a skip carries a
// `skip_reason` and nothing else.
type fixture struct {
	Description  string                     `json:"description"`
	Now          string                     `json:"now"`
	Kind         string                     `json:"kind"`
	Args         json.RawMessage            `json:"args"`
	ExpectWindow *expectWindow              `json:"expect_window"`
	Responses    map[string]json.RawMessage `json:"responses"`
	Expected     json.RawMessage            `json:"expected"`
}

type expectWindow struct {
	Start int64 `json:"start"`
	End   int64 `json:"end"`
	Step  int   `json:"step"`
}

func TestFixtures(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("testdata", "fixtures", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatal("no fixtures found")
	}

	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var f fixture
			if err := json.Unmarshal(raw, &f); err != nil {
				t.Fatalf("fixture is not valid json: %v", err)
			}
			// A fixture that queries the store must pin the window it asked
			// for, or it silently loses the start/end/step and instant-time
			// assertions while still looking complete. Every fixture satisfies
			// this today, so it guards the next one rather than the six here.
			if len(f.Responses) > 0 && f.ExpectWindow == nil {
				t.Fatal("this fixture queries the store but pins no expect_window")
			}
			now, err := time.Parse(time.RFC3339, f.Now)
			if err != nil {
				t.Fatalf("fixture `now` is not RFC3339: %v", err)
			}

			asked := map[string]bool{}
			var unexpected []string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				query := r.URL.Query().Get("query")
				if f.ExpectWindow != nil {
					switch r.URL.Path {
					case "/api/v1/query_range":
						checkParam(t, r, "start", strconv.FormatInt(f.ExpectWindow.Start, 10))
						checkParam(t, r, "end", strconv.FormatInt(f.ExpectWindow.End, 10))
						checkParam(t, r, "step", strconv.Itoa(f.ExpectWindow.Step))
					case "/api/v1/query":
						// last_over_time(...) is evaluated AT THE WINDOW END:
						// anywhere else and the enrichment silently resolves
						// nothing, while everything else still looks right.
						checkParam(t, r, "time", strconv.FormatInt(f.ExpectWindow.End, 10))
					}
				}
				body, ok := f.Responses[query]
				if !ok {
					unexpected = append(unexpected, query)
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				asked[query] = true
				w.Header().Set("Content-Type", "application/json")
				w.Write(body)
			}))
			defer srv.Close()

			client := NewClient(srv.URL, "", "", 10*time.Second)
			outcome, err := Run(context.Background(), client, f.Kind, f.Args, now)
			for _, q := range unexpected {
				t.Errorf("query not in the fixture:\n  %s", q)
			}
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			for q := range f.Responses {
				if !asked[q] {
					t.Errorf("fixture records a response that was never requested:\n  %s", q)
				}
			}

			assertJSONEqual(t, asSubmission(outcome), f.Expected)
		})
	}
}

// wireView is the submission rendered as the request body's keys. It carries no
// mapping of its own -- Outcome.AsSubmission is the one the runner uses.
type wireView struct {
	Outcome    string `json:"outcome"`
	Result     any    `json:"result,omitempty"`
	SkipReason string `json:"skip_reason,omitempty"`
}

func asSubmission(o Outcome) wireView {
	s := o.AsSubmission()
	return wireView{Outcome: s.Outcome, Result: s.Result, SkipReason: s.SkipReason}
}

func checkParam(t *testing.T, r *http.Request, name, want string) {
	t.Helper()
	if got := r.URL.Query().Get(name); got != want {
		t.Errorf("request %s = %q, want %q", name, got, want)
	}
}

// assertJSONEqual compares the marshalled outcome against the fixture's
// expectation as decoded JSON, so key order never matters but the key SET does:
// an added, renamed or dropped payload key fails.
func assertJSONEqual(t *testing.T, got any, want json.RawMessage) {
	t.Helper()
	gotRaw, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshalling the outcome: %v", err)
	}
	var gotAny, wantAny any
	if err := json.Unmarshal(gotRaw, &gotAny); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(want, &wantAny); err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(gotAny, wantAny) {
		return
	}
	gotPretty, _ := json.MarshalIndent(gotAny, "", "  ")
	wantPretty, _ := json.MarshalIndent(wantAny, "", "  ")
	t.Errorf("payload mismatch\n--- got ---\n%s\n--- want ---\n%s", gotPretty, wantPretty)
	diffKeys(t, "", gotAny, wantAny)
}

// diffKeys points at the first leaves that differ, so a one-number regression in
// a large payload does not have to be eyeballed.
func diffKeys(t *testing.T, path string, got, want any) {
	t.Helper()
	gotMap, gotOK := got.(map[string]any)
	wantMap, wantOK := want.(map[string]any)
	if !gotOK || !wantOK {
		if !reflect.DeepEqual(got, want) {
			t.Errorf("  %s: got %v, want %v", path, got, want)
		}
		return
	}
	keys := map[string]bool{}
	for k := range gotMap {
		keys[k] = true
	}
	for k := range wantMap {
		keys[k] = true
	}
	sorted := make([]string, 0, len(keys))
	for k := range keys {
		sorted = append(sorted, k)
	}
	sort.Strings(sorted)
	for _, k := range sorted {
		diffKeys(t, path+"/"+k, gotMap[k], wantMap[k])
	}
}
