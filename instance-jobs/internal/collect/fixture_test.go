package collect

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// A fixture is (recorded store responses + the request params that produced
// them) -> what the box would SUBMIT. The responses are keyed by the query's
// name -- the `-- instance-jobs: <name>` line every SELECT the box sends starts
// with -- so the fixture pins the set of queries, the request window, the
// arithmetic and the wire shape in one artifact. A query the fixture does not
// carry fails the test, and a recorded response the code never asks for fails
// it too: the set of queries is pinned in both directions. expect_sql_contains
// pins text every query must carry (the label escaping, for one).
//
// `expected` is the SUBMISSION, not the internal outcome: every submit names its
// `outcome`, a payload goes bare under `result`, and a skip carries a
// `skip_reason` and nothing else.
type fixture struct {
	Description       string                     `json:"description"`
	Now               string                     `json:"now"`
	Kind              string                     `json:"kind"`
	Args              json.RawMessage            `json:"args"`
	ExpectWindow      *expectWindow              `json:"expect_window"`
	ExpectSQLContains []string                   `json:"expect_sql_contains"`
	Responses         map[string]json.RawMessage `json:"responses"`
	Expected          json.RawMessage            `json:"expected"`
}

type expectWindow struct {
	Start int64 `json:"start"`
	End   int64 `json:"end"`
	Step  int   `json:"step"`
}

// sqlQueryName is the name in a SELECT's leading `-- instance-jobs: <name>` line.
func sqlQueryName(sql string) string {
	first, _, _ := strings.Cut(sql, "\n")
	return strings.TrimPrefix(first, "-- instance-jobs: ")
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
			// for, or it silently loses the start/end/step assertions while
			// still looking complete.
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
				if r.Method != http.MethodPost || r.URL.Path != "/sql" {
					t.Errorf("request %s %s, want POST /sql", r.Method, r.URL.Path)
				}
				b, _ := io.ReadAll(r.Body)
				sql := string(b)
				name := sqlQueryName(sql)
				if f.ExpectWindow != nil {
					if name == "query_texts" {
						// The text lookup is bounded AT THE WINDOW END:
						// anywhere else and the enrichment silently resolves
						// nothing, while everything else still looks right.
						want := "to_timestamp(" + strconv.FormatInt(f.ExpectWindow.End, 10) + ")"
						if !strings.Contains(sql, want) {
							t.Errorf("%s: SQL does not bound the lookup at %s:\n%s", name, want, sql)
						}
					} else {
						want := fmt.Sprintf(", %d, %d, %d", f.ExpectWindow.Start*1000,
							f.ExpectWindow.End*1000, f.ExpectWindow.Step*1000)
						if !strings.Contains(sql, want) {
							t.Errorf("%s: SQL does not evaluate the window %q:\n%s", name, want, sql)
						}
						for _, want := range f.ExpectSQLContains {
							if !strings.Contains(sql, want) {
								t.Errorf("%s: SQL does not contain %q:\n%s", name, want, sql)
							}
						}
					}
				}
				body, ok := f.Responses[name]
				if !ok {
					unexpected = append(unexpected, sql)
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				asked[name] = true
				w.Header().Set("Content-Type", "application/json")
				w.Write(body)
			}))
			defer srv.Close()

			client := NewClient(srv.URL, "", 10*time.Second)
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
