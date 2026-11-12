package collect

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// queryTextStore answers every lookup with body and records the statements.
func queryTextStore(t *testing.T, body string) (*Client, *[]string) {
	t.Helper()
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/sql" || r.URL.RawQuery != "" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL)
		}
		seen = append(seen, sqlText(r))
		w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return NewClient(srv.URL, "", time.Second), &seen
}

const queryTextArgs = `{"cluster":"c","node_name":"n","datname":"d","queryid":"9223372036854775807","start":100,"end":200}`

func TestNativeQueryTextExactOriginalAndScope(t *testing.T) {
	original := "/* comment */\nSELECT  $1, '☃'"
	c, seen := queryTextStore(t, rowsJSON([]string{"query", "observed_at"}, []any{original, 150}))
	out, err := QueryText(context.Background(), c, json.RawMessage(queryTextArgs), time.Unix(200, 0))
	if err != nil {
		t.Fatal(err)
	}
	res := out.Payload.(QueryTextResult)
	if !res.Available || *res.Query != original || *res.ObservedAt != 150 || res.QueryID != "9223372036854775807" || res.TextScope != "" {
		t.Fatalf("result = %+v", res)
	}
	sql := (*seen)[0]
	for _, want := range []string{"-- instance-jobs: query_text\n", "= '9223372036854775807'", "dbname = 'd'",
		"to_timestamp(100)", "to_timestamp(200)"} {
		if !strings.Contains(sql, want) {
			t.Fatalf("SQL lacks %q:\n%s", want, sql)
		}
	}
	// The wire shape the platform reads.
	raw, _ := json.Marshal(res)
	want := `{"queryid":"9223372036854775807","query":"/* comment */\nSELECT  $1, '☃'","truncated":false,"observed_at":150,"available":true}`
	if string(raw) != want {
		t.Fatalf("payload = %s\nwant      %s", raw, want)
	}
}

func TestNativeQueryTextNotAvailable(t *testing.T) {
	c, _ := queryTextStore(t, rowsJSON([]string{"query", "observed_at"}))
	out, err := QueryText(context.Background(), c, json.RawMessage(queryTextArgs), time.Unix(200, 0))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(out.Payload)
	if want := `{"queryid":"9223372036854775807","query":null,"truncated":false,"observed_at":null,"available":false}`; string(raw) != want {
		t.Fatalf("payload = %s, want %s", raw, want)
	}
}

func TestNativeQueryTextRejectsInvalidArgsAndForeignResults(t *testing.T) {
	c, seen := queryTextStore(t, noRows)
	for _, raw := range []string{
		`{"queryid":"1"}`,
		`{"cluster":"c","node_name":"n","datname":"d","queryid":"1","start":100,"end":200,"url":"http://elsewhere"}`,
		`{"cluster":"c","node_name":"n","datname":"d","queryid":"1; DROP","start":100,"end":200}`,
	} {
		if _, err := QueryText(context.Background(), c, json.RawMessage(raw), time.Unix(200, 0)); !errors.Is(err, ErrInvalidArgs) {
			t.Fatalf("%s: err = %v, want ErrInvalidArgs", raw, err)
		}
	}
	if len(*seen) != 0 {
		t.Fatal("called the store for an invalid scope")
	}
	cols := []string{"query", "observed_at"}
	for name, body := range map[string]string{
		"observed after the window":  rowsJSON(cols, []any{"q", 250}),
		"observed before the window": rowsJSON(cols, []any{"q", 100}),
		"two rows":                   rowsJSON(cols, []any{"a", 150}, []any{"b", 150}),
		"empty text":                 rowsJSON(cols, []any{"", 150}),
		"wrong types":                rowsJSON(cols, []any{1, "x"}),
		"truncated":                  `{"columns":["query","observed_at"],"rows":[],"truncated":true}`,
	} {
		c, _ := queryTextStore(t, body)
		if _, err := QueryText(context.Background(), c, json.RawMessage(queryTextArgs), time.Unix(200, 0)); err == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
}

func TestNativeQueryTextTruncatesOnARuneBoundary(t *testing.T) {
	long := strings.Repeat("a", queryTextMaxBytes-1) + "☃☃"
	c, _ := queryTextStore(t, rowsJSON([]string{"query", "observed_at"}, []any{long, 150}))
	out, err := QueryText(context.Background(), c, json.RawMessage(queryTextArgs), time.Unix(200, 0))
	if err != nil {
		t.Fatal(err)
	}
	res := out.Payload.(QueryTextResult)
	if !res.Truncated || *res.Query != strings.Repeat("a", queryTextMaxBytes-1) {
		t.Fatalf("truncated=%v len=%d", res.Truncated, len(*res.Query))
	}
}

func TestNativeQueryTextDoesNotFollowRedirects(t *testing.T) {
	reached := false
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reached = true }))
	defer destination.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()
	_, err := QueryText(context.Background(), NewClient(source.URL, "t", time.Second), json.RawMessage(queryTextArgs), time.Unix(200, 0))
	if err == nil || reached {
		t.Fatal("followed redirect or accepted redirect response")
	}
}

func TestLatestCollectedQueryTextSeparatesChartAndRetrievalTime(t *testing.T) {
	const args = `{"cluster":"c","node_name":"n","datname":"d","queryid":"1","start":100,"end":200,"lookup_mode":"latest_collected"}`
	for _, test := range []struct {
		name     string
		observed int
		valid    bool
	}{
		{"after chart", 250, true},
		{"future row", 301, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			c, seen := queryTextStore(t, rowsJSON([]string{"query", "observed_at"}, []any{"SELECT $1", test.observed}))
			out, err := QueryText(context.Background(), c, json.RawMessage(args), time.Unix(300, 0))
			if (err == nil) != test.valid {
				t.Fatalf("valid=%v error=%v", test.valid, err)
			}
			// The lookup reaches to now, not to the chart's end.
			if !strings.Contains((*seen)[0], "to_timestamp(300)") {
				t.Fatalf("SQL does not reach now:\n%s", (*seen)[0])
			}
			if err != nil {
				return
			}
			res := out.Payload.(QueryTextResult)
			if res.TextScope != "latest_collected" || res.RetrievedAt == nil || *res.RetrievedAt != 300 {
				t.Fatalf("result = %+v", res)
			}
		})
	}
}
