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

// sqlStore records each request and answers with a fixed status and body.
type sqlStore struct {
	client   *Client
	requests []*http.Request
	bodies   []string
}

func newSQLStore(t *testing.T, status int, body string) *sqlStore {
	t.Helper()
	s := &sqlStore{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.requests = append(s.requests, r)
		s.bodies = append(s.bodies, sqlText(r))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	s.client = NewClient(srv.URL, "tok", 10*time.Second)
	return s
}

func runSQLJob(s *sqlStore, args string) (Outcome, error) {
	return Run(context.Background(), s.client, KindSQL, []byte(args), time.Date(2026, 9, 18, 12, 0, 0, 0, time.UTC))
}

// The statement goes as the body of POST /sql, never into the URL, with the
// bearer token; nothing in args can move the request elsewhere.
func TestSQLJobPostsTheStatementAsTheBody(t *testing.T) {
	s := newSQLStore(t, 200, rowsJSON([]string{"x"}, []any{1}))
	q := "select 1 as x -- ?a=b&c"
	if _, err := runSQLJob(s, `{"query":"select 1 as x -- ?a=b&c","path":"/api/v1/import/prometheus","url":"http://evil/"}`); err != nil {
		t.Fatal(err)
	}
	r := s.requests[0]
	if r.Method != http.MethodPost || r.URL.Path != "/sql" || r.URL.RawQuery != "" {
		t.Fatalf("request = %s %s", r.Method, r.URL)
	}
	if s.bodies[0] != q {
		t.Fatalf("body = %q, want the statement verbatim", s.bodies[0])
	}
	if got := r.Header.Get("Authorization"); got != "Bearer tok" {
		t.Fatalf("Authorization = %q", got)
	}
	if !strings.HasPrefix(r.Header.Get("Content-Type"), "text/plain") {
		t.Fatalf("Content-Type = %q", r.Header.Get("Content-Type"))
	}
}

// The result is the store's columns, types and rows, rebuilt value by value:
// numbers exact (a BIGINT above 2^53 included), control characters gone.
func TestSQLJobReturnsRows(t *testing.T) {
	body := `{"columns":["id","txt","l","m"],"types":["UBIGINT","VARCHAR","INTEGER[]","MAP"],` +
		`"rows":[[18446744073709551557,"a\u001b[31mb\nc",[1,2.5],{"k\u0007":"v\u009b"}],[null,null,null,null]],` +
		`"row_count":2,"truncated":false,"elapsed_ms":3}`
	s := newSQLStore(t, 200, body)
	out, err := runSQLJob(s, `{"query":"select 1"}`)
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != OutcomeOK {
		t.Fatalf("status = %q", out.Status)
	}
	raw, _ := json.Marshal(out.Payload)
	want := `{"columns":["id","txt","l","m"],"types":["UBIGINT","VARCHAR","INTEGER[]","MAP"],` +
		`"rows":[[18446744073709551557,"a [31mb c",[1,2.5],{"k":"v "}],[null,null,null,null]],"truncated":false}`
	if string(raw) != want {
		t.Fatalf("payload = %s\nwant      %s", raw, want)
	}
}

func TestSQLJobArgsAreValidated(t *testing.T) {
	s := newSQLStore(t, 200, noRows)
	for _, args := range []string{
		`not json`, `{}`, `{"query":"   "}`, `{"query":42}`,
		`{"query":"` + strings.Repeat("x", maxSQLQueryBytes+1) + `"}`,
	} {
		if _, err := runSQLJob(s, args); !errors.Is(err, ErrInvalidArgs) {
			t.Fatalf("%.40s: err = %v, want ErrInvalidArgs", args, err)
		}
	}
	if len(s.requests) != 0 {
		t.Fatal("the store was called for invalid args")
	}
}

// A statement the store refuses (syntax, binder, not read-only) is the caller's
// fault: not retried, and carrying the store's own words.
func TestSQLJobRefusalIsABadArgumentWithTheStoresWords(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusForbidden} {
		s := newSQLStore(t, status, `{"error":{"type":"not_read_only","message":"only one read-only SELECT"}}`)
		_, err := runSQLJob(s, `{"query":"delete from x"}`)
		if !errors.Is(err, ErrInvalidArgs) {
			t.Fatalf("%d: err = %v, want ErrInvalidArgs", status, err)
		}
		if got := StoreMessage(err); got != "only one read-only SELECT" {
			t.Fatalf("%d: StoreMessage = %q", status, got)
		}
	}
	s := newSQLStore(t, http.StatusInternalServerError, `{"error":{"type":"internal","message":"boom"}}`)
	_, err := runSQLJob(s, `{"query":"select 1"}`)
	var ue *UpstreamError
	if errors.Is(err, ErrInvalidArgs) || !errors.As(err, &ue) || !ue.Retryable() {
		t.Fatalf("a store fault: err = %v, want a retryable UpstreamError", err)
	}
}

// The platform caps the stored result; the box drops trailing rows to fit and
// says so, and passes the store's own truncation through.
func TestSQLJobTrimsToTheBudget(t *testing.T) {
	rows := make([][]any, 0, 2000)
	for i := 0; i < 2000; i++ {
		rows = append(rows, []any{strings.Repeat("y", 1000)})
	}
	s := newSQLStore(t, 200, rowsJSON([]string{"v"}, rows...))
	out, err := runSQLJob(s, `{"query":"select v"}`)
	if err != nil {
		t.Fatal(err)
	}
	res := out.Payload.(SQLResult)
	raw, _ := json.Marshal(res)
	if !res.Truncated || len(raw) > maxSQLResultBytes || len(res.Rows) == 0 || len(res.Rows) == 2000 {
		t.Fatalf("truncated=%v rows=%d bytes=%d (budget %d)", res.Truncated, len(res.Rows), len(raw), maxSQLResultBytes)
	}

	s = newSQLStore(t, 200, `{"columns":["v"],"types":["INTEGER"],"rows":[[1]],"truncated":true}`)
	out, err = runSQLJob(s, `{"query":"select v"}`)
	if err != nil || !out.Payload.(SQLResult).Truncated {
		t.Fatalf("the store's truncation was lost: %+v %v", out.Payload, err)
	}
}

// promql_instant / promql_range are known and refused by name, without asking
// the store: an older platform still enqueues them.
func TestPromQLKindsAreRefusedByName(t *testing.T) {
	s := newSQLStore(t, 200, noRows)
	for _, kind := range []string{KindPromQLInstant, KindPromQLRange} {
		_, err := Run(context.Background(), s.client, kind, []byte(`{"query":"up"}`), time.Now())
		if !errors.Is(err, ErrPromQLNotSupported) {
			t.Fatalf("%s: err = %v, want ErrPromQLNotSupported", kind, err)
		}
	}
	if len(s.requests) != 0 {
		t.Fatal("the store was called for a PromQL kind")
	}
}
