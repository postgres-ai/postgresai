package collect

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNativeQueryTextExactOriginalAndScope(t *testing.T) {
	original := "/* comment */\nSELECT  $1, '☃'"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/query_text" || r.URL.RawQuery != "" {
			t.Error("unexpected request")
		}
		var a QueryTextArgs
		json.NewDecoder(r.Body).Decode(&a)
		if a.Database != "d" || a.QueryID != "9223372036854775807" {
			t.Error("lost scope")
		}
		observed := float64(150)
		json.NewEncoder(w).Encode(QueryTextResult{QueryID: a.QueryID, Query: &original, Available: true, ObservedAt: &observed})
	}))
	defer server.Close()
	out, err := queryText(context.Background(), json.RawMessage(`{"cluster":"c","node_name":"n","datname":"d","queryid":"9223372036854775807","start":100,"end":200}`), time.Unix(200, 0), server.URL+"/query_text", server.Client())
	if err != nil || *out.Payload.(QueryTextResult).Query != original {
		t.Fatalf("result=%v error=%v", out, err)
	}
}
func TestNativeQueryTextRejectsInvalidArgsAndForeignResults(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.Write([]byte(`{"queryid":"other","query":"secret","available":true,"observed_at":150,"truncated":false}`))
	}))
	defer server.Close()
	for _, raw := range []string{`{"queryid":"1"}`, `{"cluster":"c","node_name":"n","datname":"d","queryid":"1","start":100,"end":200,"url":"http://elsewhere"}`} {
		if _, err := queryText(context.Background(), json.RawMessage(raw), time.Unix(200, 0), server.URL, server.Client()); err == nil {
			t.Fatal("accepted bad scope")
		}
	}
	if requests != 0 {
		t.Fatal("called backend for invalid scope")
	}
	if _, err := queryText(context.Background(), json.RawMessage(`{"cluster":"c","node_name":"n","datname":"d","queryid":"1","start":100,"end":200}`), time.Unix(200, 0), server.URL, server.Client()); err == nil {
		t.Fatal("accepted foreign query")
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
	_, err := queryText(context.Background(), json.RawMessage(`{"cluster":"c","node_name":"n","datname":"d","queryid":"1","start":100,"end":200}`), time.Unix(200, 0), source.URL, queryTextHTTPClient())
	if err == nil || reached {
		t.Fatal("followed redirect or accepted redirect response")
	}
}

func TestLatestCollectedQueryTextSeparatesChartAndRetrievalTime(t *testing.T) {
	for _, test := range []struct {
		name, body string
		valid      bool
	}{
		{"after chart", `{"queryid":"1","query":"SELECT $1","available":true,"observed_at":250,"retrieved_at":300,"text_scope":"latest_collected"}`, true},
		{"future row", `{"queryid":"1","query":"SELECT $1","available":true,"observed_at":301,"retrieved_at":300,"text_scope":"latest_collected"}`, false},
		{"foreign query", `{"queryid":"2","query":"SELECT $1","available":true,"observed_at":250,"retrieved_at":300,"text_scope":"latest_collected"}`, false},
		{"missing scope", `{"queryid":"1","query":"SELECT $1","available":true,"observed_at":250,"retrieved_at":300}`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(test.body)) }))
			defer server.Close()
			_, err := queryText(context.Background(), json.RawMessage(`{"cluster":"c","node_name":"n","datname":"d","queryid":"1","start":100,"end":200,"lookup_mode":"latest_collected"}`), time.Unix(300, 0), server.URL, server.Client())
			if (err == nil) != test.valid {
				t.Fatalf("valid=%v error=%v", test.valid, err)
			}
		})
	}
}
