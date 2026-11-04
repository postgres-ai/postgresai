package collect

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"time"
)

const KindQueryText = "query_text"
const queryTextEndpoint = "http://monitoring_flask_backend:8000/query_text"

type QueryTextArgs struct {
	LookupMode string  `json:"lookup_mode,omitempty"`
	Cluster    string  `json:"cluster"`
	Node       string  `json:"node_name"`
	Database   string  `json:"datname"`
	QueryID    string  `json:"queryid"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
}
type QueryTextResult struct {
	TextScope   string   `json:"text_scope,omitempty"`
	RetrievedAt *float64 `json:"retrieved_at,omitempty"`
	QueryID     string   `json:"queryid"`
	Query       *string  `json:"query"`
	Truncated   bool     `json:"truncated"`
	ObservedAt  *float64 `json:"observed_at"`
	Available   bool     `json:"available"`
}

// QueryText contacts one fixed compose-internal endpoint. No URL or SQL is accepted.
func QueryText(ctx context.Context, args json.RawMessage, now time.Time) (Outcome, error) {
	return queryText(ctx, args, now, queryTextEndpoint, queryTextHTTPClient())
}

func queryTextHTTPClient() *http.Client {
	return &http.Client{Timeout: 12 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
}
func queryText(ctx context.Context, args json.RawMessage, now time.Time, endpoint string, client *http.Client) (Outcome, error) {
	var a QueryTextArgs
	decoder := json.NewDecoder(bytes.NewReader(args))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&a) != nil || decoder.Decode(new(any)) != io.EOF ||
		(a.LookupMode != "" && a.LookupMode != "latest_collected") ||
		len(a.Cluster) == 0 || len(a.Cluster) > 512 || len(a.Node) == 0 || len(a.Node) > 512 || len(a.Database) == 0 || len(a.Database) > 512 ||
		!regexp.MustCompile(`^-?[0-9]{1,20}$`).MatchString(a.QueryID) || a.Start < 0 || a.End <= a.Start || a.End > float64(now.Unix())+5 || a.End-a.Start > 93*86400 {
		return Outcome{}, fmt.Errorf("%w: query text scope", ErrInvalidArgs)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(args))
	if err != nil {
		return Outcome{}, fmt.Errorf("query text request failed")
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := client.Do(req)
	if err != nil {
		return Outcome{}, fmt.Errorf("query text transport failed")
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		return Outcome{}, fmt.Errorf("query text lookup failed")
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, 450001))
	if err != nil || len(raw) > 450000 {
		return Outcome{}, fmt.Errorf("query text response too large")
	}
	var result QueryTextResult
	if json.Unmarshal(raw, &result) != nil {
		return Outcome{}, fmt.Errorf("invalid query text response")
	}
	upper := a.End
	if a.LookupMode == "latest_collected" {
		if result.TextScope != "latest_collected" || (result.Available && (result.RetrievedAt == nil || *result.RetrievedAt > float64(time.Now().Unix())+5 || *result.RetrievedAt < a.End)) {
			return Outcome{}, fmt.Errorf("invalid query text retrieval scope")
		}
		if result.RetrievedAt != nil {
			upper = *result.RetrievedAt
		}
	} else if result.TextScope != "" && result.TextScope != "historical" {
		return Outcome{}, fmt.Errorf("invalid query text retrieval scope")
	}
	if result.QueryID != a.QueryID ||
		(result.Available && (result.Query == nil || len(*result.Query) > 65536 || result.ObservedAt == nil || *result.ObservedAt <= a.Start || *result.ObservedAt > upper || (a.LookupMode == "latest_collected" && *result.ObservedAt < upper-93*86400))) ||
		(!result.Available && (result.Query != nil || result.ObservedAt != nil || result.Truncated)) {
		return Outcome{}, fmt.Errorf("invalid query text response")
	}
	return Outcome{Status: OutcomeOK, Payload: result}, nil
}
