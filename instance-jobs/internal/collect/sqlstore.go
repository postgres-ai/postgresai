package collect

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
)

// sqlPath is the store's one query endpoint. It takes a single read-only SELECT
// as a text/plain body; anything else is refused by the store (not_read_only).
const sqlPath = "sql"

// sqlResponse is pgai-mon's row format:
// {"columns": [...], "types": [...], "rows": [[...], ...], "row_count", "truncated", "elapsed_ms"}.
// Numbers are kept as json.Number so a BIGINT above 2^53 survives the decode.
type sqlResponse struct {
	Columns   []string            `json:"columns"`
	Types     []string            `json:"types"`
	Rows      [][]json.RawMessage `json:"rows"`
	RowCount  int                 `json:"row_count"`
	Truncated bool                `json:"truncated"`
}

// sqlError is the store's error body: {"error": {"type": "parse_error", "message": "..."}}.
type sqlError struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// countingReader counts bytes read so postSQL can tell a body truncated at the
// cap from one that is genuinely malformed.
type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// postSQL runs one SELECT on the store and decodes the answer.
//
// Shared by every caller rather than copied, so the guards cannot drift apart:
// the auth header, the status check, the size cap and the truncated-versus-
// malformed distinction are all decided in exactly one place. The path is fixed;
// only the body varies.
func (c *Client) postSQL(ctx context.Context, sql string) (*sqlResponse, error) {
	endpoint := c.baseURL + "/" + sqlPath
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBufferString(sql))
	if err != nil {
		return nil, err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	req.Header.Set("Content-Type", "text/plain; charset=utf-8")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// One byte past the cap, so a truncated body is distinguishable below.
	limited := &countingReader{r: io.LimitReader(resp.Body, c.maxRespBytes+1)}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Read the body BEFORE giving up on it: a query the store refuses names
		// the reason in error.message, and for a typed query that is the whole
		// diagnostic (#378). Bounded far tighter than the success path; over the
		// bound the decode fails and StoreMessage stays empty.
		var body sqlError
		_ = json.NewDecoder(io.LimitReader(limited, maxErrorBodyBytes)).Decode(&body)
		return nil, &UpstreamError{
			StatusCode:   resp.StatusCode,
			Message:      fmt.Sprintf("metric store returned status %d", resp.StatusCode),
			StoreMessage: body.Error.Message,
		}
	}

	var out sqlResponse
	dec := json.NewDecoder(limited)
	dec.UseNumber()
	if err := dec.Decode(&out); err != nil {
		if limited.n > c.maxRespBytes {
			// 413, deliberately: the body scales with series cardinality, so a
			// window that overflows the cap overflows it every time. A 502
			// would be retried three times for the same answer, and would look
			// like a store fault instead of a window we cannot fetch.
			return nil, &UpstreamError{
				StatusCode: http.StatusRequestEntityTooLarge,
				Message:    "response exceeded the size limit",
			}
		}
		return nil, &UpstreamError{StatusCode: http.StatusBadGateway, Message: "failed to parse the response"}
	}
	if out.Columns == nil || out.Rows == nil {
		return nil, &UpstreamError{StatusCode: http.StatusBadGateway, Message: "the response is not a result set"}
	}
	return &out, nil
}

// querySeries runs a SELECT shaped by sumBySQL -- label columns and a `vals`
// list per row -- and returns one promSeries per row.
//
// A store answer cut at its own row or byte cap is an error, never a short
// result: a collection computed from part of the series would land as a
// confident, wrong number. 413, as for our own cap, and for the same reason.
func (c *Client) querySeries(ctx context.Context, sql string) ([]promSeries, error) {
	res, err := c.postSQL(ctx, sql)
	if err != nil {
		return nil, err
	}
	if res.Truncated {
		return nil, &UpstreamError{StatusCode: http.StatusRequestEntityTooLarge, Message: "the store truncated the result"}
	}
	vi := -1
	for i, col := range res.Columns {
		if col == "vals" {
			vi = i
		}
	}
	if vi < 0 {
		return nil, &UpstreamError{StatusCode: http.StatusBadGateway, Message: "the result has no vals column"}
	}
	out := make([]promSeries, 0, len(res.Rows))
	for _, row := range res.Rows {
		if len(row) != len(res.Columns) {
			return nil, &UpstreamError{StatusCode: http.StatusBadGateway, Message: "a result row does not match its columns"}
		}
		labels := make(map[string]string, len(row)-1)
		for i, col := range res.Columns {
			if i == vi {
				continue
			}
			var v string
			if json.Unmarshal(row[i], &v) == nil {
				labels[col] = v
			}
		}
		var raw []json.RawMessage
		if json.Unmarshal(row[vi], &raw) != nil {
			return nil, &UpstreamError{StatusCode: http.StatusBadGateway, Message: "vals is not a list"}
		}
		out = append(out, promSeries{labels: labels, values: parseValues(raw)})
	}
	return out, nil
}

// parseValues reads a list of numbers. The store writes a non-finite double as
// the string "NaN", "+Inf" or "-Inf" (JSON has no such number), which parse;
// anything else that does not parse is skipped, never zeroed. +Inf/NaN are kept
// -- only the tempfile path filters them (see AggregateTempfile).
func parseValues(raw []json.RawMessage) []float64 {
	vals := make([]float64, 0, len(raw))
	for _, r := range raw {
		var n json.Number
		if json.Unmarshal(r, &n) == nil {
			if f, err := n.Float64(); err == nil {
				vals = append(vals, f)
			}
			continue
		}
		var s string
		if json.Unmarshal(r, &s) == nil {
			if f, err := strconv.ParseFloat(s, 64); err == nil {
				vals = append(vals, f)
			}
		}
	}
	return vals
}
