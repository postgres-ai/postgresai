package collect

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"time"
	"unicode/utf8"
)

// KindQueryText returns the stored text of one statement: one queryid in one
// database, newest first. The args are identifiers only; no SQL is accepted.
const KindQueryText = "query_text"

// queryTextMaxBytes caps the text a result carries (UTF-8, cut on a rune
// boundary); a longer text is flagged truncated.
const queryTextMaxBytes = 65536

// queryTextHorizon is how far back a latest_collected lookup reaches.
const queryTextHorizon = 93 * 24 * time.Hour

var queryIDPattern = regexp.MustCompile(`^-?[0-9]{1,20}$`)

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

// QueryText answers a query_text job from the store's query_texts view.
//
// Historical (no lookup_mode): the text of the (queryid, datname) seen in
// (start, end]; observed_at is the last time in that window the store saw it.
// latest_collected: the same over (max(start, now - 93 days), now], with
// retrieved_at = now. When the newest observation has two different texts the
// answer is "not available" rather than a guess.
//
// query_texts carries no cluster or node: it holds the statements of the
// instances this box's store collects. cluster and node_name are validated as
// before and do not narrow the lookup.
func QueryText(ctx context.Context, c *Client, args json.RawMessage, now time.Time) (Outcome, error) {
	var a QueryTextArgs
	decoder := json.NewDecoder(bytes.NewReader(args))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&a) != nil || decoder.Decode(new(any)) != io.EOF ||
		(a.LookupMode != "" && a.LookupMode != "latest_collected") ||
		len(a.Cluster) == 0 || len(a.Cluster) > 512 || len(a.Node) == 0 || len(a.Node) > 512 || len(a.Database) == 0 || len(a.Database) > 512 ||
		!utf8.ValidString(a.Database) || !queryIDPattern.MatchString(a.QueryID) ||
		math.IsNaN(a.Start) || math.IsInf(a.Start, 0) || math.IsNaN(a.End) || math.IsInf(a.End, 0) ||
		a.Start < 0 || a.End <= a.Start || a.End > float64(now.Unix())+5 || a.End-a.Start > 93*86400 {
		return Outcome{}, fmt.Errorf("%w: query text scope", ErrInvalidArgs)
	}
	lo, hi := a.Start, a.End
	result := QueryTextResult{QueryID: a.QueryID}
	if a.LookupMode == "latest_collected" {
		retrieved := float64(now.Unix())
		hi = retrieved
		if floor := float64(now.Add(-queryTextHorizon).Unix()); floor > lo {
			lo = floor
		}
		result.TextScope, result.RetrievedAt = "latest_collected", &retrieved
	}

	res, err := c.postSQL(ctx, queryTextSQL(a.QueryID, a.Database, lo, hi))
	if err != nil {
		return Outcome{}, err
	}
	if res.Truncated || len(res.Rows) > 1 {
		return Outcome{}, fmt.Errorf("invalid query text response")
	}
	if len(res.Rows) == 1 {
		row := res.Rows[0]
		var text string
		var observed json.Number
		if len(row) != 2 || json.Unmarshal(row[0], &text) != nil || json.Unmarshal(row[1], &observed) != nil {
			return Outcome{}, fmt.Errorf("invalid query text response")
		}
		at, err := observed.Float64()
		if err != nil || text == "" || at <= lo || at > hi {
			return Outcome{}, fmt.Errorf("invalid query text response")
		}
		if len(text) > queryTextMaxBytes {
			cut := queryTextMaxBytes
			for cut > 0 && !utf8.RuneStart(text[cut]) {
				cut--
			}
			text, result.Truncated = text[:cut], true
		}
		result.Query, result.ObservedAt, result.Available = &text, &at, true
	}
	return Outcome{Status: OutcomeOK, Payload: result}, nil
}

// queryTextSQL renders the lookup: the texts of the statement whose stored
// interval [first_seen, last_seen] meets (lo, hi], each observed at the last
// time inside it the store saw it; of those, the newest, if it is one text.
func queryTextSQL(queryID, database string, lo, hi float64) string {
	from := "to_timestamp(" + strconv.FormatFloat(lo, 'f', -1, 64) + ")"
	to := "to_timestamp(" + strconv.FormatFloat(hi, 'f', -1, 64) + ")"
	return "-- instance-jobs: query_text\n" +
		"WITH c AS (\n" +
		"  SELECT query, epoch(least(last_seen, " + to + ")) AS observed_at\n" +
		"  FROM query_texts\n" +
		"  WHERE CAST(queryid AS VARCHAR) = " + sqlString(queryID) + " AND dbname = " + sqlString(database) + "\n" +
		"    AND query IS NOT NULL AND query <> ''\n" +
		"    AND first_seen <= " + to + " AND last_seen > " + from + "\n" +
		"), n AS (\n" +
		"  SELECT * FROM c WHERE observed_at = (SELECT max(observed_at) FROM c)\n" +
		")\n" +
		"SELECT min(query) AS query, max(observed_at) AS observed_at\n" +
		"FROM n\n" +
		"HAVING count(*) > 0 AND count(DISTINCT query) = 1"
}
