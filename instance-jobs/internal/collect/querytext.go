package collect

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// queryInfoMetric maps a queryid to its (truncated) query text. The monitoring
// stack scrapes it from monitoring_flask_backend:/query_info_metrics into the
// same store this package already queries.
const queryInfoMetric = "pgwatch_query_info"

// queryInfoTextLabel carries the longest available text, formatted by the
// backend as "<queryid> | <text>" (or a bare "<queryid>" when it has none).
const queryInfoTextLabel = "displayname_full"

const queryInfoLabelSep = " | "

// QueryInfoTexts resolves the query text for the given ids, returning
// queryid -> text (ids with no text are omitted). An empty id set issues no
// request.
//
// It is a WINDOWED lookup, not an instant query at now(): the exporter emits a
// queryid only while it was active in the last ~10 minutes at each scrape, so
// the series exists only at the timestamps that id was recently active. AAS
// windows are historical, so this evaluates
// last_over_time(pgwatch_query_info{queryid=~...}[<window>]) at time=end -- one
// small vector, one sample per id active anywhere in [start, end].
//
// Best-effort: any error is returned for the caller to log and drop.
// Enrichment must never fail a collection.
func (c *Client) QueryInfoTexts(ctx context.Context, queryIDs []string, start, end time.Time) (map[string]string, error) {
	texts := make(map[string]string)
	if len(queryIDs) == 0 {
		return texts, nil
	}

	// Regex-quote each id, join with |, then escape for the PromQL string
	// literal -- the same two-step escaping as waitEventTypeRegex.
	quoted := make([]string, 0, len(queryIDs))
	for _, id := range queryIDs {
		quoted = append(quoted, regexp.QuoteMeta(id))
	}
	idRegex := escapeLabelValue(strings.Join(quoted, "|"))

	windowSec := int(end.Sub(start) / time.Second)
	if windowSec < StepSeconds {
		windowSec = StepSeconds
	}

	q := fmt.Sprintf(`last_over_time(%s{queryid=~"%s"}[%ds])`, queryInfoMetric, idRegex, windowSec)
	series, err := c.queryInstant(ctx, q, end)
	if err != nil {
		return nil, err
	}

	for _, labels := range series {
		qid := labels["queryid"]
		if qid == "" {
			continue
		}
		// A bare "<queryid>" with no separator means the backend had no text:
		// omit it rather than emitting the id as its own text.
		text, ok := strings.CutPrefix(labels[queryInfoTextLabel], qid+queryInfoLabelSep)
		if !ok || text == "" {
			continue
		}
		texts[qid] = text
	}
	return texts, nil
}

// CollectQueryIDs returns the distinct query_ids across all three flavor maps
// and every class, in first-seen order, so the caller resolves each text once.
func CollectQueryIDs(r Result) []string {
	seen := make(map[string]struct{})
	var out []string
	add := func(m map[string][]QueryIDEntry) {
		for _, key := range QueryIDClassKeys {
			for _, e := range m[key] {
				if e.QueryID == "" {
					continue
				}
				if _, ok := seen[e.QueryID]; ok {
					continue
				}
				seen[e.QueryID] = struct{}{}
				out = append(out, e.QueryID)
			}
		}
	}
	add(r.TopQueryIDsWorst1m)
	add(r.TopQueryIDsP99)
	add(r.TopQueryIDsP999)
	return out
}

// EnrichQueryTexts stamps resolved texts onto every entry across the three
// flavor maps, matching on query_id alone (pgwatch_query_info has no datname).
// Entries without a text keep Query == "" and are omitted from the JSON.
func EnrichQueryTexts(r *Result, texts map[string]string) {
	if len(texts) == 0 {
		return
	}
	apply := func(m map[string][]QueryIDEntry) {
		for _, list := range m {
			for i := range list {
				if t, ok := texts[list[i].QueryID]; ok && t != "" {
					list[i].Query = t
				}
			}
		}
	}
	apply(r.TopQueryIDsWorst1m)
	apply(r.TopQueryIDsP99)
	apply(r.TopQueryIDsP999)
}
