package collect

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"time"
)

// queryTextMaxChars is how much of a statement's text the payload carries:
// whitespace collapsed, at most 500 characters -- the "full" display name the
// pgwatch_query_info metric carried before, so a payload is no larger than it was.
const queryTextMaxChars = 500

// QueryInfoTexts resolves the query text for the given ids, returning
// queryid -> text (ids with no text are omitted). An empty id set issues no
// request.
//
// It reads the store's query_texts view: one row per stored text of a
// (queryid, database), with first_seen / last_seen (refreshed hourly while the
// statement stays in pg_stat_statements). The text of an id is its newest one
// first seen by the end of the window, from any database -- AAS entries carry
// a datname, but the lookup matches on query_id alone, as before.
//
// Best-effort: any error is returned for the caller to log and drop.
// Enrichment must never fail a collection.
func (c *Client) QueryInfoTexts(ctx context.Context, queryIDs []string, start, end time.Time) (map[string]string, error) {
	texts := make(map[string]string)
	if len(queryIDs) == 0 {
		return texts, nil
	}
	res, err := c.postSQL(ctx, queryTextsSQL(queryIDs, end))
	if err != nil {
		return nil, err
	}
	for _, row := range res.Rows {
		if len(row) != 2 {
			continue
		}
		var qid, text string
		if json.Unmarshal(row[0], &qid) != nil || json.Unmarshal(row[1], &text) != nil {
			continue
		}
		if qid != "" && text != "" {
			texts[qid] = text
		}
	}
	return texts, nil
}

// queryTextsSQL renders the lookup. The ids are compared as text, so an id above
// 2^53 is never rounded; each is a quoted literal.
func queryTextsSQL(queryIDs []string, end time.Time) string {
	ids := make([]string, 0, len(queryIDs))
	for _, id := range queryIDs {
		ids = append(ids, sqlString(id))
	}
	return "-- instance-jobs: query_texts\n" +
		"SELECT CAST(queryid AS VARCHAR) AS queryid,\n" +
		"       left(trim(regexp_replace(arg_max(query, last_seen), '\\s+', ' ', 'g')), " + strconv.Itoa(queryTextMaxChars) + ") AS query\n" +
		"FROM query_texts\n" +
		"WHERE CAST(queryid AS VARCHAR) IN (" + strings.Join(ids, ", ") + ")\n" +
		"  AND query IS NOT NULL AND first_seen <= to_timestamp(" + strconv.FormatInt(end.Unix(), 10) + ")\n" +
		"GROUP BY 1\nORDER BY 1"
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
// flavor maps, matching on query_id alone (the lookup is per id, not per database).
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
