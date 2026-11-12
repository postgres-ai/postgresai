package collect

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"unicode/utf8"
)

// KindSQL runs one read-only SELECT somebody typed against the store and returns
// the rows. Like the PromQL kinds it replaces, the result is read back by a
// human and is never applied to anything: `sql` is not in the platform's
// c_collection_kinds, so instance_job_apply_collection is never reached.
//
// The box adds no SQL policy of its own. The store runs the text only if it is
// a single read-only statement against its own views (anything else is refused
// with not_read_only), under its own memory, time and row caps; the box picks
// the endpoint and sends the text as the request body, never as part of a URL.
const KindSQL = "sql"

// maxSQLQueryBytes bounds the statement the box forwards. The platform validates
// its own bound before enqueueing; this one keeps a malformed job from becoming
// a multi-megabyte request.
const maxSQLQueryBytes = 64 << 10

// maxSQLResultBytes is the compact-JSON budget of one submitted result. The
// platform caps `result` at 1 MiB measured as octet_length(result::text) on
// JSONB, whose text form inserts a space after every `:` and `,`: measured at up
// to ~20% more than Go's compact JSON. 850000 leaves 23% headroom. Do not raise
// it toward 1 MiB without re-measuring the jsonb text form, which is the only
// figure the platform reads.
const maxSQLResultBytes = 850000

type sqlArgs struct {
	Query string `json:"query"`
}

// SQLResult is what the box submits: the store's columns, their types and the
// rows, rebuilt value by value from the decoded answer so nothing the store
// emits reaches the platform unexamined. Truncated is set when the store cut the
// result at its own cap or the box dropped trailing rows to fit the budget.
type SQLResult struct {
	Columns   []string `json:"columns"`
	Types     []string `json:"types"`
	Rows      [][]any  `json:"rows"`
	Truncated bool     `json:"truncated"`
}

// runSQL answers one sql job.
func runSQL(ctx context.Context, c *Client, args []byte) (Outcome, error) {
	var a sqlArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return Outcome{}, fmt.Errorf("%w: args did not decode", ErrInvalidArgs)
	}
	if strings.TrimSpace(a.Query) == "" {
		return Outcome{}, fmt.Errorf("%w: query is required", ErrInvalidArgs)
	}
	if len(a.Query) > maxSQLQueryBytes || !utf8.ValidString(a.Query) {
		return Outcome{}, fmt.Errorf("%w: query is not valid UTF-8 of at most %d bytes", ErrInvalidArgs, maxSQLQueryBytes)
	}
	res, err := c.postSQL(ctx, a.Query)
	if err != nil {
		// A refusal of the STATEMENT is the caller's fault, not the store's: a
		// syntax or binder error (400) or a statement that is not one read-only
		// SELECT (403). Classed as a bad argument so it is not retried three
		// times for the same answer, and carrying the store's own words, which
		// are the whole diagnostic to someone who has just typed it (#378).
		var ue *UpstreamError
		if errors.As(err, &ue) && isCallerFault(ue.StatusCode) {
			return Outcome{}, fmt.Errorf("%w: %w", ErrInvalidArgs, ue)
		}
		return Outcome{}, err
	}
	return Outcome{Status: OutcomeOK, Payload: buildSQLResult(res)}, nil
}

// isCallerFault reports whether a store status means the submitted STATEMENT was
// wrong, as opposed to the box being pointed at the wrong thing.
func isCallerFault(status int) bool {
	return status == http.StatusBadRequest || status == http.StatusForbidden
}

// buildSQLResult copies the answer value by value, strips control characters
// from every string (the CLI renders them on a terminal), and drops trailing
// rows until the result fits maxSQLResultBytes.
func buildSQLResult(res *sqlResponse) SQLResult {
	out := SQLResult{Columns: []string{}, Types: []string{}, Rows: [][]any{}, Truncated: res.Truncated}
	for _, col := range res.Columns {
		out.Columns = append(out.Columns, StripControls(col))
	}
	for _, t := range res.Types {
		out.Types = append(out.Types, StripControls(t))
	}
	size := 0
	if b, err := json.Marshal(out); err == nil {
		size = len(b)
	}
	for _, raw := range res.Rows {
		row := make([]any, 0, len(raw))
		for _, cell := range raw {
			row = append(row, cleanValue(cell))
		}
		b, err := json.Marshal(row)
		if err != nil {
			out.Truncated = true
			break
		}
		// +1 for the comma between rows
		if size+len(b)+1 > maxSQLResultBytes {
			out.Truncated = true
			break
		}
		size += len(b) + 1
		out.Rows = append(out.Rows, row)
	}
	return out
}

// cleanValue decodes one JSON cell and rebuilds it: numbers stay exact
// (json.Number), strings lose control characters (to spaces: a statement text
// with newlines keeps its words apart), lists and maps are cleaned recursively.
func cleanValue(raw json.RawMessage) any {
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	var v any
	if dec.Decode(&v) != nil {
		return nil
	}
	return clean(v)
}

func clean(v any) any {
	switch x := v.(type) {
	case string:
		return StripControlsToSpace(x)
	case []any:
		for i := range x {
			x[i] = clean(x[i])
		}
		return x
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			out[StripControls(k)] = clean(val)
		}
		return out
	default:
		return x
	}
}
