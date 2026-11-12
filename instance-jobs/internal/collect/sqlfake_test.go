package collect

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// Test doubles for pgai-mon's POST /sql: the request is the SQL text, the answer
// is {"columns", "types", "rows", "row_count", "truncated"}.

// noRows is an empty result set.
const noRows = `{"columns":["vals"],"types":["DOUBLE[]"],"rows":[],"row_count":0,"truncated":false}`

// sqlText reads the statement a request carries.
func sqlText(r *http.Request) string {
	b, _ := io.ReadAll(r.Body)
	return string(b)
}

// isQuery reports whether sql is the box's query of that name (its leading
// `-- instance-jobs: <name>` line).
func isQuery(sql, name string) bool {
	return strings.HasPrefix(sql, "-- instance-jobs: "+name+"\n")
}

// rowsJSON renders a result set; every column is typed VARCHAR except `vals`.
func rowsJSON(cols []string, rows ...[]any) string {
	types := make([]string, len(cols))
	for i, c := range cols {
		types[i] = "VARCHAR"
		if c == "vals" {
			types[i] = "DOUBLE[]"
		}
	}
	if rows == nil {
		rows = [][]any{}
	}
	b, err := json.Marshal(map[string]any{
		"columns": cols, "types": types, "rows": rows, "row_count": len(rows), "truncated": false,
	})
	if err != nil {
		panic(err)
	}
	return string(b)
}
