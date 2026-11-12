package collect

import (
	"errors"
	"strings"
)

// The PromQL query kinds. The store is pgai-mon, which answers SQL and has no
// PromQL API, so these kinds are refused with ErrPromQLNotSupported rather than
// treated as unknown: an older platform still enqueues them, and the caller has
// to be told what to send instead. The CLI's `pgai promql` maps onto the `sql`
// kind (see sqljob.go and README.md).
const (
	KindPromQLInstant = "promql_instant"
	KindPromQLRange   = "promql_range"
)

// ErrPromQLNotSupported answers a promql_instant / promql_range job. It is
// permanent: retrying cannot make the store speak PromQL.
var ErrPromQLNotSupported = errors.New("not supported: use sql")

// StripControlsToSpace is StripControls for TEXT a human will read: the same
// runes are neutralised, but each becomes a space rather than vanishing. A
// multi-line label value -- pgwatch carries query text in labels -- otherwise
// comes back with its tokens fused ("select 1\nfrom t" -> "select 1from t"),
// which changes what the text says rather than sanitising it.
func StripControlsToSpace(s string) string {
	return strings.Map(func(r rune) rune {
		if isControl(r) {
			return ' '
		}
		return r
	}, s)
}

// StripControls removes C0, C1 and DEL, plus LINE/PARAGRAPH SEPARATOR. Applied
// to everything that comes back from the metric store and is later rendered:
// label keys and values, string sample values, and the error message.
//
// U+009B is the single-character CSI -- `ESC [` in 8-bit form -- so dropping ESC
// alone would not be enough. parseRequest REJECTS this same class for
// cluster_name and node_name; here it is stripped, because a metric label is
// not something to fail a whole collection over.
func StripControls(s string) string {
	return strings.Map(func(r rune) rune {
		if isControl(r) {
			return -1
		}
		return r
	}, s)
}

// isControl is the one definition both variants use: C0, DEL, C1, and the two
// Unicode line separators. U+009B is the single-character CSI -- `ESC [` in
// 8-bit form -- so dropping ESC alone would not be enough.
func isControl(r rune) bool {
	return r < 0x20 || r == 0x7f || (r >= 0x80 && r <= 0x9f) || r == 0x2028 || r == 0x2029
}

// StoreMessage returns the metric store's own words about why it refused a
// query, or "" when there were none. It exists so the runner can put them in
// the submitted error: a syntax error is the ONE failure where the store's
// message is the whole diagnostic, and a caller that only has ErrInvalidArgs
// can tell the user nothing more useful than "bad args" -- which points them at
// the platform, where the fault is not (#378).
//
// The text is attacker-influenced: a store echoes the submitted expression back
// inside it, and PromQL string literals take Go-style escapes. Callers MUST
// strip control characters before it reaches a terminal or a stored field.
func StoreMessage(err error) string {
	var ue *UpstreamError
	if errors.As(err, &ue) {
		return ue.StoreMessage
	}
	return ""
}
