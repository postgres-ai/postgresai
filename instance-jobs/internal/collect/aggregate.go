// Package collect queries the monitoring instance's own metric store and builds
// the AAS and TEMPFILE checkup payloads the platform stores verbatim.
//
// The crux: pgwatch emits a sample for pgwatch_wait_events_total only in a
// 1-minute slot that had waiters, so what comes back is SPARSE. The missing
// slots are real zeros and are never materialised here. Averages therefore
// divide by the slots the window should have contained and percentiles zero-pad
// up to that same count. Dividing by the returned sample count instead
// overstates AAS by 5-10x (production: present=501 against expected=4184).
package collect

import (
	"math"
	"sort"
)

// The five wait-event types we track. pgwatch emits the literal string "CPU*".
const (
	TypeCPU    = "CPU*"
	TypeIO     = "IO"
	TypeIPC    = "IPC"
	TypeLock   = "Lock"
	TypeLWLock = "LWLock"
)

// TypeToKey maps a wait_event_type label to its output JSON key.
var TypeToKey = map[string]string{
	TypeIO:     "io",
	TypeIPC:    "ipc",
	TypeLock:   "lock",
	TypeLWLock: "lwlock",
	TypeCPU:    "cpu",
}

// OrderedTypes is the canonical iteration order over wait-event types.
var OrderedTypes = []string{TypeIO, TypeIPC, TypeLock, TypeLWLock, TypeCPU}

// QueryIDClassKeys is the class-key order for top_queryids. Unlike top_events
// (which excludes CPU, having no sub-events) the queryid ranking INCLUDES cpu:
// pg_stat_activity carries a query_id for on-CPU backends.
var QueryIDClassKeys = []string{"total", "io", "ipc", "lock", "lwlock", "cpu"}

// TopQueryIDN caps ranked query_ids per (class, window). It is also the topk()
// bound applied in PromQL.
const TopQueryIDN = 5

// topEventKeys are the types included in top_events output (CPU excluded).
var topEventKeys = []string{TypeIO, TypeIPC, TypeLock, TypeLWLock}

const (
	qP99  = 0.99
	qP999 = 0.999
)

// Metrics holds the six numeric AAS values for one flavor (avg, worst1m, p99,
// p999). Total is the per-timestamp sum series, not the sum of the five.
type Metrics struct {
	Total  float64 `json:"total"`
	IO     float64 `json:"io"`
	IPC    float64 `json:"ipc"`
	Lock   float64 `json:"lock"`
	LWLock float64 `json:"lwlock"`
	CPU    float64 `json:"cpu"`
}

// TypeSeries is the present samples for one wait_event_type.
type TypeSeries struct {
	Type    string
	Samples []float64
}

// EventSeries is the present samples for one (type, event) pair.
type EventSeries struct {
	Type    string
	Event   string
	Samples []float64
}

// QueryIDSeries is the present samples for one (class, query_id, datname).
type QueryIDSeries struct {
	Class   string
	QueryID string
	Datname string
	Samples []float64
}

// QueryIDEntry is one ranked query_id. QueryID is a STRING and is never parsed
// to a number, so an id above 2^53 round-trips byte-identical. Query is the
// best-effort text and is omitted when unavailable.
type QueryIDEntry struct {
	QueryID string  `json:"queryid"`
	AAS     float64 `json:"aas"`
	Datname string  `json:"datname"`
	Query   string  `json:"query,omitempty"`
}

// AggregateInput is everything the aggregator needs.
type AggregateInput struct {
	ExpectedSlots int
	PerType       []TypeSeries
	TotalSeries   []float64
	PerEvent      []EventSeries
	PerQueryID    []QueryIDSeries
}

// Result holds the four flavors plus the rankings.
type Result struct {
	Avg     Metrics
	Worst1m Metrics
	P99     Metrics
	P999    Metrics

	TopWorst1m map[string][]string
	TopP99     map[string][]string
	TopP999    map[string][]string

	TopQueryIDsWorst1m map[string][]QueryIDEntry
	TopQueryIDsP99     map[string][]QueryIDEntry
	TopQueryIDsP999    map[string][]QueryIDEntry
}

// avg returns sum(present samples) / expectedSlots. It divides by expectedSlots
// and NOT by len(samples), because the omitted slots are real zeros. The
// denominator is bumped so it never falls below len(samples) -- query_range can
// return expectedSlots+1 boundary points -- which keeps avg <= worst1m.
func avg(samples []float64, expectedSlots int) float64 {
	n := expectedSlots
	if len(samples) > n {
		n = len(samples)
	}
	if n == 0 {
		return 0
	}
	var sum float64
	for _, v := range samples {
		sum += v
	}
	return sum / float64(n)
}

// worst1m returns the max present sample (0 if none). Missing slots are zeros,
// which never raise a max, so sparse emission does not affect it.
func worst1m(samples []float64) float64 {
	m := 0.0
	for _, v := range samples {
		if v > m {
			m = v
		}
	}
	return m
}

// quantilePadded computes the q-quantile of the present samples after padding
// them with zeros up to expectedSlots, interpolating linearly between order
// statistics (numpy's default "linear" method).
func quantilePadded(samples []float64, expectedSlots int, q float64) float64 {
	n := expectedSlots
	if len(samples) > n {
		n = len(samples)
	}
	if n == 0 {
		return 0
	}

	// Zeros sort first, so sorting the present samples and treating the leading
	// (n - len) ranks as zero is the padded, sorted slice.
	present := make([]float64, len(samples))
	copy(present, samples)
	sort.Float64s(present)

	padZeros := n - len(present)
	at := func(i int) float64 {
		if i < padZeros {
			return 0
		}
		return present[i-padZeros]
	}

	h := q * float64(n-1)
	lo := int(math.Floor(h))
	hi := int(math.Ceil(h))
	if lo == hi {
		return at(lo)
	}
	return at(lo) + (h-float64(lo))*(at(hi)-at(lo))
}

// round1 rounds to one decimal place.
func round1(v float64) float64 {
	return math.Round(v*10) / 10
}

// Aggregate computes every AAS metric from the input series.
func Aggregate(in AggregateInput) Result {
	es := in.ExpectedSlots

	byType := make(map[string][]float64, len(in.PerType))
	for _, ts := range in.PerType {
		byType[ts.Type] = ts.Samples
	}

	setMetric := func(m *Metrics, key string, v float64) {
		switch key {
		case "io":
			m.IO = v
		case "ipc":
			m.IPC = v
		case "lock":
			m.Lock = v
		case "lwlock":
			m.LWLock = v
		case "cpu":
			m.CPU = v
		}
	}

	var resAvg, resWorst, resP99, resP999 Metrics
	for _, typ := range OrderedTypes {
		key := TypeToKey[typ]
		samples := byType[typ]
		setMetric(&resAvg, key, round1(avg(samples, es)))
		setMetric(&resWorst, key, round1(worst1m(samples)))
		setMetric(&resP99, key, round1(quantilePadded(samples, es, qP99)))
		setMetric(&resP999, key, round1(quantilePadded(samples, es, qP999)))
	}

	// Total comes from the per-timestamp sum series, which is the correct "total
	// active sessions" semantics for max and the quantiles; for avg it tracks the
	// sum of the per-type avgs and is authoritative when they diverge.
	resAvg.Total = round1(avg(in.TotalSeries, es))
	resWorst.Total = round1(worst1m(in.TotalSeries))
	resP99.Total = round1(quantilePadded(in.TotalSeries, es, qP99))
	resP999.Total = round1(quantilePadded(in.TotalSeries, es, qP999))

	return Result{
		Avg:                resAvg,
		Worst1m:            resWorst,
		P99:                resP99,
		P999:               resP999,
		TopWorst1m:         topEvents(in.PerEvent, es, "worst1m"),
		TopP99:             topEvents(in.PerEvent, es, "p99"),
		TopP999:            topEvents(in.PerEvent, es, "p999"),
		TopQueryIDsWorst1m: topQueryIDs(in.PerQueryID, es, "worst1m"),
		TopQueryIDsP99:     topQueryIDs(in.PerQueryID, es, "p99"),
		TopQueryIDsP999:    topQueryIDs(in.PerQueryID, es, "p999"),
	}
}

// seriesScore is the ranking statistic for one series under a flavor. The
// per-series window stat is computed here rather than as a PromQL
// *_over_time quantile because PromQL would divide by the present-sample count
// and so could not do the zero-padding this package depends on.
func seriesScore(samples []float64, expectedSlots int, flavor string) float64 {
	switch flavor {
	case "worst1m":
		return worst1m(samples)
	case "p99":
		return quantilePadded(samples, expectedSlots, qP99)
	case "p999":
		return quantilePadded(samples, expectedSlots, qP999)
	default:
		return 0
	}
}

// topEvents ranks each type's events and returns the top 3 as pre-formatted
// "Name(value)" strings.
func topEvents(perEvent []EventSeries, expectedSlots int, flavor string) map[string][]string {
	type scored struct {
		event string
		score float64
	}
	byType := make(map[string][]scored)
	for _, ev := range perEvent {
		byType[ev.Type] = append(byType[ev.Type],
			scored{event: ev.Event, score: seriesScore(ev.Samples, expectedSlots, flavor)})
	}

	out := make(map[string][]string, len(topEventKeys))
	for _, typ := range topEventKeys {
		items := byType[typ]
		sort.SliceStable(items, func(i, j int) bool {
			if items[i].score != items[j].score {
				return items[i].score > items[j].score
			}
			return items[i].event < items[j].event
		})
		list := []string{}
		for i := 0; i < len(items) && i < 3; i++ {
			if items[i].score <= 0 {
				break
			}
			list = append(list, FormatEvent(items[i].event, items[i].score))
		}
		out[TypeToKey[typ]] = list
	}
	return out
}

// FormatEvent formats an event name and value as "Name(value)" with the value
// at one decimal, e.g. "DataFileRead(89.0)".
func FormatEvent(name string, value float64) string {
	return name + "(" + formatFloat1(round1(value)) + ")"
}

// topQueryIDs ranks each class's query_ids and returns up to TopQueryIDN entries
// per class, AAS descending. Series scoring at or below zero are dropped: after
// zero-padding they had no activity in the window.
func topQueryIDs(perQueryID []QueryIDSeries, expectedSlots int, flavor string) map[string][]QueryIDEntry {
	type scored struct {
		qid   string
		dn    string
		score float64
	}
	byClass := make(map[string][]scored)
	for _, s := range perQueryID {
		byClass[s.Class] = append(byClass[s.Class],
			scored{qid: s.QueryID, dn: s.Datname, score: seriesScore(s.Samples, expectedSlots, flavor)})
	}

	out := make(map[string][]QueryIDEntry, len(QueryIDClassKeys))
	for _, key := range QueryIDClassKeys {
		items := byClass[key]
		sort.SliceStable(items, func(i, j int) bool {
			if items[i].score != items[j].score {
				return items[i].score > items[j].score
			}
			if items[i].qid != items[j].qid {
				return items[i].qid < items[j].qid
			}
			return items[i].dn < items[j].dn
		})
		list := []QueryIDEntry{}
		for i := 0; i < len(items) && i < TopQueryIDN; i++ {
			if items[i].score <= 0 {
				break
			}
			list = append(list, QueryIDEntry{
				QueryID: items[i].qid,
				AAS:     round1(items[i].score),
				Datname: items[i].dn,
			})
		}
		out[key] = list
	}
	return out
}
