package collect

import (
	"context"
	"math"
	"strconv"
	"time"
)

func itoa(n int64) string { return strconv.FormatInt(n, 10) }

// Observation describes actual raw metric sample time. Range evaluation times
// and period boundaries cannot establish that the collector observed a source.
type Observation struct {
	Source      string `json:"source"`
	ObservedAt  string `json:"observed_at"`
	SampleCount int    `json:"sample_count"`
}

func (c *Client) queryObservation(ctx context.Context, kind string, f SeriesFilter, start, end time.Time) *Observation {
	// PromQL: max(timestamp(<metric>{<selector>})) -- per step, the newest raw
	// sample time (seconds) of the series the collection read.
	name, metric, matchers := "aas_observation", "pgwatch_wait_events_total", f.labelMatchers()
	if kind == KindTempfile {
		name, metric, matchers = "tempfile_observation", "pgwatch_db_stats_temp_bytes", f.baseMatchers()
	}
	matrix, err := c.queryRange(ctx, start, end, func(s, e int64) string {
		return maxTimestampSQL(name, metric, matchers, s, e)
	})
	if err != nil {
		return nil
	}
	var latest float64
	count := 0
	for _, s := range matrix {
		for _, v := range s.values {
			if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 || v < float64(start.Unix()) || v > float64(end.Unix()) {
				continue
			}
			count++
			if v > latest {
				latest = v
			}
		}
	}
	if count == 0 {
		return nil
	}
	return &Observation{Source: "prometheus_sample_timestamp", ObservedAt: time.Unix(int64(latest), 0).UTC().Format(time.RFC3339), SampleCount: count}
}

// maxTimestampSQL renders max(timestamp(metric{matchers})) over the steps of
// [startMs, endMs] in the shape querySeries reads: one row, its values in `vals`.
func maxTimestampSQL(name, metric string, matchers []matcher, startMs, endMs int64) string {
	s, e := alignRange(startMs, endMs)
	return "-- instance-jobs: " + name + "\n" +
		"WITH a AS (\n  SELECT t, max(value) AS v\n  FROM prom_timestamp(" + sqlString(metric) + ", " +
		matchersSQL(matchers) + ", " + itoa(s) + ", " + itoa(e) + ", " + itoa(stepMs) + ")\n  GROUP BY t\n)\n" +
		"SELECT list(v ORDER BY t) AS vals\nFROM a\nHAVING count(*) > 0"
}
