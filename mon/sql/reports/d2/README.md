# Dashboard_1_Node_performance_overview.json (census "D2") in SQL

Every PromQL target maps to one file here plus parameters (see mon/grafana/cases.py, ids `D2/...`).
Common parameters: cluster, node, db_re (the `$db_name` regex), start_ms, end_ms, step_ms,
interval_ms (`$__interval`), rate_interval_ms (`$__rate_interval`).

| Panel / ref | File | Params |
|---|---|---|
| 38 A/B/C Active session history | ash.sql | variant a/b/c |
| 4 A, 39 A Sessions | sum_by_state_gt0.sql | non_idle |
| 4 D, 39 D, 33 A, 34 A, 35 A, 50 A/B, host raw panels (55 F/G/H, 56 B-E, 62-67) | raw_selector.sql | metric, matchers |
| 7 A/B, 6 A, 10 A/B, 25 A | sum_irate.sql | metric, divisor |
| 48 A/B latency | latency_ratio.sql | metric |
| 12 A/B, 47 A-C, 14 A/B, 49 A/B, 29 A | irate_series.sql | metric |
| 14 C, 46 A, 51 A, 52 A | ../last_over_time.sql | metric, matchers, window_ms |
| 32 A Locks by mode | locks_rate.sql | |
| 34 B/C, 35 B/C | max_last_over_time.sql | metric, gt0 |
| 34 D, 35 D, 46 B | constant.sql | value |
| 41 A/B/C bgwriter x block_size | irate_times_setting.sql | metric, matchers |
| 42 A Vacuum timeline | vacuum_timeline.sql | |
| 44 A-E logical size | sum_series.sql | metric |
| 54 A-D, 55 B-D, 55 E, 54 F host and/unless | host_and_unless.sql | metric, others, op |
| 54 E, 54 G CPU idle/other | host_cpu.sql | variant |
| 55 A Memory used | host_mem_used.sql | |
| 56 A Storage used | host_disk_used.sql | |
