//! Expand the compiler's window parameters before both cache lookup and execution.
use crate::query::QueryError;

const PREFIX: &str = "-- pgai-window-v1 ";

fn invalid() -> QueryError {
    QueryError {
        kind: "invalid_input",
        message: "invalid pgai-window-v1 header or parameter".into(),
    }
}

/// Plain SQL is unchanged. Only the compiler's leading header opts into parameter expansion.
/// The executed SQL is the cache key, so raw bounds used elsewhere remain part of its identity.
pub fn expand(sql: &str) -> Result<String, QueryError> {
    let Some(rest) = sql.strip_prefix(PREFIX) else {
        return Ok(sql.to_string());
    };
    let (header, body) = rest.split_once('\n').ok_or_else(invalid)?;
    let fields: Vec<_> = header.split_whitespace().collect();
    if fields.len() != 4 || !matches!(fields[3], "range" | "instant") {
        return Err(invalid());
    }
    let number = |s: &str| s.parse::<i64>().map_err(|_| invalid());
    let (from, to, interval) = (number(fields[0])?, number(fields[1])?, number(fields[2])?);
    let span = to.checked_sub(from).filter(|s| *s >= 0).ok_or_else(invalid)?;
    if interval <= 0 {
        return Err(invalid());
    }
    let instant = fields[3] == "instant";
    let day = 86_400_000;
    let bucket = if !instant && span >= 90 * day {
        day
    } else if !instant && span >= 7 * day {
        3_600_000
    } else {
        1
    };
    let step = interval
        .checked_add(bucket - 1)
        .and_then(|v| (v / bucket).checked_mul(bucket))
        .ok_or_else(invalid)?;
    let end = to - to % step;
    let start = if instant { end } else { from - from % step };
    let rate = if bucket > 1 {
        step
    } else {
        interval.checked_add(5000).ok_or_else(invalid)?.max(20_000)
    };
    // Grafana's $__range rounds to whole seconds, independently of the aligned grid.
    let range = span.checked_add(500).map(|v| v / 1000 * 1000).ok_or_else(invalid)?;
    let params = [
        ("start_ms", start),
        ("end_ms", end),
        ("step_ms", if instant { 300_000 } else { step }),
        ("interval_ms", step),
        ("rate_interval_ms", rate),
        ("range_ms", range),
    ];
    substitute(body, &params)
}

fn ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 128
}

/// Skip SQL quotes and comments, including nested block comments and dollar-quoted strings.
/// Reserved parameters inside labels are data, not window values.
fn substitute(sql: &str, params: &[(&str, i64)]) -> Result<String, QueryError> {
    let b = sql.as_bytes();
    let mut out = String::with_capacity(sql.len());
    let (mut i, mut copied) = (0, 0);
    while i < b.len() {
        match b[i] {
            b'\'' | b'"' => {
                let quote = b[i];
                let escaped = quote == b'\'' && i > 0 && matches!(b[i - 1], b'e' | b'E') && (i < 2 || !ident(b[i - 2]));
                i += 1;
                while i < b.len() {
                    if escaped && b[i] == b'\\' {
                        i = (i + 2).min(b.len());
                    } else if b[i] == quote {
                        i += 1;
                        if i < b.len() && b[i] == quote {
                            i += 1;
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            b'-' if b.get(i + 1) == Some(&b'-') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let mut depth = 1;
                i += 2;
                while i < b.len() && depth > 0 {
                    if b[i..].starts_with(b"/*") {
                        depth += 1;
                        i += 2;
                    } else if b[i..].starts_with(b"*/") {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            b'$' if i == 0 || (!ident(b[i - 1]) && b[i - 1] != b'$') => {
                let mut end = i + 1;
                while end < b.len() && ident(b[end]) {
                    end += 1;
                }
                if b.get(end) == Some(&b'$') {
                    let delimiter = &sql[i..=end];
                    i = sql[end + 1..]
                        .find(delimiter)
                        .map_or(b.len(), |p| end + 1 + p + delimiter.len());
                } else if let Some(name) = sql[i..end].strip_prefix("$__pgai_") {
                    let value = params.iter().find(|(key, _)| *key == name).ok_or_else(invalid)?.1;
                    out.push_str(&sql[copied..i]);
                    out.push_str(&format!("CAST({value} AS BIGINT)"));
                    i = end;
                    copied = i;
                } else {
                    i = end;
                }
            }
            _ => i += 1,
        }
    }
    out.push_str(&sql[copied..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_semantics_and_boundaries() {
        let query = |from, to, iv, mode| {
            expand(&format!("{PREFIX}{from} {to} {iv} {mode}\nSELECT $__pgai_start_ms, $__pgai_end_ms, $__pgai_step_ms, $__pgai_interval_ms, $__pgai_rate_interval_ms, $__pgai_range_ms")).unwrap()
        };
        let expected = |values: &[i64]| {
            format!(
                "SELECT {}",
                values
                    .iter()
                    .map(|v| format!("CAST({v} AS BIGINT)"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        assert_eq!(
            query(1001, 61001, 30000, "range"),
            expected(&[0, 60000, 30000, 30000, 35000, 60000])
        );
        assert_eq!(query(1002, 61002, 30000, "range"), query(1001, 61001, 30000, "range"));
        assert_eq!(
            query(1001, 61001, 30000, "instant"),
            expected(&[60000, 60000, 300000, 30000, 35000, 60000])
        );
        for (span, iv, step) in [
            (604799999, 600000, 600000),
            (604800000, 600000, 3600000),
            (2592000000, 7200001, 10800000),
            (7776000000, 7200000, 86400000),
        ] {
            assert_eq!(
                query(0, span, iv, "range"),
                expected(&[
                    0,
                    span - span % step,
                    step,
                    step,
                    if span < 604800000 { iv + 5000 } else { step },
                    (span + 500) / 1000 * 1000
                ])
            );
        }
        assert_eq!(
            query(-61001, -1001, 30000, "range"),
            expected(&[-60000, 0, 30000, 30000, 35000, 60000])
        );
        assert!(query(0, 1499, 1000, "range").ends_with("CAST(1000 AS BIGINT)"));
        assert!(query(0, 1500, 1000, "range").ends_with("CAST(2000 AS BIGINT)"));
    }

    #[test]
    fn preserves_literals_comments_and_plain_sql() {
        let literal = r#"'$__pgai_end_ms' 'a''$__pgai_end_ms' foo$__pgai_end_ms é$__pgai_end_ms "foo$__pgai_end_ms" $$ $__pgai_end_ms $$ $tag$ $__pgai_end_ms $tag$ E'\' $__pgai_end_ms' /* nested /* $__pgai_end_ms */ x */ -- $__pgai_end_ms
"#;
        let sql = format!("SELECT {literal}, $__pgai_end_ms, 61001");
        assert_eq!(expand(&sql).unwrap(), sql);
        assert_eq!(
            expand(&format!("{PREFIX}1001 61001 30000 range\n{sql}")).unwrap(),
            format!("SELECT {literal}, CAST(60000 AS BIGINT), 61001")
        );
    }

    #[test]
    fn invalid_headers_and_unknown_parameters_fail() {
        for header in [
            "0 1 0 range",
            "2 1 1 range",
            "0 1 1 bad",
            "0 1 range",
            "0 1 1 range extra",
            "0 9223372036854775807 1 range",
            "0 1 9223372036854775807 instant",
        ] {
            assert!(expand(&format!("{PREFIX}{header}\nSELECT 1")).is_err(), "{header}");
        }
        assert!(expand(&format!("{PREFIX}0 1 1 range")).is_err());
        assert!(expand(&format!("{PREFIX}0 1 1 range\nSELECT $__pgai_unknown")).is_err());
    }
}
