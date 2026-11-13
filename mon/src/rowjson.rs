//! Result rows to JSON in Rust, from the Arrow batches DuckDB streams.
//!
//! The query is prepared once and its Arrow output is written here, in the same text DuckDB's
//! `to_json` produces. Calling `to_json` in SQL would need a `DESCRIBE <query>` first, which binds
//! every table function twice per query. The output:
//! - VARCHAR: a JSON string; BOOLEAN; integers; DOUBLE and FLOAT as the shortest text that reads
//!   back to the same value (NaN and ±Inf as the strings "NaN", "+Inf", "-Inf", the Prometheus
//!   API convention);
//! - LIST and ARRAY as arrays, STRUCT as objects, MAP as objects (keys as strings);
//! - DATE, TIME, TIMESTAMP[_S|_MS|_NS] and TIMESTAMP WITH TIME ZONE as DuckDB's text form
//!   (`2026-11-09 10:00:00.5`, `+00` for UTC), DECIMAL as a number;
//! - anything else (INTERVAL, BLOB, UUID, ...) as a JSON string of Arrow's text form, best
//!   effort (`fallback` counts these).
use arrow::array::*;
use arrow::datatypes::{DataType, TimeUnit};
use duckdb::core::{LogicalTypeHandle, LogicalTypeId};
use std::fmt::Write;

/// DuckDB's name for a logical type, as DESCRIBE prints it.
pub fn type_name(t: &LogicalTypeHandle) -> String {
    use LogicalTypeId as L;
    match t.id() {
        L::Boolean => "BOOLEAN".into(),
        L::Tinyint => "TINYINT".into(),
        L::Smallint => "SMALLINT".into(),
        L::Integer => "INTEGER".into(),
        L::Bigint => "BIGINT".into(),
        L::UTinyint => "UTINYINT".into(),
        L::USmallint => "USMALLINT".into(),
        L::UInteger => "UINTEGER".into(),
        L::UBigint => "UBIGINT".into(),
        L::Float => "FLOAT".into(),
        L::Double => "DOUBLE".into(),
        L::Timestamp => "TIMESTAMP".into(),
        L::TimestampS => "TIMESTAMP_S".into(),
        L::TimestampMs => "TIMESTAMP_MS".into(),
        L::TimestampNs => "TIMESTAMP_NS".into(),
        L::TimestampTZ => "TIMESTAMP WITH TIME ZONE".into(),
        L::Date => "DATE".into(),
        L::Time => "TIME".into(),
        L::Interval => "INTERVAL".into(),
        L::Hugeint => "HUGEINT".into(),
        L::UHugeint => "UHUGEINT".into(),
        L::Varchar => "VARCHAR".into(),
        L::Blob => "BLOB".into(),
        L::Uuid => "UUID".into(),
        L::Decimal => format!("DECIMAL({},{})", t.decimal_width(), t.decimal_scale()),
        L::List => format!("{}[]", type_name(&t.child(0))),
        L::Map => format!("MAP({}, {})", type_name(&t.child(0)), type_name(&t.child(1))),
        L::Struct => {
            let f: Vec<String> = (0..t.num_children())
                .map(|i| format!("{} {}", ident(&t.child_name(i)), type_name(&t.child(i))))
                .collect();
            format!("STRUCT({})", f.join(", "))
        }
        other => format!("{other:?}").to_uppercase(),
    }
}

/// A struct field name as DuckDB prints it: bare when it is a plain identifier (keywords are
/// not checked: DuckDB quotes those, this does not).
fn ident(s: &str) -> String {
    let plain = !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain {
        s.to_string()
    } else {
        format!("\"{}\"", s.replace('"', "\"\""))
    }
}

/// Count of values written through the best-effort fallback (diagnostics).
pub static FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn push_f64(out: &mut String, x: f64) {
    if x.is_nan() {
        out.push_str("\"NaN\"");
    } else if x == f64::INFINITY {
        out.push_str("\"+Inf\"");
    } else if x == f64::NEG_INFINITY {
        out.push_str("\"-Inf\"");
    } else {
        // shortest round-trip text (1.0, 0.1, 1e20); same value as DuckDB's to_json
        let mut b = ryu_like(x);
        if !b.contains(['.', 'e', 'E']) {
            b.push_str(".0");
        }
        out.push_str(&b);
    }
}

/// Shortest text that reads back as `x` (serde_json's float writer: valid JSON, e.g. 1e20).
fn ryu_like(x: f64) -> String {
    serde_json::to_string(&x).unwrap_or_default()
}

fn push_str(out: &mut String, s: &str) {
    // serde_json escaping (control chars, quote, backslash); always valid JSON
    let _ = write!(out, "{}", serde_json::Value::String(s.to_string()));
}

const DAYS_0_TO_1970: i64 = 719_468;

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + DAYS_0_TO_1970;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn push_date(out: &mut String, days: i64) {
    let (y, m, d) = civil(days);
    if y < 0 {
        // DuckDB writes BC years as "0001-01-01 (BC)"; rare enough to leave to the fallback text
        let _ = write!(out, "{:04}-{:02}-{:02} (BC)", 1 - y, m, d);
    } else {
        let _ = write!(out, "{y:04}-{m:02}-{d:02}");
    }
}

/// HH:MM:SS[.fraction], DuckDB style: micro (or nano) digits with trailing zeros dropped.
fn push_time(out: &mut String, sub_day: i64, per_sec: i64) {
    let secs = sub_day / per_sec;
    let frac = sub_day % per_sec;
    let _ = write!(out, "{:02}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60);
    if frac != 0 {
        let digits = if per_sec == 1_000_000_000 { 9 } else { 6 };
        let f = if per_sec == 1_000_000_000 {
            frac
        } else {
            frac * (1_000_000 / per_sec)
        };
        let s = format!("{f:0digits$}");
        out.push('.');
        out.push_str(s.trim_end_matches('0'));
    }
}

fn push_ts(out: &mut String, v: i64, unit: &TimeUnit, tz: bool) {
    let per_sec = match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    };
    let per_day = per_sec * 86_400;
    let days = v.div_euclid(per_day);
    let rem = v.rem_euclid(per_day);
    out.push('"');
    push_date(out, days);
    out.push(' ');
    push_time(out, rem, per_sec);
    if tz {
        out.push_str("+00");
    }
    out.push('"');
}

fn push_decimal(out: &mut String, v: i128, scale: i8) {
    if scale <= 0 {
        let _ = write!(out, "{v}");
        return;
    }
    let s = scale as u32;
    let p = 10i128.pow(s);
    let sign = if v < 0 { "-" } else { "" };
    let a = v.unsigned_abs();
    let _ = write!(out, "{sign}{}.{:0w$}", a / p as u128, a % p as u128, w = s as usize);
}

/// Append the JSON text of `a[i]`.
pub fn push_value(out: &mut String, a: &dyn Array, i: usize) {
    if a.is_null(i) {
        out.push_str("null");
        return;
    }
    macro_rules! int {
        ($t:ty) => {{
            let _ = write!(out, "{}", a.as_any().downcast_ref::<$t>().unwrap().value(i));
        }};
    }
    match a.data_type() {
        DataType::Null => out.push_str("null"),
        DataType::Utf8 => push_str(out, a.as_any().downcast_ref::<StringArray>().unwrap().value(i)),
        DataType::LargeUtf8 => push_str(out, a.as_any().downcast_ref::<LargeStringArray>().unwrap().value(i)),
        DataType::Utf8View => push_str(out, a.as_any().downcast_ref::<StringViewArray>().unwrap().value(i)),
        DataType::Boolean => out.push_str(if a.as_any().downcast_ref::<BooleanArray>().unwrap().value(i) {
            "true"
        } else {
            "false"
        }),
        DataType::Int8 => int!(Int8Array),
        DataType::Int16 => int!(Int16Array),
        DataType::Int32 => int!(Int32Array),
        DataType::Int64 => int!(Int64Array),
        DataType::UInt8 => int!(UInt8Array),
        DataType::UInt16 => int!(UInt16Array),
        DataType::UInt32 => int!(UInt32Array),
        DataType::UInt64 => int!(UInt64Array),
        DataType::Float64 => push_f64(out, a.as_any().downcast_ref::<Float64Array>().unwrap().value(i)),
        // DuckDB widens FLOAT to double text through its shortest float form
        // FLOAT: DuckDB's to_json widens to double (0.1 -> 0.10000000149011612)
        DataType::Float32 => push_f64(out, a.as_any().downcast_ref::<Float32Array>().unwrap().value(i) as f64),
        DataType::Decimal128(_, s) => {
            push_decimal(out, a.as_any().downcast_ref::<Decimal128Array>().unwrap().value(i), *s)
        }
        DataType::Date32 => {
            // DuckDB's infinite dates are +/- i32::MAX
            match a.as_any().downcast_ref::<Date32Array>().unwrap().value(i) {
                i32::MAX => out.push_str("\"infinity\""),
                v if v == -i32::MAX => out.push_str("\"-infinity\""),
                v => {
                    out.push('"');
                    push_date(out, v as i64);
                    out.push('"');
                }
            }
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            out.push('"');
            push_time(
                out,
                a.as_any().downcast_ref::<Time64MicrosecondArray>().unwrap().value(i),
                1_000_000,
            );
            out.push('"');
        }
        DataType::Timestamp(u, tz) => {
            let v = match u {
                TimeUnit::Second => a.as_any().downcast_ref::<TimestampSecondArray>().unwrap().value(i),
                TimeUnit::Millisecond => a.as_any().downcast_ref::<TimestampMillisecondArray>().unwrap().value(i),
                TimeUnit::Microsecond => a.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap().value(i),
                TimeUnit::Nanosecond => a.as_any().downcast_ref::<TimestampNanosecondArray>().unwrap().value(i),
            };
            // DuckDB's infinite timestamps are +/- i64::MAX
            if v == i64::MAX {
                out.push_str("\"infinity\"");
            } else if v == -i64::MAX || v == i64::MIN {
                out.push_str("\"-infinity\"");
            } else {
                push_ts(out, v, u, tz.is_some());
            }
        }
        DataType::List(_) => list(out, a.as_any().downcast_ref::<ListArray>().unwrap().value(i).as_ref()),
        DataType::LargeList(_) => list(
            out,
            a.as_any().downcast_ref::<LargeListArray>().unwrap().value(i).as_ref(),
        ),
        DataType::FixedSizeList(_, _) => list(
            out,
            a.as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .value(i)
                .as_ref(),
        ),
        DataType::Struct(fields) => {
            let s = a.as_any().downcast_ref::<StructArray>().unwrap();
            out.push('{');
            for (j, f) in fields.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                push_str(out, f.name());
                out.push(':');
                push_value(out, s.column(j).as_ref(), i);
            }
            out.push('}');
        }
        DataType::Map(_, _) => {
            let m = a.as_any().downcast_ref::<MapArray>().unwrap();
            let e = m.value(i);
            let (k, v) = (e.column(0), e.column(1));
            out.push('{');
            for j in 0..e.len() {
                if j > 0 {
                    out.push(',');
                }
                // keys are written as strings, whatever their type
                let mut ks = String::new();
                push_value(&mut ks, k.as_ref(), j);
                if ks.starts_with('"') {
                    out.push_str(&ks);
                } else {
                    push_str(out, &ks);
                }
                out.push(':');
                push_value(out, v.as_ref(), j);
            }
            out.push('}');
        }
        _ => {
            FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let s = arrow::util::display::array_value_to_string(a, i).unwrap_or_default();
            push_str(out, &s);
        }
    }
}

fn list(out: &mut String, a: &dyn Array) {
    out.push('[');
    for j in 0..a.len() {
        if j > 0 {
            out.push(',');
        }
        push_value(out, a, j);
    }
    out.push(']');
}
