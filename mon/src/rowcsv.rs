//! CSV fields from DuckDB's Arrow batches. Timestamps use epoch milliseconds for Grafana.
use arrow::array::*;
use arrow::datatypes::{DataType, TimeUnit};
use std::fmt::Write;

pub fn push_field(out: &mut String, s: &str) {
    if s.contains([',', '"', '\r', '\n']) {
        out.push('"');
        for c in s.chars() {
            if c == '"' {
                out.push('"');
            }
            out.push(c);
        }
        out.push('"');
    } else {
        out.push_str(s);
    }
}

pub fn push_value(out: &mut String, a: &dyn Array, i: usize) {
    if a.is_null(i) {
        return;
    }
    match a.data_type() {
        DataType::Utf8 => push_field(out, a.as_any().downcast_ref::<StringArray>().unwrap().value(i)),
        DataType::LargeUtf8 => push_field(out, a.as_any().downcast_ref::<LargeStringArray>().unwrap().value(i)),
        DataType::Timestamp(unit, _) => {
            let (v, per_ms) = match unit {
                TimeUnit::Second => (
                    a.as_any().downcast_ref::<TimestampSecondArray>().unwrap().value(i),
                    -1000,
                ),
                TimeUnit::Millisecond => (
                    a.as_any().downcast_ref::<TimestampMillisecondArray>().unwrap().value(i),
                    1,
                ),
                TimeUnit::Microsecond => (
                    a.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap().value(i),
                    1000,
                ),
                TimeUnit::Nanosecond => (
                    a.as_any().downcast_ref::<TimestampNanosecondArray>().unwrap().value(i),
                    1_000_000,
                ),
            };
            // DuckDB's infinite timestamps are +/- i64::MAX (rowjson.rs): not a number of milliseconds
            if v == i64::MAX {
                out.push_str("infinity");
            } else if v == -i64::MAX || v == i64::MIN {
                out.push_str("-infinity");
            } else if per_ms < 0 {
                let _ = write!(out, "{}", i128::from(v) * 1000);
            } else {
                let _ = write!(out, "{}", v / per_ms);
            }
        }
        _ => {
            // Reuse numeric and nested-value formatting; remove JSON string escaping before CSV quoting.
            let mut s = String::new();
            crate::rowjson::push_value(&mut s, a, i);
            if s.starts_with('"') {
                s = serde_json::from_str::<String>(&s).unwrap();
            }
            push_field(out, &s);
        }
    }
}
