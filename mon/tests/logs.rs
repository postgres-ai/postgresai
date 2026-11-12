//! auto_explain / jsonlog parsing on real lines captured from a PostgreSQL jsonlog.
use pgai_mon::logs::*;

const FIXTURE: &str = include_str!("fixtures/pglog_sample.json");

#[test]
fn parses_real_auto_explain_lines() {
    let mut plans = 0;
    for line in FIXTURE.lines() {
        let p = parse_line(line, None).expect("every fixture line parses");
        assert!(p.log.ts > 1_700_000_000_000);
        if let Some((ev, def)) = p.plan {
            plans += 1;
            assert_eq!(p.log.plan_hash.as_deref(), Some(ev.plan_hash.as_str()));
            assert!(
                ev.queryid.is_some(),
                "PG17 auto_explain verbose prints Query Identifier"
            );
            assert!(ev.total_ms > 0.0);
            assert!(!ev.top_node.is_empty());
            // query text and parameters are not stored; the message no longer has the plan
            let v: serde_json::Value = serde_json::from_str(&def.plan).unwrap();
            assert!(v.get("Query Text").is_none());
            assert!(v.get("Plan").is_some());
            assert!(!p.log.message.contains('{'));
            assert!(!p.log.raw.contains("Query Text"));
        }
    }
    assert!(plans >= 3, "fixture has several plans");
}

#[test]
fn plan_hash_ignores_literals_and_costs_but_not_shape() {
    let line = FIXTURE.lines().find(|l| l.contains("ModifyTable")).unwrap();
    let a = parse_line(line, None).unwrap().plan.unwrap().0;
    // same plan, other literal values, costs and timings
    let b_line = line
        .replace("53000", "777")
        .replace("1987.50", "12.00")
        .replace("243.176", "999.9");
    let b = parse_line(&b_line, None).unwrap().plan.unwrap().0;
    assert_eq!(a.plan_hash, b.plan_hash);
    assert_eq!(b.total_ms, 999.9);
    // different shape
    let c_line = line.replace("Function Scan", "Seq Scan");
    let c = parse_line(&c_line, None).unwrap().plan.unwrap().0;
    assert_ne!(a.plan_hash, c.plan_hash);
}

#[test]
fn plan_literals_are_redacted() {
    let line = FIXTURE
        .lines()
        .find(|l| l.contains("generate_series(1, 53000)"))
        .unwrap();
    let def = parse_line(line, None).unwrap().plan.unwrap().1;
    // expression strings lose their literals; numeric plan fields (rows, buffers) stay
    assert!(!def.plan.contains("generate_series(1, 53000)"), "{}", def.plan);
    assert!(def.plan.contains("generate_series(?, ?)"), "{}", def.plan);
    assert!(def.plan.contains("\"Function Name\":\"generate_series\"")); // names kept
}

#[test]
fn fingerprint_matches_pgss_normalized_text() {
    let raw = "SELECT * FROM app.t1 WHERE id = 42 AND name = 'x''y' AND k IN (1, 2, 3)";
    let pgss = "select * from app.t1 where id = $1 and name = $2 and k in ($3, $4, $5)";
    assert_eq!(fingerprint(raw), fingerprint(pgss));
    assert_ne!(fingerprint(raw), fingerprint("select * from app.t2 where id = $1"));
}

#[test]
fn pg14_plan_links_to_queryid_by_text() {
    // Strip the Query Identifier, as PG14/15 print no such line.
    let line = FIXTURE.lines().find(|l| l.contains("ModifyTable")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
    let msg = v["message"].as_str().unwrap().to_string();
    let i = msg.find('{').unwrap();
    let mut plan: serde_json::Value = serde_json::from_str(&msg[i..]).unwrap();
    let qt = plan["Query Text"].as_str().unwrap().to_string();
    plan.as_object_mut().unwrap().remove("Query Identifier");
    v["message"] = serde_json::Value::String(format!("{}{}", &msg[..i], plan));
    let mut map = std::collections::HashMap::new();
    map.insert(fingerprint_text(&qt), 4242_i64);
    let p = parse_line(&v.to_string(), Some(&map)).unwrap();
    assert_eq!(p.plan.unwrap().0.queryid, Some(4242));
}
