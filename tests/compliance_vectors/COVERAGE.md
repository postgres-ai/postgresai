# Compliance vector coverage

Tracks which functions have compliance vectors and their verification status.

| Function | Vector File | Python | TypeScript | Notes |
|----------|-------------|--------|------------|-------|
| `_parse_memory_value()` | memory_parsing.json | ✅ | ⬜ | Overflow is TS-only |
| `_analyze_memory_settings()` | memory_analysis.json | ✅ | ⬜ | Tested via golden snapshots |
| `_build_qid_regex()` | query_id_validation.json | ✅ | ⬜ | Security-critical |
| `get_all_nodes()` | - | ✅ | ⬜ | Contract test (shape, not computation) |
| `_densify()` | - | ✅ | ⬜ | Property tests only |

## Review process
- Vector changes require review from Python maintainer AND TS migration lead
- New vectors must include `python_verified` date after tests pass
- `typescript_verified` set when TS harness passes all cases
- Production mismatches → new vector case + snapshot update

## Test summary

| Category | Count | Files |
|----------|-------|-------|
| Compliance Vectors | 44 | test_compliance.py |
| Property Tests | 13 | test_property.py |
| Golden Snapshots | 16 | test_golden_snapshots.py |
| Contract Tests | 4 | test_golden_snapshots.py |
| **Total** | **77** | |

Last updated: 2026-01-23
