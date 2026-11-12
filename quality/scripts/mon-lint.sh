#!/usr/bin/env bash
# rustfmt + clippy gate for the pgai-mon crate (mon/), against a recorded baseline.
#
#   quality/scripts/mon-lint.sh          check: fail on any NEW finding
#   quality/scripts/mon-lint.sh record   rewrite the baseline from the current tree
#
# The crate came in with unformatted files and clippy warnings. The baseline
# (quality/mon-lint-baseline.txt) lists them: a file rustfmt would change, and
# the number of compiler/clippy warnings per file. The check fails when a file
# not in the baseline needs formatting, or when a file has more warnings than
# recorded; fewer is reported so the baseline can be tightened. Once the crate
# is clean, delete the baseline and this becomes `cargo fmt --check` plus
# `cargo clippy -- -D warnings`.
set -Eeuo pipefail

mode="${1:-check}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
baseline="${root}/quality/mon-lint-baseline.txt"
cd "${root}/mon"

current="$(mktemp)"
clippy_out="$(mktemp)"
trap 'rm -f "${current}" "${clippy_out}"' EXIT

# rustfmt prints absolute paths of the files it would change; keep them crate-relative.
cargo fmt --check -- -l 2>/dev/null | sed "s|^${PWD}/||" | sort -u | sed 's/^/fmt /' >> "${current}" || true

# Short format: "src/x.rs:LINE:COL: warning: ...". The same warning can be reported for the
# lib and its test build: count each location once.
if ! cargo clippy --all-targets --locked --message-format=short > "${clippy_out}" 2>&1; then
  cat "${clippy_out}"
  echo "mon-lint: cargo clippy failed (compile error, not a lint)" >&2
  exit 1
fi
grep -E '^[^ ]+\.rs:[0-9]+:[0-9]+: warning' "${clippy_out}" | sort -u \
  | awk -F: '{n[$1]++} END {for (f in n) print "clippy " f " " n[f]}' | sort >> "${current}" || true

if [[ "${mode}" == "record" ]]; then
  {
    echo "# pgai-mon lint baseline: quality/scripts/mon-lint.sh record"
    echo "# fmt <file>             rustfmt would change this file"
    echo "# clippy <file> <count>  warnings in this file"
    cat "${current}"
  } > "${baseline}"
  echo "mon-lint: recorded $(grep -c . "${current}" || true) entries in ${baseline#"${root}"/}"
  exit 0
fi

[[ -f "${baseline}" ]] || { echo "mon-lint: no baseline at ${baseline}" >&2; exit 1; }

fail=0
while read -r kind file count; do
  case "${kind}" in
    fmt)
      if ! grep -qxF "fmt ${file}" "${baseline}"; then
        echo "NEW  rustfmt: ${file} is not formatted (run: cd mon && cargo fmt)"
        fail=1
      fi
      ;;
    clippy)
      base="$(awk -v f="${file}" '$1 == "clippy" && $2 == f {print $3}' "${baseline}")"
      if (( count > ${base:-0} )); then
        echo "NEW  clippy: ${file} has ${count} warnings, baseline ${base:-0}"
        fail=1
      fi
      ;;
  esac
done < "${current}"

# Tightening hints: findings in the baseline that are gone.
while read -r kind file count; do
  case "${kind}" in
    fmt)
      if ! grep -qxF "fmt ${file}" "${current}"; then
        echo "GONE rustfmt: ${file} is formatted now"
      fi
      ;;
    clippy)
      now="$(awk -v f="${file}" '$1 == "clippy" && $2 == f {print $3}' "${current}")"
      if (( ${now:-0} < count )); then
        echo "GONE clippy: ${file} ${count} -> ${now:-0} warnings"
      fi
      ;;
  esac
done < <(grep -v '^#' "${baseline}")

if (( fail )); then
  echo ""
  grep -E '^[^ ]+\.rs:[0-9]+:[0-9]+: warning' "${clippy_out}" | sort -u || true
  echo "mon-lint: new findings (above). Fix them; record a new baseline only for findings you did not add." >&2
  exit 1
fi
echo "mon-lint: no new rustfmt or clippy findings"
