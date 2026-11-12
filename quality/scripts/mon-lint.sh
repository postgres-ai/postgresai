#!/usr/bin/env bash
# rustfmt + clippy gate for the pgai-mon crate (mon/): no formatting diff, no warnings.
# The three style lints the crate allows are in mon/Cargo.toml ([lints.clippy]).
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}/mon"

cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
echo "mon-lint: clean"
