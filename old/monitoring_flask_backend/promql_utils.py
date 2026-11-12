"""Shared PromQL escaping utilities.

Used by both monitoring_flask_backend.app and reporter.postgres_reports to
avoid duplicate implementations.
"""

import re


def escape_promql_label(value: str) -> str:
    """Escape a value for safe use inside PromQL label matchers (double-quoted strings)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def escape_promql_regex_literal(value: str) -> str:
    """Escape a value so it matches literally inside a PromQL regex label matcher (=~ or !~).

    Escapes both PromQL double-quoted string metacharacters and RE2 regex
    metacharacters, so the value is treated as a literal string rather than a
    regex pattern.  Use this when a user-supplied value is embedded inside a
    hand-crafted regex anchor, e.g.:
        instance=~".*{escape_promql_regex_literal(node_name)}.*"
    """
    value = escape_promql_label(value)
    return re.sub(r'([.+*?|()\[\]{}^$])', r'\\\1', value)
