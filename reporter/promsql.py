"""Compile the reporter's PromQL forms to pgai-mon SQL.

The reporter's checks are written as PromQL strings. pgai-mon serves SQL (DuckDB) with
PromQL-shaped table functions (`prom_last_over_time`, `prom_increase`, ...; mon/src/eval/eval.sql,
mon/sql/macros.sql) that reproduce VictoriaMetrics v1.140 semantics. This module turns one PromQL
expression plus an evaluation grid into one SQL statement over those functions, so the reporter
needs no VictoriaMetrics and the same string can be checked against VictoriaMetrics.

It is closed: it accepts the subset the reporter uses (selectors, the rollups below, aggregations
with `by`, `topk`, scalar and one-to-one vector arithmetic, comparisons, `and`/`or`/`unless` with
`on`, `vector`, `clamp_min`, `timestamp`, `offset`, subqueries) and raises Unsupported for
anything else, never a guess.

Every compiled node is a pair of relations: rows (k UBIGINT, t BIGINT ms, value DOUBLE) and labels
(k, labels MAP, name VARCHAR), one labels row per k. Rows never carry a MAP (that is what keeps the
7-day subqueries small); grouping and matching keys are computed once per series on the labels
relation. k is the series_id for a selector, else a hash of the result's label set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

__all__ = ["Unsupported", "parse", "compile_instant", "compile_range", "duration_ms"]

INSTANT_STEP_MS = 300_000  # VictoriaMetrics' default step for /api/v1/query: also the lookback
SILENCE_MS = 300_000       # maxSilenceInterval in VictoriaMetrics' subquery start


class Unsupported(ValueError):
    """The expression is outside the reporter's PromQL subset."""


# ----------------------------------------------------------------------------------------------
# Lexer
# ----------------------------------------------------------------------------------------------
_UNITS = {"ms": 1, "s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000, "y": 31_536_000_000}
_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<dur>(?:\d+(?:ms|s|m|h|d|w|y))+)(?![\w.])
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<ident>[a-zA-Z_][a-zA-Z0-9_:]*)
  | (?P<str>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
  | (?P<op>=~|!~|==|!=|<=|>=|[-+*/%^<>=(){}\[\],:])
""", re.VERBOSE)


@dataclass
class Tok:
    kind: str
    text: str


def _lex(src: str) -> List[Tok]:
    out, pos = [], 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m:
            raise Unsupported(f"cannot read {src[pos:pos + 20]!r}")
        pos = m.end()
        if m.lastgroup != "ws":
            out.append(Tok(m.lastgroup, m.group()))
    out.append(Tok("eof", ""))
    return out


def duration_ms(text: str) -> int:
    """'3h', '35d', '1h30m', '5m' (and a bare number of seconds) in milliseconds."""
    text = str(text).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return int(round(float(text) * 1000))
    parts = re.findall(r"(\d+)(ms|s|m|h|d|w|y)", text)
    if not parts or "".join(a + b for a, b in parts) != text:
        raise Unsupported(f"bad duration {text!r}")
    return sum(int(a) * _UNITS[b] for a, b in parts)


def _unquote(s: str) -> str:
    body = s[1:-1]
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), body)


# ----------------------------------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------------------------------
@dataclass
class Num:
    value: float


@dataclass
class Sel:
    metric: str
    matchers: List[Tuple[str, str, str]]
    range_ms: Optional[int] = None
    offset_ms: int = 0


@dataclass
class Sub:
    expr: "Node"
    window_ms: int
    step_ms: int
    offset_ms: int = 0


@dataclass
class Call:
    fn: str
    args: List["Node"]


@dataclass
class Agg:
    op: str
    expr: "Node"
    by: List[str] = field(default_factory=list)
    param: Optional["Node"] = None


@dataclass
class Bin:
    op: str
    lhs: "Node"
    rhs: "Node"
    on: Optional[List[str]] = None


Node = Union[Num, Sel, Sub, Call, Agg, Bin]

_AGGS = {"sum", "min", "max", "avg", "count", "group", "topk"}
_ROLLUPS = {"last_over_time", "max_over_time", "min_over_time", "avg_over_time", "sum_over_time",
            "count_over_time", "quantile_over_time", "tlast_over_time", "tmax_over_time", "increase",
            "rate", "resets"}
_FUNCS = _ROLLUPS | {"timestamp", "clamp_min", "vector"}
# precedence and associativity (PromQL): or < and/unless < comparisons < +- < */% < ^
_PREC = {"or": 1, "and": 2, "unless": 2, "==": 3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
         "+": 4, "-": 4, "*": 5, "/": 5, "%": 5, "^": 6}
_CMP = {"==", "!=", "<", ">", "<=", ">="}
_SET = {"and", "or", "unless"}


class _Parser:
    def __init__(self, src: str):
        self.toks = _lex(src)
        self.i = 0

    def peek(self, k: int = 0) -> Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> Tok:
        t = self.next()
        if t.text != text:
            raise Unsupported(f"expected {text!r}, got {t.text!r}")
        return t

    def parse(self) -> Node:
        n = self.expr(0)
        if self.peek().kind != "eof":
            raise Unsupported(f"unexpected {self.peek().text!r}")
        return n

    def binop(self) -> Optional[str]:
        t = self.peek()
        if t.kind == "op" and t.text in _PREC:
            return t.text
        if t.kind == "ident" and t.text in _SET:
            return t.text
        return None

    def expr(self, min_prec: int) -> Node:
        lhs = self.unary()
        while True:
            op = self.binop()
            if op is None or _PREC[op] < min_prec:
                break
            self.next()
            on = None
            if self.peek().text == "bool":
                raise Unsupported("bool modifier")
            if self.peek().text in ("on", "ignoring"):
                if self.next().text == "ignoring":
                    raise Unsupported("ignoring()")
                on = self.labels()
            if self.peek().text in ("group_left", "group_right"):
                raise Unsupported(self.peek().text)
            nxt = _PREC[op] + (0 if op == "^" else 1)
            rhs = self.expr(nxt)
            lhs = Bin(op, lhs, rhs, on)
        return lhs

    def unary(self) -> Node:
        t = self.peek()
        if t.text == "-" and t.kind == "op":
            self.next()
            n = self.unary()
            if isinstance(n, Num):
                return Num(-n.value)
            return Bin("*", Num(-1.0), n)
        if t.text == "+" and t.kind == "op":
            self.next()
            return self.unary()
        return self.postfix(self.primary())

    def postfix(self, n: Node) -> Node:
        while True:
            t = self.peek()
            if t.text == "[":
                self.next()
                w = duration_ms(self.next().text)
                if self.peek().text == ":":
                    self.next()
                    step = duration_ms(self.next().text) if self.peek().text != "]" else None
                    self.expect("]")
                    if step is None:
                        raise Unsupported("subquery without a step")
                    n = Sub(n, w, step)
                else:
                    self.expect("]")
                    if not isinstance(n, Sel) or n.range_ms is not None or n.offset_ms:
                        raise Unsupported("range on a non-selector")
                    n.range_ms = w
            elif t.kind == "ident" and t.text == "offset":
                self.next()
                off = duration_ms(self.next().text)
                if isinstance(n, (Sel, Sub)) and not n.offset_ms:
                    n.offset_ms = off
                else:
                    raise Unsupported("offset here")
            else:
                return n

    def labels(self) -> List[str]:
        self.expect("(")
        out = []
        while self.peek().text != ")":
            t = self.next()
            if t.kind != "ident":
                raise Unsupported(f"label name {t.text!r}")
            out.append(t.text)
            if self.peek().text == ",":
                self.next()
        self.expect(")")
        return out

    def primary(self) -> Node:
        t = self.next()
        if t.kind == "num":
            return Num(float(t.text))
        if t.text == "(":
            n = self.expr(0)
            self.expect(")")
            return n
        if t.text == "{":
            raise Unsupported("selector without a metric name")
        if t.kind != "ident":
            raise Unsupported(f"unexpected {t.text!r}")
        name = t.text
        if name in _AGGS and self.peek().text in ("(", "by", "without"):
            return self.aggregation(name)
        if self.peek().text == "(":
            if name not in _FUNCS:
                raise Unsupported(f"function {name}")
            self.next()
            args = []
            while self.peek().text != ")":
                args.append(self.expr(0))
                if self.peek().text == ",":
                    self.next()
            self.expect(")")
            return Call(name, args)
        matchers = []
        if self.peek().text == "{":
            self.next()
            while self.peek().text != "}":
                k = self.next()
                if k.kind != "ident":
                    raise Unsupported(f"label name {k.text!r}")
                op = self.next()
                if op.text not in ("=", "!=", "=~", "!~"):
                    raise Unsupported(f"matcher op {op.text!r}")
                v = self.next()
                if v.kind != "str":
                    raise Unsupported("matcher value must be a string")
                if k.text == "__name__":
                    raise Unsupported("__name__ matcher")
                matchers.append((k.text, op.text, _unquote(v.text)))
                if self.peek().text == ",":
                    self.next()
            self.expect("}")
        return Sel(name, matchers)

    def aggregation(self, op: str) -> Agg:
        by: Optional[List[str]] = None
        if self.peek().text == "without":
            raise Unsupported("without()")
        if self.peek().text == "by":
            self.next()
            by = self.labels()
        self.expect("(")
        param = None
        if op == "topk":
            param = self.expr(0)
            self.expect(",")
        e = self.expr(0)
        self.expect(")")
        if self.peek().text == "by":
            if by is not None:
                raise Unsupported("by twice")
            self.next()
            by = self.labels()
        elif self.peek().text == "without":
            raise Unsupported("without()")
        if op == "topk" and by:
            raise Unsupported("topk by")
        return Agg(op, e, by or [], param)


def parse(src: str) -> Node:
    return _Parser(src).parse()


# ----------------------------------------------------------------------------------------------
# SQL generation
# ----------------------------------------------------------------------------------------------
def _q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _matchers(ms: List[Tuple[str, str, str]]) -> str:
    if not ms:
        return "CAST([] AS VARCHAR[][])"
    return "[" + ", ".join(f"[{_q(k)}, {_q(op)}, {_q(v)}]" for k, op, v in ms) + "]"


def _keys(keys: List[str]) -> str:
    return "CAST([" + ", ".join(_q(k) for k in keys) + "] AS VARCHAR[])"


def _num(v: float) -> str:
    if v != v:
        return "CAST('nan' AS DOUBLE)"
    if v in (float("inf"), float("-inf")):
        return f"CAST('{'inf' if v > 0 else '-inf'}' AS DOUBLE)"
    return f"CAST({v!r} AS DOUBLE)"


def _align_down(x: int, step: int) -> int:
    return x - x % step


def _align_up(x: int, step: int) -> int:
    return x + step - x % step if x % step else x


@dataclass(frozen=True)
class Grid:
    start: int  # ms
    end: int
    step: int

    def shifted(self, off: int) -> "Grid":
        return Grid(self.start - off, self.end - off, self.step)


# Rollups whose result keeps the metric name on VictoriaMetrics (measured on v1.140.0).
_KEEP_NAME = {"last_over_time", "max_over_time", "min_over_time", "quantile_over_time"}
# Rollups pgai-mon evaluates natively (eval.sql); the rest are SQL here.
_NATIVE = {"last_over_time", "max_over_time", "min_over_time", "avg_over_time", "sum_over_time",
           "count_over_time", "tlast_over_time", "increase", "rate"}
_OVER = {"last_over_time": None, "max_over_time": "max(v)", "min_over_time": "min(v)",
         "avg_over_time": "avg(v)", "sum_over_time": "sum(v)", "count_over_time": "CAST(count(*) AS DOUBLE)"}
_AGG_SQL = {"sum": "sum(r.value)", "min": "min(r.value)", "max": "max(r.value)", "avg": "avg(r.value)",
            "count": "CAST(count(*) AS DOUBLE)", "group": "CAST(1 AS DOUBLE)"}
_LKEY = "hash(list_sort(map_entries(labels)))"


class _Compiler:
    def __init__(self):
        self.ctes: List[str] = []
        self.n = 0

    def cte(self, sql: str) -> str:
        self.n += 1
        name = f"c{self.n}"
        self.ctes.append(f"{name} AS MATERIALIZED (\n{sql}\n)")
        return name

    # -- scalars ---------------------------------------------------------------------------
    def scalar(self, n: Node) -> Optional[float]:
        """The value of a constant expression, or None if n is a vector."""
        if isinstance(n, Num):
            return n.value
        if isinstance(n, Bin) and n.op in ("+", "-", "*", "/") and n.on is None:
            a, b = self.scalar(n.lhs), self.scalar(n.rhs)
            if a is not None and b is not None:
                if n.op == "+":
                    return a + b
                if n.op == "-":
                    return a - b
                if n.op == "*":
                    return a * b
                return a / b if b else (float("nan") if a == 0 or a != a else float("inf") * (1 if a > 0 else -1))
        return None

    # -- vectors: return (rows, labels) CTE names --------------------------------------------
    def vec(self, n: Node, g: Grid) -> Tuple[str, str]:
        if isinstance(n, Sel):
            if n.range_ms is not None:
                raise Unsupported("a range vector outside a rollup function")
            return self.rollup("last", n, g)
        if isinstance(n, Call):
            return self.call(n, g)
        if isinstance(n, Agg):
            return self.agg(n, g)
        if isinstance(n, Bin):
            return self.binary(n, g)
        if isinstance(n, Sub):
            raise Unsupported("a subquery outside a rollup function")
        raise Unsupported(f"scalar where a vector is needed: {n}")

    def labels_of(self, rows: str, metric: str, matchers, keep_name: bool) -> str:
        name = _q(metric) if keep_name else "CAST(NULL AS VARCHAR)"
        return self.cte(
            f"SELECT series_id AS k, labels, {name} AS name FROM series\n"
            f"WHERE metric = {_q(metric)} AND prom_match(labels, {_matchers(matchers)})\n"
            f"  AND series_id IN (SELECT k FROM {rows})")

    def rollup(self, fn: str, s: Sel, g: Grid, phi: Optional[float] = None) -> Tuple[str, str]:
        """fn(s[range]) on grid g; fn 'last' = a bare selector, 'timestamp' = timestamp(selector)."""
        off = s.offset_ms
        gg = g.shifted(off)
        m, ms = _q(s.metric), _matchers(s.matchers)
        w = s.range_ms or 0
        args = f"{m}, {ms}, {gg.start}, {gg.end}, {gg.step}"
        if fn in ("last", "timestamp"):
            src = f"SELECT series_id AS k, t + {off} AS t, value FROM prom_{fn}({args})"
            keep = fn == "last"
        elif fn == "quantile_over_time":
            src = f"SELECT series_id AS k, t + {off} AS t, value FROM prom_quantile_over_time({_num(phi)}, {args}, {w})"
            keep = True
        elif fn in _NATIVE:
            src = f"SELECT series_id AS k, t + {off} AS t, value FROM prom_{fn}({args}, {w})"
            keep = fn in _KEEP_NAME
        elif fn == "tmax_over_time":
            # VictoriaMetrics returns the last timestamp of the maximum (seconds)
            src = (f"SELECT series_id AS k, t + {off} AS t, arg_max(ts, (v, ts)) / 1e3 AS value\n"
                   f"FROM prom_window({args}, {w}) GROUP BY series_id, t")
            keep = False
        elif fn == "resets":
            # rollupResets: drops of the raw value inside (t - w, t]; the first in-window sample is
            # compared with the previous one only when that is a usable prev (within maxPrev).
            fetch = gg.start - SILENCE_MS - max(w, gg.step)
            src = (
                f"WITH e AS (SELECT series_id, t, n, has_prev, first_v, prev_v\n"
                f"           FROM prom_edges({args}, {w}, true, false)),\n"
                f"     d AS (SELECT series_id, ts_ms FROM (\n"
                f"             SELECT series_id, ts_ms, v < lag(v) OVER (PARTITION BY series_id ORDER BY ts_ms) AS dropped\n"
                f"             FROM prom_select({m}, {ms}, {fetch}, {gg.end})) WHERE dropped),\n"
                f"     c AS (SELECT e.series_id, e.t, count(d.ts_ms) AS c FROM e\n"
                f"           LEFT JOIN d ON d.series_id = e.series_id AND d.ts_ms > e.t - {w} AND d.ts_ms <= e.t\n"
                f"           GROUP BY e.series_id, e.t)\n"
                f"SELECT series_id AS k, t + {off} AS t, value FROM (\n"
                f"  SELECT e.series_id, e.t,\n"
                f"         CASE WHEN e.n = 0 THEN CASE WHEN e.has_prev THEN 0.0 END\n"
                f"              ELSE CAST(c.c - CASE WHEN NOT e.has_prev AND e.first_v < e.prev_v THEN 1 ELSE 0 END AS DOUBLE)\n"
                f"         END AS value\n"
                f"  FROM e JOIN c USING (series_id, t)) WHERE value IS NOT NULL")
            keep = False
        else:
            raise Unsupported(f"rollup {fn}")
        rows = self.cte(f"SELECT k, t, value FROM ({src}) WHERE NOT isnan(value)")
        return rows, self.labels_of(rows, s.metric, s.matchers, keep)

    def call(self, c: Call, g: Grid) -> Tuple[str, str]:
        fn, args = c.fn, c.args
        if fn == "vector":
            v = self.scalar(args[0]) if len(args) == 1 else None
            if v is None:
                raise Unsupported("vector() of a non-constant")
            rows = self.cte(f"SELECT hash(MAP {{}}::MAP(VARCHAR, VARCHAR)) AS k, t, {_num(v)} AS value\n"
                            f"FROM (SELECT unnest(range({g.start}, {g.end} + 1, {g.step})) AS t)")
            labels = self.cte("SELECT hash(MAP {}::MAP(VARCHAR, VARCHAR)) AS k, MAP {}::MAP(VARCHAR, VARCHAR) AS labels, "
                              "CAST(NULL AS VARCHAR) AS name")
            return rows, labels
        if fn == "clamp_min":
            lo = self.scalar(args[1]) if len(args) == 2 else None
            if lo is None:
                raise Unsupported("clamp_min with a non-constant bound")
            r, l = self.vec(args[0], g)
            return self.cte(f"SELECT k, t, greatest(value, {_num(lo)}) AS value FROM {r}"), l
        if fn == "timestamp":
            if len(args) != 1:
                raise Unsupported("timestamp() arity")
            a = args[0]
            if isinstance(a, Sel) and a.range_ms is None:
                return self.rollup("timestamp", a, g)
            r, l = self.vec(a, g)   # of an expression: the evaluation time of each point
            return (self.cte(f"SELECT k, t, t / 1e3 AS value FROM {r}"),
                    self.cte(f"SELECT k, labels, CAST(NULL AS VARCHAR) AS name FROM {l}"))
        if fn in _ROLLUPS:
            phi = None
            if fn == "quantile_over_time":
                if len(args) != 2:
                    raise Unsupported("quantile_over_time arity")
                phi = self.scalar(args[0])
                if phi is None:
                    raise Unsupported("quantile_over_time with a non-constant phi")
                args = args[1:]
            if len(args) != 1:
                raise Unsupported(f"{fn} arity")
            a = args[0]
            if isinstance(a, Sel) and a.range_ms is not None:
                return self.rollup(fn, a, g, phi)
            if isinstance(a, Sub):
                return self.subquery(fn, a, g, phi)
            raise Unsupported(f"{fn} of {type(a).__name__}")
        raise Unsupported(f"function {fn}")

    def subquery(self, fn: str, s: Sub, g: Grid, phi: Optional[float]) -> Tuple[str, str]:
        """fn((expr)[W:S]): expr as a range query on VictoriaMetrics' aligned subquery grid, then
        fn over the inner points in (t - W, t] for every outer t."""
        if fn not in _OVER and fn not in ("quantile_over_time",):
            raise Unsupported(f"{fn} over a subquery")
        gg = g.shifted(s.offset_ms)
        w, st = s.window_ms, s.step_ms
        inner = Grid(_align_down(gg.start - w - st - SILENCE_MS, st), _align_up(gg.end + st, st), st)
        r, l = self.vec(s.expr, inner)
        if fn == "last_over_time":
            agg = "arg_max(v, ts)"
        elif fn == "quantile_over_time":
            agg = f"quantile_cont(v, {_num(phi)})"
        else:
            agg = _OVER[fn]
        rows = self.cte(
            f"SELECT k, t + {s.offset_ms} AS t, {agg} AS value FROM (\n"
            f"  SELECT k, t AS ts, value AS v, unnest(prom_points(t, {gg.start}, {gg.step}, {w}, {gg.end})) AS t FROM {r})\n"
            f"GROUP BY k, t")
        if fn in _KEEP_NAME:
            return rows, l
        return rows, self.cte(f"SELECT k, labels, CAST(NULL AS VARCHAR) AS name FROM {l}")

    def agg(self, a: Agg, g: Grid) -> Tuple[str, str]:
        r, l = self.vec(a.expr, g)
        if a.op == "topk":
            k = self.scalar(a.param)
            if k is None:
                raise Unsupported("topk with a non-constant k")
            rows = self.cte(
                f"SELECT r.k, r.t, r.value FROM {r} r JOIN {l} l USING (k)\n"
                f"QUALIFY row_number() OVER (PARTITION BY r.t ORDER BY r.value DESC, coalesce(l.name, ''),\n"
                f"                           list_sort(map_entries(l.labels))) <= {int(k)}")
            return rows, l
        by = _keys(a.by)
        m = self.cte(f"SELECT k, hash(prom_by(labels, {by})) AS g, prom_by(labels, {by}) AS gl FROM {l}")
        rows = self.cte(f"SELECT m.g AS k, r.t, {_AGG_SQL[a.op]} AS value\n"
                        f"FROM {r} r JOIN {m} m USING (k) GROUP BY m.g, r.t")
        labels = self.cte(f"SELECT DISTINCT ON (g) g AS k, gl AS labels, CAST(NULL AS VARCHAR) AS name FROM {m}")
        return rows, labels

    def binary(self, b: Bin, g: Grid) -> Tuple[str, str]:
        op = b.op
        ls, rs = self.scalar(b.lhs), self.scalar(b.rhs)
        if op in _SET:
            if ls is not None or rs is not None:
                raise Unsupported(f"{op} with a scalar")
            return self.set_op(b, g)
        if b.on is not None:
            raise Unsupported(f"on() with {op}")
        if op in ("%", "^"):
            raise Unsupported(f"operator {op}")
        if ls is not None and rs is not None:
            raise Unsupported("a constant expression where a vector is needed")
        if op in _CMP:
            if ls is not None:
                raise Unsupported("scalar on the left of a comparison")
            if rs is None:
                raise Unsupported("vector-vector comparison")
            r, l = self.vec(b.lhs, g)
            sqlop = {"==": "=", "!=": "<>"}.get(op, op)
            return self.cte(f"SELECT k, t, value FROM {r} WHERE value {sqlop} {_num(rs)}"), l
        if ls is not None or rs is not None:
            r, l = self.vec(b.rhs if ls is not None else b.lhs, g)
            expr = f"{_num(ls)} {op} value" if ls is not None else f"value {op} {_num(rs)}"
            rows = self.cte(f"SELECT k, t, value FROM (SELECT k, t, {expr} AS value FROM {r}) WHERE NOT isnan(value)")
            return rows, self.cte(f"SELECT k, labels, CAST(NULL AS VARCHAR) AS name FROM {l}")
        # one-to-one on the full label set (the metric name is not a label)
        ra, la = self.vec(b.lhs, g)
        rb, lb = self.vec(b.rhs, g)
        ma = self.cte(f"SELECT k, {_LKEY} AS mk, labels FROM {la}")
        mb = self.cte(f"SELECT k, {_LKEY} AS mk FROM {lb}")
        rows = self.cte(
            f"SELECT k, t, value FROM (\n"
            f"  SELECT ma.mk AS k, a.t, a.value {op} b.value AS value\n"
            f"  FROM {ra} a JOIN {ma} ma USING (k)\n"
            f"  JOIN (SELECT mb.mk, b.t, b.value FROM {rb} b JOIN {mb} mb USING (k)) b ON b.mk = ma.mk AND b.t = a.t)\n"
            f"WHERE NOT isnan(value)")
        labels = self.cte(f"SELECT DISTINCT ON (mk) mk AS k, labels, CAST(NULL AS VARCHAR) AS name FROM {ma}")
        return rows, labels

    def set_op(self, b: Bin, g: Grid) -> Tuple[str, str]:
        ra, la = self.vec(b.lhs, g)
        rb, lb = self.vec(b.rhs, g)
        key = f"hash(prom_by(labels, {_keys(b.on)}))" if b.on is not None else _LKEY
        ma = self.cte(f"SELECT k, {key} AS mk FROM {la}")
        mb = self.cte(f"SELECT k, {key} AS mk FROM {lb}")
        rbk = self.cte(f"SELECT DISTINCT mb.mk, b.t FROM {rb} b JOIN {mb} mb USING (k)")
        if b.op in ("and", "unless"):
            neg = "NOT " if b.op == "unless" else ""
            rows = self.cte(
                f"SELECT a.k, a.t, a.value FROM {ra} a JOIN {ma} ma USING (k)\n"
                f"WHERE {neg}EXISTS (SELECT 1 FROM {rbk} x WHERE x.mk = ma.mk AND x.t = a.t)")
            return rows, la
        rak = self.cte(f"SELECT DISTINCT ma.mk, a.t FROM {ra} a JOIN {ma} ma USING (k)")
        rows = self.cte(
            f"SELECT k, t, value FROM {ra}\n"
            f"UNION ALL\n"
            f"SELECT b.k, b.t, b.value FROM {rb} b JOIN {mb} mb USING (k)\n"
            f"WHERE NOT EXISTS (SELECT 1 FROM {rak} x WHERE x.mk = mb.mk AND x.t = b.t)")
        labels = self.cte(f"SELECT DISTINCT ON (k) k, labels, name FROM (SELECT * FROM {la} UNION ALL SELECT * FROM {lb})")
        return rows, labels

    def finish(self, rows: str, labels: str) -> str:
        body = ",\n".join(self.ctes)
        return (f"WITH {body}\n"
                # One row per series, its points as lists: a row per point would repeat the labels
                # and reach pgai-mon's row cap at (series x steps).
                f"SELECT l.name, CAST(to_json(l.labels) AS VARCHAR) AS labels, r.t, r.value\n"
                f"FROM (SELECT k, list(t ORDER BY t) AS t, list(value ORDER BY t) AS value FROM {rows} GROUP BY k) r\n"
                f"JOIN {labels} l USING (k)\n"
                f"ORDER BY r.k")


def _compile(src: str, g: Grid) -> str:
    node = parse(src)
    c = _Compiler()
    if c.scalar(node) is not None:
        raise Unsupported("a scalar query")
    rows, labels = c.vec(node, g)
    return c.finish(rows, labels)


def compile_instant(src: str, at_ms: int, step_ms: int = INSTANT_STEP_MS) -> str:
    """SQL for an instant query at at_ms (VictoriaMetrics /api/v1/query: start = end = time)."""
    return _compile(src, Grid(int(at_ms), int(at_ms), int(step_ms)))


def range_bounds(start_ms: int, end_ms: int, step_ms: int) -> Tuple[int, int]:
    """VictoriaMetrics' AdjustStartEnd for /api/v1/query_range with its cache on (the default):
    with 50 or more points the grid is aligned to the step."""
    if (end_ms - start_ms) // step_ms + 1 < 50:
        return start_ms, end_ms
    s = _align_down(start_ms, step_ms)
    e = _align_up(end_ms, step_ms)
    e -= step_ms * max(0, (e - s) // step_ms - (end_ms - start_ms) // step_ms)
    return s, e


def compile_range(src: str, start_ms: int, end_ms: int, step_ms: int) -> str:
    """SQL for a range query (VictoriaMetrics /api/v1/query_range)."""
    s, e = range_bounds(int(start_ms), int(end_ms), int(step_ms))
    return _compile(src, Grid(s, e, int(step_ms)))
