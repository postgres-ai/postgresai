"""pgai-mon client for the reporter: metrics, query texts and index definitions over POST /sql.

The reporter's checks keep their PromQL strings; `query_instant` / `query_range` here compile them
(reporter/promsql.py) and return the same shapes as VictoriaMetrics' /api/v1/query and
/api/v1/query_range, so the callers' parsing is unchanged. Query texts and index definitions,
which came from the pgwatch Postgres sink, come from pgai-mon's `query_texts` and `objects` views.
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

import requests

from reporter import promsql
from reporter.logger import logger


def format_value(v: float) -> str:
    """A sample value as VictoriaMetrics prints it (Go strconv.FormatFloat(v, 'f', -1, 64)):
    the shortest digits that read back as v, in plain decimal notation."""
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "+Inf"
    if v == float("-inf"):
        return "-Inf"
    s = format(Decimal(repr(float(v))).normalize(), "f")
    return "0" if s == "-0" else s


def _ts(ms: int):
    """Epoch ms as the seconds number VictoriaMetrics returns (an int when whole)."""
    return ms // 1000 if ms % 1000 == 0 else ms / 1000


def _value(v: Any) -> float:
    """A value from pgai-mon's JSON (non-finite doubles come as "+Inf", "-Inf", "NaN") or from
    format_value()."""
    if isinstance(v, str):
        return {"+Inf": float("inf"), "-Inf": float("-inf"), "Inf": float("inf")}.get(v) or float(v)
    return float(v)


def _order(node: promsql.Node) -> str:
    """VictoriaMetrics sorts an instant topk result by value (descending), everything else by labels."""
    return "value" if isinstance(node, promsql.Agg) and node.op == "topk" else "labels"


class MonError(RuntimeError):
    pass


class MonClient:
    """POST /sql on pgai-mon. `clock` returns the evaluation time for instant queries (epoch s)."""

    def __init__(self, url: str, token: Optional[str] = None, timeout: float = 120.0,
                 clock: Optional[Callable[[], float]] = None, session: Optional[requests.Session] = None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.clock = clock or time.time
        self.session = session or requests.Session()
        self.headers = {"content-type": "application/json"}
        if token:
            self.headers["authorization"] = f"Bearer {token}"

    def sql(self, text: str) -> List[Dict[str, Any]]:
        """Rows of one read-only SQL statement as dicts. Raises MonError on any failure."""
        try:
            r = self.session.post(f"{self.url}/sql", json={"sql": text}, headers=self.headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise MonError(f"pgai-mon unreachable: {e}") from e
        try:
            body = r.json()
        except ValueError:
            raise MonError(f"pgai-mon HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code != 200 or "error" in body:
            err = body.get("error")
            raise MonError(f"pgai-mon HTTP {r.status_code}: {err if err is not None else body}")
        cols = body["columns"]
        return [dict(zip(cols, row)) for row in body["rows"]]

    def healthy(self) -> bool:
        try:
            return self.session.get(f"{self.url}/health", headers=self.headers, timeout=10).status_code == 200
        except requests.RequestException:
            return False

    # -- metrics ---------------------------------------------------------------------------
    def _series(self, sql: str):
        series: Dict[tuple, Dict[str, Any]] = {}
        for row in self.sql(sql):
            labels = _labels(row["labels"])
            if row["name"] is not None:
                labels["__name__"] = row["name"]
            key = tuple(sorted(labels.items()))
            series.setdefault(key, {"metric": labels, "points": []})["points"].append((row["t"], _value(row["value"])))
        return series

    def query_instant(self, query: str, at: Optional[float] = None) -> Dict[str, Any]:
        """/api/v1/query: {'status': 'success', 'data': {'resultType': 'vector', 'result': [...]}},
        or {'status': 'error', ...} (never an exception)."""
        at_ms = int(round((self.clock() if at is None else at) * 1000))
        try:
            node = promsql.parse(query)
            series = self._series(promsql.compile_instant(query, at_ms))
        except (promsql.Unsupported, MonError) as e:
            logger.error(f"Query failed: {e} (query: {query})")
            return {"status": "error", "errorType": type(e).__name__, "error": str(e)}
        result = [{"metric": s["metric"], "value": [_ts(at_ms), format_value(s["points"][-1][1])]}
                  for s in series.values()]
        if _order(node) == "value":  # ties as pgai-mon's topk breaks them: by name, then label set
            result.sort(key=lambda r: (-_value(r["value"][1]), r["metric"].get("__name__", ""),
                                       sorted((k, v) for k, v in r["metric"].items() if k != "__name__")))
        else:
            result.sort(key=lambda r: sorted(r["metric"].items()))
        return {"status": "success", "data": {"resultType": "vector", "result": result}}

    def query_range(self, query: str, start: float, end: float, step: str) -> List[Dict[str, Any]]:
        """/api/v1/query_range's data.result: [{'metric': {...}, 'values': [[t, 'v'], ...]}]; [] on error."""
        step_ms = promsql.duration_ms(step)
        try:
            series = self._series(promsql.compile_range(query, int(round(start * 1000)), int(round(end * 1000)), step_ms))
        except (promsql.Unsupported, MonError) as e:
            logger.error(f"Range query failed: {e} (query: {query})")
            return []
        result = [{"metric": s["metric"], "values": [[_ts(t), format_value(v)] for t, v in sorted(s["points"])]}
                  for s in series.values()]
        result.sort(key=lambda r: sorted(r["metric"].items()))
        return result

    # -- what the pgwatch Postgres sink held ---------------------------------------------------
    def query_texts(self, query_text_limit: int = 655360, db_names: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
        """{dbname: {queryid: text}}, the latest text per (database, queryid)."""
        where = ""
        if db_names:
            where = "AND dbname IN (" + ", ".join(promsql._q(d) for d in db_names) + ")"
        rows = self.sql(f"""
            SELECT dbname, CAST(queryid AS VARCHAR) AS queryid, query FROM query_texts
            WHERE queryid IS NOT NULL AND query IS NOT NULL {where}
            QUALIFY row_number() OVER (PARTITION BY dbname, queryid ORDER BY last_seen DESC, query) = 1""")
        out: Dict[str, Dict[str, str]] = {}
        for r in rows:
            text = r["query"] or ""
            if len(text) > query_text_limit:
                text = text[:query_text_limit] + "..."
            out.setdefault(r["dbname"], {})[r["queryid"]] = text
        return out

    def index_definitions(self, db_name: Optional[str] = None) -> Dict[str, str]:
        """{index name: definition} for one database, or {'db.index': definition} for all; the
        current definition (an index dropped since has none)."""
        where = f"AND dbname = {promsql._q(db_name)}" if db_name else ""
        rows = self.sql(f"""
            SELECT dbname, name, definition FROM objects
            WHERE kind = 'index' AND valid_to IS NULL {where}
            ORDER BY dbname, name, schema""")
        out: Dict[str, str] = {}
        for r in rows:
            key = r["name"] if db_name else f"{r['dbname']}.{r['name']}"
            out.setdefault(key, r["definition"])
        return out


def _labels(v: Any) -> Dict[str, str]:
    import json
    if isinstance(v, dict):
        return dict(v)
    return json.loads(v) if v else {}
