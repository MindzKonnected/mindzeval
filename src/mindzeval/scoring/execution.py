"""Execution scoring: get the agent's rows, compare them, and replay without the display cap.

Two answers are reported side by side and never merged:

* **as shown**: the rows the user actually received (possibly cut short by a
  display LIMIT). This drives ``correct`` and the verdict.
* **uncapped**: the agent's own SQL re-run without its trailing display LIMIT.
  This answers "was the query logic right?" so a UI row cap isn't mistaken for a
  reasoning error. Skipped when the reference itself has a LIMIT (top-N
  questions), because then the LIMIT is part of the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mindzeval.db import Database, QueryResult
from mindzeval.errors import DatabaseError, SQLGuardError
from mindzeval.schema.case import CompareMode, Tolerance, Turn
from mindzeval.schema.response import AgentResponse
from mindzeval.scoring.compare import Comparison, Table, compare

_DISPLAY_LIMIT = re.compile(r"\s+LIMIT\s+(\d+)(?:\s+OFFSET\s+\d+)?\s*;?\s*$", re.IGNORECASE)
_REFERENCE_LIMIT = re.compile(
    r"\bLIMIT\s+\d+|\bFETCH\s+(?:FIRST|NEXT)\b|\bTOP\s*\(?\s*\d+", re.IGNORECASE
)
REPLAY_MODES = frozenset({CompareMode.SET, CompareMode.ORDERED, CompareMode.TABLE})


@dataclass
class ExecutionScore:
    comparison: Comparison | None = None
    """The rows the user received, compared with the reference."""

    rows_source: str = ""
    """``agent`` (rows returned by the agent) or ``replayed`` (MindzEval ran the agent's SQL)."""

    sql_error: str | None = None
    """The agent's SQL failed or was refused when MindzEval ran it."""

    sql_refused_unsafe: bool = False
    truncated: bool = False
    truncated_subset: bool = False
    """Truncated, and every row shown was correct. Strong evidence, not proof."""

    uncapped: Comparison | None = None
    uncapped_status: str = ""
    """``compared``, ``skipped: <why>`` or ``error: <what>`` (an evaluator-side issue)."""

    @property
    def scored(self) -> bool:
        return self.comparison is not None

    @property
    def correct(self) -> bool:
        return self.comparison is not None and self.comparison.correct

    @property
    def uncapped_correct(self) -> bool | None:
        return None if self.uncapped is None else self.uncapped.correct


def strip_display_limit(sql: str) -> tuple[str, int | None]:
    """Remove a trailing ``LIMIT n [OFFSET m]``. Returns the SQL and the limit removed."""
    match = _DISPLAY_LIMIT.search(sql)
    if not match:
        return sql, None
    return sql[: match.start()].rstrip(), int(match.group(1))


def reference_declares_limit(sql: str | None) -> bool:
    return bool(sql) and bool(_REFERENCE_LIMIT.search(sql or ""))


def score_execution(
    turn: Turn,
    response: AgentResponse,
    reference: QueryResult,
    *,
    db: Database | None,
    tolerance: Tolerance,
    uncapped_replay: bool = True,
) -> ExecutionScore:
    score = ExecutionScore()
    mode = turn.compare
    assert mode is not None and mode is not CompareMode.NONE

    agent_table: Table | None = None
    if response.rows is not None:
        agent_table = Table.from_dicts(response.rows, response.columns)
        score.rows_source = "agent"
    elif response.sql and db is not None:
        try:
            replay = db.fetch(response.sql, _params(response.params))
        except SQLGuardError as exc:
            score.sql_error = f"refused: {exc}"
            score.sql_refused_unsafe = exc.unsafe
        except DatabaseError as exc:
            score.sql_error = str(exc)
        else:
            agent_table = Table(replay.columns, replay.rows)
            score.rows_source = "replayed"
    if agent_table is None:
        return score

    is_static_count = (
        turn.reference_sql is None
        and isinstance(turn.expected_rows, int)
        and mode is not CompareMode.SCALAR
        and not reference.rows
    )

    if is_static_count:
        assert isinstance(turn.expected_rows, int)
        score.comparison = _compare_static_row_count(
            mode, turn.expected_rows, agent_table, turn.key
        )
    else:
        ref_table = Table(reference.columns, reference.rows)
        score.comparison = compare(mode, ref_table, agent_table, key=turn.key, tolerance=tolerance)

    stripped, limit = strip_display_limit(response.sql) if response.sql else (None, None)
    score.truncated = bool(response.truncated) or (
        limit is not None
        and len(agent_table.rows) >= limit
        and not reference_declares_limit(turn.reference_sql)
    )
    c = score.comparison
    score.truncated_subset = (
        score.truncated and not c.correct and c.spurious_n == 0 and c.agent_n > 0
    )

    if not uncapped_replay:
        score.uncapped_status = "skipped: disabled in config"
    elif mode not in REPLAY_MODES:
        score.uncapped_status = f"skipped: not needed for {mode} cases"
    elif not response.sql:
        score.uncapped_status = "skipped: the agent returned no SQL"
    elif reference_declares_limit(turn.reference_sql):
        score.uncapped_status = (
            "skipped: the reference has its own LIMIT, so the agent's is part of the answer"
        )
    elif limit is None:
        score.uncapped_status = "skipped: the agent's SQL has no trailing LIMIT"
    elif db is None:
        score.uncapped_status = "skipped: no database configured"
    else:
        try:
            full = db.fetch(stripped or "", _params(response.params))
        except (SQLGuardError, DatabaseError) as exc:
            score.uncapped_status = f"error: {exc}"
        else:
            if is_static_count:
                assert isinstance(turn.expected_rows, int)
                score.uncapped = _compare_static_row_count(
                    mode, turn.expected_rows, Table(full.columns, full.rows), turn.key
                )
            else:
                score.uncapped = compare(
                    mode,
                    ref_table,
                    Table(full.columns, full.rows),
                    key=turn.key,
                    tolerance=tolerance,
                )
            score.uncapped_status = "compared"
    return score


def _compare_static_row_count(
    mode: CompareMode,
    expected_n: int,
    agent_table: Table,
    key: list[str] | None,
) -> Comparison:
    agent_n = len(agent_table.rows)
    missing_keys = [k for k in (key or []) if agent_table.column_index(k) is None]
    if missing_keys:
        return Comparison(
            mode=mode,
            correct=False,
            code="missing_columns",
            reason=f"agent output is missing key column(s): {', '.join(missing_keys)}",
            reference_n=expected_n,
            agent_n=agent_n,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            set_match=False,
            ordered_match=False,
        )
    if agent_n == expected_n:
        return Comparison(
            mode=mode,
            correct=True,
            code="match",
            reason=f"matched {expected_n} expected rows",
            reference_n=expected_n,
            agent_n=agent_n,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            set_match=True,
            ordered_match=True,
        )
    if agent_n < expected_n:
        rec = round(agent_n / expected_n, 4) if expected_n else 0.0
        f1_val = round(2 * rec / (1.0 + rec), 4) if rec else 0.0
        return Comparison(
            mode=mode,
            correct=False,
            code="missing_rows",
            reason=f"expected {expected_n} rows, got {agent_n} (missing {expected_n - agent_n})",
            reference_n=expected_n,
            agent_n=agent_n,
            precision=1.0,
            recall=rec,
            f1=f1_val,
            set_match=False,
            ordered_match=False,
            missing_n=expected_n - agent_n,
        )
    prec = round(expected_n / agent_n, 4) if agent_n else 0.0
    f1_val = round(2 * prec / (1.0 + prec), 4) if prec else 0.0
    return Comparison(
        mode=mode,
        correct=False,
        code="extra_rows",
        reason=f"expected {expected_n} rows, got {agent_n} (extra {agent_n - expected_n})",
        reference_n=expected_n,
        agent_n=agent_n,
        precision=prec,
        recall=1.0,
        f1=f1_val,
        set_match=False,
        ordered_match=False,
        spurious_n=agent_n - expected_n,
    )


def _params(params: dict[str, Any] | list[Any] | None) -> Any:
    return params or None
