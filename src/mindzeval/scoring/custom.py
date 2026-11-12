"""Custom scorers: teams add their own checks as plain Python.

A scorer is any callable that takes a :class:`ScoreContext` and returns one of:

* ``True`` / ``False``: passed or failed,
* ``None``: not applicable to this turn (ignored),
* a :class:`CustomResult` with a name, pass/fail, optional score and reason.

Register it in ``mindzeval.yaml``::

    scoring:
      custom: [my_checks:mentions_units]

A failed custom check fails the turn, like a built-in check. An exception in a
scorer is reported as an evaluator error (the scorer's bug), not an agent failure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mindzeval.db import QueryResult
from mindzeval.schema.case import Case, Turn
from mindzeval.schema.response import AgentResponse


@dataclass(frozen=True)
class ScoreContext:
    case: Case
    turn: Turn
    turn_index: int
    response: AgentResponse
    reference: QueryResult | None
    """The reference query's result, when the turn has one."""
    params: dict[str, Any]
    execution: Any = None
    """The built-in :class:`~mindzeval.scoring.execution.ExecutionScore`, if scored."""


@dataclass(frozen=True)
class CustomResult:
    name: str
    passed: bool | None
    score: float | None = None
    reason: str = ""


Scorer = Callable[[ScoreContext], "CustomResult | bool | None"]


def run_scorer(name: str, scorer: Scorer, context: ScoreContext) -> CustomResult | None:
    """Run one scorer and normalise its result. Exceptions propagate to the caller."""
    value = scorer(context)
    if value is None:
        return None
    if isinstance(value, CustomResult):
        return value
    if isinstance(value, bool):
        return CustomResult(name=name, passed=value)
    raise TypeError(
        f"custom scorer '{name}' returned {type(value).__name__}; expected bool, None or CustomResult"
    )
