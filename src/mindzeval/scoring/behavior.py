"""Behaviour and trace checks: did the agent answer, ask or decline when it should?

Scored in both directions. Expecting ``answer`` and getting ``clarify`` fails,
just as expecting ``clarify`` and getting an answer does, so an agent that asks
about everything can't score well. When the agent doesn't report ``status`` the
check is skipped and reported as skipped, never guessed from the answer text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mindzeval.schema.case import Turn
from mindzeval.schema.response import AgentResponse, Behavior

_MISSING = object()


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class BehaviorScore:
    expected: Behavior | None = None
    actual: Behavior | None = None
    checks: list[Check] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool | None:
        """``None`` when nothing could be checked."""
        if not self.checks:
            return None
        return all(c.passed for c in self.checks)

    @property
    def behavior_check(self) -> Check | None:
        return next((c for c in self.checks if c.name == "behavior"), None)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def score_behavior(turn: Turn, response: AgentResponse, *, enabled: bool = True) -> BehaviorScore:
    score = BehaviorScore(expected=turn.expected_behavior, actual=response.status)
    expect = turn.expect

    if enabled and score.expected is not None:
        if response.status is None:
            score.skipped.append("behavior: the agent did not report a status")
        else:
            ok = response.status is score.expected
            score.checks.append(
                Check("behavior", ok, f"expected {score.expected}, got {response.status}")
            )

    if enabled and expect.reason is not None:
        if response.reason is None:
            score.skipped.append("reason: the agent did not report a reason")
        else:
            ok = response.reason.strip().lower() == expect.reason.strip().lower()
            score.checks.append(
                Check("reason", ok, f"expected {expect.reason!r}, got {response.reason!r}")
            )

    if expect.trace:
        if response.trace is None:
            score.skipped.append("trace: the agent returned no trace")
        else:
            for path, wanted in expect.trace.items():
                got = trace_value(response.trace, path)
                if got is _MISSING:
                    score.checks.append(Check(f"trace.{path}", False, "not present in the trace"))
                else:
                    score.checks.append(
                        Check(
                            f"trace.{path}",
                            _equal(got, wanted),
                            f"expected {wanted!r}, got {got!r}",
                        )
                    )
    return score


def trace_value(trace: Any, path: str) -> Any:
    current = trace
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _equal(got: Any, wanted: Any) -> bool:
    if isinstance(got, str) and isinstance(wanted, str):
        return got.strip().lower() == wanted.strip().lower()
    return bool(got == wanted)
