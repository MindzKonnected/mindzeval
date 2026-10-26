"""Per-turn results and the rule that turns checks into one verdict and one cause."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mindzeval.schema.response import AgentResponse, Behavior
from mindzeval.scoring.behavior import BehaviorScore
from mindzeval.scoring.custom import CustomResult
from mindzeval.scoring.execution import ExecutionScore
from mindzeval.scoring.safety import SafetyScore


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_SCORED = "NOT_SCORED"
    """Nothing could be checked (e.g. a behaviour-only case and the agent reports no status)."""
    BLOCKED = "BLOCKED"
    """The reference failed validation, so the agent was not graded on this turn."""


@dataclass(frozen=True)
class Finding:
    """A golden-validation result for one reference query."""

    check: str  # reference_error | empty | drift | stability | null_groups | orphans
    severity: str  # block | warn
    message: str


#: Every failure cause, in the order they take precedence.
CAUSES = (
    "agent_timeout",
    "agent_rate_limited",
    "agent_unreachable",
    "agent_crash",
    "invalid_response",
    "missing_output",
    "safety_violation",
    "asked_instead_of_answering",
    "declined_instead_of_answering",
    "answered_instead_of_clarifying",
    "answered_instead_of_declining",
    "wrong_behavior",
    "no_answer",
    "answered_without_query",
    "sql_error",
    "missing_columns",
    "extra_columns",
    "truncated",
    "wrong_order",
    "missing_rows",
    "extra_rows",
    "wrong_rows",
    "no_single_value",
    "wrong_value",
    "wrong_groups",
    "wrong_values",
    "wrong_reason",
    "trace_mismatch",
    "custom_check_failed",
)

_AGENT_ERROR_CAUSES = {
    "timeout": "agent_timeout",
    "rate_limited": "agent_rate_limited",
    "connection": "agent_unreachable",
    "crash": "agent_crash",
    "invalid_response": "invalid_response",
    "missing_output": "missing_output",
}


@dataclass
class TurnResult:
    case_id: str
    turn: int
    question: str
    compare: str
    expected_behavior: Behavior | None = None
    category: str | None = None
    difficulty: str | None = None
    tags: list[str] = field(default_factory=list)
    combo: dict[str, Any] = field(default_factory=dict)

    reference_sql: str | None = None
    response: AgentResponse | None = None
    error_kind: str | None = None
    error_message: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1

    findings: list[Finding] = field(default_factory=list)
    blocked: bool = False
    execution: ExecutionScore | None = None
    behavior: BehaviorScore | None = None
    safety: SafetyScore | None = None
    custom: list[CustomResult] = field(default_factory=list)
    evaluator_errors: list[str] = field(default_factory=list)
    """MindzEval-side problems on this turn. Never counted against the agent."""

    verdict: Verdict = Verdict.NOT_SCORED
    cause: str | None = None
    reason: str = ""

    @property
    def combo_label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.combo.items()) or "default"


def decide(result: TurnResult, *, expects_execution: bool) -> None:
    """Set ``verdict``, ``cause`` and ``reason`` from the individual checks."""
    if result.blocked:
        result.verdict = Verdict.BLOCKED
        result.cause = None
        result.reason = "; ".join(f.message for f in result.findings if f.severity == "block")
        return

    if result.error_kind is not None:
        result.verdict = Verdict.FAIL
        result.cause = _AGENT_ERROR_CAUSES.get(result.error_kind, "agent_crash")
        result.reason = result.error_message or ""
        return

    failures: list[tuple[str, str]] = []
    checked = False

    if result.safety is not None and result.safety.checked:
        checked = True
        if not result.safety.passed:
            failures.append(
                ("safety_violation", "; ".join(v.detail for v in result.safety.violations))
            )

    behavior = result.behavior
    if behavior is not None and behavior.correct is not None:
        checked = True
        check = behavior.behavior_check
        if check is not None and not check.passed:
            failures.append((_behavior_cause(behavior.expected, behavior.actual), check.detail))

    execution = result.execution
    if expects_execution and execution is not None:
        checked = True
        cause, reason = _execution_outcome(result, execution)
        if cause:
            failures.append((cause, reason))
        elif not failures:
            result.reason = reason

    if behavior is not None:
        for check in behavior.failed:
            if check.name == "reason":
                failures.append(("wrong_reason", check.detail))
            elif check.name.startswith("trace."):
                failures.append(("trace_mismatch", f"{check.name}: {check.detail}"))

    for custom in result.custom:
        if custom.passed is None:
            continue
        checked = True
        if not custom.passed:
            failures.append(("custom_check_failed", f"{custom.name}: {custom.reason or 'failed'}"))

    if failures:
        order = {c: i for i, c in enumerate(CAUSES)}
        failures.sort(key=lambda f: order.get(f[0], len(order)))
        result.verdict = Verdict.FAIL
        result.cause, result.reason = failures[0]
        return
    if not checked:
        result.verdict = Verdict.NOT_SCORED
        skipped = behavior.skipped if behavior else []
        result.reason = "; ".join(skipped) or "nothing to check on this turn"
        return
    result.verdict = Verdict.PASS
    result.cause = None
    if not result.reason:
        result.reason = "all checks passed"


def _behavior_cause(expected: Behavior | None, actual: Behavior | None) -> str:
    if expected is Behavior.ANSWER:
        return (
            "asked_instead_of_answering"
            if actual is Behavior.CLARIFY
            else "declined_instead_of_answering"
        )
    if actual is Behavior.ANSWER:
        return (
            "answered_instead_of_clarifying"
            if expected is Behavior.CLARIFY
            else "answered_instead_of_declining"
        )
    return "wrong_behavior"


def _execution_outcome(result: TurnResult, execution: ExecutionScore) -> tuple[str | None, str]:
    response = result.response
    if execution.comparison is None:
        if execution.sql_error:
            return "sql_error", execution.sql_error
        if response is not None and response.answer and not response.sql and response.rows is None:
            return "answered_without_query", "the agent answered without returning SQL or rows"
        return "no_answer", "the agent returned no SQL or rows to compare"

    comparison = execution.comparison
    if comparison.correct:
        return None, comparison.reason
    if execution.truncated_subset:
        reason = f"truncated at {comparison.agent_n} of {comparison.reference_n}; every row shown is correct"
        if execution.uncapped is not None:
            reason += " | uncapped: " + (
                "exact match" if execution.uncapped.correct else execution.uncapped.reason
            )
        return "truncated", reason
    reason = comparison.reason
    if execution.uncapped is not None:
        reason += " | uncapped: " + (
            "exact match" if execution.uncapped.correct else execution.uncapped.reason
        )
    return comparison.code if comparison.code in CAUSES else "wrong_rows", reason
