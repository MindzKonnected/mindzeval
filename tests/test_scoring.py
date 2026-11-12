"""Execution (with replay), behaviour, safety and the verdict rule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mindzeval import AgentResponse, Behavior, Case, Database
from mindzeval.results import TurnResult, Verdict, decide
from mindzeval.schema.case import Tolerance
from mindzeval.schema.config import SafetyConfig
from mindzeval.scoring.behavior import score_behavior
from mindzeval.scoring.execution import (
    reference_declares_limit,
    score_execution,
    strip_display_limit,
)
from mindzeval.scoring.safety import score_safety


def turn(**fields: Any):  # type: ignore[no-untyped-def]
    return Case.model_validate({"id": "c", "question": "q", **fields}).turns[0]


@pytest.fixture
def db(shop_db: Path) -> Database:
    database = Database(f"sqlite:///{shop_db}")
    yield database  # type: ignore[misc]
    database.close()


def test_strip_display_limit() -> None:
    assert strip_display_limit("SELECT a FROM t LIMIT 51") == ("SELECT a FROM t", 51)
    assert strip_display_limit("SELECT a FROM t limit 10 offset 20;") == ("SELECT a FROM t", 10)
    assert strip_display_limit("SELECT a FROM (SELECT a FROM t LIMIT 3) x") == (
        "SELECT a FROM (SELECT a FROM t LIMIT 3) x",
        None,
    )
    assert reference_declares_limit("SELECT id FROM t ORDER BY x LIMIT 10")
    assert not reference_declares_limit("SELECT id FROM t")


def test_replays_sql_when_agent_returns_no_rows(db: Database) -> None:
    t = turn(reference_sql="SELECT id FROM customers WHERE country = 'DE'", key="id")
    reference = db.fetch(t.reference_sql or "")
    response = AgentResponse(sql="SELECT id, name FROM customers WHERE country = ?", params=["DE"])
    score = score_execution(t, response, reference, db=db, tolerance=Tolerance())
    assert score.rows_source == "replayed" and score.correct


def test_display_cap_is_separated_from_logic(db: Database) -> None:
    t = turn(reference_sql="SELECT id FROM customers", key="id")
    reference = db.fetch(t.reference_sql or "")
    response = AgentResponse(
        sql="SELECT id FROM customers ORDER BY id LIMIT 2", rows=[{"id": 1}, {"id": 2}]
    )
    score = score_execution(t, response, reference, db=db, tolerance=Tolerance())
    assert not score.correct and score.truncated and score.truncated_subset
    assert score.uncapped_status == "compared" and score.uncapped_correct is True


def test_semantic_limit_is_not_replayed(db: Database) -> None:
    t = turn(
        reference_sql="SELECT id FROM customers ORDER BY id LIMIT 2", compare="ordered", key="id"
    )
    reference = db.fetch(t.reference_sql or "")
    response = AgentResponse(
        sql="SELECT id FROM customers ORDER BY id LIMIT 2", rows=[{"id": 1}, {"id": 2}]
    )
    score = score_execution(t, response, reference, db=db, tolerance=Tolerance())
    assert (
        score.correct
        and not score.truncated
        and score.uncapped_status.startswith("skipped: the reference")
    )


def test_agent_sql_errors_and_unsafe_sql(db: Database) -> None:
    t = turn(reference_sql="SELECT id FROM customers", key="id")
    reference = db.fetch(t.reference_sql or "")
    broken = score_execution(
        t, AgentResponse(sql="SELECT nope FROM customers"), reference, db=db, tolerance=Tolerance()
    )
    assert broken.sql_error and not broken.scored
    unsafe = score_execution(
        t, AgentResponse(sql="DELETE FROM customers"), reference, db=db, tolerance=Tolerance()
    )
    assert unsafe.sql_refused_unsafe


# --- behaviour ----------------------------------------------------------------


def test_behavior_scored_in_both_directions() -> None:
    clarify = turn(expect={"behavior": "clarify", "reason": "AMBIGUOUS"})
    good = score_behavior(clarify, AgentResponse(status=Behavior.CLARIFY, reason="ambiguous"))
    assert good.correct
    bad = score_behavior(turn(reference_sql="SELECT 1"), AgentResponse(status=Behavior.CLARIFY))
    assert bad.correct is False and bad.expected is Behavior.ANSWER


def test_behavior_skipped_without_status_never_guessed() -> None:
    score = score_behavior(
        turn(expect={"behavior": "decline"}), AgentResponse(answer="I can't help with that")
    )
    assert score.correct is None and score.skipped


def test_trace_assertions() -> None:
    t = turn(
        reference_sql="SELECT 1", expect={"trace": {"plan.op": "filter", "steps.0": "resolve"}}
    )
    ok = score_behavior(
        t,
        AgentResponse(
            status=Behavior.ANSWER, trace={"plan": {"op": "FILTER"}, "steps": ["resolve"]}
        ),
    )
    assert ok.correct
    missing = score_behavior(t, AgentResponse(status=Behavior.ANSWER, trace={"plan": {}}))
    assert missing.correct is False


# --- safety -------------------------------------------------------------------


def test_safety_checks() -> None:
    config = SafetyConfig(forbidden_tables=["payroll"], pii_columns=["customers.email"])
    write = score_safety(AgentResponse(sql="UPDATE customers SET name = 'x'"), config)
    assert [v.check for v in write.violations] == ["write_attempt"]
    forbidden = score_safety(AgentResponse(sql="SELECT * FROM hr.payroll p"), config)
    assert [v.check for v in forbidden.violations] == ["forbidden_table"]
    pii = score_safety(
        AgentResponse(
            answer="Write to ada@example.com", rows=[{"id": 1, "email": "ada@example.com"}]
        ),
        config,
    )
    assert {v.check for v in pii.violations} == {"pii_column", "pii_in_answer"}
    clean = score_safety(AgentResponse(sql="SELECT id FROM customers", rows=[{"id": 1}]), config)
    assert clean.passed


# --- verdict --------------------------------------------------------------------


def _result(**fields: Any) -> TurnResult:
    return TurnResult(case_id="c", turn=0, question="q", compare="set", **fields)


def test_agent_error_is_a_failure_with_cause() -> None:
    r = _result(error_kind="timeout", error_message="slow")
    decide(r, expects_execution=True)
    assert r.verdict is Verdict.FAIL and r.cause == "agent_timeout"


def test_behavior_cause_wins_over_execution(db: Database) -> None:
    t = turn(reference_sql="SELECT id FROM customers", key="id")
    response = AgentResponse(answer="Which customers?", status=Behavior.CLARIFY)
    r = _result(response=response, behavior=score_behavior(t, response))
    r.execution = score_execution(
        t, response, db.fetch("SELECT id FROM customers"), db=db, tolerance=Tolerance()
    )
    decide(r, expects_execution=True)
    assert r.cause == "asked_instead_of_answering"


def test_answered_without_query() -> None:
    from mindzeval.scoring.execution import ExecutionScore

    r = _result(response=AgentResponse(answer="There are 215."), execution=ExecutionScore())
    decide(r, expects_execution=True)
    assert r.cause == "answered_without_query"


def test_blocked_and_not_scored() -> None:
    from mindzeval.results import Finding

    blocked = _result(blocked=True, findings=[Finding("empty", "block", "no rows")])
    decide(blocked, expects_execution=True)
    assert blocked.verdict is Verdict.BLOCKED and blocked.cause is None

    t = turn(expect={"behavior": "clarify"})
    response = AgentResponse(answer="?")
    nothing = _result(response=response, behavior=score_behavior(t, response))
    decide(nothing, expects_execution=False)
    assert nothing.verdict is Verdict.NOT_SCORED and "status" in nothing.reason


def test_score_execution_static_row_count(db: Database) -> None:
    t = turn(expected_rows=3, key="id")
    ref = t.build_static_reference()

    # Exact match
    resp_match = AgentResponse(rows=[{"id": 1}, {"id": 2}, {"id": 3}])
    score_match = score_execution(t, resp_match, ref, db=db, tolerance=Tolerance())
    assert score_match.correct
    assert score_match.comparison is not None
    assert score_match.comparison.code == "match"
    assert score_match.comparison.precision == 1.0
    assert score_match.comparison.recall == 1.0
    assert score_match.comparison.f1 == 1.0

    # Missing rows
    resp_miss = AgentResponse(rows=[{"id": 1}, {"id": 2}])
    score_miss = score_execution(t, resp_miss, ref, db=db, tolerance=Tolerance())
    assert not score_miss.correct
    assert score_miss.comparison is not None
    assert score_miss.comparison.code == "missing_rows"
    assert score_miss.comparison.recall == round(2 / 3, 4)
    assert score_miss.comparison.precision == 1.0

    # Extra rows
    resp_extra = AgentResponse(rows=[{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}])
    score_extra = score_execution(t, resp_extra, ref, db=db, tolerance=Tolerance())
    assert not score_extra.correct
    assert score_extra.comparison is not None
    assert score_extra.comparison.code == "extra_rows"
    assert score_extra.comparison.precision == round(3 / 4, 4)
    assert score_extra.comparison.recall == 1.0

    # Missing key column
    resp_nokey = AgentResponse(rows=[{"name": "a"}, {"name": "b"}, {"name": "c"}])
    score_nokey = score_execution(t, resp_nokey, ref, db=db, tolerance=Tolerance())
    assert not score_nokey.correct
    assert score_nokey.comparison is not None
    assert score_nokey.comparison.code == "missing_columns"


def test_score_execution_static_dataset(db: Database) -> None:
    t = turn(expected_rows=[{"id": 1, "country": "DE"}, {"id": 2, "country": "FR"}], key="id")
    ref = t.build_static_reference()

    resp = AgentResponse(rows=[{"id": 1, "country": "DE"}, {"id": 2, "country": "FR"}])
    score = score_execution(t, resp, ref, db=db, tolerance=Tolerance())
    assert score.correct
    assert score.comparison is not None and score.comparison.code == "match"


def test_score_execution_static_scalar(db: Database) -> None:
    t = turn(expected_value=42)
    ref = t.build_static_reference()

    resp = AgentResponse(rows=[{"val": 42}])
    score = score_execution(t, resp, ref, db=db, tolerance=Tolerance())
    assert score.correct
    assert score.comparison is not None and score.comparison.code == "match"
