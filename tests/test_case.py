from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mindzeval import Behavior, Case, CompareMode, Tolerance, ToleranceOverride

SQL = "SELECT id FROM customers"


def case(**fields: Any) -> Case:
    return Case.model_validate({"id": "c1", **fields})


def test_single_turn_case_becomes_one_turn() -> None:
    c = case(question="Which customers?", reference_sql=SQL, key="id", tags="a, b")
    assert len(c.turns) == 1 and not c.is_multi_turn
    turn = c.turns[0]
    assert turn.key == ["id"]
    assert turn.compare is CompareMode.SET
    assert turn.expected_behavior is Behavior.ANSWER
    assert c.tags == ["a", "b"]


@pytest.mark.parametrize(
    ("fields", "mode"),
    [
        ({"reference_sql": SQL}, CompareMode.TABLE),
        ({"reference_sql": SQL, "key": "id"}, CompareMode.SET),
        ({"reference_sql": SQL, "compare": "scalar"}, CompareMode.SCALAR),
        ({"expect": {"behavior": "clarify"}}, CompareMode.NONE),
    ],
)
def test_compare_mode_is_inferred(fields: dict[str, Any], mode: CompareMode) -> None:
    assert case(question="q", **fields).turns[0].compare is mode


def test_golden_sql_and_behaviour_spellings_are_accepted() -> None:
    c = case(question="q", golden_sql=SQL)
    assert c.turns[0].reference_sql == SQL
    c = case(question="q", expect={"behaviour": "decline"})
    assert c.turns[0].expect.behavior is Behavior.DECLINE


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"compare": "set"}, "needs a reference_sql"),
        ({"reference_sql": SQL, "compare": "ordered"}, "needs a key"),
        ({"reference_sql": SQL, "expect": {"behavior": "clarify"}}, "cannot also have"),
        ({}, "checks nothing"),
        ({"reference_sql": SQL, "expect_empty": True, "expected_rows": 3}, "expect_empty"),
        ({"reference_sql": "   "}, "must not be blank"),
        ({"reference_sql": SQL, "key": []}, "at least one column"),
    ],
)
def test_turn_rules(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        case(question="q", **fields)


def test_case_needs_question_or_turns() -> None:
    with pytest.raises(ValidationError, match="either a 'question' or a list of 'turns'"):
        case(tags=["x"])


def test_invalid_case_id() -> None:
    with pytest.raises(ValidationError):
        case(id="bad id!", question="q", reference_sql=SQL)


def test_unknown_case_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown field"):
        case(question="q", reference_sql=SQL, owner="me")


def test_multi_turn_inherits_case_defaults() -> None:
    c = case(
        key="id",
        compare="set",
        turns=[
            {"question": "All customers", "reference_sql": SQL},
            {"question": "Only in DE", "reference_sql": SQL + " WHERE country='DE'"},
            {"question": "How many?", "reference_sql": "SELECT 1", "compare": "scalar"},
        ],
    )
    assert c.is_multi_turn
    assert [t.compare for t in c.turns] == [CompareMode.SET, CompareMode.SET, CompareMode.SCALAR]
    assert all(t.key == ["id"] for t in c.turns)


def test_turn_fields_at_case_level_are_rejected_with_turns() -> None:
    with pytest.raises(ValidationError, match="belong inside each turn"):
        case(question="q", turns=[{"question": "q", "reference_sql": SQL}])


def test_tolerance_override_applies_only_set_fields() -> None:
    base = Tolerance()
    merged = ToleranceOverride(relative=0.01, extra_columns="forbid").apply(base)
    assert merged.relative == 0.01 and merged.extra_columns == "forbid"
    assert merged.absolute == base.absolute
    assert ToleranceOverride().apply(base) is base


def test_static_expected_without_reference_sql() -> None:
    c1 = case(question="Startups", expected_rows=160, key="org_id")
    t1 = c1.turns[0]
    assert t1.compare is CompareMode.SET
    assert t1.expected_behavior is Behavior.ANSWER
    assert t1.has_static_expected
    assert t1.reference_sql is None

    c2 = case(question="Startups", expected_rows=160)
    t2 = c2.turns[0]
    assert t2.compare is CompareMode.TABLE

    c3 = case(
        question="Startups",
        expected_rows=[{"org_id": "1"}, {"org_id": "2"}],
        key="org_id",
    )
    t3 = c3.turns[0]
    assert t3.compare is CompareMode.SET
    ref3 = t3.build_static_reference()
    assert ref3.row_count == 2
    assert "org_id" in ref3.columns

    c4 = case(question="Count", expected_value=42)
    t4 = c4.turns[0]
    assert t4.compare is CompareMode.SCALAR
    ref4 = t4.build_static_reference()
    assert ref4.rows == [(42,)]

    c5 = case(question="Empty", expect_empty=True)
    t5 = c5.turns[0]
    assert t5.has_static_expected
    assert t5.compare is CompareMode.TABLE
