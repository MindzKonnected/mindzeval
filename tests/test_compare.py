from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from mindzeval.schema.case import CompareMode, Tolerance
from mindzeval.scoring.compare import Table, compare

T = Tolerance()


def ref(columns: tuple[str, ...], *rows: tuple[Any, ...]) -> Table:
    return Table(columns, list(rows))


def agent(rows: list[dict[str, Any]]) -> Table:
    return Table.from_dicts(rows)


# --- set / ordered ----------------------------------------------------------


def test_set_ignores_order_extra_columns_and_duplicates() -> None:
    r = ref(("id",), (1,), (2,), (3,))
    a = agent([{"ID": "3", "name": "c"}, {"ID": 1.0, "name": "a"}, {"ID": 2}, {"ID": 2}])
    c = compare(CompareMode.SET, r, a, key=["id"], tolerance=T)
    assert c.correct and c.code == "match" and c.precision == c.recall == 1.0


def test_set_reports_direction_of_error() -> None:
    r = ref(("id",), (1,), (2,), (3,), (4,))
    c = compare(
        CompareMode.SET, r, agent([{"id": 1}, {"id": 2}, {"id": 9}]), key=["id"], tolerance=T
    )
    assert not c.correct and c.code == "wrong_rows"
    assert (c.missing_n, c.spurious_n) == (2, 1)
    assert c.precision == pytest.approx(2 / 3) and c.recall == pytest.approx(0.5)
    assert c.missing == ["3", "4"] and c.spurious == ["9"]


def test_reference_key_falls_back_to_first_column() -> None:
    r = ref(("customer_id",), (1,), (2,))
    c = compare(CompareMode.SET, r, agent([{"id": 1}, {"id": 2}]), key=["id"], tolerance=T)
    assert c.correct


def test_missing_key_column_in_agent_result() -> None:
    c = compare(
        CompareMode.SET, ref(("id",), (1,)), agent([{"name": "a"}]), key=["id"], tolerance=T
    )
    assert c.code == "missing_columns" and "name" in c.reason


def test_composite_key() -> None:
    r = ref(("a", "b"), (1, "x"), (1, "y"))
    c = compare(
        CompareMode.SET,
        r,
        agent([{"a": 1, "b": "y"}, {"a": 1, "b": "x"}]),
        key=["a", "b"],
        tolerance=T,
    )
    assert c.correct


def test_ordered_distinguishes_wrong_order_from_wrong_records() -> None:
    r = ref(("id",), (1,), (2,), (3,))
    swapped = compare(
        CompareMode.ORDERED, r, agent([{"id": 2}, {"id": 1}, {"id": 3}]), key=["id"], tolerance=T
    )
    assert swapped.code == "wrong_order" and swapped.set_match and not swapped.ordered_match
    wrong = compare(
        CompareMode.ORDERED, r, agent([{"id": 1}, {"id": 2}, {"id": 4}]), key=["id"], tolerance=T
    )
    assert wrong.code == "wrong_rows" and not wrong.set_match


def test_empty_both_sides_is_a_match() -> None:
    c = compare(CompareMode.SET, ref(("id",)), agent([]), key=["id"], tolerance=T)
    assert c.correct and c.f1 == 1.0


# --- scalar -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "rows", "correct"),
    [
        (Decimal("215"), [{"total_count": 215}], True),
        (215, [{"n": "215"}], True),
        (0.1 + 0.2, [{"x": 0.3}], True),
        (215, [{"label": "fintech", "count": 215}], True),
        (215, [{"count": 216}], False),
        ("DE", [{"country": "DE"}], True),
    ],
)
def test_scalar(reference: Any, rows: list[dict[str, Any]], correct: bool) -> None:
    c = compare(CompareMode.SCALAR, ref(("n",), (reference,)), agent(rows), key=None, tolerance=T)
    assert c.correct is correct


def test_scalar_without_single_value() -> None:
    c = compare(
        CompareMode.SCALAR, ref(("n",), (1,)), agent([{"a": 1, "b": 2}]), key=None, tolerance=T
    )
    assert c.code == "no_single_value"


# --- grouped ----------------------------------------------------------------


def test_grouped_matches_positionally_when_names_differ() -> None:
    r = ref(("group_key", "metric"), ("Fintech", 215), ("Health", 525))
    a = agent(
        [{"industry": "Health", "companies": 525}, {"industry": "Fintech", "companies": 215.0}]
    )
    assert compare(CompareMode.GROUPED, r, a, key=None, tolerance=T).correct


def test_grouped_detects_extra_group_and_value_mismatch() -> None:
    r = ref(("g", "m"), ("A", 1), ("B", 2))
    extra = compare(
        CompareMode.GROUPED,
        r,
        agent([{"g": "A", "m": 1}, {"g": "B", "m": 2}, {"g": None, "m": 7}]),
        key=None,
        tolerance=T,
    )
    assert extra.code == "wrong_groups" and extra.spurious_n == 1 and extra.recall == 1.0
    values = compare(
        CompareMode.GROUPED,
        r,
        agent([{"g": "A", "m": 1}, {"g": "B", "m": 99}]),
        key=None,
        tolerance=T,
    )
    assert values.code == "wrong_values" and values.mismatched_n == 1


def test_grouped_float_tolerance() -> None:
    r = ref(("g", "avg"), ("Series A", Decimal("4649677.4193548387")))
    a = agent([{"g": "Series A", "avg": 4649677.419354839}])
    assert compare(CompareMode.GROUPED, r, a, key=None, tolerance=T).correct


# --- table ------------------------------------------------------------------


def test_table_aligns_columns_by_name_then_values() -> None:
    r = ref(("month", "revenue"), ("2024-01", 10.5), ("2024-02", 20))
    a = agent(
        [
            {"m": "2024-02", "total_revenue": 20, "note": "x"},
            {"m": "2024-01", "total_revenue": 10.5, "note": "y"},
        ]
    )
    c = compare(CompareMode.TABLE, r, a, key=None, tolerance=T)
    assert c.correct, c.reason


def test_table_extra_columns_can_be_forbidden() -> None:
    r = ref(("a",), (1,))
    strict = Tolerance(extra_columns="forbid")
    assert (
        compare(CompareMode.TABLE, r, agent([{"a": 1, "b": 2}]), key=None, tolerance=strict).code
        == "extra_columns"
    )


def test_table_row_order_strict() -> None:
    r = ref(("a",), (1,), (2,))
    strict = Tolerance(row_order="strict")
    c = compare(CompareMode.TABLE, r, agent([{"a": 2}, {"a": 1}]), key=None, tolerance=strict)
    assert c.code == "wrong_order"


def test_table_duplicates_policy() -> None:
    r = ref(("a",), (1,), (1,))
    assert compare(CompareMode.TABLE, r, agent([{"a": 1}]), key=None, tolerance=T).correct
    counted = Tolerance(duplicates="count")
    assert not compare(CompareMode.TABLE, r, agent([{"a": 1}]), key=None, tolerance=counted).correct


def test_text_normalisation_options() -> None:
    r = ref(("name",), ("Ada ",))
    assert compare(CompareMode.TABLE, r, agent([{"name": "Ada"}]), key=None, tolerance=T).correct
    assert not compare(
        CompareMode.TABLE, r, agent([{"name": "ada"}]), key=None, tolerance=T
    ).correct
    loose = Tolerance(case_sensitive=False)
    assert compare(
        CompareMode.TABLE, r, agent([{"name": "ada"}]), key=None, tolerance=loose
    ).correct
    nulls = Tolerance(null_equals_empty=True)
    assert compare(
        CompareMode.TABLE, ref(("x",), (None,)), agent([{"x": ""}]), key=None, tolerance=nulls
    ).correct


def test_large_integers_are_compared_exactly() -> None:
    r = ref(("a",), (1234567,))
    assert not compare(CompareMode.TABLE, r, agent([{"a": 1234568}]), key=None, tolerance=T).correct
