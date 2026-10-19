from __future__ import annotations

import pytest
from pydantic import ValidationError

from mindzeval import AgentResponse, Behavior


def test_plain_string_is_an_answer() -> None:
    response = AgentResponse.model_validate("There are 3 customers.")
    assert response.answer == "There are 3 customers."
    assert response.sql is None


def test_rows_as_dicts_infer_columns_in_order() -> None:
    response = AgentResponse(rows=[{"id": 1, "name": "Ada"}, {"id": 2, "name": "Bo", "x": 1}])
    assert response.columns == ["id", "name", "x"]


def test_rows_as_lists_need_columns() -> None:
    with pytest.raises(ValidationError, match="need a 'columns' list"):
        AgentResponse.model_validate({"rows": [[1, "Ada"]]})


def test_rows_as_lists_are_converted() -> None:
    response = AgentResponse.model_validate(
        {"columns": ["id", "name"], "rows": [[1, "Ada"], (2, "Bo")]}
    )
    assert response.rows == [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Bo"}]


def test_row_width_must_match_columns() -> None:
    with pytest.raises(ValidationError, match="row 1 has 1 values but 2 columns"):
        AgentResponse.model_validate({"columns": ["id", "name"], "rows": [[1, "Ada"], [2]]})


def test_duplicate_columns_are_rejected_with_a_fix() -> None:
    with pytest.raises(ValidationError, match="alias them"):
        AgentResponse.model_validate({"columns": ["name", "name"], "rows": [["a", "b"]]})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="SQL"):
        AgentResponse.model_validate({"answer": "x", "SQL": "SELECT 1"})


def test_status_values() -> None:
    assert AgentResponse(status="clarify").status is Behavior.CLARIFY  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AgentResponse.model_validate({"status": "HITL"})
