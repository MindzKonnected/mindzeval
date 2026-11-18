"""Keep the examples in the docs honest: they must load with the real loader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindzeval import load_cases, load_config

DOCS = Path(__file__).resolve().parents[1] / "docs" / "reference"
GUIDES = Path(__file__).resolve().parents[1] / "docs" / "guides"


def _first_yaml_block(text: str, after: str) -> str:
    section = text[text.index(after) :]
    match = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert match, f"no yaml block after {after!r}"
    return match.group(1)


def test_full_config_example_loads(tmp_path: Path) -> None:
    text = (DOCS / "config.md").read_text(encoding="utf-8")
    path = tmp_path / "mindzeval.yaml"
    path.write_text(_first_yaml_block(text, "## Full example"), encoding="utf-8")
    env = {"AGENT_TOKEN": "t", "DB_PASSWORD": "p", "OPENAI_API_KEY": "k"}
    config = load_config(path, environ=env)
    assert len(config.matrix_combinations) == 4
    assert config.safety.pii_columns == ["email", "customers.phone"]


def test_case_format_static_expected_examples_load(tmp_path: Path) -> None:
    text = (DOCS / "case-format.md").read_text(encoding="utf-8")
    yaml_snippet = _first_yaml_block(text, "## Static expected results")
    path = tmp_path / "static_cases.yaml"
    path.write_text(yaml_snippet, encoding="utf-8")
    cases = load_cases([str(path)])
    assert len(cases) == 3

    # 1. Scalar expectation
    c1 = cases[0]
    assert c1.id == "total_users"
    assert c1.turns[0].compare.value == "scalar"
    assert c1.turns[0].expected_value == 42

    # 2. Tabular row expectation
    c2 = cases[1]
    assert c2.id == "top_products"
    assert c2.turns[0].compare.value == "set"
    assert isinstance(c2.turns[0].expected_rows, list)
    assert len(c2.turns[0].expected_rows) == 2

    # 3. Expected empty result
    c3 = cases[2]
    assert c3.id == "expired_trials"
    assert c3.turns[0].expect_empty is True


def test_langchain_adapter_snippet_extraction() -> None:
    @dataclass
    class Action:
        tool: str
        tool_input: Any

    mock_result = {
        "output": "Found 3 customers.",
        "intermediate_steps": [
            (Action(tool="other_tool", tool_input="foo"), "res1"),
            (Action(tool="sql_db_query", tool_input={"query": "SELECT id FROM c"}), "res2"),
        ],
    }

    # Follows the exact logic documented in connecting-agents.md
    sql = None
    for action, _observation in mock_result.get("intermediate_steps", []):
        if getattr(action, "tool", None) == "sql_db_query":
            tool_input = getattr(action, "tool_input", {})
            sql = (
                tool_input["query"]
                if isinstance(tool_input, dict) and "query" in tool_input
                else tool_input
            )

    res = {"answer": mock_result.get("output", ""), "sql": sql, "status": "answer"}
    assert res["answer"] == "Found 3 customers."
    assert res["sql"] == "SELECT id FROM c"
    assert res["status"] == "answer"


def test_crewai_adapter_snippet_extraction() -> None:
    @dataclass
    class MockCrewOutput:
        raw: str
        json_dict: dict[str, Any] | None = None
        pydantic: Any = None

    # Case A: with json_dict
    out_json = MockCrewOutput(raw="2 records", json_dict={"sql": "SELECT 2", "rows": [{"x": 1}]})
    data = out_json.json_dict or (
        out_json.pydantic.model_dump() if getattr(out_json, "pydantic", None) else {}
    )
    res_a = {
        "answer": out_json.raw,
        "sql": data.get("sql") if isinstance(data, dict) else None,
        "rows": data.get("rows") if isinstance(data, dict) else None,
    }
    assert res_a["sql"] == "SELECT 2"
    assert res_a["rows"] == [{"x": 1}]

    # Case B: with pydantic
    @dataclass
    class PydanticModel:
        def model_dump(self) -> dict[str, Any]:
            return {"sql": "SELECT 3", "rows": [{"y": 2}]}

    out_pyd = MockCrewOutput(raw="3 records", pydantic=PydanticModel())
    data_b = out_pyd.json_dict or (
        out_pyd.pydantic.model_dump() if getattr(out_pyd, "pydantic", None) else {}
    )
    res_b = {
        "answer": out_pyd.raw,
        "sql": data_b.get("sql") if isinstance(data_b, dict) else None,
        "rows": data_b.get("rows") if isinstance(data_b, dict) else None,
    }
    assert res_b["sql"] == "SELECT 3"
    assert res_b["rows"] == [{"y": 2}]
