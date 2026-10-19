from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mindzeval import CompareMode, ConfigError, load_cases, load_config, load_project
from mindzeval.schema import HttpAgentConfig, PythonAgentConfig

Write = Callable[[str, str], Path]

CONFIG = """\
project: shop-agent
agent:
  type: python
  target: my_agent:answer
database:
  url: sqlite:///shop.db
datasets: cases/*.yaml
"""

CASES = """\
- id: customers_de
  question: Which customers are in Germany?
  reference_sql: SELECT id FROM customers WHERE country = 'DE'
  key: id
  tags: [smoke]
- id: order_count
  question: How many orders are there?
  reference_sql: SELECT count(*) FROM orders
  compare: scalar
"""


def _issues(exc: pytest.ExceptionInfo[ConfigError]) -> list[str]:
    return [str(i) for i in exc.value.issues]


# --- config -----------------------------------------------------------------


def test_load_project(write: Write, tmp_path: Path) -> None:
    config_path = write("mindzeval.yaml", CONFIG)
    write("cases/basic.yaml", CASES)
    project = load_project(config_path)
    assert isinstance(project.config.agent, PythonAgentConfig)
    assert [c.id for c in project.cases] == ["customers_de", "order_count"]
    assert project.cases[0].source == f"{tmp_path / 'cases' / 'basic.yaml'}:1"
    url = project.config.database.url.get_secret_value()  # type: ignore[union-attr]
    assert url == "sqlite:///" + (tmp_path / "shop.db").resolve().as_posix()


def test_config_errors_carry_line_numbers(write: Write) -> None:
    path = write(
        "mindzeval.yaml",
        """\
agent:
  type: http
  url: ftp://wrong
  timeout_s: -1
datasets: []
""",
    )
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    issues = exc.value.issues
    by_path = {i.path: i for i in issues}
    assert by_path["agent.url"].line == 3
    assert by_path["agent.timeout_s"].line == 4
    assert by_path["datasets"].line == 5
    assert len(issues) == 3


def test_unknown_agent_type(write: Write) -> None:
    path = write("mindzeval.yaml", "agent: {type: carrier-pigeon}\ndatasets: x.yaml\n")
    with pytest.raises(ConfigError, match="carrier-pigeon"):
        load_config(path)


def test_env_vars_are_substituted(write: Write) -> None:
    path = write(
        "mindzeval.yaml",
        """\
agent:
  type: http
  url: ${AGENT_URL}
  headers: {Authorization: "Bearer ${TOKEN}"}
  params: {model: "${MODEL:-gpt-5}", price: "$$5"}
database:
  url: postgresql://reader:${DB_PASSWORD}@db.internal/shop
datasets: x.yaml
""",
    )
    env = {"AGENT_URL": "https://agent.internal/ask", "TOKEN": "t0k", "DB_PASSWORD": "pw"}
    config = load_config(path, environ=env)
    assert isinstance(config.agent, HttpAgentConfig)
    assert config.agent.url == "https://agent.internal/ask"
    assert config.agent.headers["Authorization"] == "Bearer t0k"
    assert config.agent.params == {"model": "gpt-5", "price": "$5"}
    assert "pw" not in repr(config)
    assert config.database is not None
    assert config.database.url.get_secret_value() == "postgresql://reader:pw@db.internal/shop"


def test_missing_env_var_is_reported_with_line(write: Write) -> None:
    path = write(
        "mindzeval.yaml",
        "agent:\n  type: http\n  url: ${AGENT_URL}\ndatasets: x.yaml\n",
    )
    with pytest.raises(ConfigError) as exc:
        load_config(path, environ={})
    assert exc.value.issues[0].line == 3
    assert "AGENT_URL" in exc.value.issues[0].message


def test_env_vars_left_alone_when_not_resolving(write: Write) -> None:
    path = write(
        "mindzeval.yaml",
        "agent:\n  type: http\n  url: https://${HOST}/ask\ndatasets: x.yaml\n",
    )
    config = load_config(path, resolve_env=False, environ={})
    assert isinstance(config.agent, HttpAgentConfig)
    assert config.agent.url == "https://${HOST}/ask"


@pytest.mark.parametrize(
    ("snippet", "field"),
    [
        ("database: {url: 'postgresql://u:hunter2@h/db'}", "database.url"),
        (
            "agent: {type: http, url: 'https://a', headers: {Authorization: 'Bearer abc'}}",
            "agent.headers.Authorization",
        ),
        ("judge: {provider: openai, model: m, api_key: sk-123}", "judge.api_key"),
    ],
)
def test_literal_secrets_are_refused(write: Write, snippet: str, field: str) -> None:
    base = "agent: {type: python, target: 'a:b'}\n" if not snippet.startswith("agent") else ""
    path = write("mindzeval.yaml", base + snippet + "\ndatasets: x.yaml\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path, resolve_env=False)
    assert [i.path for i in exc.value.issues] == [field]


def test_matrix_combinations_and_limits(write: Write) -> None:
    path = write(
        "mindzeval.yaml",
        """\
agent: {type: python, target: 'a:b'}
datasets: x.yaml
matrix:
  model: [gpt-5, claude]
  prompt: [v1, v2]
""",
    )
    config = load_config(path)
    assert config.matrix_combinations == [
        {"model": "gpt-5", "prompt": "v1"},
        {"model": "gpt-5", "prompt": "v2"},
        {"model": "claude", "prompt": "v1"},
        {"model": "claude", "prompt": "v2"},
    ]

    too_big = write(
        "big.yaml",
        "agent: {type: python, target: 'a:b'}\ndatasets: x.yaml\n"
        "matrix: {a: [1,2,3,4,5,6,7,8,9], b: [1,2,3,4,5,6,7,8]}\n",
    )
    with pytest.raises(ConfigError, match="72 combinations"):
        load_config(too_big)


def test_invalid_yaml_reports_position(write: Write) -> None:
    path = write("mindzeval.yaml", "agent:\n  type: python\n target: x\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert exc.value.issues[0].line is not None
    assert "invalid YAML" in exc.value.issues[0].message


# --- datasets ---------------------------------------------------------------


def test_case_errors_carry_case_id_and_line(write: Write, tmp_path: Path) -> None:
    write(
        "cases.yaml",
        """\
- id: ok_case
  question: Fine
  reference_sql: SELECT 1
- id: bad_case
  question: Ranked list
  reference_sql: SELECT id FROM t
  compare: ordered
- id: worse
  question: q
  expect: {behavior: maybe}
""",
    )
    with pytest.raises(ConfigError) as exc:
        load_cases(["cases.yaml"], base_dir=tmp_path)
    issues = exc.value.issues
    assert len(issues) == 2
    assert issues[0].path == "case 'bad_case'" and issues[0].line == 4
    assert "needs a key" in issues[0].message
    assert issues[1].path == "case 'worse': expect.behavior" and issues[1].line == 10


def test_dataset_defaults(write: Write, tmp_path: Path) -> None:
    write(
        "cases.yaml",
        """\
defaults:
  key: customer_id
  tags: [regression]
  category: filter
cases:
  - id: a
    question: q
    reference_sql: SELECT customer_id FROM c
    tags: [smoke]
  - id: b
    question: how many
    reference_sql: SELECT count(*) FROM c
    compare: scalar
    category: count
""",
    )
    a, b = load_cases(["cases.yaml"], base_dir=tmp_path)
    assert a.turns[0].compare is CompareMode.SET and a.turns[0].key == ["customer_id"]
    assert a.tags == ["regression", "smoke"] and a.category == "filter"
    assert b.turns[0].compare is CompareMode.SCALAR and b.category == "count"


def test_config_default_key_only_fills_keyed_modes(write: Write, tmp_path: Path) -> None:
    write(
        "cases.yaml",
        """\
- {id: a, question: q, reference_sql: SELECT id FROM t, compare: ordered}
- {id: b, question: q, reference_sql: "SELECT id, x FROM t"}
""",
    )
    a, b = load_cases(["cases.yaml"], base_dir=tmp_path, default_key=["id"])
    assert a.turns[0].key == ["id"]
    assert b.turns[0].compare is CompareMode.TABLE and b.turns[0].key is None


def test_duplicate_ids_across_files(write: Write, tmp_path: Path) -> None:
    write("one.yaml", "- {id: dup, question: q, reference_sql: SELECT 1}\n")
    write("two.yaml", "- {id: dup, question: q, reference_sql: SELECT 2}\n")
    with pytest.raises(ConfigError, match="duplicate case id 'dup'"):
        load_cases(["*.yaml"], base_dir=tmp_path)


def test_unmatched_pattern(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no dataset files match"):
        load_cases(["missing/*.yaml"], base_dir=tmp_path)


def test_jsonl_dataset(write: Write, tmp_path: Path) -> None:
    write(
        "cases.jsonl",
        '{"id": "a", "question": "q", "reference_sql": "SELECT 1"}\n'
        "\n"
        '{"id": "b", "question": "q", "expect": {"behavior": "decline"}}\n',
    )
    cases = load_cases(["cases.jsonl"], base_dir=tmp_path)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[1].source is not None and cases[1].source.endswith(":3")


def test_csv_dataset_single_and_multi_turn(write: Write, tmp_path: Path) -> None:
    write(
        "cases.csv",
        "id,turn,question,reference_sql,compare,key,behavior,tags,meta.owner\n"
        "single,,Which customers?,SELECT id FROM c,,id,,smoke;regression,ana\n"
        "convo,1,All orders,SELECT id FROM o,,id,,,\n"
        "convo,2,Only big ones,SELECT id FROM o WHERE total > 10,,id,,,\n"
        "vague,,Show me stuff,,,,clarify,,\n",
    )
    single, convo, vague = load_cases(["cases.csv"], base_dir=tmp_path)
    assert single.tags == ["smoke", "regression"] and single.metadata == {"owner": "ana"}
    assert single.turns[0].compare is CompareMode.SET
    assert convo.is_multi_turn and convo.turns[1].question == "Only big ones"
    assert vague.turns[0].compare is CompareMode.NONE


def test_csv_unknown_column(write: Write, tmp_path: Path) -> None:
    write("cases.csv", "id,question,owner\na,q,me\n")
    with pytest.raises(ConfigError, match="meta"):
        load_cases(["cases.csv"], base_dir=tmp_path)


def test_csv_repeated_id_without_turn_column(write: Write, tmp_path: Path) -> None:
    write("cases.csv", "id,question,reference_sql\na,q,SELECT 1\na,q2,SELECT 2\n")
    with pytest.raises(ConfigError) as exc:
        load_cases(["cases.csv"], base_dir=tmp_path)
    assert exc.value.issues[0].line == 3


def test_bad_dataset_shape(write: Write, tmp_path: Path) -> None:
    write("cases.yaml", "defaults: {owner: x}\ncases: []\n")
    with pytest.raises(ConfigError, match="may only set"):
        load_cases(["cases.yaml"], base_dir=tmp_path)
