"""End-to-end: quickstart project, runner, reports, matrix and the CLI."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from mindzeval import load_project
from mindzeval.cli import EXIT_FAILED, EXIT_OK, main
from mindzeval.quickstart import create_quickstart
from mindzeval.report import build_rollup, check_thresholds, write_reports
from mindzeval.results import Verdict
from mindzeval.runner import run, select_cases

Write = Callable[[str, str], Path]


@pytest.fixture
def quick(tmp_path: Path) -> Path:
    directory = tmp_path / "qs"
    create_quickstart(directory)
    return directory


def test_quickstart_run_produces_expected_verdicts(quick: Path) -> None:
    project = load_project(quick / "mindzeval.yaml")
    result = run(project)
    outcome = {(r.case_id, r.turn): (r.verdict, r.cause) for r in result.results}

    assert outcome[("customers_in_germany", 0)] == (Verdict.PASS, None)
    assert outcome[("orders_over_100", 0)] == (Verdict.FAIL, "extra_rows")
    assert outcome[("completed_orders", 0)] == (Verdict.FAIL, "truncated")
    assert outcome[("signups_2024", 0)] == (Verdict.FAIL, "asked_instead_of_answering")
    assert outcome[("emails_of_german_customers", 0)] == (Verdict.FAIL, "safety_violation")
    assert outcome[("customers_with_orders", 0)] == (Verdict.FAIL, "wrong_value")
    assert all(outcome[("germany_then_recent_then_count", i)][0] is Verdict.PASS for i in range(3))
    assert (
        outcome[("too_vague", 0)][0] is Verdict.PASS
        and outcome[("out_of_scope", 0)][0] is Verdict.PASS
    )

    rollup = build_rollup(result)
    assert rollup["execution"]["correct_but_capped_cases"] == ["completed_orders"]
    assert rollup["evaluator"] == {"errors": 0, "blocked_turns": 0, "validation_warnings": 1}
    flawed = next(r for r in result.results if r.case_id == "customers_with_orders")
    assert [f.check for f in flawed.findings] == ["orphans"]


def test_reports_are_written(quick: Path, tmp_path: Path) -> None:
    project = load_project(quick / "mindzeval.yaml")
    result = run(project, cases=select_cases(project.cases, tags=["smoke"]))
    written = write_reports(result, output_dir=tmp_path / "out", formats=["csv", "html"])
    assert set(written) == {"csv", "json", "html"}

    rows = list(csv.DictReader(written["csv"].open(encoding="utf-8")))
    assert {r["case_id"] for r in rows} == {
        "customers_in_germany",
        "customers_in_france_count",
        "top_3_customers_by_spend",
    }
    document = json.loads(written["json"].read_text(encoding="utf-8"))
    assert document["totals"]["turns"] == 3 and len(document["turns"]) == 3
    assert document["metadata"]["mindzeval_version"]
    assert "<html" in written["html"].read_text(encoding="utf-8")


def test_matrix_run_passes_params_and_compares(write: Write, shop_db: Path) -> None:
    write(
        "agent_mod.py",
        "def answer(question, params):\n"
        "    country = 'DE' if params['model'] == 'good' else 'FR'\n"
        "    return {'answer': 'ok', 'sql': f\"SELECT id FROM customers WHERE country = '{country}'\", 'status': 'answer'}\n",
    )
    write(
        "cases.yaml",
        "- id: de\n  question: Who is in DE?\n"
        "  reference_sql: SELECT id FROM customers WHERE country = 'DE'\n  key: id\n",
    )
    config = write(
        "mindzeval.yaml",
        f"agent: {{type: python, target: 'agent_mod:answer'}}\n"
        f"database: {{url: 'sqlite:///{shop_db.as_posix()}'}}\n"
        "datasets: cases.yaml\nmatrix: {model: [good, bad]}\nthresholds: {execution_accuracy: 0.9}\n",
    )
    project = load_project(config)
    result = run(project)
    verdicts = {r.combo["model"]: r.verdict for r in result.results}
    assert verdicts == {"good": Verdict.PASS, "bad": Verdict.FAIL}
    rollup = build_rollup(result)
    assert len(rollup["matrix"]["differing_turns"]) == 1
    failures = check_thresholds(result, project.config.thresholds)
    assert len(failures) == 1 and "model=bad" in failures[0]


def test_agent_errors_dont_stop_the_run(write: Write) -> None:
    write(
        "agent_mod2.py",
        "def answer(question):\n    if 'boom' in question:\n        raise RuntimeError('agent bug')\n    return {'answer': 'no', 'status': 'decline'}\n",
    )
    write(
        "cases.yaml",
        "- {id: a, question: boom, expect: {behavior: decline}}\n- {id: b, question: fine, expect: {behavior: decline}}\n",
    )
    config = write(
        "mindzeval.yaml",
        "agent: {type: python, target: 'agent_mod2:answer', retries: 0}\ndatasets: cases.yaml\n",
    )
    result = run(load_project(config))
    by_id = {r.case_id: r for r in result.results}
    assert by_id["a"].cause == "agent_crash" and "agent bug" in by_id["a"].reason
    assert by_id["b"].verdict is Verdict.PASS


def test_custom_scorer_and_its_failures(write: Write) -> None:
    write(
        "checks.py",
        "from mindzeval.scoring.custom import CustomResult\n"
        "def polite(ctx):\n    return CustomResult('polite', 'please' in (ctx.response.answer or ''), reason='needs please')\n"
        "def broken(ctx):\n    raise ValueError('scorer bug')\n",
    )
    write(
        "agent_mod3.py",
        "def answer(question):\n    return {'answer': 'no thanks', 'status': 'decline'}\n",
    )
    write("cases.yaml", "- {id: a, question: q, expect: {behavior: decline}}\n")
    config = write(
        "mindzeval.yaml",
        "agent: {type: python, target: 'agent_mod3:answer'}\ndatasets: cases.yaml\n"
        "scoring: {custom: ['checks:polite', 'checks:broken']}\n",
    )
    result = run(load_project(config)).results[0]
    assert result.cause == "custom_check_failed"
    assert result.evaluator_errors and "scorer bug" in result.evaluator_errors[0]


def test_cli_quickstart_and_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "qs"
    assert main(["quickstart", str(directory)]) == EXIT_OK
    assert main(["quickstart", str(directory)]) != EXIT_OK  # refuses to overwrite
    monkeypatch.chdir(directory)
    assert main(["validate"]) == EXIT_OK
    assert main(["run", "--list"]) == EXIT_OK
    assert main(["run", "-q", "--tag", "smoke"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "execution accuracy" in out and "csv   report" in out


def test_cli_thresholds_fail_the_run(quick: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = quick / "mindzeval.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("pass_rate: 0.5", "pass_rate: 0.99"),
        encoding="utf-8",
    )
    monkeypatch.chdir(quick)
    assert main(["run", "-q"]) == EXIT_FAILED
