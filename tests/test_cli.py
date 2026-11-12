from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mindzeval import __version__
from mindzeval.cli import EXIT_OK, EXIT_USAGE, main

Write = Callable[[str, str], Path]


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_USAGE
    assert "lint" in capsys.readouterr().out


def test_lint_ok_without_secrets(write: Write, capsys: pytest.CaptureFixture[str]) -> None:
    config = write(
        "mindzeval.yaml",
        """\
agent: {type: http, url: 'https://${HOST}/ask'}
database: {url: 'postgresql://reader:${DB_PASSWORD}@db/shop'}
datasets: cases.yaml
matrix: {model: [a, b]}
""",
    )
    write(
        "cases.yaml",
        "- {id: a, question: q, reference_sql: SELECT id FROM t, key: id}\n"
        "- {id: b, question: vague, expect: {behavior: clarify}}\n",
    )
    assert main(["lint", "-c", str(config)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "config OK" in out
    assert "2 (2 turns, 0 multi-turn)" in out
    assert "2 combinations -> 4 agent calls" in out


def test_lint_reports_problems(write: Write, capsys: pytest.CaptureFixture[str]) -> None:
    config = write("mindzeval.yaml", "agent: {type: python, target: nope}\ndatasets: x.yaml\n")
    assert main(["lint", "-c", str(config)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "agent.target" in err


def test_lint_needs_database_for_reference_sql(
    write: Write, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write("mindzeval.yaml", "agent: {type: python, target: 'a:b'}\ndatasets: c.yaml\n")
    write("c.yaml", "- {id: a, question: q, reference_sql: SELECT 1}\n")
    assert main(["lint", "-c", str(config)]) == EXIT_USAGE
    assert "no 'database'" in capsys.readouterr().err
