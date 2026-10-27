"""Command-line interface.

Commands:
    mindzeval quickstart [DIR]   create a runnable example project
    mindzeval lint               check config and datasets (no connections, no secrets)
    mindzeval validate           check reference queries against the database
    mindzeval run                evaluate the agent and write reports

Exit codes:
    0  success
    1  the evaluation ran but did not meet its thresholds, or validation blocked cases
    2  usage, config or dataset error: nothing was evaluated
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from mindzeval._version import __version__
from mindzeval.errors import ConfigError, MindzEvalError
from mindzeval.loading import Project, load_project
from mindzeval.results import TurnResult, Verdict
from mindzeval.schema.case import Case
from mindzeval.schema.config import HttpAgentConfig, OutputsAgentConfig, PythonAgentConfig

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_output()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.handler(args))
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    except MindzEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindzeval",
        description="Evaluate SQL agents: accuracy, behaviour and safety.",
    )
    parser.add_argument("--version", action="version", version=f"mindzeval {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    quick = commands.add_parser("quickstart", help="create a runnable example project")
    quick.add_argument("directory", nargs="?", default="mindzeval-quickstart")
    quick.add_argument("--force", action="store_true", help="overwrite existing files")
    quick.set_defaults(handler=_cmd_quickstart)

    lint = commands.add_parser(
        "lint",
        help="check the config and dataset files without connecting to anything",
        description="Check mindzeval.yaml and every dataset it references. Does not connect to "
        "the agent or the database, and does not need environment variables.",
    )
    _config_arg(lint)
    lint.set_defaults(handler=_cmd_lint)

    validate = commands.add_parser(
        "validate",
        help="check every reference query against the database",
        description="Run every reference query and report empty results, drift, unstable "
        "results, NULL groups and orphan-inflated answers. Does not call the agent.",
    )
    _config_arg(validate)
    _filter_args(validate)
    validate.set_defaults(handler=_cmd_validate)

    run = commands.add_parser("run", help="evaluate the agent and write reports")
    _config_arg(run)
    _filter_args(run)
    run.add_argument("--name", help="run name used in report file names")
    run.add_argument("--output", help="report directory (default: output.dir in the config)")
    run.add_argument(
        "--list", action="store_true", help="list what would run, without calling the agent"
    )
    run.add_argument("-q", "--quiet", action="store_true", help="don't print a line per turn")
    run.set_defaults(handler=_cmd_run)
    return parser


def _config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", default=None, help="config file (default: ./mindzeval.yaml)"
    )


def _filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tag", action="append", default=[], help="only cases with this tag (repeatable)"
    )
    parser.add_argument(
        "--case", action="append", default=[], help="only this case id (repeatable)"
    )
    parser.add_argument("--limit", type=int, help="at most this many cases")


def _selected(project: Project, args: argparse.Namespace) -> list[Case]:
    from mindzeval.runner import select_cases

    return select_cases(project.cases, tags=args.tag, case_ids=args.case, limit=args.limit)


# ---------------------------------------------------------------------------


def _cmd_quickstart(args: argparse.Namespace) -> int:
    from mindzeval.quickstart import create_quickstart

    directory = Path(args.directory)
    created = create_quickstart(directory, force=args.force)
    print(f"created {len(created)} files in {directory}")
    for path in created:
        print(f"  {path}")
    print(f"\nnext:\n  cd {directory}\n  mindzeval run")
    return EXIT_OK


def _cmd_lint(args: argparse.Namespace) -> int:
    project = load_project(args.config, resolve_env=False)
    config, cases = project.config, project.cases
    turns = sum(len(c.turns) for c in cases)
    multi = sum(1 for c in cases if c.is_multi_turn)
    modes = Counter(str(t.compare) for c in cases for t in c.turns)

    print("config OK")
    print(f"  project  : {config.project or '(unnamed)'}")
    print(f"  agent    : {_describe_agent(project)}")
    print(f"  database : {'configured' if config.database else 'not configured'}")
    print(f"  cases    : {len(cases)} ({turns} turns, {multi} multi-turn)")
    print("  compare  : " + ", ".join(f"{m} {n}" for m, n in sorted(modes.items())))
    combos = len(config.matrix_combinations)
    if combos > 1:
        print(f"  matrix   : {combos} combinations -> {combos * turns} agent calls per run")

    needs_db = any(t.reference_sql for c in cases for t in c.turns)
    if needs_db and config.database is None:
        print("error: cases have reference_sql but no 'database' is configured", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    from mindzeval.db import Database
    from mindzeval.validate import validate_cases

    project = load_project(args.config)
    config = project.config
    cases = _selected(project, args)
    if config.database is None:
        print("error: no 'database' is configured", file=sys.stderr)
        return EXIT_USAGE
    with Database(
        config.database.url.get_secret_value(),
        statement_timeout_s=config.database.statement_timeout_s,
        dialect=config.database.dialect,
    ) as db:
        report = validate_cases(cases, db, config.validation)

    ok = 0
    for case in cases:
        for index, _turn in enumerate(case.turns):
            check = report.get(case.id, index)
            if check is None:
                continue
            label = f"{case.id}" + (f"#{index}" if case.is_multi_turn else "")
            if not check.findings:
                ok += 1
                rows = check.result.row_count if check.result else 0
                print(f"  ok     {label:50s} {rows} rows")
                continue
            for finding in check.findings:
                print(f"  {finding.severity:6s} {label:50s} {finding.check}: {finding.message}")
    blocked = len(report.blocked)
    print(
        f"\n{len(report.turns)} reference queries: {ok} ok, {len(report.warnings)} with warnings, {blocked} blocked"
    )
    return EXIT_FAILED if blocked else EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    from mindzeval.report import build_rollup, check_thresholds, format_summary, write_reports
    from mindzeval.runner import run_project

    project = load_project(args.config)
    config = project.config
    cases = _selected(project, args)
    turns = sum(len(c.turns) for c in cases)
    combos = config.matrix_combinations

    print(f"MindzEval {__version__}  project: {config.project or '(unnamed)'}")
    print(f"agent: {_describe_agent(project)}")
    print(
        f"cases: {len(cases)}  turns: {turns}  combinations: {len(combos)}  agent calls: {turns * len(combos)}"
    )
    if args.list:
        for combo in combos:
            label = ", ".join(f"{k}={v}" for k, v in combo.items())
            if label:
                print(f"\n[{label}]")
            for case in cases:
                for index, turn in enumerate(case.turns):
                    tag = f"#{index}" if case.is_multi_turn else ""
                    print(f"  {case.id + tag:50s} [{turn.compare}] {turn.question[:60]}")
        return EXIT_OK
    if not cases:
        print("no cases selected")
        return EXIT_USAGE
    print("-" * 64)

    def on_start(case_id: str, turn: int, current: int, total: int) -> None:
        if args.quiet:
            return
        label = case_id + (f"#{turn}" if turn else "")
        combo = f" [{label}]" if len(combos) > 1 else ""
        if sys.stdout.isatty():
            sys.stdout.write(f"\r\033[K[{current}/{total}] RUNNING  {label}{combo}...")
            sys.stdout.flush()

    def progress(result: TurnResult, done: int, total: int) -> None:
        if args.quiet:
            return
        mark = {"PASS": "PASS", "FAIL": "FAIL", "NOT_SCORED": "----", "BLOCKED": "BLKD"}[
            str(result.verdict)
        ]
        label = result.case_id + (f"#{result.turn}" if result.turn else "")
        combo = f" [{result.combo_label}]" if len(combos) > 1 else ""
        detail = result.cause or ""
        if result.verdict is not Verdict.PASS:
            detail = f"{detail}: {result.reason}" if detail else result.reason
        if sys.stdout.isatty():
            sys.stdout.write(f"\r\033[K[{done}/{total}] {mark:4s}  {label}{combo}  {detail[:90]}\n")
            sys.stdout.flush()
        else:
            print(f"[{done}/{total}] {mark:4s}  {label}{combo}  {detail[:90]}")

    run = asyncio.run(
        run_project(project, cases=cases, name=args.name, progress=progress, on_start=on_start)
    )
    rollup = build_rollup(run)
    print()
    print(format_summary(rollup))

    output = Path(args.output) if args.output else config.resolve_path(config.output.dir)
    written = write_reports(run, output_dir=output, formats=config.output.formats)
    for kind, path in written.items():
        print(f"{kind:5s} report: {path}")

    failures = check_thresholds(run, config.thresholds)
    if failures:
        print("\nTHRESHOLDS NOT MET")
        for failure in failures:
            print(f"  - {failure}")
        return EXIT_FAILED
    return EXIT_OK


def _describe_agent(project: Project) -> str:
    agent = project.config.agent
    if isinstance(agent, PythonAgentConfig):
        return f"python ({agent.target})"
    if isinstance(agent, HttpAgentConfig):
        return f"http ({agent.method} {agent.url})"
    if isinstance(agent, OutputsAgentConfig):
        return f"pre-generated outputs ({agent.path})"
    return "unknown"  # pragma: no cover


def _utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
