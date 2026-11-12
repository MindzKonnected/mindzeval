"""Run an evaluation: validate references, call the agent, score every turn.

Design rules:

* One agent call per turn; every check scores that single response.
* Each case gets its own session; turns of a multi-turn case share it and see
  the conversation so far.
* One failing case never stops the run. Agent failures are scored as failures;
  MindzEval-side problems are recorded as evaluator errors and never counted
  against the agent.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import subprocess
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mindzeval._version import __version__
from mindzeval.adapters import (
    Adapter,
    AgentError,
    AgentRequest,
    Message,
    build_adapter,
    load_target,
)
from mindzeval.db import Database
from mindzeval.errors import ConfigError, Issue
from mindzeval.loading import Project
from mindzeval.results import Finding, TurnResult, decide
from mindzeval.schema.case import Case, CompareMode, Tolerance
from mindzeval.schema.config import Config, OutputsAgentConfig
from mindzeval.scoring.behavior import score_behavior
from mindzeval.scoring.custom import CustomResult, ScoreContext, Scorer, run_scorer
from mindzeval.scoring.execution import score_execution
from mindzeval.scoring.safety import score_safety
from mindzeval.validate import ValidationReport, validate_cases

Progress = Callable[[TurnResult, int, int], None]
OnStart = Callable[[str, int, int, int], None]


@dataclass
class RunResult:
    run_id: str
    name: str
    started_at: str
    finished_at: str
    config: Config
    results: list[TurnResult]
    combos: list[dict[str, Any]]
    validation: ValidationReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def select_cases(
    cases: Sequence[Case],
    *,
    tags: Iterable[str] = (),
    case_ids: Iterable[str] = (),
    limit: int | None = None,
) -> list[Case]:
    """Filter by tag (any of), case id (any of) and a maximum count, keeping file order."""
    wanted_tags = set(tags)
    wanted_ids = set(case_ids)
    selected = [
        c
        for c in cases
        if (not wanted_tags or wanted_tags & set(c.tags)) and (not wanted_ids or c.id in wanted_ids)
    ]
    unknown = wanted_ids - {c.id for c in cases}
    if unknown:
        raise ConfigError([Issue("--case", f"no case with id {sorted(unknown)}")])
    return selected[:limit] if limit else selected


async def run_project(
    project: Project,
    *,
    cases: Sequence[Case] | None = None,
    name: str | None = None,
    adapter: Adapter | None = None,
    database: Database | None = None,
    progress: Progress | None = None,
    on_start: OnStart | None = None,
) -> RunResult:
    config = project.config
    selected = list(cases if cases is not None else project.cases)
    combos = config.matrix_combinations
    started = _now()
    run_id = uuid.uuid4().hex[:12]

    needs_db = any(t.reference_sql for c in selected for t in c.turns)
    own_db = database is None and needs_db
    if needs_db and database is None:
        if config.database is None:
            raise ConfigError(
                [Issue("database", "cases have reference_sql but no database is configured")]
            )
        database = Database(
            config.database.url.get_secret_value(),
            statement_timeout_s=config.database.statement_timeout_s,
            dialect=config.database.dialect,
        )
    own_adapter = adapter is None
    agent = adapter or build_adapter(config)
    scorers = [(target, load_target(target, config.base_dir)) for target in config.scoring.custom]

    try:
        validation = (
            await asyncio.to_thread(validate_cases, selected, database, config.validation)
            if database is not None
            else None
        )
        total = sum(len(c.turns) for c in selected) * len(combos)
        done = 0
        started_count = 0
        results: list[TurnResult] = []
        lock = asyncio.Lock()

        async def report_start(case_id: str, turn: int) -> None:
            nonlocal started_count
            async with lock:
                started_count += 1
                curr = started_count
                if on_start is not None:
                    on_start(case_id, turn, curr, total)

        async def report(result: TurnResult) -> None:
            nonlocal done
            async with lock:
                done += 1
                results.append(result)
                if progress is not None:
                    progress(result, done, total)

        concurrency = (
            8 if isinstance(config.agent, OutputsAgentConfig) else config.agent.concurrency
        )
        semaphore = asyncio.Semaphore(concurrency)
        base_params = (
            {} if isinstance(config.agent, OutputsAgentConfig) else dict(config.agent.params)
        )

        async def one_case(case: Case, combo_index: int, combo: dict[str, Any]) -> None:
            async with semaphore:
                await _run_case(
                    case,
                    combo=combo,
                    params={**base_params, **combo},
                    session_id=f"mz-{run_id}-{combo_index}-{case.id}",
                    agent=agent,
                    db=database,
                    config=config,
                    validation=validation,
                    scorers=scorers,
                    report=report,
                    report_start=report_start,
                )

        await asyncio.gather(
            *(one_case(case, i, combo) for i, combo in enumerate(combos) for case in selected)
        )
    finally:
        if own_adapter:
            await agent.aclose()
        if own_db and database is not None:
            database.close()

    order = {(c.id): i for i, c in enumerate(selected)}
    combo_order = {json.dumps(c, sort_keys=True, default=str): i for i, c in enumerate(combos)}
    results.sort(
        key=lambda r: (
            combo_order.get(json.dumps(r.combo, sort_keys=True, default=str), 0),
            order.get(r.case_id, 0),
            r.turn,
        )
    )
    return RunResult(
        run_id=run_id,
        name=name or config.project or "mindzeval",
        started_at=started,
        finished_at=_now(),
        config=config,
        results=results,
        combos=combos,
        validation=validation,
        metadata=_metadata(config, selected, agent),
    )


async def _run_case(
    case: Case,
    *,
    combo: dict[str, Any],
    params: dict[str, Any],
    session_id: str,
    agent: Adapter,
    db: Database | None,
    config: Config,
    validation: ValidationReport | None,
    scorers: list[tuple[str, Scorer]],
    report: Callable[[TurnResult], Any],
    report_start: Callable[[str, int], Any] | None = None,
) -> None:
    history: list[Message] = []
    for index, turn in enumerate(case.turns):
        check = validation.get(case.id, index) if validation else None
        findings = list(check.findings) if check else []
        if (
            turn.reference_sql is None
            and turn.has_static_expected
            and not any(f.check == "static_expected" for f in findings)
        ):
            findings.append(
                Finding(
                    "static_expected",
                    "warn",
                    "no reference_sql provided; evaluated against static expected result",
                )
            )

        result = TurnResult(
            case_id=case.id,
            turn=index,
            question=turn.question,
            compare=str(turn.compare),
            expected_behavior=turn.expected_behavior,
            category=case.category,
            difficulty=case.difficulty,
            tags=list(case.tags),
            combo=dict(combo),
            findings=findings,
            blocked=bool(check and check.blocked),
            reference_sql=turn.reference_sql,
        )

        if result.blocked and not case.is_multi_turn:
            decide(result, expects_execution=False)
            await report(result)
            return

        if report_start is not None:
            await report_start(case.id, index)

        request = AgentRequest(
            question=turn.question,
            history=list(history),
            session_id=session_id,
            case_id=case.id,
            turn=index,
            params=params,
        )
        started = time.perf_counter()
        try:
            response = await agent.call(request)
        except AgentError as exc:
            result.latency_ms = (time.perf_counter() - started) * 1000
            result.error_kind, result.error_message = str(exc.kind), exc.message
            decide(result, expects_execution=bool(turn.reference_sql or turn.has_static_expected))
            await report(result)
            history.append(Message("user", turn.question))
            continue
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.response = response

        if not result.blocked:
            await asyncio.to_thread(_score, result, case, index, check, db, config, scorers, params)
        decide(
            result,
            expects_execution=bool(turn.reference_sql or turn.has_static_expected)
            and not result.blocked,
        )
        await report(result)

        history.append(Message("user", turn.question))
        if response.answer:
            history.append(Message("assistant", response.answer))


def _score(
    result: TurnResult,
    case: Case,
    index: int,
    check: Any,
    db: Database | None,
    config: Config,
    scorers: list[tuple[str, Scorer]],
    params: dict[str, Any],
) -> None:
    turn = case.turns[index]
    response = result.response
    assert response is not None
    dialect = db.dialect if db is not None else None

    result.safety = score_safety(response, config.safety, dialect=dialect)
    result.behavior = score_behavior(turn, response, enabled=config.scoring.behavior)

    reference = check.result if check is not None else None
    if reference is None and turn.reference_sql is None and turn.has_static_expected:
        reference = turn.build_static_reference()

    if (
        (turn.reference_sql is not None or turn.has_static_expected)
        and turn.compare is not CompareMode.NONE
        and reference is not None
    ):
        tolerance = config.scoring.tolerance.apply(Tolerance())
        if turn.tolerance is not None:
            tolerance = turn.tolerance.apply(tolerance)
        try:
            result.execution = score_execution(
                turn,
                response,
                reference,
                db=db,
                tolerance=tolerance,
                uncapped_replay=config.scoring.uncapped_replay,
            )
        except Exception as exc:
            result.evaluator_errors.append(f"execution scoring failed: {type(exc).__name__}: {exc}")
        else:
            comparison = result.execution.comparison
            if comparison is not None and comparison.code.startswith("reference_"):
                result.blocked = True
                result.findings.append(Finding("reference_shape", "block", comparison.reason))

    context = ScoreContext(
        case=case,
        turn=turn,
        turn_index=index,
        response=response,
        reference=reference,
        params=params,
        execution=result.execution,
    )
    for name, scorer in scorers:
        try:
            outcome = run_scorer(name, scorer, context)
        except Exception as exc:
            result.evaluator_errors.append(
                f"custom scorer {name} failed: {type(exc).__name__}: {exc}"
            )
            continue
        if isinstance(outcome, CustomResult):
            result.custom.append(outcome)


def run(project: Project, **kwargs: Any) -> RunResult:
    """Synchronous wrapper around :func:`run_project`."""
    return asyncio.run(run_project(project, **kwargs))


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _metadata(config: Config, cases: Sequence[Case], agent: Adapter) -> dict[str, Any]:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.model_dump_json().encode())
    return {
        "mindzeval_version": __version__,
        "agent": getattr(agent, "name", "agent"),
        "agent_type": config.agent.type,
        "params": {} if isinstance(config.agent, OutputsAgentConfig) else config.agent.params,
        "dataset_hash": digest.hexdigest()[:16],
        "cases": len(cases),
        "git_commit": _git_commit(config),
    }


def _git_commit(config: Config) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.base_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
