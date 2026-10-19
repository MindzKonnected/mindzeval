"""Turn a run into numbers, files and a console summary.

Two files are always meaningful together:

* a per-turn CSV, one row per question, for finding out *why* a case failed;
* a JSON file with the rollup (headline numbers and slices), run metadata and
  every turn's detail, for trends, CI and uploading to the app later.

An optional HTML report gives a readable summary for people who don't open CSVs.
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mindzeval.results import TurnResult, Verdict
from mindzeval.runner import RunResult
from mindzeval.schema.config import Thresholds

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Rollup
# ---------------------------------------------------------------------------


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.fmean(clean), 4) if clean else None


def summarise(results: list[TurnResult]) -> dict[str, Any]:
    """Headline numbers and slices for one set of turn results."""
    scored = [r for r in results if r.verdict in (Verdict.PASS, Verdict.FAIL)]
    passed = [r for r in scored if r.verdict is Verdict.PASS]

    execution = [r for r in results if r.compare != "none" and not r.blocked]
    exec_correct = [r for r in execution if r.execution is not None and r.execution.correct]
    comparisons = [
        r.execution.comparison for r in execution if r.execution and r.execution.comparison
    ]

    uncapped = [r for r in execution if r.execution and r.execution.uncapped is not None]
    rescued = [
        r
        for r in uncapped
        if r.execution and not r.execution.correct and r.execution.uncapped_correct
    ]
    truncated = [r for r in execution if r.execution and r.execution.truncated]

    behavior = [
        r for r in results if r.behavior is not None and r.behavior.behavior_check is not None
    ]
    behavior_ok = [
        r
        for r in behavior
        if r.behavior and r.behavior.behavior_check and r.behavior.behavior_check.passed
    ]
    behavior_skipped = sum(
        1
        for r in results
        if r.behavior is not None and any(s.startswith("behavior:") for s in r.behavior.skipped)
    )

    safety_checked = [r for r in results if r.safety is not None and r.safety.checked]
    violations = Counter(v.check for r in safety_checked if r.safety for v in r.safety.violations)

    latencies = [r.latency_ms for r in results if r.latency_ms]
    usage = [
        r.response.usage for r in results if r.response is not None and r.response.usage is not None
    ]

    def slice_by(key: str) -> dict[str, Any]:
        groups: dict[str, list[TurnResult]] = defaultdict(list)
        for r in scored:
            values = getattr(r, key)
            for value in values if isinstance(values, list) else [values]:
                groups[str(value) if value else "(none)"].append(r)
        return {
            k: {
                "total": len(v),
                "passed": sum(1 for r in v if r.verdict is Verdict.PASS),
                "pass_rate": _rate(sum(1 for r in v if r.verdict is Verdict.PASS), len(v)),
            }
            for k, v in sorted(groups.items())
        }

    blocked = [r for r in results if r.verdict is Verdict.BLOCKED]
    warnings = sum(1 for r in results if not r.blocked for f in r.findings)
    evaluator_errors = sum(len(r.evaluator_errors) for r in results)

    return {
        "totals": {
            "turns": len(results),
            "scored": len(scored),
            "passed": len(passed),
            "failed": len(scored) - len(passed),
            "not_scored": sum(1 for r in results if r.verdict is Verdict.NOT_SCORED),
            "blocked": len(blocked),
        },
        "headline": {
            "pass_rate": _rate(len(passed), len(scored)),
            "execution_accuracy": _rate(len(exec_correct), len(execution)),
            "uncapped_accuracy": _rate(
                sum(1 for r in uncapped if r.execution and r.execution.uncapped_correct),
                len(uncapped),
            ),
            "behavior_accuracy": _rate(len(behavior_ok), len(behavior)),
            "precision": _mean(c.precision for c in comparisons),
            "recall": _mean(c.recall for c in comparisons),
            "f1": _mean(c.f1 for c in comparisons),
        },
        "execution": {
            "turns": len(execution),
            "correct": len(exec_correct),
            "truncated": len(truncated),
            "truncated_but_correct_so_far": sum(
                1 for r in truncated if r.execution and r.execution.truncated_subset
            ),
            "uncapped_compared": len(uncapped),
            "correct_but_capped": len(rescued),
            "correct_but_capped_cases": sorted({r.case_id for r in rescued}),
        },
        "behavior": {
            "checked": len(behavior),
            "correct": len(behavior_ok),
            "skipped_no_status": behavior_skipped,
        },
        "safety": {
            "turns_checked": len(safety_checked),
            "turns_with_violations": sum(
                1 for r in safety_checked if r.safety and not r.safety.passed
            ),
            "violations": dict(violations),
        },
        "by_category": slice_by("category"),
        "by_difficulty": slice_by("difficulty"),
        "by_compare": slice_by("compare"),
        "by_tag": slice_by("tags"),
        "causes": dict(Counter(r.cause for r in results if r.cause).most_common()),
        "cost": {
            "total_seconds": round(sum(latencies) / 1000, 1) if latencies else 0,
            "mean_latency_ms": round(statistics.fmean(latencies)) if latencies else None,
            "p95_latency_ms": round(_percentile(latencies, 95)) if latencies else None,
            "max_latency_ms": round(max(latencies)) if latencies else None,
            "input_tokens": sum(u.input_tokens or 0 for u in usage) or None,
            "output_tokens": sum(u.output_tokens or 0 for u in usage) or None,
            "cost_usd": round(sum(u.cost_usd or 0 for u in usage), 4) or None,
        },
        "evaluator": {
            "errors": evaluator_errors,
            "blocked_turns": len(blocked),
            "validation_warnings": warnings,
        },
    }


def build_rollup(run: RunResult) -> dict[str, Any]:
    rollup: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.run_id,
        "name": run.name,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "metadata": run.metadata,
        **summarise(run.results),
    }
    if len(run.combos) > 1:
        rollup["matrix"] = _matrix(run)
    return rollup


def _matrix(run: RunResult) -> dict[str, Any]:
    by_combo: dict[str, list[TurnResult]] = defaultdict(list)
    for r in run.results:
        by_combo[r.combo_label].append(r)
    combos = {label: summarise(results)["headline"] for label, results in by_combo.items()}
    verdicts: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for r in run.results:
        verdicts[(r.case_id, r.turn)][r.combo_label] = str(r.verdict)
    differing = [
        {"case_id": case_id, "turn": turn, "verdicts": v}
        for (case_id, turn), v in verdicts.items()
        if len(set(v.values())) > 1
    ]
    return {"combinations": combos, "differing_turns": differing}


def check_thresholds(run: RunResult, thresholds: Thresholds) -> list[str]:
    """Return a message for every threshold that is not met (per matrix combination)."""
    failures: list[str] = []
    groups: dict[str, list[TurnResult]] = defaultdict(list)
    for r in run.results:
        groups[r.combo_label].append(r)
    for label, results in groups.items():
        summary = summarise(results)
        prefix = f"[{label}] " if len(groups) > 1 else ""
        for metric in ("execution_accuracy", "uncapped_accuracy", "behavior_accuracy", "pass_rate"):
            minimum = getattr(thresholds, metric)
            value = summary["headline"][metric]
            if minimum is not None and value is not None and value < minimum:
                failures.append(
                    f"{prefix}{metric} {value:.1%} is below the threshold {minimum:.1%}"
                )
        problems = summary["evaluator"]["errors"] + summary["evaluator"]["blocked_turns"]
        if problems > thresholds.max_evaluator_errors:
            failures.append(
                f"{prefix}{problems} evaluator error(s) or blocked turn(s); "
                f"the limit is {thresholds.max_evaluator_errors}"
            )
    return failures


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (k - lower)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "combo", "case_id", "turn", "category", "difficulty", "tags", "compare",
    "verdict", "cause", "reason", "question",
    "expected_behavior", "status",
    "correct", "reference_n", "agent_n", "precision", "recall", "f1",
    "set_match", "ordered_match", "missing_n", "extra_n", "missing_examples", "extra_examples",
    "rows_source", "truncated", "truncated_subset",
    "uncapped_status", "uncapped_correct", "uncapped_precision", "uncapped_recall",
    "sql_error", "behavior_checks", "safety_violations", "custom_checks",
    "validation", "evaluator_errors", "error_kind",
    "latency_ms", "input_tokens", "output_tokens", "cost_usd",
    "answer", "sql",
]  # fmt: skip


def _one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def _strip_tables(text: str | None) -> str:
    """Drop markdown table lines from an answer so the CSV keeps the prose readable."""
    lines = [
        line for line in (text or "").splitlines()
        if not (line.strip().startswith("|") and line.strip().endswith("|"))
    ]  # fmt: skip
    return _one_line("\n".join(lines))


def turn_row(r: TurnResult) -> dict[str, Any]:
    e = r.execution
    c = e.comparison if e else None
    u = e.uncapped if e else None
    resp = r.response
    usage = resp.usage if resp else None

    def blank(value: Any) -> Any:
        return "" if value is None else value

    return {
        "combo": r.combo_label,
        "case_id": r.case_id,
        "turn": r.turn,
        "category": r.category or "",
        "difficulty": r.difficulty or "",
        "tags": " ".join(r.tags),
        "compare": r.compare,
        "verdict": str(r.verdict),
        "cause": r.cause or "",
        "reason": _one_line(r.reason),
        "question": _one_line(r.question),
        "expected_behavior": blank(r.expected_behavior),
        "status": blank(resp.status if resp else None),
        "correct": blank(c.correct if c else None),
        "reference_n": blank(c.reference_n if c else None),
        "agent_n": blank(c.agent_n if c else None),
        "precision": blank(c.precision if c else None),
        "recall": blank(c.recall if c else None),
        "f1": blank(c.f1 if c else None),
        "set_match": blank(c.set_match if c else None),
        "ordered_match": blank(c.ordered_match if c else None),
        "missing_n": blank(c.missing_n if c else None),
        "extra_n": blank(c.spurious_n if c else None),
        "missing_examples": " ; ".join(c.missing) if c else "",
        "extra_examples": " ; ".join(c.spurious) if c else "",
        "rows_source": e.rows_source if e else "",
        "truncated": blank(e.truncated if e else None),
        "truncated_subset": blank(e.truncated_subset if e else None),
        "uncapped_status": e.uncapped_status if e else "",
        "uncapped_correct": blank(u.correct if u else None),
        "uncapped_precision": blank(u.precision if u else None),
        "uncapped_recall": blank(u.recall if u else None),
        "sql_error": _one_line(e.sql_error if e else None),
        "behavior_checks": " ; ".join(
            f"{'ok' if ch.passed else 'FAIL'} {ch.name}: {ch.detail}"
            for ch in (r.behavior.checks if r.behavior else [])
        )
        + (
            " ; skipped " + " ; ".join(r.behavior.skipped)
            if r.behavior and r.behavior.skipped
            else ""
        ),
        "safety_violations": " ; ".join(
            f"{v.check}: {v.detail}" for v in (r.safety.violations if r.safety else [])
        ),
        "custom_checks": " ; ".join(
            f"{x.name}={'pass' if x.passed else 'fail'}{f' ({x.reason})' if x.reason else ''}"
            for x in r.custom
        ),
        "validation": " ; ".join(f"{f.severity} {f.check}: {f.message}" for f in r.findings),
        "evaluator_errors": " ; ".join(r.evaluator_errors),
        "error_kind": r.error_kind or "",
        "latency_ms": round(r.latency_ms) if r.latency_ms else "",
        "input_tokens": blank(usage.input_tokens if usage else None),
        "output_tokens": blank(usage.output_tokens if usage else None),
        "cost_usd": blank(usage.cost_usd if usage else None),
        "answer": _strip_tables(resp.answer if resp else None),
        "sql": _one_line(resp.sql if resp else None),
    }


def write_reports(run: RunResult, *, output_dir: Path, formats: Iterable[str]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run.name)
    base = output_dir / f"{safe_name}-{stamp}"
    rollup = build_rollup(run)
    written: dict[str, Path] = {}
    formats = set(formats) | {"json"}

    if "csv" in formats:
        path = base.with_suffix(".csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for r in run.results:
                writer.writerow(turn_row(r))
        written["csv"] = path

    path = base.with_suffix(".json")
    document = {**rollup, "turns": [turn_row(r) for r in run.results]}
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    written["json"] = path

    if "html" in formats:
        path = base.with_suffix(".html")
        path.write_text(render_html(run, rollup), encoding="utf-8")
        written["html"] = path
    return written


# ---------------------------------------------------------------------------
# Console and HTML
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def format_summary(rollup: dict[str, Any]) -> str:
    t, h, e, ev = rollup["totals"], rollup["headline"], rollup["execution"], rollup["evaluator"]
    lines = [
        "=" * 64,
        f"MINDZEVAL  {rollup['name']}  (run {rollup['run_id']})",
        "=" * 64,
        f"turns {t['turns']}   scored {t['scored']}   passed {t['passed']}   failed {t['failed']}"
        f"   not scored {t['not_scored']}   blocked {t['blocked']}",
        "",
        f"  pass rate            {_pct(h['pass_rate'])}",
        f"  execution accuracy   {_pct(h['execution_accuracy'])}   ({e['correct']}/{e['turns']})",
    ]
    if e["uncapped_compared"]:
        lines.append(
            f"  uncapped accuracy    {_pct(h['uncapped_accuracy'])}   "
            f"({e['uncapped_compared']} replayed without the display LIMIT)"
        )
    if e["correct_but_capped"]:
        lines.append(
            f"  correct but capped   {e['correct_but_capped']} case(s): the query was right, the display cut it"
        )
    lines += [
        f"  behavior accuracy    {_pct(h['behavior_accuracy'])}"
        + (
            f"   ({rollup['behavior']['skipped_no_status']} skipped: no status)"
            if rollup["behavior"]["skipped_no_status"]
            else ""
        ),
        f"  precision / recall   {_pct(h['precision'])} / {_pct(h['recall'])}   (F1 {_pct(h['f1'])})",
    ]
    s = rollup["safety"]
    if s["turns_checked"]:
        lines.append(
            f"  safety               {s['turns_with_violations']} turn(s) with violations {s['violations'] or ''}"
        )

    for title, key in (
        ("BY CATEGORY", "by_category"),
        ("BY DIFFICULTY", "by_difficulty"),
        ("BY COMPARE MODE", "by_compare"),
    ):
        if rollup[key] and list(rollup[key]) != ["(none)"]:
            lines += ["", title]
            lines += [
                f"  {k:18s} {v['passed']:4d}/{v['total']:<4d} {_pct(v['pass_rate'])}"
                for k, v in rollup[key].items()
            ]
    if rollup["causes"]:
        lines += ["", "FAILURE CAUSES"]
        lines += [f"  {k:30s} {v}" for k, v in rollup["causes"].items()]
    if "matrix" in rollup:
        lines += ["", "MATRIX"]
        for label, headline in rollup["matrix"]["combinations"].items():
            lines.append(
                f"  {label:30s} pass {_pct(headline['pass_rate'])}  exec {_pct(headline['execution_accuracy'])}"
                f"  behavior {_pct(headline['behavior_accuracy'])}"
            )
        lines.append(
            f"  {len(rollup['matrix']['differing_turns'])} turn(s) differ between combinations"
        )
    c = rollup["cost"]
    if c["mean_latency_ms"]:
        lines += [
            "",
            f"LATENCY  mean {c['mean_latency_ms']} ms   p95 {c['p95_latency_ms']} ms   total {c['total_seconds']} s",
        ]
    if ev["errors"] or ev["blocked_turns"] or ev["validation_warnings"]:
        lines += [
            "",
            "EVALUATOR HEALTH (dataset or MindzEval problems, not the agent's)",
            f"  blocked turns        {ev['blocked_turns']}",
            f"  validation warnings  {ev['validation_warnings']}",
            f"  evaluator errors     {ev['errors']}",
        ]
    lines.append("=" * 64)
    return "\n".join(lines)


def render_html(run: RunResult, rollup: dict[str, Any]) -> str:
    esc = html.escape
    h = rollup["headline"]
    cards = "".join(
        f"<div class=card><div class=v>{_pct(h[k])}</div><div class=k>{label}</div></div>"
        for k, label in (
            ("pass_rate", "pass rate"),
            ("execution_accuracy", "execution accuracy"),
            ("uncapped_accuracy", "uncapped accuracy"),
            ("behavior_accuracy", "behavior accuracy"),
        )
    )
    causes = "".join(f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k, v in rollup["causes"].items())
    rows = []
    for r in run.results:
        if r.verdict is Verdict.PASS:
            continue
        resp = r.response
        detail = (
            f"<p><b>Question:</b> {esc(r.question)}</p>"
            f"<p><b>Reason:</b> {esc(r.reason)}</p>"
            + (
                f"<p><b>Answer:</b> {esc(resp.answer or '')}</p><pre>{esc(resp.sql or '')}</pre>"
                if resp
                else ""
            )
            + "".join(f"<p class=warn>{esc(f.check)}: {esc(f.message)}</p>" for f in r.findings)
        )
        rows.append(
            f"<tr class={r.verdict.lower()}><td>{esc(r.case_id)}#{r.turn}</td><td>{esc(r.combo_label)}</td>"
            f"<td>{r.verdict}</td><td>{esc(r.cause or '')}</td>"
            f"<td><details><summary>{esc(_one_line(r.reason)[:120])}</summary>{detail}</details></td></tr>"
        )
    style = (
        "body{font-family:system-ui,Segoe UI,sans-serif;margin:2rem;color:#111}"
        ".card{display:inline-block;border:1px solid #ddd;border-radius:8px;padding:1rem 1.5rem;margin:0 1rem 1rem 0}"
        ".v{font-size:1.8rem;font-weight:700}.k{color:#555}"
        "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #eee;padding:.4rem;text-align:left;vertical-align:top}"
        "tr.fail td:nth-child(3){color:#b91c1c;font-weight:600}tr.blocked td:nth-child(3){color:#92400e}"
        "pre{background:#f6f6f6;padding:.5rem;white-space:pre-wrap}.warn{color:#92400e}"
    )
    return (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        f"<title>MindzEval report: {esc(run.name)}</title><style>{style}</style>"
        f"<h1>MindzEval report: {esc(run.name)}</h1>"
        f"<p>Run {esc(run.run_id)} &middot; {esc(run.started_at)} &middot; "
        f"{rollup['totals']['turns']} turns &middot; MindzEval {esc(str(run.metadata.get('mindzeval_version')))}</p>"
        f"{cards}<h2>Failure causes</h2><table><tr><th>cause</th><th>turns</th></tr>{causes}</table>"
        "<h2>Turns that did not pass</h2><table><tr><th>case</th><th>combination</th><th>verdict</th>"
        f"<th>cause</th><th>details</th></tr>{''.join(rows)}</table></html>"
    )
