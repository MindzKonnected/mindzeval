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
import re
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
    "answer", "sql", "reference_sql",
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
        "reference_sql": _one_line(r.reference_sql),
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


_SQL_TOKEN = re.compile(
    r"(?P<comment>--[^\n]*|/\*.*?\*/)"
    r"|(?P<string>'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")"
    r"|(?P<number>\b\d+(?:\.\d+)?\b)"
    r"|(?P<keyword>\b(?:SELECT|FROM|WHERE|(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN|ON|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION\s+ALL|UNION|EXISTS|IN|NOT\s+IN|NOT|AND|OR|IS\s+NOT\s+NULL|IS\s+NULL|IS|NULL|LIKE|ILIKE|AS|DISTINCT|CASE|WHEN|THEN|ELSE|END|WITH|FETCH\s+FIRST|FETCH\s+NEXT|ROWS\s+ONLY|ASC|DESC)\b)"
    r"|(?P<func>\b(?:COUNT|SUM|AVG|MIN|MAX|ROUND|COALESCE|DATE|SUBSTR|TRIM|LOWER|UPPER|LENGTH)\b(?=\s*\())",
    re.IGNORECASE | re.DOTALL,
)


def _highlight_sql(sql: str | None) -> str:
    """Format SQL with syntax highlighting tokens, fully HTML-escaped."""
    if not sql or not sql.strip():
        return '<span class="tok-empty">(no query)</span>'
    parts: list[str] = []
    last_end = 0
    for match in _SQL_TOKEN.finditer(sql):
        start, end = match.span()
        if start > last_end:
            parts.append(html.escape(sql[last_end:start]))
        if match.group("comment"):
            parts.append(f'<span class="tok-cmt">{html.escape(match.group(0))}</span>')
        elif match.group("string"):
            parts.append(f'<span class="tok-str">{html.escape(match.group(0))}</span>')
        elif match.group("number"):
            parts.append(f'<span class="tok-num">{html.escape(match.group(0))}</span>')
        elif match.group("keyword"):
            parts.append(f'<span class="tok-kwd">{html.escape(match.group(0).upper())}</span>')
        elif match.group("func"):
            parts.append(f'<span class="tok-fn">{html.escape(match.group(0).upper())}</span>')
        else:
            parts.append(html.escape(match.group(0)))
        last_end = end
    if last_end < len(sql):
        parts.append(html.escape(sql[last_end:]))
    return "".join(parts)


def render_html(run: RunResult, rollup: dict[str, Any]) -> str:
    """Render a comprehensive, self-contained interactive mini-dashboard."""
    esc = html.escape
    t = rollup["totals"]
    h = rollup["headline"]
    e = rollup["execution"]
    c = rollup["cost"]
    ev = rollup["evaluator"]
    b = rollup["behavior"]
    m = run.metadata
    # Build KPI cards HTML
    rescued_badge = (
        f'<div class="card-pill accent">✨ {e["correct_but_capped"]} query rescued by uncapped replay</div>'
        if e.get("correct_but_capped")
        else ""
    )

    total_tokens = (c.get("input_tokens") or 0) + (c.get("output_tokens") or 0)
    cost_val = c.get("cost_usd") or 0.0

    kpi_cards = f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-val {"val-pass" if (h["pass_rate"] or 0) >= 0.8 else "val-fail"}">{_pct(h["pass_rate"])}</div>
        <div class="kpi-sub">{t["passed"]} of {t["scored"]} scored turns passed</div>
        <div class="kpi-bar"><div class="kpi-bar-fill" style="width: {round((h["pass_rate"] or 0) * 100)}%"></div></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Execution Accuracy <button type="button" class="metric-info-btn" onclick="openMetricsModal(event, 'accuracy')" title="What is Execution Accuracy and how is it calculated?">i</button></div>
        <div class="kpi-val">{_pct(h["execution_accuracy"])}</div>
        <div class="kpi-sub">{e["correct"]} of {e["turns"]} query results match</div>
        <div class="kpi-bar"><div class="kpi-bar-fill" style="width: {round((h["execution_accuracy"] or 0) * 100)}%"></div></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Uncapped Accuracy <button type="button" class="metric-info-btn" onclick="openMetricsModal(event, 'accuracy')" title="What is Uncapped Accuracy and how is it calculated?">i</button></div>
        <div class="kpi-val">{_pct(h["uncapped_accuracy"])}</div>
        <div class="kpi-sub">{e["uncapped_compared"]} replayed without display cap</div>
        {rescued_badge}
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Behavior Accuracy</div>
        <div class="kpi-val">{_pct(h["behavior_accuracy"])}</div>
        <div class="kpi-sub">{b["correct"]} of {b["checked"]} behavior checks passed</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Precision / Recall / F1 <button type="button" class="metric-info-btn" onclick="openMetricsModal(event, 'f1')" title="What are Precision, Recall & F1 and how are they calculated?">i</button></div>
        <div class="kpi-val">{_pct(h["f1"])} <span class="kpi-unit">F1</span></div>
        <div class="kpi-sub">P: {_pct(h["precision"])} &middot; R: {_pct(h["recall"])}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Latency & Cost</div>
        <div class="kpi-val">{c["mean_latency_ms"] or 0}<span class="kpi-unit">ms</span></div>
        <div class="kpi-sub">p95: {c["p95_latency_ms"] or 0}ms &middot; {total_tokens} tok &middot; ${cost_val:.4f}</div>
      </div>
    </div>
    """

    # Breakdown slices
    def make_bars(items: dict[str, Any]) -> str:
        lines = []
        for name, data in items.items():
            if name == "(none)" and len(items) == 1:
                continue
            rate = data.get("pass_rate") or 0.0
            pct_text = _pct(data.get("pass_rate"))
            passed = data.get("passed", 0)
            total = data.get("total", 0)
            color_class = "pass" if rate >= 0.8 else ("warn" if rate >= 0.5 else "fail")
            lines.append(
                f'<div class="slice-row">'
                f'  <div class="slice-name">{esc(str(name))}</div>'
                f'  <div class="slice-bar-track"><div class="slice-bar-fill fill-{color_class}" style="width: {round(rate * 100)}%"></div></div>'
                f'  <div class="slice-stat">{passed}/{total} <span class="slice-pct">({pct_text})</span></div>'
                f"</div>"
            )
        return "".join(lines) or '<div class="muted">No data available</div>'

    causes_chips = (
        "".join(
            f'<span class="cause-chip cause-clickable" onclick="filterByCause(\'{esc(cause)}\')" title="Click to filter queries by \'{esc(cause)}\'">'
            f'<span class="cause-name">{esc(cause)}</span><span class="cause-count">{cnt}</span></span>'
            for cause, cnt in rollup["causes"].items()
        )
        or '<span class="muted">None (all passed)</span>'
    )

    # Evaluator health banner
    evaluator_banner = ""
    if ev["errors"] or ev["blocked_turns"] or ev["validation_warnings"]:
        evaluator_banner = f"""
        <div class="alert-banner alert-warning">
          <div class="alert-icon">⚠️</div>
          <div class="alert-body">
            <strong>Golden Validator & Dataset Health Notice:</strong>
            {ev["blocked_turns"]} blocked turn(s), {ev["validation_warnings"]} reference warning(s), {ev["errors"]} evaluator error(s).
            These represent answer key or query drift issues, <em>not</em> agent bugs.
          </div>
        </div>
        """

    # Matrix section if multiple combinations
    matrix_section = ""
    if "matrix" in rollup and len(run.combos) > 1:
        matrix_rows = []
        for combo_label, headline in rollup["matrix"]["combinations"].items():
            matrix_rows.append(
                f"<tr><td><strong>{esc(combo_label)}</strong></td>"
                f"<td>{_pct(headline['pass_rate'])}</td>"
                f"<td>{_pct(headline['execution_accuracy'])}</td>"
                f"<td>{_pct(headline['behavior_accuracy'])}</td>"
                f"<td>{_pct(headline['f1'])}</td></tr>"
            )
        matrix_section = f"""
        <div class="section-card">
          <h3 class="section-title">Matrix Combinations Side-by-Side</h3>
          <div class="table-wrap">
            <table class="data-table">
              <thead><tr><th>Combination</th><th>Pass Rate</th><th>Exec Accuracy</th><th>Behavior Accuracy</th><th>F1</th></tr></thead>
              <tbody>{"".join(matrix_rows)}</tbody>
            </table>
          </div>
          <p class="matrix-diff-note">{len(rollup["matrix"]["differing_turns"])} turn(s) produced differing verdicts between combinations.</p>
        </div>
        """

    # Build Turn Cards
    turn_cards = []
    category_set = set()
    verdict_counts = Counter(r.verdict for r in run.results)

    for r in run.results:
        if r.category:
            category_set.add(r.category)
        resp = r.response
        e_score = r.execution
        c_comp = e_score.comparison if e_score else None
        uncapped_comp = e_score.uncapped if e_score else None

        # Search index content
        search_terms = (
            f"{r.case_id} {r.turn} {r.question} {' '.join(r.tags)} {r.category or ''} "
            f"{r.difficulty or ''} {r.compare} {r.verdict} {r.cause or ''} {r.reason} "
            f"{resp.sql if resp and resp.sql else ''} {r.reference_sql or ''} "
            f"{resp.answer if resp and resp.answer else ''}"
        ).lower()

        # Verdict classes
        v_class = {
            Verdict.PASS: "pass",
            Verdict.FAIL: "fail",
            Verdict.BLOCKED: "blocked",
            Verdict.NOT_SCORED: "not-scored",
        }.get(r.verdict, "not-scored")

        # Tags badges
        tag_pills = "".join(f'<span class="badge badge-tag">{esc(t)}</span>' for t in r.tags)
        cat_badge = f'<span class="badge badge-cat">{esc(r.category)}</span>' if r.category else ""
        diff_badge = (
            f'<span class="badge badge-diff">{esc(r.difficulty)}</span>' if r.difficulty else ""
        )
        mode_badge = f'<span class="badge badge-mode">{esc(r.compare)}</span>'
        lat_badge = (
            f'<span class="badge badge-lat">{round(r.latency_ms)}ms</span>' if r.latency_ms else ""
        )
        combo_badge = (
            f'<span class="badge badge-combo">{esc(r.combo_label)}</span>'
            if len(run.combos) > 1
            else ""
        )

        # Evaluated Queries Side-by-Side
        agent_sql_raw = (resp.sql if resp and resp.sql else "").strip()
        agent_sql_highlighted = _highlight_sql(agent_sql_raw)

        ref_sql_raw = (r.reference_sql or "").strip()
        ref_sql_highlighted = _highlight_sql(ref_sql_raw)

        # SQL parameters
        params_html = ""
        if resp and resp.params:
            params_str = json.dumps(resp.params, default=str)
            params_html = f'<div class="query-params"><span class="query-meta-tag">params:</span> <code>{esc(params_str)}</code></div>'

        # Rows source badge
        source_badge = (
            f'<span class="query-source-tag source-{e_score.rows_source}">{esc(e_score.rows_source)}</span>'
            if e_score and e_score.rows_source
            else ""
        )

        # SQL error banner
        sql_err_html = ""
        if e_score and e_score.sql_error:
            sql_err_html = f'<div class="query-error"><strong>SQL Execution Error:</strong> {esc(e_score.sql_error)}</div>'

        # Uncapped query details if applicable
        uncapped_block = ""
        if e_score and uncapped_comp is not None and agent_sql_raw:
            unc_correct = uncapped_comp.correct
            unc_badge = f'<span class="badge badge-{"pass" if unc_correct else "fail"}">Uncapped: {"MATCH" if unc_correct else "MISMATCH"}</span>'
            uncapped_block = f"""
            <div class="uncapped-box">
              <div class="uncapped-header">
                <strong>Uncapped Replay (display LIMIT removed):</strong>
                {unc_badge}
                <span class="muted">{esc(e_score.uncapped_status)} &middot; {esc(uncapped_comp.reason)}</span>
              </div>
            </div>
            """

        # Data differences & metrics
        diff_html = ""
        if c_comp is not None:
            missing_chips = "".join(
                f'<span class="diff-chip diff-missing">-{esc(m)}</span>' for m in c_comp.missing[:8]
            )
            if c_comp.missing_n > 8:
                missing_chips += (
                    f'<span class="diff-chip diff-more">+{c_comp.missing_n - 8} more missing</span>'
                )

            spurious_chips = "".join(
                f'<span class="diff-chip diff-spurious">+{esc(s)}</span>'
                for s in c_comp.spurious[:8]
            )
            if c_comp.spurious_n > 8:
                spurious_chips += (
                    f'<span class="diff-chip diff-more">+{c_comp.spurious_n - 8} more extra</span>'
                )

            accuracy_str = "100%" if c_comp.correct else "0%"
            accuracy_class = "metric-pass" if c_comp.correct else "metric-fail"
            precision_html = (
                f'<span class="diff-metric">Precision: <strong>{_pct(c_comp.precision)}</strong>'
                f'<button type="button" class="metric-info-btn" onclick="openMetricsModal(event, \'precision\')" title="What is Precision and how is it calculated?">i</button></span>'
                if c_comp.precision is not None
                else ""
            )
            recall_html = (
                f'<span class="diff-metric">Recall: <strong>{_pct(c_comp.recall)}</strong>'
                f'<button type="button" class="metric-info-btn" onclick="openMetricsModal(event, \'recall\')" title="What is Recall and how is it calculated?">i</button></span>'
                if c_comp.recall is not None
                else ""
            )
            f1_html = (
                f'<span class="diff-metric">F1: <strong>{_pct(c_comp.f1)}</strong>'
                f'<button type="button" class="metric-info-btn" onclick="openMetricsModal(event, \'f1\')" title="What is F1 Score and how is it calculated?">i</button></span>'
                if c_comp.f1 is not None
                else ""
            )

            mismatched_row = (
                f'<div class="diff-row"><span class="diff-lbl">Mismatches:</span> <div class="diff-chips"><span class="diff-chip diff-missing">{c_comp.mismatched_n} group value(s) differ</span></div></div>'
                if c_comp.mismatched_n > 0
                else ""
            )

            diff_html = f"""
            <div class="data-diff-box">
              <div class="diff-header">
                <span class="diff-metric">Reference rows: <strong>{c_comp.reference_n}</strong></span>
                <span class="diff-metric">Agent rows: <strong>{c_comp.agent_n}</strong></span>
                <span class="diff-metric">Accuracy: <strong class="{accuracy_class}">{accuracy_str}</strong><button type="button" class="metric-info-btn" onclick="openMetricsModal(event, 'accuracy')" title="What is Accuracy and how is it calculated?">i</button></span>
                {precision_html}
                {recall_html}
                {f1_html}
              </div>
              {f'<div class="diff-row"><span class="diff-lbl">Missing items:</span> <div class="diff-chips">{missing_chips}</div></div>' if missing_chips else ""}
              {f'<div class="diff-row"><span class="diff-lbl">Extra items:</span> <div class="diff-chips">{spurious_chips}</div></div>' if spurious_chips else ""}
              {mismatched_row}
            </div>
            """

        # Answer text
        answer_text = (resp.answer if resp and resp.answer else "").strip()
        answer_html = (
            f"""
        <div class="answer-box">
          <div class="answer-header"><strong>Agent Response Text:</strong></div>
          <div class="answer-content">{esc(answer_text) if answer_text else '<span class="muted">(no text response)</span>'}</div>
        </div>
        """
            if answer_text
            else ""
        )

        # Behavior & Safety Audit Details
        audit_items = []
        if r.behavior and (r.behavior.expected is not None or r.behavior.actual is not None):
            b_chk = r.behavior.behavior_check
            status_text = (
                f"expected {r.behavior.expected or 'none'}, got {r.behavior.actual or 'none'}"
            )
            audit_items.append(
                f'<div class="audit-item"><span class="audit-name">Behavior:</span> '
                f'<span class="badge badge-{"pass" if b_chk and b_chk.passed else "fail"}">{esc(status_text)}</span></div>'
            )
            for chk in r.behavior.checks:
                if chk.name != "behavior":
                    audit_items.append(
                        f'<div class="audit-item"><span class="audit-name">{esc(chk.name)}:</span> {esc(chk.detail)}</div>'
                    )
            for skip in r.behavior.skipped:
                audit_items.append(
                    f'<div class="audit-item muted"><span class="audit-name">Skipped:</span> {esc(skip)}</div>'
                )

        if r.safety and r.safety.checked:
            s_pass = r.safety.passed
            audit_items.append(
                f'<div class="audit-item"><span class="audit-name">Safety:</span> '
                f'<span class="badge badge-{"pass" if s_pass else "fail"}">{"PASSED" if s_pass else "VIOLATIONS"}</span> '
                f"{'; '.join(esc(v.detail) for v in r.safety.violations)}</div>"
            )

        if r.findings:
            findings_text = " ; ".join(f"{esc(f.check)}: {esc(f.message)}" for f in r.findings)
            audit_items.append(
                f'<div class="audit-item audit-warning"><span class="audit-name">Validation:</span> {findings_text}</div>'
            )

        audit_html = (
            f'<div class="audit-box"><div class="audit-header"><strong>Audits & Behavioral Verification:</strong></div>{"".join(audit_items)}</div>'
            if audit_items
            else ""
        )

        # Verdict explanation callout
        cause_html = f'<span class="cause-tag">{esc(r.cause)}</span>' if r.cause else ""
        reason_text = esc(r.reason) if r.reason else "All checks passed"
        rescue_note = ""
        if e_score and e_score.truncated_subset and e_score.uncapped_correct:
            rescue_note = '<div class="rescue-note">✨ <strong>Logic Correct:</strong> UI display limit truncated the query, but uncapped replay matched the reference exactly.</div>'

        card_html = f"""
        <div class="turn-card status-{v_class}" data-verdict="{r.verdict}" data-category="{esc(r.category or "")}" data-combo="{esc(r.combo_label)}" data-search="{esc(search_terms)}">
          <div class="turn-header" onclick="toggleCard(this.parentElement)">
            <div class="turn-header-left">
              <span class="badge badge-verdict badge-{v_class}">{r.verdict}</span>
              <span class="turn-id">{esc(r.case_id)}<span class="turn-num">#{r.turn}</span></span>
              <span class="turn-q">{esc(r.question)}</span>
            </div>
            <div class="turn-header-right">
              {combo_badge}
              {cat_badge}
              {diff_badge}
              {mode_badge}
              {lat_badge}
              {tag_pills}
              <span class="chevron">&#9662;</span>
            </div>
          </div>
          <div class="turn-body">
            <div class="verdict-banner banner-{v_class}">
              <div class="verdict-title">{cause_html} <span>{reason_text}</span></div>
              {rescue_note}
            </div>

            <!-- Evaluated Queries Side-by-Side -->
            <div class="queries-section">
              <div class="query-col">
                <div class="query-card">
                  <div class="query-card-header">
                    <div class="query-card-title">
                      <span class="icon-sql">⚡</span> Agent Query
                      {source_badge}
                    </div>
                    {'<button class="btn-copy" onclick="copyCardSql(this)">Copy SQL</button>' if agent_sql_raw else ""}
                  </div>
                  <pre class="query-code"><code>{agent_sql_highlighted}</code></pre>
                  {params_html}
                  {sql_err_html}
                </div>
              </div>

              <div class="query-col">
                <div class="query-card">
                  <div class="query-card-header">
                    <div class="query-card-title">
                      <span class="icon-ref">🎯</span> Reference Query (Golden Answer)
                      <span class="query-source-tag source-golden">mode: {esc(r.compare)}</span>
                    </div>
                    {'<button class="btn-copy" onclick="copyCardSql(this)">Copy SQL</button>' if ref_sql_raw else ""}
                  </div>
                  <pre class="query-code"><code>{ref_sql_highlighted}</code></pre>
                </div>
              </div>
            </div>

            {uncapped_block}
            {diff_html}
            {answer_html}
            {audit_html}
          </div>
        </div>
        """
        turn_cards.append(card_html)

    category_options = "".join(
        f'<option value="{esc(c)}">{esc(c)}</option>' for c in sorted(category_set)
    )
    combos_options = (
        "".join(
            f'<option value="{esc(c)}">{esc(c)}</option>'
            for c in sorted({r.combo_label for r in run.results})
        )
        if len(run.combos) > 1
        else ""
    )

    combo_filter_html = (
        f'<div class="filter-group"><label>Combo:</label><select id="combo-filter" onchange="filterTurns()"><option value="all">All Combinations</option>{combos_options}</select></div>'
        if combos_options
        else ""
    )

    style = """
    :root {
      --bg-canvas: #090d16;
      --bg-surface: #0f172a;
      --bg-surface-elevated: #162033;
      --bg-surface-hover: #1e293b;
      --bg-code: #070a12;
      --border: rgba(255, 255, 255, 0.08);
      --border-subtle: rgba(255, 255, 255, 0.04);
      --border-focus: rgba(99, 102, 241, 0.5);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --brand: #6366f1;
      --brand-glow: rgba(99, 102, 241, 0.15);
      --pass: #10b981;
      --pass-bg: rgba(16, 185, 129, 0.1);
      --pass-border: rgba(16, 185, 129, 0.25);
      --fail: #f43f5e;
      --fail-bg: rgba(244, 63, 94, 0.1);
      --fail-border: rgba(244, 63, 94, 0.25);
      --blocked: #f59e0b;
      --blocked-bg: rgba(245, 158, 11, 0.1);
      --blocked-border: rgba(245, 158, 11, 0.25);
      --not-scored: #64748b;
      --not-scored-bg: rgba(100, 116, 139, 0.1);
      --not-scored-border: rgba(100, 116, 139, 0.25);
      --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.4);
      --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.5);
    }

    [data-theme="light"] {
      --bg-canvas: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-elevated: #f1f5f9;
      --bg-surface-hover: #e2e8f0;
      --bg-code: #0f172a;
      --border: #e2e8f0;
      --border-subtle: #edf2f7;
      --border-focus: #6366f1;
      --text: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;
      --brand: #4f46e5;
      --brand-glow: rgba(79, 70, 229, 0.1);
      --pass: #059669;
      --pass-bg: #ecfdf5;
      --pass-border: #a7f3d0;
      --fail: #e11d48;
      --fail-bg: #fff1f2;
      --fail-border: #fecdd3;
      --blocked: #d97706;
      --blocked-bg: #fffbeb;
      --blocked-border: #fde68a;
      --not-scored: #475569;
      --not-scored-bg: #f1f5f9;
      --not-scored-border: #cbd5e1;
      --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07);
      --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
      background: var(--bg-canvas);
      color: var(--text);
      line-height: 1.5;
      padding: 0 0 5rem 0;
      transition: background .2s ease, color .2s ease;
    }

    .container { max-width: 1440px; margin: 0 auto; padding: 0 1.5rem; }

    /* Top Nav / Header */
    .top-header {
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(12px);
    }
    .top-header-inner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 70px;
    }
    .brand-group { display: flex; align-items: center; gap: .75rem; }
    .brand-icon {
      width: 38px; height: 38px;
      background: linear-gradient(135deg, #6366f1, #a855f7);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-weight: 800; font-size: 1.2rem;
      box-shadow: 0 4px 12px rgba(99,102,241,0.3);
    }
    .brand-title { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
    .brand-name { color: var(--brand); font-weight: 600; margin-left: .5rem; font-size: 1rem; }

    .header-controls { display: flex; align-items: center; gap: .75rem; }
    .btn-theme {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      color: var(--text);
      padding: .45rem .85rem;
      border-radius: 8px;
      font-size: .85rem;
      cursor: pointer;
      display: flex; align-items: center; gap: .4rem;
      transition: all .15s ease;
    }
    .btn-theme:hover { background: var(--bg-surface-hover); }

    /* Meta Ribbon */
    .meta-ribbon {
      margin-top: 1.5rem;
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-subtle);
    }
    .meta-pills { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; }
    .pill {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: .78rem;
      padding: .25rem .6rem;
      border-radius: 6px;
      display: inline-flex; align-items: center; gap: .35rem;
    }
    .pill strong { color: var(--text); }
    .pill-btn { cursor: pointer; }
    .pill-btn:hover { background: var(--bg-surface-hover); color: var(--text); }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 1rem;
      margin: 1.5rem 0;
    }
    .kpi-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: var(--shadow-sm);
      transition: transform .15s ease, box-shadow .15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
    .kpi-label { font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; color: var(--text-dim); font-weight: 600; margin-bottom: .25rem; }
    .kpi-val { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; color: var(--text); }
    .kpi-unit { font-size: 1rem; font-weight: 600; color: var(--text-dim); }
    .val-pass { color: var(--pass); }
    .val-fail { color: var(--fail); }
    .kpi-sub { font-size: .8rem; color: var(--text-muted); margin-top: .4rem; }
    .kpi-bar { height: 4px; background: var(--bg-surface-elevated); border-radius: 2px; margin-top: .6rem; overflow: hidden; }
    .kpi-bar-fill { height: 100%; background: var(--brand); border-radius: 2px; }
    .card-pill { font-size: .72rem; padding: .15rem .45rem; border-radius: 4px; margin-top: .5rem; display: inline-block; }
    .card-pill.accent { background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); font-weight: 500; }

    /* Alert Banner */
    .alert-banner {
      display: flex; align-items: flex-start; gap: .75rem;
      padding: 1rem 1.25rem; border-radius: 10px; margin: 1.5rem 0;
      font-size: .88rem;
    }
    .alert-warning { background: var(--blocked-bg); border: 1px solid var(--blocked-border); color: #fde68a; }
    [data-theme="light"] .alert-warning { color: #92400e; }
    .alert-icon { font-size: 1.2rem; }

    /* Breakdowns & Slices */
    .breakdowns-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }
    .section-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .section-title { font-size: .92rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); margin-bottom: 1rem; }
    .slice-row { display: flex; align-items: center; gap: .75rem; margin-bottom: .6rem; font-size: .85rem; }
    .slice-name { width: 100px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text); }
    .slice-bar-track { flex: 1; height: 6px; background: var(--bg-surface-elevated); border-radius: 3px; overflow: hidden; }
    .slice-bar-fill { height: 100%; border-radius: 3px; }
    .fill-pass { background: var(--pass); }
    .fill-warn { background: var(--blocked); }
    .fill-fail { background: var(--fail); }
    .slice-stat { width: 85px; text-align: right; font-size: .8rem; color: var(--text-dim); }
    .slice-pct { color: var(--text); font-weight: 600; }

    .causes-wrap { display: flex; flex-wrap: wrap; gap: .5rem; }
    .cause-chip {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: .3rem .6rem;
      font-size: .8rem;
      display: inline-flex; align-items: center; gap: .4rem;
    }
    .cause-name { color: var(--text); }
    .cause-count {
      background: var(--fail-bg); color: var(--fail);
      border: 1px solid var(--fail-border);
      padding: .05rem .35rem; border-radius: 4px; font-weight: 700; font-size: .75rem;
    }

    /* Matrix Table */
    .table-wrap { overflow-x: auto; margin-top: .5rem; }
    .data-table { width: 100%; border-collapse: collapse; font-size: .85rem; text-align: left; }
    .data-table th, .data-table td { padding: .6rem .8rem; border-bottom: 1px solid var(--border); }
    .data-table th { background: var(--bg-surface-elevated); color: var(--text-dim); font-size: .75rem; text-transform: uppercase; font-weight: 600; }
    .matrix-diff-note { font-size: .8rem; color: var(--text-dim); margin-top: .75rem; }

    /* Explorer Controls Toolbar */
    .explorer-header {
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: 1rem; margin: 2.5rem 0 1rem 0;
    }
    .explorer-title-group { display: flex; align-items: center; gap: .75rem; }
    .explorer-title { font-size: 1.25rem; font-weight: 700; }
    .explorer-count { font-size: .85rem; color: var(--text-dim); }

    .toolbar {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: .75rem 1rem;
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: .75rem;
      margin-bottom: 1.25rem;
    }
    .toolbar-left { display: flex; flex-wrap: wrap; align-items: center; gap: .75rem; flex: 1; }
    .search-box {
      position: relative; min-width: 240px; flex: 1; max-width: 380px;
    }
    .search-box input {
      width: 100%;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: .5rem .75rem .5rem 2rem;
      color: var(--text);
      font-size: .85rem;
      outline: none;
      transition: border-color .15s ease;
    }
    .search-box input:focus { border-color: var(--border-focus); }
    .search-icon { position: absolute; left: .65rem; top: .6rem; color: var(--text-dim); font-size: .85rem; }

    .filter-tabs { display: flex; background: var(--bg-surface-elevated); border-radius: 8px; padding: 2px; }
    .filter-tab {
      background: none; border: none; color: var(--text-muted);
      padding: .4rem .75rem; border-radius: 6px; font-size: .8rem; font-weight: 600;
      cursor: pointer; transition: all .15s ease;
    }
    .filter-tab:hover { color: var(--text); }
    .filter-tab.active { background: var(--brand); color: #fff; }

    .filter-select {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      color: var(--text);
      padding: .45rem .75rem;
      border-radius: 8px;
      font-size: .8rem;
      outline: none;
      cursor: pointer;
    }
    .btn-action {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      color: var(--text);
      padding: .45rem .85rem;
      border-radius: 8px;
      font-size: .8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all .15s ease;
    }
    .btn-action:hover { background: var(--bg-surface-hover); }

    /* Turn Cards (Accordions) */
    .turn-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: .75rem;
      overflow: hidden;
      box-shadow: var(--shadow-sm);
      transition: border-color .15s ease;
    }
    .turn-card.status-pass { border-left: 4px solid var(--pass); }
    .turn-card.status-fail { border-left: 4px solid var(--fail); }
    .turn-card.status-blocked { border-left: 4px solid var(--blocked); }
    .turn-card.status-not-scored { border-left: 4px solid var(--not-scored); }

    .turn-header {
      padding: .85rem 1.25rem;
      display: flex; align-items: center; justify-content: space-between;
      gap: 1rem;
      cursor: pointer;
      user-select: none;
      background: var(--bg-surface);
      transition: background .15s ease;
    }
    .turn-header:hover { background: var(--bg-surface-hover); }

    .turn-header-left { display: flex; align-items: center; gap: .75rem; flex: 1; min-width: 0; }
    .turn-id { font-weight: 700; font-size: .88rem; color: var(--text); white-space: nowrap; }
    .turn-num { color: var(--text-dim); font-weight: 500; font-size: .8rem; }
    .turn-q {
      color: var(--text-muted);
      font-size: .88rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex: 1;
    }
    .turn-header-right { display: flex; align-items: center; gap: .5rem; flex-shrink: 0; }

    .chevron { font-size: .75rem; color: var(--text-dim); transition: transform .2s ease; margin-left: .25rem; }
    .turn-card.open .chevron { transform: rotate(180deg); }

    /* Badges */
    .badge {
      display: inline-flex; align-items: center; gap: .25rem;
      padding: .15rem .5rem; border-radius: 6px;
      font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
    }
    .badge-pass { background: var(--pass-bg); color: var(--pass); border: 1px solid var(--pass-border); }
    .badge-fail { background: var(--fail-bg); color: var(--fail); border: 1px solid var(--fail-border); }
    .badge-blocked { background: var(--blocked-bg); color: var(--blocked); border: 1px solid var(--blocked-border); }
    .badge-not-scored { background: var(--not-scored-bg); color: var(--not-scored); border: 1px solid var(--not-scored-border); }

    .badge-mode { background: var(--bg-surface-elevated); color: var(--text-dim); border: 1px solid var(--border); }
    .badge-cat { background: rgba(99,102,241,0.1); color: #818cf8; border: 1px solid rgba(99,102,241,0.25); }
    .badge-diff { background: var(--bg-surface-elevated); color: var(--text-muted); border: 1px solid var(--border); }
    .badge-lat { background: var(--bg-surface-elevated); color: var(--text-dim); border: 1px solid var(--border); }
    .badge-tag { background: var(--bg-surface-elevated); color: var(--text-dim); border: 1px solid var(--border); text-transform: none; }
    .badge-combo { background: rgba(168,85,247,0.1); color: #c084fc; border: 1px solid rgba(168,85,247,0.25); text-transform: none; }

    /* Turn Body */
    .turn-body {
      display: none;
      padding: 1.25rem;
      border-top: 1px solid var(--border);
      background: var(--bg-canvas);
    }
    .turn-card.open .turn-body { display: block; }

    .verdict-banner {
      padding: .85rem 1rem; border-radius: 8px; margin-bottom: 1.25rem; font-size: .88rem;
    }
    .banner-pass { background: var(--pass-bg); border: 1px solid var(--pass-border); color: var(--text); }
    .banner-fail { background: var(--fail-bg); border: 1px solid var(--fail-border); color: var(--text); }
    .banner-blocked { background: var(--blocked-bg); border: 1px solid var(--blocked-border); color: var(--text); }
    .banner-not-scored { background: var(--not-scored-bg); border: 1px solid var(--not-scored-border); color: var(--text); }
    .cause-tag {
      background: var(--fail); color: #fff; font-weight: 700; font-size: .75rem;
      padding: .15rem .45rem; border-radius: 4px; display: inline-block; margin-right: .5rem;
    }
    .rescue-note {
      margin-top: .4rem; padding-top: .4rem; border-top: 1px solid var(--border-subtle);
      color: #38bdf8; font-size: .82rem;
    }

    /* Queries Section */
    .queries-section {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 1rem;
      margin-bottom: 1.25rem;
    }
    .query-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex; flex-direction: column;
    }
    .query-card-header {
      padding: .5rem .85rem;
      background: var(--bg-surface-elevated);
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
      font-size: .8rem;
    }
    .query-card-title { display: flex; align-items: center; gap: .5rem; font-weight: 600; color: var(--text); }
    .query-source-tag { font-size: .7rem; padding: .1rem .4rem; border-radius: 4px; text-transform: uppercase; font-weight: 700; }
    .source-agent { background: rgba(16,185,129,0.15); color: #34d399; }
    .source-replayed { background: rgba(59,130,246,0.15); color: #60a5fa; }
    .source-golden { background: rgba(168,85,247,0.15); color: #c084fc; }

    .btn-copy {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-dim);
      font-size: .75rem;
      padding: .2rem .5rem;
      border-radius: 5px;
      cursor: pointer;
      transition: all .15s ease;
    }
    .btn-copy:hover { color: var(--text); border-color: var(--border-focus); }
    .btn-copy.copied { background: var(--pass-bg); color: var(--pass); border-color: var(--pass-border); }

    .query-code {
      background: var(--bg-code);
      padding: 1rem;
      font-family: ui-monospace, "SFMono-Regular", "SF Mono", Menlo, Consolas, monospace;
      font-size: .83rem;
      color: #e2e8f0;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: break-word;
      flex: 1;
      line-height: 1.5;
    }
    .query-params {
      padding: .5rem 1rem;
      background: var(--bg-surface-elevated);
      border-top: 1px solid var(--border);
      font-size: .78rem;
      color: var(--text-muted);
    }
    .query-meta-tag { color: var(--text-dim); font-weight: 600; }
    .query-error {
      background: var(--fail-bg);
      border-top: 1px solid var(--fail-border);
      color: var(--fail);
      padding: .6rem 1rem;
      font-size: .8rem;
    }

    /* SQL Token Highlighting */
    .tok-kwd { color: #93c5fd; font-weight: 700; }
    .tok-fn { color: #c084fc; font-weight: 600; }
    .tok-str { color: #86efac; }
    .tok-num { color: #fdba74; }
    .tok-cmt { color: #64748b; font-style: italic; }
    .tok-empty { color: #64748b; font-style: italic; }

    /* Uncapped box */
    .uncapped-box {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: .75rem 1rem;
      margin-bottom: 1.25rem;
      font-size: .85rem;
    }
    .uncapped-header { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }

    /* Data Diff Box */
    .data-diff-box {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: .85rem 1rem;
      margin-bottom: 1.25rem;
      font-size: .85rem;
    }
    .diff-header { display: flex; flex-wrap: wrap; gap: 1.5rem; margin-bottom: .6rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: .5rem; }
    .diff-metric { color: var(--text-muted); }
    .diff-metric strong { color: var(--text); }
    .diff-row { display: flex; align-items: center; gap: .75rem; margin-top: .4rem; }
    .diff-lbl { color: var(--text-dim); font-size: .8rem; width: 100px; flex-shrink: 0; }
    .diff-chips { display: flex; flex-wrap: wrap; gap: .35rem; }
    .diff-chip { font-size: .75rem; padding: .15rem .45rem; border-radius: 4px; font-family: monospace; }
    .diff-missing { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .diff-spurious { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
    .diff-more { background: var(--bg-surface-elevated); color: var(--text-dim); border: 1px solid var(--border); font-size: .7rem; }

    /* Answer Box */
    .answer-box {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: .85rem 1rem;
      margin-bottom: 1.25rem;
      font-size: .85rem;
    }
    .answer-header { color: var(--text-dim); font-size: .78rem; text-transform: uppercase; margin-bottom: .35rem; }
    .answer-content { color: var(--text); }

    /* Audit Box */
    .audit-box {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: .85rem 1rem;
      font-size: .82rem;
    }
    .audit-header { color: var(--text-dim); font-size: .75rem; text-transform: uppercase; margin-bottom: .4rem; }
    .audit-item { margin-bottom: .3rem; display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
    .audit-name { color: var(--text-dim); font-weight: 600; }
    .audit-warning { color: #f59e0b; }

    .muted { color: var(--text-dim); }

    /* Metrics Info Buttons */
    .metric-info-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 17px;
      height: 17px;
      border-radius: 50%;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      color: var(--text-dim);
      font-size: 11px;
      font-weight: 700;
      font-style: italic;
      font-family: serif;
      cursor: pointer;
      margin-left: 6px;
      vertical-align: middle;
      padding: 0;
      line-height: 1;
      transition: all 0.15s ease;
    }
    .metric-info-btn:hover {
      background: var(--brand);
      color: #ffffff;
      border-color: var(--brand);
      transform: scale(1.15);
    }
    .btn-guide-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      margin-left: 0.75rem;
    }
    .btn-guide-pill:hover {
      background: var(--brand);
      color: #ffffff;
      border-color: var(--brand);
    }
    .metric-info-btn-inline {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 15px;
      height: 15px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.15);
      font-size: 10px;
      font-weight: 700;
      font-style: italic;
      font-family: serif;
    }
    .metric-pass { color: var(--pass) !important; }
    .metric-fail { color: var(--fail) !important; }

    /* Modal Backdrop & Dialog */
    .modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
      z-index: 2000;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }
    .modal-backdrop.open {
      display: flex;
    }
    .modal-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      width: 100%;
      max-width: 840px;
      max-height: 88vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
      animation: modalSlideUp 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalSlideUp {
      from { transform: translateY(12px) scale(0.98); opacity: 0; }
      to { transform: translateY(0) scale(1); opacity: 1; }
    }
    .modal-header {
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 1rem;
    }
    .modal-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text);
      margin: 0 0 0.25rem 0;
    }
    .modal-subtitle {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin: 0;
      line-height: 1.4;
    }
    .modal-close-btn {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 1.4rem;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      line-height: 1;
      padding: 0;
      transition: all 0.15s ease;
      flex-shrink: 0;
    }
    .modal-close-btn:hover {
      background: var(--fail);
      color: #fff;
      border-color: var(--fail);
    }
    .modal-tabs {
      display: flex;
      gap: 0.4rem;
      padding: 0.75rem 1.5rem;
      border-bottom: 1px solid var(--border-subtle);
      background: var(--bg-canvas);
      overflow-x: auto;
    }
    .modal-tab {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.4rem 0.85rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
    }
    .modal-tab:hover {
      color: var(--text);
      border-color: var(--border-focus);
    }
    .modal-tab.active {
      background: var(--brand);
      color: #ffffff;
      border-color: var(--brand);
    }
    .modal-body {
      padding: 1.5rem;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    .modal-footer {
      padding: 0.85rem 1.5rem;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: flex-end;
      background: var(--bg-surface);
      border-radius: 0 0 16px 16px;
    }
    .btn-modal-close {
      background: var(--brand);
      color: #ffffff;
      border: none;
      padding: 0.5rem 1.25rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s ease;
    }
    .btn-modal-close:hover {
      opacity: 0.9;
    }

    /* Explainer Cards inside Modal */
    .metric-explainer-card {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .explainer-head {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      margin-bottom: 0.85rem;
    }
    .explainer-icon {
      font-size: 1.4rem;
      width: 42px;
      height: 42px;
      background: var(--bg-canvas);
      border: 1px solid var(--border);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .explainer-title {
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text);
      margin: 0;
    }
    .explainer-tagline {
      font-size: 0.82rem;
      color: var(--brand);
      font-weight: 500;
    }
    .explainer-body {
      font-size: 0.88rem;
      line-height: 1.55;
      color: var(--text-muted);
    }
    .explainer-body p {
      margin: 0 0 0.65rem 0;
    }
    .explainer-body ul {
      margin: 0.4rem 0 0.85rem 1.25rem;
      padding: 0;
    }
    .explainer-body li {
      margin-bottom: 0.35rem;
    }
    .formula-box {
      background: var(--bg-canvas);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.65rem 0.9rem;
      margin: 0.75rem 0;
    }
    .formula-label {
      display: block;
      font-size: 0.75rem;
      text-transform: uppercase;
      color: var(--text-dim);
      font-weight: 700;
      letter-spacing: 0.04em;
      margin-bottom: 0.25rem;
    }
    .formula-box code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: var(--text);
      font-size: 0.86rem;
      font-weight: 600;
    }
    .example-box {
      background: rgba(99, 102, 241, 0.08);
      border-left: 3px solid var(--brand);
      padding: 0.85rem 1.1rem;
      border-radius: 0 8px 8px 0;
      margin-top: 0.85rem;
      font-size: 0.85rem;
    }
    .example-title {
      font-weight: 700;
      color: var(--text);
      margin-bottom: 0.35rem;
    }
    .summary-table-wrap {
      overflow-x: auto;
    }
    .metrics-summary-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.83rem;
      background: var(--bg-surface-elevated);
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .metrics-summary-table th {
      background: var(--bg-canvas);
      text-align: left;
      padding: 0.75rem 1rem;
      font-weight: 700;
      color: var(--text);
      border-bottom: 1px solid var(--border);
    }
    .metrics-summary-table td {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-muted);
      vertical-align: top;
      line-height: 1.45;
    }
    .metrics-summary-table tr:last-child td {
      border-bottom: none;
    }

    .cause-clickable {
      cursor: pointer;
      transition: all .15s ease;
    }
    .cause-clickable:hover {
      transform: translateY(-1px);
      box-shadow: var(--shadow-sm);
      border-color: var(--fail);
    }
    .cards-empty-state {
      text-align: center;
      padding: 3.5rem 1.5rem;
      background: var(--bg-surface);
      border: 1px dashed var(--border);
      border-radius: 12px;
      margin: 1rem 0;
    }
    .empty-icon { font-size: 2rem; margin-bottom: .5rem; }
    .empty-title { font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: .25rem; }
    .empty-desc { font-size: .85rem; color: var(--text-muted); margin-bottom: 1rem; }

    /* Responsive adjustments */
    @media (max-width: 768px) {
      .top-header-inner { flex-direction: column; height: auto; padding: 1rem 0; gap: .75rem; }
      .queries-section { grid-template-columns: 1fr; }
      .toolbar { flex-direction: column; align-items: stretch; }
      .search-box { max-width: 100%; }
      .turn-header { flex-direction: column; align-items: flex-start; gap: .5rem; }
      .turn-header-left { width: 100%; }
      .turn-header-right { width: 100%; justify-content: flex-start; }
    }

    /* Print styles for PDF and reports */
    @media print {
      body { background: #fff !important; color: #000 !important; }
      .top-header, .toolbar, .btn-theme, .btn-copy, .btn-action, .btn-guide-pill, .metric-info-btn, .modal-backdrop { display: none !important; }
      .turn-card .turn-body { display: block !important; }
      .turn-card { break-inside: avoid; border: 1px solid #ccc !important; margin-bottom: 1rem !important; }
      .container { max-width: 100% !important; padding: 0 !important; }
      .kpi-card, .section-card, .data-diff-box, .query-card { break-inside: avoid; border: 1px solid #ddd !important; }
    }
    """

    script = """
    function toggleTheme() {
      const doc = document.documentElement;
      const current = doc.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      doc.setAttribute('data-theme', next);
      localStorage.setItem('mindzeval-theme', next);
    }

    (function initTheme() {
      const saved = localStorage.getItem('mindzeval-theme');
      if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
      } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        document.documentElement.setAttribute('data-theme', 'light');
      } else {
        document.documentElement.setAttribute('data-theme', 'dark');
      }
    })();

    function toggleCard(card) {
      card.classList.toggle('open');
    }

    function toggleAll(expand) {
      document.querySelectorAll('.turn-card').forEach(card => {
        if (expand) card.classList.add('open');
        else card.classList.remove('open');
      });
    }

    function setVerdictFilter(btn) {
      document.querySelectorAll('.filter-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterTurns();
    }

    function filterTurns() {
      const query = document.getElementById('search-input').value.toLowerCase().trim();
      const activeTab = document.querySelector('.filter-tab.active');
      const verdictFilter = activeTab ? activeTab.dataset.verdict : 'ALL';
      const catEl = document.getElementById('cat-filter');
      const categoryFilter = catEl ? catEl.value : 'all';
      const comboEl = document.getElementById('combo-filter');
      const comboFilter = comboEl ? comboEl.value : 'all';

      let visible = 0;
      document.querySelectorAll('.turn-card').forEach(card => {
        const v = card.dataset.verdict;
        const cat = card.dataset.category;
        const combo = card.dataset.combo;
        const search = card.dataset.search;

        const matchVerdict = (verdictFilter === 'ALL' || v === verdictFilter);
        const matchCat = (categoryFilter === 'all' || cat === categoryFilter);
        const matchCombo = (comboFilter === 'all' || combo === comboFilter);
        const matchQuery = (!query || search.includes(query));

        if (matchVerdict && matchCat && matchCombo && matchQuery) {
          card.style.display = '';
          visible++;
        } else {
          card.style.display = 'none';
        }
      });
      document.getElementById('visible-count').textContent = visible;
      const emptyEl = document.getElementById('cards-empty');
      if (emptyEl) {
        emptyEl.style.display = (visible === 0) ? 'block' : 'none';
      }
    }

    function copyText(btn, text) {
      function onCopied() {
        const orig = btn.innerHTML;
        btn.innerHTML = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.innerHTML = orig;
          btn.classList.remove('copied');
        }, 1500);
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(onCopied).catch(() => fallbackCopy(text, onCopied));
      } else {
        fallbackCopy(text, onCopied);
      }
    }

    function fallbackCopy(text, cb) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.top = '0';
      ta.style.left = '0';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      try {
        document.execCommand('copy');
        if (cb) cb();
      } catch (err) {
        console.warn('Copy failed', err);
      }
      document.body.removeChild(ta);
    }

    function copyCardSql(btn) {
      const card = btn.closest('.query-card');
      const codeEl = card ? card.querySelector('.query-code code') : null;
      if (!codeEl) return;
      copyText(btn, codeEl.innerText || codeEl.textContent);
    }

    function copyRunId(btn, text) {
      copyText(btn, text);
    }

    function filterByCause(cause) {
      const searchInput = document.getElementById('search-input');
      if (searchInput) {
        searchInput.value = cause;
        const activeTab = document.querySelector('.filter-tab.active');
        if (activeTab && activeTab.dataset.verdict === 'PASS') {
          const allTab = document.querySelector('.filter-tab[data-verdict="ALL"]');
          if (allTab) {
            document.querySelectorAll('.filter-tab').forEach(b => b.classList.remove('active'));
            allTab.classList.add('active');
          }
        }
        filterTurns();
        const explorer = document.querySelector('.explorer-header');
        if (explorer) explorer.scrollIntoView({ behavior: 'smooth' });
      }
    }

    function resetFilters() {
      const searchInput = document.getElementById('search-input');
      if (searchInput) searchInput.value = '';
      const allTab = document.querySelector('.filter-tab[data-verdict="ALL"]');
      if (allTab) {
        document.querySelectorAll('.filter-tab').forEach(b => b.classList.remove('active'));
        allTab.classList.add('active');
      }
      const catEl = document.getElementById('cat-filter');
      if (catEl) catEl.value = 'all';
      const comboEl = document.getElementById('combo-filter');
      if (comboEl) comboEl.value = 'all';
      filterTurns();
    }

    function openMetricsModal(event, tab) {
      if (event) {
        event.stopPropagation();
      }
      const modal = document.getElementById('metrics-modal');
      if (modal) {
        modal.classList.add('open');
        document.body.style.overflow = 'hidden';
        switchMetricsTab(tab || 'all');
      }
    }

    function closeMetricsModal(event) {
      if (event) {
        event.stopPropagation();
      }
      const modal = document.getElementById('metrics-modal');
      if (modal) {
        modal.classList.remove('open');
        document.body.style.overflow = '';
      }
    }

    function switchMetricsTab(tabName) {
      document.querySelectorAll('.modal-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tabName);
      });
      const sections = ['summary', 'accuracy', 'precision', 'recall', 'f1'];
      sections.forEach(sec => {
        const el = document.getElementById('section-' + sec);
        if (!el) return;
        if (tabName === 'all') {
          el.style.display = 'block';
        } else {
          el.style.display = (sec === tabName) ? 'block' : 'none';
        }
      });
      const modalBody = document.querySelector('.modal-body');
      if (modalBody) modalBody.scrollTop = 0;
    }

    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        const modal = document.getElementById('metrics-modal');
        if (modal && modal.classList.contains('open')) {
          closeMetricsModal();
        } else if (document.activeElement && document.activeElement.id === 'search-input') {
          document.activeElement.value = '';
          filterTurns();
          document.activeElement.blur();
        }
      }
    });
    """

    metrics_modal_html = """
  <div id="metrics-modal" class="modal-backdrop" onclick="closeMetricsModal(event)" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <div class="modal-card" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div>
          <h2 id="modal-title" class="modal-title">Understanding Evaluation Metrics</h2>
          <p class="modal-subtitle">A simple, non-technical guide to how MindzEval measures AI accuracy, completeness, and data quality.</p>
        </div>
        <button type="button" class="modal-close-btn" onclick="closeMetricsModal(event)" aria-label="Close dialog">&times;</button>
      </div>

      <div class="modal-tabs">
        <button type="button" class="modal-tab active" data-tab="all" onclick="switchMetricsTab('all')">All Metrics</button>
        <button type="button" class="modal-tab" data-tab="accuracy" onclick="switchMetricsTab('accuracy')">🎯 Accuracy</button>
        <button type="button" class="modal-tab" data-tab="precision" onclick="switchMetricsTab('precision')">✨ Precision</button>
        <button type="button" class="modal-tab" data-tab="recall" onclick="switchMetricsTab('recall')">🔍 Recall</button>
        <button type="button" class="modal-tab" data-tab="f1" onclick="switchMetricsTab('f1')">⚖️ F1 Score</button>
      </div>

      <div class="modal-body">
        <!-- Quick Summary Table -->
        <div class="modal-section" id="section-summary">
          <div class="summary-table-wrap">
            <table class="metrics-summary-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>The Simple Question</th>
                  <th>What 100% Means</th>
                  <th>What Hurts the Score</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Accuracy</strong></td>
                  <td>Did the query return the exact right answer?</td>
                  <td>The output table perfectly matches the verified golden answer.</td>
                  <td>Wrong SQL logic, syntax error, or wrong sorting/grouping.</td>
                </tr>
                <tr>
                  <td><strong>Precision</strong></td>
                  <td>Are all returned records clean and relevant?</td>
                  <td>Zero unwanted or extra junk records were returned.</td>
                  <td>Broad filters that bring in unwanted rows (false positives).</td>
                </tr>
                <tr>
                  <td><strong>Recall</strong></td>
                  <td>Did the query miss any required records?</td>
                  <td>Every single record that was supposed to be found was retrieved.</td>
                  <td>Overly strict conditions or wrong joins leaving data behind.</td>
                </tr>
                <tr>
                  <td><strong>F1 Score</strong></td>
                  <td>Is the answer both complete AND clean?</td>
                  <td>Perfect balance: found all answers with zero extra clutter.</td>
                  <td>Drops whenever rows are missing OR extra unwanted rows appear.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- ACCURACY SECTION -->
        <div class="modal-section" id="section-accuracy">
          <div class="metric-explainer-card">
            <div class="explainer-head">
              <span class="explainer-icon">🎯</span>
              <div>
                <h3 class="explainer-title">Accuracy (Execution &amp; Uncapped)</h3>
                <span class="explainer-tagline">"Did the query produce the exact right answer on screen?"</span>
              </div>
            </div>
            <div class="explainer-body">
              <p>
                <strong>What is it?</strong> Accuracy is the simplest pass/fail standard. For an individual question, the answer is either <strong>100% accurate</strong> (it matches what a human database expert verified) or <strong>0%</strong> (something was off).
              </p>
              <div class="formula-box">
                <span class="formula-label">Overall Execution Accuracy formula:</span>
                <code>Execution Accuracy = (Number of Correct Queries / Total Queries Evaluated) &times; 100%</code>
              </div>
              <div class="example-box">
                <div class="example-title">💡 What is "Uncapped Accuracy"? (The Display Limit Rescuer)</div>
                <p>
                  Often, an AI application adds <code>LIMIT 5</code> or <code>LIMIT 10</code> to its SQL query so the table looks neat inside a chat message on a screen.
                </p>
                <p>
                  If the golden answer has 14 rows, but the AI's preview query only displays 5 rows, strict execution accuracy would fail it for "missing 9 rows."
                </p>
                <p>
                  <strong>MindzEval solves this:</strong> It automatically replays the query without the preview limit. If all 14 rows match, MindzEval awards <strong>Uncapped Accuracy</strong>, confirming that the AI's SQL logic was 100% sound.
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- PRECISION SECTION -->
        <div class="modal-section" id="section-precision">
          <div class="metric-explainer-card">
            <div class="explainer-head">
              <span class="explainer-icon">✨</span>
              <div>
                <h3 class="explainer-title">Precision</h3>
                <span class="explainer-tagline">"Out of everything the AI gave you, how much of it was actually right?"</span>
              </div>
            </div>
            <div class="explainer-body">
              <p>
                <strong>What is it?</strong> Precision measures <strong>cleanliness and purity</strong>. It checks whether the AI returned unwanted extra rows or irrelevant data.
              </p>
              <ul>
                <li><strong>100% Precision:</strong> Every single row returned was genuinely supposed to be there. No unwanted noise.</li>
                <li><strong>Low Precision:</strong> The AI included extra rows that don't belong (for instance, including customers from France when you only asked for Germany).</li>
              </ul>
              <div class="formula-box">
                <span class="formula-label">How it is calculated:</span>
                <code>Precision = (Correct Matching Rows Returned / Total Rows Returned by AI) &times; 100%</code>
              </div>
              <div class="example-box">
                <div class="example-title">🍎 Real-World Example:</div>
                <p>
                  You ask: <em>"Which customers are in Germany?"</em><br>
                  The AI returns <strong>5 customer records</strong>: 4 from Germany and 1 from France.<br>
                  Since 4 out of 5 are correct: <strong>Precision = 4 / 5 = 80.0%</strong>. (The French customer is an extra unwanted row).
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- RECALL SECTION -->
        <div class="modal-section" id="section-recall">
          <div class="metric-explainer-card">
            <div class="explainer-head">
              <span class="explainer-icon">🔍</span>
              <div>
                <h3 class="explainer-title">Recall</h3>
                <span class="explainer-tagline">"Did the AI find all the records it was supposed to find?"</span>
              </div>
            </div>
            <div class="explainer-body">
              <p>
                <strong>What is it?</strong> Recall measures <strong>completeness</strong>. It checks whether anything was accidentally missed or left behind.
              </p>
              <ul>
                <li><strong>100% Recall:</strong> Nothing was forgotten. The AI captured every single relevant record in the database.</li>
                <li><strong>Low Recall:</strong> The AI missed some records (for example, its WHERE clause was too strict, or it joined tables incorrectly).</li>
              </ul>
              <div class="formula-box">
                <span class="formula-label">How it is calculated:</span>
                <code>Recall = (Correct Matching Rows Returned / Total Expected Rows in Database) &times; 100%</code>
              </div>
              <div class="example-box">
                <div class="example-title">🍎 Real-World Example:</div>
                <p>
                  There are actually <strong>4 German customers</strong> in the database.<br>
                  The AI returns only <strong>3 customers</strong> (it forgot the 4th one).<br>
                  Since it found 3 out of 4: <strong>Recall = 3 / 4 = 75.0%</strong>. (1 customer was missing).
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- F1 SCORE SECTION -->
        <div class="modal-section" id="section-f1">
          <div class="metric-explainer-card">
            <div class="explainer-head">
              <span class="explainer-icon">⚖️</span>
              <div>
                <h3 class="explainer-title">F1 Score</h3>
                <span class="explainer-tagline">"The balanced quality score combining completeness and cleanliness"</span>
              </div>
            </div>
            <div class="explainer-body">
              <p>
                <strong>Why do we need F1 instead of just Precision or Recall?</strong><br>
                Because an AI could easily "game" Recall by returning the entire database (Recall = 100%, but 99% is useless junk!). Or it could "game" Precision by returning only 1 super-safe customer (Precision = 100%, but it missed the other 50 customers!).
              </p>
              <p>
                The <strong>F1 Score</strong> is the harmonic blend of Precision and Recall. It requires an AI to be <strong>both complete and clean</strong> to score well. If either Precision or Recall drops, F1 drops immediately.
              </p>
              <div class="formula-box">
                <span class="formula-label">How it is calculated:</span>
                <code>F1 = 2 &times; (Precision &times; Recall) / (Precision + Recall)</code>
              </div>
              <div class="example-box">
                <div class="example-title">🎯 The Takeaway:</div>
                <p>
                  Think of <strong>F1</strong> as the overall report card grade for data retrieval:
                  <br>• <strong>100% F1:</strong> Perfect answer. Found all correct rows and zero extra rows.
                  <br>• <strong>80-95% F1:</strong> Strong answer with minor row discrepancies or missing edge-cases.
                  <br>• <strong>Below 70% F1:</strong> The query either missed significant data or returned lots of irrelevant rows.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="modal-footer">
        <button type="button" class="btn-modal-close" onclick="closeMetricsModal(event)">Got it</button>
      </div>
    </div>
  </div>
    """

    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MindzEval Dashboard: {esc(run.name)}</title>
  <style>{style}</style>
</head>
<body>

  <header class="top-header">
    <div class="container top-header-inner">
      <div class="brand-group">
        <div class="brand-icon">⚡</div>
        <div>
          <span class="brand-title">MindzEval</span>
          <span class="brand-name">{esc(run.name)}</span>
        </div>
      </div>
      <div class="header-controls">
        <button class="btn-theme" onclick="toggleTheme()" title="Toggle Dark / Light theme">🌓 Theme</button>
      </div>
    </div>
  </header>

  <main class="container">
    <div class="meta-ribbon">
      <div class="meta-pills">
        <span class="pill pill-btn" onclick="copyRunId(this, '{esc(run.run_id)}')" title="Click to copy Run ID">
          Run <strong>{esc(run.run_id)}</strong> 📋
        </span>
        <span class="pill">⏱️ <strong>{esc(run.started_at)}</strong></span>
        {f'<span class="pill">⏳ <strong>{c["total_seconds"]}s</strong></span>' if c.get("total_seconds") else ""}
        <span class="pill">🤖 Agent: <strong>{esc(str(m.get("agent_type", "python")))}</strong> ({esc(str(m.get("agent", "target")))})</span>
        {f'<span class="pill">🌿 Commit: <strong>{esc(str(m.get("git_commit")))}</strong></span>' if m.get("git_commit") else ""}
        <span class="pill">📦 Cases: <strong>{len(run.results)} turns</strong></span>
        <span class="pill">v{esc(str(m.get("mindzeval_version", "")))}</span>
      </div>
    </div>

    {kpi_cards}
    {evaluator_banner}

    <!-- Slices & Breakdowns -->
    <div class="breakdowns-grid">
      <div class="section-card">
        <h3 class="section-title">Performance by Category</h3>
        {make_bars(rollup.get("by_category", {}))}
      </div>
      <div class="section-card">
        <h3 class="section-title">Performance by Compare Mode</h3>
        {make_bars(rollup.get("by_compare", {}))}
      </div>
      <div class="section-card">
        <h3 class="section-title">Failure Causes Distribution</h3>
        <div class="causes-wrap">
          {causes_chips}
        </div>
      </div>
    </div>

    {matrix_section}

    <!-- Evaluated Cases & Queries Explorer -->
    <div class="explorer-header">
      <div class="explorer-title-group">
        <h2 class="explorer-title">Evaluated Queries & Cases</h2>
        <span class="explorer-count">Showing <span id="visible-count">{len(run.results)}</span> of {len(run.results)} turns</span>
        <button type="button" class="btn-guide-pill" onclick="openMetricsModal(event)" title="Understanding Precision, Accuracy, and F1">
          <span class="metric-info-btn-inline">i</span> Metrics Guide
        </button>
      </div>
      <div class="explorer-actions">
        <button class="btn-action" onclick="toggleAll(true)">Expand All</button>
        <button class="btn-action" onclick="toggleAll(false)">Collapse All</button>
      </div>
    </div>

    <div class="toolbar">
      <div class="toolbar-left">
        <div class="search-box">
          <span class="search-icon">🔍</span>
          <input type="text" id="search-input" oninput="filterTurns()" placeholder="Search questions, case IDs, SQL, tags...">
        </div>
        <div class="filter-tabs">
          <button class="filter-tab active" data-verdict="ALL" onclick="setVerdictFilter(this)">All ({len(run.results)})</button>
          <button class="filter-tab" data-verdict="PASS" onclick="setVerdictFilter(this)">Passed ({verdict_counts[Verdict.PASS]})</button>
          <button class="filter-tab" data-verdict="FAIL" onclick="setVerdictFilter(this)">Failed ({verdict_counts[Verdict.FAIL]})</button>
          {f'<button class="filter-tab" data-verdict="BLOCKED" onclick="setVerdictFilter(this)">Blocked ({verdict_counts[Verdict.BLOCKED]})</button>' if verdict_counts[Verdict.BLOCKED] else ""}
          {f'<button class="filter-tab" data-verdict="NOT_SCORED" onclick="setVerdictFilter(this)">Not Scored ({verdict_counts[Verdict.NOT_SCORED]})</button>' if verdict_counts[Verdict.NOT_SCORED] else ""}
        </div>
      </div>
      <div class="toolbar-right">
        {f'<select id="cat-filter" class="filter-select" onchange="filterTurns()"><option value="all">All Categories</option>{category_options}</select>' if category_options else ""}
        {combo_filter_html}
      </div>
    </div>

    <div id="cards-container">
      {"".join(turn_cards)}
      <div id="cards-empty" class="cards-empty-state" style="display:none">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">No matching test cases found</div>
        <div class="empty-desc">Try clearing your search query or switching your filters.</div>
        <button type="button" class="btn-action" onclick="resetFilters()">Reset All Filters</button>
      </div>
    </div>

  </main>

  {metrics_modal_html}

  <script>{script}</script>
</body>
</html>
"""
