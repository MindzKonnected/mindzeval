"""Load and validate the config file and dataset files.

Every problem is reported with its file, line and field, and all problems in a
file are collected before raising, so a user fixes them in one pass rather than
one error at a time.

Supported dataset formats: YAML (``.yaml``/``.yml``), JSON, JSONL and CSV.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr, ValidationError

from mindzeval.db import resolve_sqlite_url
from mindzeval.errors import ConfigError, Issue
from mindzeval.schema.case import Case
from mindzeval.schema.config import Config

CONFIG_FILENAMES = ("mindzeval.yaml", "mindzeval.yml")
DATASET_SUFFIXES = (".yaml", ".yml", ".json", ".jsonl", ".csv")

#: Fields a dataset file's ``defaults`` block may set for every case in the file.
DATASET_DEFAULTS = frozenset({"key", "compare", "tolerance", "tags", "category", "difficulty"})

#: CSV columns and where they go. ``meta.<name>`` columns go to ``metadata``.
CSV_CASE_COLUMNS = frozenset({"id", "tags", "category", "difficulty", "notes"})
CSV_TURN_COLUMNS = frozenset(
    {
        "question",
        "reference_sql",
        "golden_sql",
        "compare",
        "key",
        "expected_rows",
        "expected_value",
        "expected",
        "expected_result",
        "expect_empty",
    }
)
CSV_EXPECT_COLUMNS = {"behavior": "behavior", "behaviour": "behavior", "reason": "reason"}

Path_ = tuple[str | int, ...]
Positions = dict[Path_, tuple[int, int | None]]

_ENV_PATTERN = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_URL_PASSWORD = re.compile(r"^[A-Za-z][\w+.-]*://[^/@:]*:([^@/]*)@")
_SECRET_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "proxy-authorization"})


@dataclass(frozen=True)
class Project:
    """A loaded config and all its cases."""

    config: Config
    cases: list[Case]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_config(start: str | Path | None = None) -> Path:
    """Return ``mindzeval.yaml`` (or ``.yml``) in ``start`` (default: the working directory)."""
    directory = Path(start) if start else Path.cwd()
    for name in CONFIG_FILENAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise ConfigError(
        [Issue(str(directory), f"no config file found (looked for {', '.join(CONFIG_FILENAMES)})")]
    )


def load_config(
    path: str | Path,
    *,
    resolve_env: bool = True,
    environ: Mapping[str, str] | None = None,
) -> Config:
    """Load and validate a config file.

    With ``resolve_env=False``, ``${VAR}`` references are left as written, which
    lets a config be checked on a machine that doesn't have the secrets.
    """
    path = Path(path)
    source = str(path)
    data, positions = _read_structured(path)
    if not isinstance(data, dict):
        raise ConfigError([Issue(source, "the config file must be a mapping of settings")])

    issues = _secrets_written_inline(data, source, positions)
    if resolve_env:
        data = _interpolate(data, (), source, positions, environ or os.environ, issues)
    if issues:
        raise ConfigError(issues)

    try:
        config = Config.model_validate({**data, "base_dir": path.resolve().parent})
    except ValidationError as exc:
        raise ConfigError(_issues_from(exc, source, data, positions)) from None

    if resolve_env and config.database is not None:
        url = resolve_sqlite_url(config.database.url.get_secret_value(), config.base_dir)
        config.database.url = SecretStr(url)
    return config


def load_cases(
    patterns: Iterable[str],
    *,
    base_dir: str | Path = ".",
    default_key: list[str] | None = None,
) -> list[Case]:
    """Load every case from the dataset files matching ``patterns``.

    ``default_key`` fills in ``key`` for ``set`` / ``ordered`` cases that don't
    name one. Case ids must be unique across all files.
    """
    base = Path(base_dir)
    issues: list[Issue] = []
    files: list[Path] = []
    for pattern in patterns:
        matched = _expand(pattern, base)
        if not matched:
            issues.append(Issue(pattern, "no dataset files match this path or pattern"))
        files.extend(p for p in matched if p not in files)

    cases: list[Case] = []
    first_seen: dict[str, str] = {}
    for file in files:
        try:
            loaded = _load_dataset_file(file, default_key)
        except ConfigError as exc:
            issues.extend(exc.issues)
            continue
        for case in loaded:
            if case.id in first_seen:
                issues.append(
                    Issue(
                        case.source or str(file),
                        f"duplicate case id '{case.id}' (first defined at {first_seen[case.id]})",
                    )
                )
                continue
            first_seen[case.id] = case.source or str(file)
            cases.append(case)

    if issues:
        raise ConfigError(issues)
    return cases


def load_project(
    config_path: str | Path | None = None,
    *,
    resolve_env: bool = True,
    environ: Mapping[str, str] | None = None,
) -> Project:
    """Load a config and every case it references. Reports config and dataset issues together."""
    path = Path(config_path) if config_path else find_config()
    config = load_config(path, resolve_env=resolve_env, environ=environ)
    cases = load_cases(config.datasets, base_dir=config.base_dir, default_key=config.scoring.key)
    return Project(config=config, cases=cases)


# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------


def _expand(pattern: str, base: Path) -> list[Path]:
    full = Path(pattern) if Path(pattern).is_absolute() else base / pattern
    if any(ch in pattern for ch in "*?["):
        matches = sorted(Path(p) for p in glob.glob(str(full), recursive=True))
        return [p for p in matches if p.is_file() and p.suffix.lower() in DATASET_SUFFIXES]
    return [full] if full.is_file() else []


def _load_dataset_file(path: Path, default_key: list[str] | None) -> list[Case]:
    source = str(path)
    suffix = path.suffix.lower()
    if suffix not in DATASET_SUFFIXES:
        raise ConfigError(
            [Issue(source, f"unsupported dataset format '{suffix}' (use {DATASET_SUFFIXES})")]
        )

    if suffix == ".csv":
        raw_cases, positions = _read_csv(path)
        defaults: dict[str, Any] = {}
    else:
        data, positions = _read_jsonl(path) if suffix == ".jsonl" else _read_structured(path)
        raw_cases, defaults, positions = _split_defaults(data, source, positions)

    issues: list[Issue] = []
    cases: list[Case] = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            issues.append(_issue_at(source, positions, (index,), "each case must be a mapping"))
            continue
        merged = _apply_defaults(raw, defaults, default_key)
        try:
            case = Case.model_validate(merged)
        except ValidationError as exc:
            label = raw.get("id")
            issues.extend(
                _issues_from(exc, source, merged, positions, prefix=(index,), label=label)
            )
            continue
        line = positions.get((index,), (None, None))[0]
        case._source = f"{source}:{line}" if line else source
        cases.append(case)

    if issues:
        raise ConfigError(issues)
    return cases


def _split_defaults(
    data: Any, source: str, positions: Positions
) -> tuple[list[Any], dict[str, Any], Positions]:
    """Accept either a list of cases or ``{defaults: {...}, cases: [...]}``."""
    if data is None:
        return [], {}, positions
    if isinstance(data, list):
        return data, {}, positions
    if isinstance(data, dict):
        unknown = sorted(set(data) - {"defaults", "cases"})
        if unknown or "cases" not in data:
            raise ConfigError(
                [
                    Issue(
                        source,
                        "a dataset file must be a list of cases, or a mapping with "
                        "'cases' and optional 'defaults'"
                        + (f" (unexpected keys: {unknown})" if unknown else ""),
                        line=1,
                    )
                ]
            )
        defaults = data.get("defaults") or {}
        bad = sorted(set(defaults) - DATASET_DEFAULTS) if isinstance(defaults, dict) else []
        if not isinstance(defaults, dict) or bad:
            raise ConfigError(
                [
                    _issue_at(
                        source,
                        positions,
                        ("defaults",),
                        f"'defaults' may only set {sorted(DATASET_DEFAULTS)}"
                        + (f"; not {bad}" if bad else ""),
                    )
                ]
            )
        cases = data.get("cases") or []
        if not isinstance(cases, list):
            raise ConfigError([_issue_at(source, positions, ("cases",), "'cases' must be a list")])
        # Re-root positions so case paths start at their index, like a plain list.
        shifted = {p[1:]: v for p, v in positions.items() if p[:1] == ("cases",)}
        return cases, defaults, shifted
    raise ConfigError([Issue(source, "a dataset file must contain a list of cases", line=1)])


def _apply_defaults(
    raw: dict[str, Any], defaults: dict[str, Any], default_key: list[str] | None
) -> dict[str, Any]:
    merged = dict(raw)
    for name, value in defaults.items():
        if name == "tags":
            own = merged.get("tags") or []
            own = [own] if isinstance(own, str) else list(own)
            extra = [value] if isinstance(value, str) else list(value)
            merged["tags"] = list(dict.fromkeys([*extra, *own]))
        else:
            merged.setdefault(name, value)

    if default_key:
        turns = merged.get("turns")
        targets: list[Any] = turns if isinstance(turns, list) else [merged]
        for target in targets:
            if not isinstance(target, dict) or target.get("key") or merged.get("key"):
                continue
            mode = target.get("compare") or merged.get("compare")
            if mode in ("set", "ordered"):
                target["key"] = list(default_key)
    return merged


# ---------------------------------------------------------------------------
# Readers that remember positions
# ---------------------------------------------------------------------------


def _read_structured(path: Path) -> tuple[Any, Positions]:
    """Read YAML or JSON, recording the line and column of every key and item."""
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError([Issue(source, "file not found")]) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError([Issue(source, f"cannot read file: {exc}")]) from None

    try:
        node = yaml.compose(text, Loader=yaml.SafeLoader)
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(
            [
                Issue(
                    source,
                    f"invalid {'JSON' if path.suffix.lower() == '.json' else 'YAML'}: {problem}",
                    line=mark.line + 1 if mark else None,
                    column=mark.column + 1 if mark else None,
                )
            ]
        ) from None

    positions: Positions = {}
    if node is not None:
        _record_positions(node, (), positions)
    return data, positions


def _record_positions(node: yaml.Node, path: Path_, positions: Positions) -> None:
    positions.setdefault(path, (node.start_mark.line + 1, node.start_mark.column + 1))
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode):
                child = (*path, key_node.value)
                positions[child] = (key_node.start_mark.line + 1, key_node.start_mark.column + 1)
                _record_positions(value_node, child, positions)
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            _record_positions(item, (*path, index), positions)


def _read_jsonl(path: Path) -> tuple[list[Any], Positions]:
    source = str(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError([Issue(source, f"cannot read file: {exc}")]) from None

    items: list[Any] = []
    positions: Positions = {}
    issues: list[Issue] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(Issue(source, f"invalid JSON: {exc.msg}", line=number, column=exc.colno))
            continue
        positions[(len(items),)] = (number, None)
        items.append(item)
    if issues:
        raise ConfigError(issues)
    return items, positions


def _read_csv(path: Path) -> tuple[list[dict[str, Any]], Positions]:
    """Read cases from CSV. Rows sharing an ``id`` (with a ``turn`` column) form one case."""
    source = str(path)
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ConfigError([Issue(source, f"cannot read file: {exc}")]) from None

    with handle:
        reader = csv.DictReader(handle)
        header = [h.strip() for h in (reader.fieldnames or [])]
        allowed = CSV_CASE_COLUMNS | CSV_TURN_COLUMNS | set(CSV_EXPECT_COLUMNS) | {"turn"}
        unknown = [h for h in header if h not in allowed and not h.startswith("meta.")]
        if "id" not in header or unknown:
            message = (
                "a CSV dataset needs an 'id' column"
                if "id" not in header
                else f"unknown column(s) {unknown}; prefix custom columns with 'meta.'"
            )
            raise ConfigError([Issue(source, message, line=1)])

        grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
        for row in reader:
            clean = {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}
            grouped.setdefault(clean.get("id", ""), []).append((reader.line_num, clean))

    cases: list[dict[str, Any]] = []
    positions: Positions = {}
    issues: list[Issue] = []
    for case_id, rows in grouped.items():
        index = len(cases)
        first_line, first = rows[0]
        case: dict[str, Any] = {"id": case_id}
        for name in CSV_CASE_COLUMNS - {"id"}:
            if first.get(name):
                case[name] = first[name]
        meta = {k[5:]: v for k, v in first.items() if k.startswith("meta.") and v}
        if meta:
            case["metadata"] = meta
        positions[(index,)] = (first_line, None)
        for name in case:
            positions[(index, name)] = (first_line, None)

        has_turn_column = any(r.get("turn") for _, r in rows)
        if len(rows) > 1 and not has_turn_column:
            issues.append(
                Issue(
                    source,
                    f"case id '{case_id}' appears on {len(rows)} rows; add a 'turn' "
                    "column to make it a multi-turn case, or give each row a unique id",
                    line=rows[1][0],
                )
            )
            continue

        turns: list[dict[str, Any]] = []
        ordered = sorted(rows, key=lambda item: _as_int(item[1].get("turn"), item[0]))
        for line, row in ordered:
            turn = _csv_turn(row)
            turns.append(turn)
            for name in turn:
                path_ = (
                    (index, name) if not has_turn_column else (index, "turns", len(turns) - 1, name)
                )
                positions[path_] = (line, None)
            if has_turn_column:
                positions[(index, "turns", len(turns) - 1)] = (line, None)

        if has_turn_column:
            case["turns"] = turns
        else:
            case.update(turns[0])
        cases.append(case)

    if issues:
        raise ConfigError(issues)
    return cases, positions


def _csv_turn(row: dict[str, str]) -> dict[str, Any]:
    turn: dict[str, Any] = {}
    for name in CSV_TURN_COLUMNS:
        value = row.get(name)
        if not value:
            continue
        if name == "key":
            turn["key"] = [k.strip() for k in value.split(",") if k.strip()]
        elif name == "expected_rows":
            turn[name] = _as_int(value, value)
        elif name == "expect_empty":
            turn[name] = value.lower() in ("1", "true", "yes", "y")
        else:
            turn[name] = value
    expect = {target: row[col] for col, target in CSV_EXPECT_COLUMNS.items() if row.get(col)}
    if expect:
        turn["expect"] = expect
    return turn


def _as_int(value: Any, fallback: Any) -> Any:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Environment variables and inline secrets
# ---------------------------------------------------------------------------


def _interpolate(
    value: Any,
    path: Path_,
    source: str,
    positions: Positions,
    environ: Mapping[str, str],
    issues: list[Issue],
) -> Any:
    if isinstance(value, dict):
        return {
            k: _interpolate(v, (*path, k), source, positions, environ, issues)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _interpolate(v, (*path, i), source, positions, environ, issues)
            for i, v in enumerate(value)
        ]
    if not isinstance(value, str) or "$" not in value:
        return value

    def substitute(match: re.Match[str]) -> str:
        if match.group(0) == "$$":
            return "$"
        name, default = match.group(1), match.group(2)
        found = environ.get(name)
        if found:
            return found
        if default is not None:
            return default
        issues.append(
            _issue_at(source, positions, path, f"environment variable '{name}' is not set")
        )
        return match.group(0)

    return _ENV_PATTERN.sub(substitute, value)


def _secrets_written_inline(data: dict[str, Any], source: str, positions: Positions) -> list[Issue]:
    """Refuse credentials written literally into the config file."""
    issues: list[Issue] = []
    hint = "use an environment variable reference like ${NAME} instead of a literal value"

    database = data.get("database")
    if isinstance(database, dict) and isinstance(database.get("url"), str):
        match = _URL_PASSWORD.match(database["url"])
        if match and match.group(1) and "${" not in match.group(1):
            issues.append(
                _issue_at(
                    source,
                    positions,
                    ("database", "url"),
                    f"the database password is written in the config file; {hint}",
                )
            )

    agent = data.get("agent")
    headers = agent.get("headers") if isinstance(agent, dict) else None
    if isinstance(headers, dict):
        for name, value in headers.items():
            if (
                str(name).lower() in _SECRET_HEADERS
                and isinstance(value, str)
                and "${" not in value
            ):
                issues.append(
                    _issue_at(
                        source,
                        positions,
                        ("agent", "headers", name),
                        f"header '{name}' holds a credential; {hint}",
                    )
                )

    judge = data.get("judge")
    api_key = judge.get("api_key") if isinstance(judge, dict) else None
    if isinstance(api_key, str) and "${" not in api_key:
        issues.append(
            _issue_at(source, positions, ("judge", "api_key"), f"the API key is literal; {hint}")
        )
    return issues


# ---------------------------------------------------------------------------
# Mapping validation errors back to file positions
# ---------------------------------------------------------------------------


def _issues_from(
    exc: ValidationError,
    source: str,
    data: Any,
    positions: Positions,
    *,
    prefix: Path_ = (),
    label: Any = None,
) -> list[Issue]:
    issues: list[Issue] = []
    for error in exc.errors(include_url=False):
        found, shown = _locate(data, tuple(error["loc"]))
        message = str(error["msg"]).removeprefix("Value error, ")
        line, column = _nearest(positions, (*prefix, *found))
        field = ".".join(str(p) for p in shown)
        if label is not None:
            field = f"case '{label}'" + (f": {field}" if field else "")
        issues.append(Issue(source, message, path=field, line=line, column=column))
    return issues


def _locate(data: Any, loc: Path_) -> tuple[Path_, Path_]:
    """Follow a pydantic error location through the raw data.

    Returns the part of the path that exists in the file (for line numbers) and
    the path to show the user, with internal elements removed: union tags (the
    value of a ``type`` field) and the implicit ``turns.0`` of single-turn cases.
    """
    found: list[str | int] = []
    shown: list[str | int] = []
    current = data
    parts = list(loc)
    if isinstance(data, dict) and "turns" not in data and parts[:2] == ["turns", 0]:
        parts = parts[2:]

    for part in parts:
        if isinstance(current, dict):
            if part in current:
                found.append(part)
                shown.append(part)
                current = current[part]
                continue
            if part == current.get("type"):
                continue
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            found.append(part)
            shown.append(part)
            current = current[part]
            continue
        shown.append(part)
        current = None
    return tuple(found), tuple(shown)


def _nearest(positions: Positions, path: Path_) -> tuple[int | None, int | None]:
    while True:
        if path in positions:
            return positions[path]
        if not path:
            return None, None
        path = path[:-1]


def _issue_at(source: str, positions: Positions, path: Path_, message: str) -> Issue:
    line, column = _nearest(positions, path)
    return Issue(source, message, path=".".join(str(p) for p in path), line=line, column=column)
