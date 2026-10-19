"""Compare the agent's result with the reference result.

SQL text is never compared. Each mode compares the *data* the two queries
return, under the tolerance policy, and explains any difference.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mindzeval.schema.case import CompareMode, Tolerance
from mindzeval.scoring.values import as_number, normalise, normalise_key, numbers_equal

MAX_EXAMPLES = 10


@dataclass(frozen=True)
class Table:
    """A result set: column names and rows, from either side."""

    columns: tuple[str, ...]
    rows: list[tuple[Any, ...]]

    @classmethod
    def from_dicts(
        cls, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None
    ) -> Table:
        cols = tuple(columns) if columns else tuple(dict.fromkeys(k for r in rows for k in r))
        return cls(cols, [tuple(r.get(c) for c in cols) for r in rows])

    def column_index(self, name: str) -> int | None:
        """Index of ``name``, matched case-insensitively and ignoring a table prefix."""
        wanted = name.lower()
        for i, column in enumerate(self.columns):
            if column.lower() == wanted:
                return i
        for i, column in enumerate(self.columns):
            if column.lower().rsplit(".", 1)[-1] == wanted:
                return i
        return None


@dataclass
class Comparison:
    """The result of comparing one agent result with its reference."""

    mode: CompareMode
    correct: bool = False
    reason: str = ""
    code: str = ""
    """Short machine-readable outcome, e.g. ``match``, ``missing_rows``, ``wrong_order``."""

    reference_n: int = 0
    agent_n: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    set_match: bool | None = None
    ordered_match: bool | None = None
    missing: list[str] = field(default_factory=list)
    """Examples in the reference but not in the agent's result (capped)."""
    spurious: list[str] = field(default_factory=list)
    """Examples in the agent's result but not in the reference (capped)."""
    missing_n: int = 0
    spurious_n: int = 0
    mismatched_n: int = 0
    """Grouped mode: groups present on both sides with different values."""


def compare(
    mode: CompareMode,
    reference: Table,
    agent: Table,
    *,
    key: Sequence[str] | None,
    tolerance: Tolerance,
) -> Comparison:
    """Compare ``agent`` against ``reference`` under ``mode``."""
    if mode in (CompareMode.SET, CompareMode.ORDERED):
        return _compare_keyed(mode, reference, agent, list(key or []), tolerance)
    if mode is CompareMode.SCALAR:
        return _compare_scalar(reference, agent, tolerance)
    if mode is CompareMode.GROUPED:
        return _compare_grouped(reference, agent, tolerance)
    if mode is CompareMode.TABLE:
        return _compare_table(reference, agent, tolerance)
    raise ValueError(f"mode {mode} has nothing to compare")


# ---------------------------------------------------------------------------
# set / ordered
# ---------------------------------------------------------------------------


def _key_indices(table: Table, key: list[str], *, fallback_positional: bool) -> list[int] | None:
    indices = [table.column_index(k) for k in key]
    if all(i is not None for i in indices):
        return [i for i in indices if i is not None]
    if fallback_positional and len(table.columns) >= len(key):
        # A reference query that selects only the identifier may name it differently.
        return list(range(len(key)))
    return None


def _ids(table: Table, indices: list[int], tolerance: Tolerance) -> list[str]:
    """Record identities in order, duplicates dropped (joins may repeat a record)."""
    seen: dict[str, None] = {}
    for row in table.rows:
        values = [row[i] for i in indices]
        if all(v is None for v in values):
            continue
        seen.setdefault(normalise_key(values, tolerance), None)
    return list(seen)


def _compare_keyed(
    mode: CompareMode, reference: Table, agent: Table, key: list[str], tolerance: Tolerance
) -> Comparison:
    result = Comparison(mode=mode)
    ref_idx = _key_indices(reference, key, fallback_positional=True)
    agent_idx = _key_indices(agent, key, fallback_positional=False)
    if ref_idx is None:
        result.code, result.reason = "reference_missing_key", f"the reference has no column {key}"
        return result
    ref_ids = _ids(reference, ref_idx, tolerance)
    result.reference_n = len(ref_ids)
    if agent_idx is None and not agent.rows:
        agent_idx = []  # an empty result has nothing to identify; compare as empty
    if agent_idx is None:
        result.agent_n = len(agent.rows)
        result.code = "missing_columns"
        result.reason = (
            f"the agent's result has no {', '.join(key)} column "
            f"(columns: {', '.join(agent.columns) or 'none'})"
        )
        return result

    agent_ids = _ids(agent, agent_idx, tolerance) if agent_idx else []
    result.agent_n = len(agent_ids)
    _set_metrics(result, ref_ids, agent_ids)

    if mode is CompareMode.ORDERED:
        result.ordered_match = ref_ids == agent_ids
        result.correct = result.ordered_match
        if result.correct:
            result.code, result.reason = "match", f"{len(ref_ids)} records, order matches"
        elif result.set_match:
            position = next(
                i for i, (a, b) in enumerate(zip(ref_ids, agent_ids, strict=True)) if a != b
            )
            result.code = "wrong_order"
            result.reason = (
                f"right records, wrong order (first difference at position {position + 1})"
            )
        else:
            _describe_difference(result)
        return result

    result.correct = bool(result.set_match)
    if result.correct:
        result.code, result.reason = "match", f"{len(ref_ids)} records match"
    else:
        _describe_difference(result)
    return result


def _set_metrics(
    result: Comparison, reference: Sequence[Hashable], agent: Sequence[Hashable]
) -> None:
    ref_set, agent_set = set(reference), set(agent)
    hits = len(ref_set & agent_set)
    if not ref_set and not agent_set:
        result.precision = result.recall = result.f1 = 1.0
    else:
        result.precision = hits / len(agent_set) if agent_set else 0.0
        result.recall = hits / len(ref_set) if ref_set else 0.0
        total = result.precision + result.recall
        result.f1 = 2 * result.precision * result.recall / total if total else 0.0
    missing = ref_set - agent_set
    spurious = agent_set - ref_set
    result.set_match = not missing and not spurious
    result.missing_n, result.spurious_n = len(missing), len(spurious)
    result.missing = [_show(v) for v in _ordered(reference, missing)][:MAX_EXAMPLES]
    result.spurious = [_show(v) for v in _ordered(agent, spurious)][:MAX_EXAMPLES]


def _describe_difference(result: Comparison) -> None:
    m, s = result.missing_n, result.spurious_n
    if m and not s:
        result.code = "missing_rows"
    elif s and not m:
        result.code = "extra_rows"
    else:
        result.code = "wrong_rows"
    result.reason = (
        f"{m} missing, {s} extra (reference {result.reference_n}, agent {result.agent_n})"
    )


# ---------------------------------------------------------------------------
# scalar
# ---------------------------------------------------------------------------


def _scalar_of(table: Table, prefer: str | None) -> tuple[bool, Any]:
    """The single value of a result, if it has one."""
    if not table.rows:
        return False, None
    row = table.rows[0]
    if len(row) == 1:
        return True, row[0]
    if prefer is not None:
        index = table.column_index(prefer)
        if index is not None:
            return True, row[index]
    numbers = [v for v in row if as_number(v) is not None]
    if len(numbers) == 1:
        return True, numbers[0]
    return False, None


def _compare_scalar(reference: Table, agent: Table, tolerance: Tolerance) -> Comparison:
    result = Comparison(mode=CompareMode.SCALAR)
    found_ref, ref_value = _scalar_of(reference, None)
    result.reference_n = 1 if found_ref else 0
    found, value = _scalar_of(agent, reference.columns[0] if reference.columns else None)
    result.agent_n = 1 if found else 0
    if not found:
        result.code = "no_single_value"
        result.reason = (
            "the agent's result has no single value to compare "
            f"({len(agent.rows)} rows, columns: {', '.join(agent.columns) or 'none'})"
        )
        return result
    ref_num, agent_num = as_number(ref_value), as_number(value)
    if ref_num is not None and agent_num is not None:
        result.correct = numbers_equal(ref_num, agent_num, tolerance)
    else:
        result.correct = normalise(ref_value, tolerance) == normalise(value, tolerance)
    result.code = "match" if result.correct else "wrong_value"
    result.reason = f"reference {_show(ref_value)}, agent {_show(value)}"
    return result


# ---------------------------------------------------------------------------
# grouped
# ---------------------------------------------------------------------------


def _grouped_pairs(
    table: Table, group_cols: list[int], metric_col: int, tolerance: Tolerance
) -> dict[Hashable, Any]:
    pairs: dict[Hashable, Any] = {}
    for row in table.rows:
        group = tuple(normalise(row[i], tolerance) for i in group_cols)
        pairs[group if len(group) > 1 else group[0]] = row[metric_col]
    return pairs


def _agent_group_layout(reference: Table, agent: Table) -> tuple[list[int], int] | None:
    """Find the agent's group and metric columns: by name first, then by position."""
    ref_groups = list(reference.columns[:-1])
    by_name = [agent.column_index(c) for c in ref_groups]
    metric_by_name = agent.column_index(reference.columns[-1])
    if all(i is not None for i in by_name) and metric_by_name is not None:
        return [i for i in by_name if i is not None], metric_by_name
    width = len(ref_groups)
    if len(agent.columns) < width + 1:
        return None
    groups = list(range(width))
    # The metric is the last column holding numbers.
    for index in range(len(agent.columns) - 1, width - 1, -1):
        if any(as_number(row[index]) is not None for row in agent.rows):
            return groups, index
    return groups, len(agent.columns) - 1


def _compare_grouped(reference: Table, agent: Table, tolerance: Tolerance) -> Comparison:
    result = Comparison(mode=CompareMode.GROUPED)
    if len(reference.columns) < 2:
        result.code = "reference_shape"
        result.reason = "a grouped reference needs group column(s) followed by a value column"
        return result
    ref_pairs = _grouped_pairs(
        reference, list(range(len(reference.columns) - 1)), len(reference.columns) - 1, tolerance
    )
    result.reference_n = len(ref_pairs)
    layout = _agent_group_layout(reference, agent)
    if layout is None and not agent.rows:
        layout = ([], 0)
    if layout is None:
        result.agent_n = len(agent.rows)
        result.code = "missing_columns"
        result.reason = f"the agent's result has too few columns ({', '.join(agent.columns)})"
        return result
    agent_pairs = _grouped_pairs(agent, layout[0], layout[1], tolerance) if agent.rows else {}
    result.agent_n = len(agent_pairs)
    _set_metrics(result, list(ref_pairs), list(agent_pairs))

    mismatched = []
    for group in ref_pairs.keys() & agent_pairs.keys():
        a, b = ref_pairs[group], agent_pairs[group]
        na, nb = as_number(a), as_number(b)
        same = (
            numbers_equal(na, nb, tolerance)
            if na is not None and nb is not None
            else (normalise(a, tolerance) == normalise(b, tolerance))
        )
        if not same:
            mismatched.append(group)
    result.mismatched_n = len(mismatched)
    result.correct = bool(result.set_match) and not mismatched
    if result.correct:
        result.code, result.reason = "match", f"{len(ref_pairs)} groups match"
    elif result.set_match:
        example = mismatched[0]
        result.code = "wrong_values"
        result.reason = (
            f"same groups, {len(mismatched)} with different values "
            f"(e.g. {_show(example)}: reference {_show(ref_pairs[example])}, "
            f"agent {_show(agent_pairs[example])})"
        )
    else:
        result.code = "wrong_groups"
        result.reason = (
            f"group sets differ: {result.missing_n} missing, {result.spurious_n} extra"
            + (f", {len(mismatched)} with different values" if mismatched else "")
        )
    return result


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def _align_columns(
    reference: Table, agent: Table, tolerance: Tolerance
) -> tuple[list[int], list[str]]:
    """Map each reference column to an agent column: by name, then by identical values."""
    mapping: list[int | None] = [agent.column_index(c) for c in reference.columns]
    used = {i for i in mapping if i is not None}
    for position, index in enumerate(mapping):
        if index is not None:
            continue
        wanted = Counter(normalise(r[position], tolerance) for r in reference.rows)
        for candidate in range(len(agent.columns)):
            if candidate in used:
                continue
            values = Counter(normalise(r[candidate], tolerance) for r in agent.rows)
            if values == wanted or (
                tolerance.duplicates == "ignore" and set(values) == set(wanted)
            ):
                mapping[position] = candidate
                used.add(candidate)
                break
    missing = [reference.columns[i] for i, m in enumerate(mapping) if m is None]
    return [m for m in mapping if m is not None], missing


def _compare_table(reference: Table, agent: Table, tolerance: Tolerance) -> Comparison:
    result = Comparison(mode=CompareMode.TABLE)
    mapping, missing_cols = _align_columns(reference, agent, tolerance)
    if not agent.rows:
        mapping, missing_cols = [], []
    if missing_cols:
        result.reference_n, result.agent_n = len(reference.rows), len(agent.rows)
        result.code = "missing_columns"
        result.reason = f"no agent column matches reference column(s) {missing_cols}"
        return result
    extra = len(agent.columns) - len(mapping) if agent.rows else 0
    if extra and tolerance.extra_columns == "forbid":
        result.reference_n, result.agent_n = len(reference.rows), len(agent.rows)
        names = [c for i, c in enumerate(agent.columns) if i not in mapping]
        result.code = "extra_columns"
        result.reason = f"the agent returned extra column(s) {names} and extra_columns is 'forbid'"
        return result

    ref_rows = [tuple(normalise(v, tolerance) for v in row) for row in reference.rows]
    agent_rows = [tuple(normalise(row[i], tolerance) for i in mapping) for row in agent.rows]
    if tolerance.duplicates == "ignore":
        ref_rows = list(dict.fromkeys(ref_rows))
        agent_rows = list(dict.fromkeys(agent_rows))
    result.reference_n, result.agent_n = len(ref_rows), len(agent_rows)

    ref_count, agent_count = Counter(ref_rows), Counter(agent_rows)
    hits = sum((ref_count & agent_count).values())
    if not ref_rows and not agent_rows:
        result.precision = result.recall = result.f1 = 1.0
    else:
        result.precision = hits / len(agent_rows) if agent_rows else 0.0
        result.recall = hits / len(ref_rows) if ref_rows else 0.0
        total = result.precision + result.recall
        result.f1 = 2 * result.precision * result.recall / total if total else 0.0
    missing = ref_count - agent_count
    spurious = agent_count - ref_count
    result.missing_n, result.spurious_n = sum(missing.values()), sum(spurious.values())
    result.missing = [_show(r) for r in list(missing.elements())[:MAX_EXAMPLES]]
    result.spurious = [_show(r) for r in list(spurious.elements())[:MAX_EXAMPLES]]
    result.set_match = not missing and not spurious

    if tolerance.row_order == "strict":
        result.ordered_match = ref_rows == agent_rows
        result.correct = result.ordered_match
        if result.set_match and not result.ordered_match:
            result.code, result.reason = "wrong_order", "right rows, wrong order"
            return result
    else:
        result.correct = result.set_match

    if result.correct:
        result.code, result.reason = "match", f"{len(ref_rows)} rows match"
    else:
        _describe_difference(result)
    return result


# ---------------------------------------------------------------------------


def _ordered(sequence: Sequence[Hashable], subset: set[Hashable]) -> list[Hashable]:
    return [v for v in sequence if v in subset]


def _show(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\x1f", "|")
    if isinstance(value, tuple):
        return "(" + ", ".join(_show(v) for v in value) + ")"
    return repr(value) if value is None else str(value)
