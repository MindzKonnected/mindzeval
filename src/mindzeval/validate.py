"""Golden validator: check every reference query before any agent is graded against it.

A wrong answer key doesn't just miss bugs, it manufactures them. Each reference
query is run live and checked for:

* ``reference_error``: it fails, or isn't a single read query (always blocks)
* ``empty``: it returns nothing, unless the case says empty is correct
* ``drift``: its row count (or, for scalar cases, its value) no longer matches
  ``expected_rows``
* ``stability``: two runs return different results
* ``null_groups``: a grouped reference has a NULL group
* ``orphans``: the answer changes when rows referencing missing records are
  excluded, i.e. it counts links rather than real records

Each check's severity (block / warn / off) comes from ``validation`` in the config.
The reference result is kept and reused for scoring, so each query runs once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal

import sqlglot
from sqlalchemy import inspect as sa_inspect
from sqlglot import exp

from mindzeval.db import Database, QueryResult
from mindzeval.errors import DatabaseError, SQLGuardError
from mindzeval.results import Finding
from mindzeval.schema.case import Case, CompareMode, Turn
from mindzeval.schema.config import Severity, ValidationConfig, parse_relationship
from mindzeval.sql.guard import referenced_tables

Ref = tuple[str | None, str, str]  # schema, table, column


@dataclass
class TurnValidation:
    case_id: str
    turn: int
    result: QueryResult | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity == Severity.BLOCK for f in self.findings)


@dataclass
class ValidationReport:
    turns: dict[tuple[str, int], TurnValidation] = field(default_factory=dict)

    def get(self, case_id: str, turn: int) -> TurnValidation | None:
        return self.turns.get((case_id, turn))

    @property
    def blocked(self) -> list[TurnValidation]:
        return [t for t in self.turns.values() if t.blocked]

    @property
    def warnings(self) -> list[TurnValidation]:
        return [t for t in self.turns.values() if not t.blocked and t.findings]


def validate_cases(
    cases: Iterable[Case], db: Database, config: ValidationConfig
) -> ValidationReport:
    report = ValidationReport()
    probe = _OrphanProbe(db, config)
    for case in cases:
        for index, turn in enumerate(case.turns):
            if turn.reference_sql is None:
                continue
            report.turns[(case.id, index)] = validate_turn(turn, case.id, index, db, config, probe)
    return report


def validate_turn(
    turn: Turn,
    case_id: str,
    index: int,
    db: Database,
    config: ValidationConfig,
    probe: _OrphanProbe | None = None,
) -> TurnValidation:
    out = TurnValidation(case_id=case_id, turn=index)
    sql = turn.reference_sql or ""
    try:
        result = db.fetch(sql)
    except (SQLGuardError, DatabaseError) as exc:
        out.findings.append(
            Finding("reference_error", Severity.BLOCK, f"reference query failed: {exc}")
        )
        return out
    out.result = result

    def add(check: str, severity: Severity, message: str) -> None:
        if severity is not Severity.OFF:
            out.findings.append(Finding(check, severity, message))

    scalar_value = result.rows[0][0] if result.rows and result.rows[0] else None
    is_empty = result.row_count == 0 or (
        turn.compare is CompareMode.SCALAR and scalar_value is None
    )
    if is_empty and not turn.expect_empty:
        add(
            "empty",
            config.empty,
            "the reference returns no data; set expect_empty if that is correct",
        )
    elif turn.expect_empty and not is_empty:
        add(
            "empty",
            config.empty,
            f"expect_empty is set but the reference returns {result.row_count} rows",
        )

    if turn.expected_rows is not None:
        if turn.compare is CompareMode.SCALAR:
            actual = _as_int(scalar_value)
            label = "value"
        else:
            actual, label = result.row_count, "row count"
        if actual != turn.expected_rows:
            add(
                "drift",
                config.drift,
                f"expected {label} {turn.expected_rows}, the reference now gives {actual}",
            )

    if config.stability is not Severity.OFF:
        try:
            again = db.fetch(sql)
        except (SQLGuardError, DatabaseError) as exc:
            add("stability", config.stability, f"second run failed: {exc}")
        else:
            if _content(again) != _content(result):
                add(
                    "stability",
                    config.stability,
                    "two runs of the reference returned different results",
                )

    if turn.compare is CompareMode.GROUPED and config.null_groups is not Severity.OFF:
        group_width = max(1, len(result.columns) - 1)
        nulls = sum(1 for row in result.rows if any(v is None for v in row[:group_width]))
        if nulls:
            add(
                "null_groups",
                config.null_groups,
                f"{nulls} group(s) have a NULL key; decide whether NULL is a group or filtered out, "
                "and make the reference say so explicitly",
            )

    if probe is not None and config.orphans is not Severity.OFF:
        message = probe.check(sql, result)
        if message:
            add("orphans", config.orphans, message)
    return out


def _projected_columns(sql: str, dialect: str | None) -> set[str]:
    """Column names used in the outermost SELECT list (including inside aggregates)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return set()
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return set()
    return {c.name.lower() for e in select.expressions for c in e.find_all(exp.Column)}


def _content(result: QueryResult) -> list[str]:
    return sorted(repr(row) for row in result.rows)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _OrphanProbe:
    """Find references to missing records and test whether an answer depends on them."""

    def __init__(self, db: Database, config: ValidationConfig) -> None:
        self._db = db
        self._configured = [parse_relationship(r) for r in config.relationships]
        self._fk_cache: dict[tuple[str | None, str], list[tuple[Ref, Ref]]] = {}
        self._orphans: dict[tuple[Ref, Ref], int] = {}

    def check(self, sql: str, result: QueryResult) -> str | None:
        tables = referenced_tables(sql, dialect=self._db.dialect)
        table_names = {t.lower() for _, t in tables}
        projected = _projected_columns(sql, self._db.dialect)
        touched: list[tuple[Ref, Ref, int]] = []
        for schema, table in tables:
            for child, parent in self._relationships(schema, table):
                # Only relevant when the answer is about the *referenced* records:
                # the reference column is selected / counted, or the parent table is joined.
                if child[2].lower() not in projected and parent[1].lower() not in table_names:
                    continue
                count = self._orphan_count(child, parent)
                if count:
                    touched.append((child, parent, count))
        if not touched:
            return None
        rewritten = self._exclude_orphans(sql, [(c, p) for c, p, _ in touched])
        if rewritten is None:
            return None
        try:
            filtered = self._db.fetch(rewritten)
        except (SQLGuardError, DatabaseError):
            return None
        if _content(filtered) == _content(result):
            return None
        detail = ", ".join(f"{_name(c)} ({n} orphaned)" for c, _, n in touched)
        return (
            f"the answer changes when references to missing records are excluded ({detail}); "
            "it may be counting link rows rather than real records. Anchor the query on the "
            "entity table and filter through links with EXISTS or an inner join"
        )

    def _relationships(self, schema: str | None, table: str) -> list[tuple[Ref, Ref]]:
        key = (schema.lower() if schema else None, table.lower())
        if key in self._fk_cache:
            return self._fk_cache[key]
        found: list[tuple[Ref, Ref]] = []
        for child, parent in self._configured:
            if child[1].lower() == key[1] and (child[0] or "").lower() in ("", key[0] or ""):
                found.append((child, parent))
        try:
            inspector = sa_inspect(self._db.engine)
            for fk in inspector.get_foreign_keys(table, schema=schema):
                cols, refs = fk.get("constrained_columns") or [], fk.get("referred_columns") or []
                if len(cols) == 1 and len(refs) == 1 and fk.get("referred_table"):
                    found.append(
                        (
                            (schema, table, cols[0]),
                            (fk.get("referred_schema"), fk["referred_table"], refs[0]),
                        )
                    )
        except Exception:
            pass
        self._fk_cache[key] = found
        return found

    def _orphan_count(self, child: Ref, parent: Ref) -> int:
        if (child, parent) in self._orphans:
            return self._orphans[(child, parent)]
        dialect = self._db.dialect
        c, p = _quote(child, dialect), _quote(parent, dialect)
        sql = (
            f"SELECT count(*) FROM {c[0]} AS _c WHERE _c.{c[1]} IS NOT NULL AND NOT EXISTS "
            f"(SELECT 1 FROM {p[0]} AS _p WHERE _p.{p[1]} = _c.{c[1]})"
        )
        try:
            count = int(self._db.fetch(sql).rows[0][0])
        except (SQLGuardError, DatabaseError, TypeError, ValueError, IndexError):
            count = 0
        self._orphans[(child, parent)] = count
        return count

    def _exclude_orphans(self, sql: str, links: list[tuple[Ref, Ref]]) -> str | None:
        dialect = self._db.dialect
        try:
            tree = sqlglot.parse_one(sql, read=dialect)
        except sqlglot.errors.ParseError:
            return None

        by_table: dict[tuple[str, str], list[tuple[Ref, Ref]]] = {}
        for child, parent in links:
            by_table.setdefault(((child[0] or "").lower(), child[1].lower()), []).append(
                (child, parent)
            )

        def replace(node: exp.Expression) -> exp.Expression:
            if not isinstance(node, exp.Table) or not node.name or node.alias in ("_o", "_p"):
                return node
            matches = by_table.get(((node.db or "").lower(), node.name.lower())) or by_table.get(
                ("", node.name.lower())
            )
            if not matches:
                return node
            alias = node.alias_or_name
            source = exp.table_(node.name, db=node.db or None, quoted=True).sql(dialect=dialect)
            conditions = []
            for child, parent in matches:
                p = _quote(parent, dialect)
                col = _ident(child[2], dialect)
                conditions.append(
                    f"(_o.{col} IS NULL OR EXISTS (SELECT 1 FROM {p[0]} AS _p WHERE _p.{p[1]} = _o.{col}))"
                )
            sub = f"(SELECT _o.* FROM {source} AS _o WHERE {' AND '.join(conditions)})"
            wrapper = sqlglot.parse_one(
                f"SELECT 1 FROM {sub} AS {_ident(alias, dialect)}", read=dialect
            )
            subquery = wrapper.find(exp.Subquery)
            return subquery if subquery is not None else node

        try:
            return tree.transform(replace).sql(dialect=dialect)
        except Exception:
            return None


def _ident(name: str, dialect: str | None) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect=dialect)


def _quote(ref: Ref, dialect: str | None) -> tuple[str, str]:
    schema, table, column = ref
    qualified = (f"{_ident(schema, dialect)}." if schema else "") + _ident(table, dialect)
    return qualified, _ident(column, dialect)


def _name(ref: Ref) -> str:
    return ".".join(p for p in ref if p)
