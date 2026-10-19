"""Refuse any SQL that is not a single read-only query.

Applied to every statement MindzEval itself executes: reference queries, replays
of the agent's SQL, and validator probes. The check parses the statement rather
than matching keywords, so a column named ``updated_at`` passes while a
``DELETE`` hidden inside a CTE does not.

This is one layer of defence, not the only one. The database layer also opens
read-only transactions with a statement timeout, and the connection should use a
read-only login.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from mindzeval.errors import SQLGuardError

#: SQLAlchemy backend name -> sqlglot dialect.
_DIALECTS = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "redshift": "redshift",
    "sqlite": "sqlite",
    "mysql": "mysql",
    "mariadb": "mysql",
    "mssql": "tsql",
    "oracle": "oracle",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "duckdb": "duckdb",
    "databricks": "databricks",
    "trino": "trino",
    "clickhouse": "clickhouse",
}

#: Expression types that change data, schema, session state or permissions.
_WRITE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Drop",
    "Create",
    "Alter",
    "TruncateTable",
    "Grant",
    "Revoke",
    "Copy",
    "Set",
    "Command",
    "Into",
    "Pragma",
    "Use",
    "Transaction",
    "Commit",
    "Rollback",
    "LoadData",
    "Analyze",
)
_WRITE_TYPES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _WRITE_NAMES if hasattr(exp, name)
)

#: Functions that sleep, touch files, open connections or change server state.
DANGEROUS_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "set_config",
        "nextval",
        "setval",
        "load_extension",
        "readfile",
        "writefile",
        "sleep",
        "benchmark",
        "load_file",
        "xp_cmdshell",
        "system",
    }
)


def dialect_from_url(url: str) -> str | None:
    """sqlglot dialect for a SQLAlchemy URL such as ``postgresql+psycopg2://...``."""
    scheme = url.split("://", 1)[0] if "://" in url else url.split(":", 1)[0]
    backend = scheme.split("+", 1)[0].lower()
    return _DIALECTS.get(backend)


def check_read_only(
    sql: str,
    *,
    dialect: str | None = None,
    forbidden_functions: Iterable[str] = (),
) -> exp.Query:
    """Return the parsed query if ``sql`` is one read-only statement, else raise.

    Raises :class:`~mindzeval.errors.SQLGuardError` with a reason the user can act on.
    """
    if not sql or not sql.strip():
        raise SQLGuardError("the SQL statement is empty", "empty")

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as exc:
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise SQLGuardError(f"the SQL could not be parsed: {first}", "parse") from None

    if not statements:
        raise SQLGuardError("the SQL statement is empty", "empty")
    if len(statements) > 1:
        unsafe = any(not isinstance(s, exp.Query) or _first_write(s) for s in statements)
        raise SQLGuardError(
            f"expected one statement, found {len(statements)}", "write" if unsafe else "multiple"
        )

    statement = statements[0]
    if not isinstance(statement, exp.Query):
        kind = type(statement).__name__.upper()
        raise SQLGuardError(f"only SELECT / WITH queries are allowed, not {kind}", "not_query")

    write = _first_write(statement)
    if write is not None:
        kind = "SELECT ... INTO" if isinstance(write, exp.Into) else type(write).__name__.upper()
        raise SQLGuardError(f"the query contains a write or session statement ({kind})", "write")

    if statement.find(exp.Lock) is not None:
        raise SQLGuardError("row locks (FOR UPDATE / FOR SHARE) are not allowed", "lock")

    blocked = DANGEROUS_FUNCTIONS | {f.lower() for f in forbidden_functions}
    for func in statement.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name in blocked:
            raise SQLGuardError(f"the function '{name}' is not allowed", "function")

    return statement


def referenced_tables(sql: str, *, dialect: str | None = None) -> list[tuple[str | None, str]]:
    """``(schema, table)`` pairs referenced by ``sql``, excluding CTE names. Empty if unparsable."""
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except ParseError:
        return []
    if parsed is None:
        return []
    ctes = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    seen: dict[tuple[str | None, str], None] = {}
    for table in parsed.find_all(exp.Table):
        if not table.name or (not table.db and table.name.lower() in ctes):
            continue
        seen.setdefault((table.db or None, table.name), None)
    return list(seen)


def _first_write(statement: Any) -> exp.Expression | None:
    for node in statement.walk():
        if isinstance(node, _WRITE_TYPES):
            return node
    return None
