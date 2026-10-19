"""Read-only query execution for reference SQL, replays and validator probes.

Three layers keep execution safe:

1. :func:`~mindzeval.sql.check_read_only` refuses anything but one read query.
2. Each query runs in a read-only transaction (Postgres) or a read-only
   connection (SQLite), with a statement timeout, and is always rolled back.
3. The connection should use a login that only has read access.

Queries run through the driver's own cursor, so the agent's SQL is executed with
its own bind-parameter style (``%(name)s`` for psycopg2, ``?`` / ``:name`` for
SQLite) exactly as the agent ran it.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, NoSuchModuleError
from sqlalchemy.pool import NullPool

from mindzeval.errors import DatabaseError, QueryTimeoutError
from mindzeval.sql.guard import check_read_only, dialect_from_url

SUPPORTED_BACKENDS = ("postgresql", "sqlite")
_PG_QUERY_CANCELED = "57014"


@dataclass(frozen=True)
class QueryResult:
    """Rows returned by one query, with column names in order."""

    columns: tuple[str, ...]
    rows: list[tuple[Any, ...]]
    elapsed_ms: float = 0.0
    sql: str = field(default="", repr=False)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def as_dicts(self) -> list[dict[str, Any]]:
        """Rows as dicts. Repeated column names get a ``_2``, ``_3`` suffix."""
        names = _unique_names(self.columns)
        return [dict(zip(names, row, strict=True)) for row in self.rows]


def resolve_sqlite_url(url: str, base_dir: str | Path) -> str:
    """Make a relative SQLite file path absolute against ``base_dir``. Other URLs pass through."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    path = url[len(prefix) :]
    if not path or path.startswith(":memory:") or path.startswith("/") or Path(path).is_absolute():
        return url
    return prefix + (Path(base_dir) / path).resolve().as_posix()


class Database:
    """A read-only connection to the customer's database.

    ``url`` is a SQLAlchemy URL. Pass ``engine`` instead to reuse an existing
    SQLAlchemy engine (useful in tests); read-only protection still applies per query.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        engine: Engine | None = None,
        statement_timeout_s: float = 30.0,
        dialect: str | None = None,
        forbidden_functions: Iterable[str] = (),
    ) -> None:
        if (url is None) == (engine is None):
            raise ValueError("pass exactly one of 'url' or 'engine'")
        self.statement_timeout_s = statement_timeout_s
        self._forbidden = tuple(forbidden_functions)
        self._engine = engine if engine is not None else _create_engine(str(url))
        backend = self._engine.dialect.name
        if backend not in SUPPORTED_BACKENDS:
            self._engine.dispose()
            raise DatabaseError(
                f"the '{backend}' database is not supported yet "
                f"(supported: {', '.join(SUPPORTED_BACKENDS)})"
            )
        self.backend = backend
        self.dialect = dialect or dialect_from_url(backend) or backend

    @property
    def engine(self) -> Engine:
        return self._engine

    def fetch(
        self,
        sql: str,
        params: Sequence[Any] | dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> QueryResult:
        """Run one read-only query and return its rows.

        Raises :class:`~mindzeval.errors.SQLGuardError` if the SQL is not a single
        read query, :class:`~mindzeval.errors.QueryTimeoutError` on timeout and
        :class:`~mindzeval.errors.DatabaseError` for any other failure.
        """
        check_read_only(sql, dialect=self.dialect, forbidden_functions=self._forbidden)
        timeout = timeout_s if timeout_s is not None else self.statement_timeout_s

        try:
            raw = self._engine.raw_connection()
        except Exception as exc:
            raise DatabaseError(f"could not connect to the database: {_first_line(exc)}") from exc

        started = time.perf_counter()
        try:
            cursor = raw.cursor()
            try:
                if self.backend == "postgresql":
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(f"SET LOCAL statement_timeout = {max(1, int(timeout * 1000))}")
                    rows, columns = _run(cursor, sql, params)
                else:
                    rows, columns = _run_sqlite(raw.driver_connection, cursor, sql, params, timeout)
            finally:
                cursor.close()
        except QueryTimeoutError:
            raise
        except Exception as exc:
            if getattr(exc, "pgcode", None) == _PG_QUERY_CANCELED:
                raise QueryTimeoutError(f"the query ran longer than {timeout:g}s") from exc
            raise DatabaseError(_first_line(exc)) from exc
        finally:
            try:
                raw.rollback()
            finally:
                raw.close()

        elapsed = (time.perf_counter() - started) * 1000.0
        return QueryResult(columns=columns, rows=rows, elapsed_ms=elapsed, sql=sql)

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _create_engine(url: str) -> Engine:
    try:
        parsed = make_url(url)
    except ArgumentError:
        raise DatabaseError("the database URL is not valid") from None

    if parsed.get_backend_name() == "sqlite":
        database = parsed.database or ""
        if not database or database == ":memory:":
            raise DatabaseError("an in-memory SQLite database has no data to evaluate; use a file")
        path = Path(database)
        if not path.is_file():
            raise DatabaseError(f"SQLite database file not found: {path}")
        uri = f"{path.resolve().as_uri()}?mode=ro"

        def connect() -> sqlite3.Connection:
            # mode=ro opens the file read-only and never creates it.
            return sqlite3.connect(uri, uri=True, check_same_thread=False)

        return create_engine("sqlite://", creator=connect, poolclass=NullPool)

    try:
        return create_engine(url, pool_pre_ping=True)
    except NoSuchModuleError:
        raise DatabaseError(
            f"no driver installed for '{parsed.drivername}'; for Postgres install "
            "'mindzeval[postgres]'"
        ) from None
    except ModuleNotFoundError as exc:
        raise DatabaseError(
            f"the database driver '{exc.name}' is not installed; for Postgres install "
            "'mindzeval[postgres]'"
        ) from None


def _run(
    cursor: Any, sql: str, params: Sequence[Any] | dict[str, Any] | None
) -> tuple[list[tuple[Any, ...]], tuple[str, ...]]:
    # Drivers only interpolate placeholders when params are passed. Passing an
    # empty container would make literal '%' characters (LIKE '%x') fail.
    if params:
        cursor.execute(sql, params)
    else:
        cursor.execute(sql)
    description = cursor.description or ()
    columns = tuple(str(d[0]) for d in description)
    rows = [tuple(r) for r in cursor.fetchall()] if description else []
    return rows, columns


def _run_sqlite(
    connection: Any,
    cursor: Any,
    sql: str,
    params: Sequence[Any] | dict[str, Any] | None,
    timeout: float,
) -> tuple[list[tuple[Any, ...]], tuple[str, ...]]:
    deadline = time.monotonic() + timeout
    timed_out = False

    def check_deadline() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1  # non-zero aborts the query
        return 0

    cursor.execute("PRAGMA query_only = ON")
    connection.set_progress_handler(check_deadline, 10_000)
    try:
        return _run(cursor, sql, params)
    except sqlite3.OperationalError as exc:
        if timed_out:
            raise QueryTimeoutError(f"the query ran longer than {timeout:g}s") from exc
        raise
    finally:
        connection.set_progress_handler(None, 0)


def _unique_names(columns: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for column in columns:
        counts[column] = counts.get(column, 0) + 1
        names.append(column if counts[column] == 1 else f"{column}_{counts[column]}")
    return names


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:300] if text else type(exc).__name__
