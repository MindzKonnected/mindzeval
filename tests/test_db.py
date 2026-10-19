from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from mindzeval import Database, DatabaseError, QueryTimeoutError, SQLGuardError
from mindzeval.db import resolve_sqlite_url


def test_fetch_returns_columns_and_rows(shop_db: Path) -> None:
    with Database(f"sqlite:///{shop_db}") as db:
        result = db.fetch("SELECT id, name FROM customers WHERE country = ? ORDER BY id", ["DE"])
    assert result.columns == ("id", "name")
    assert result.rows == [(1, "Ada"), (3, "Cy")]
    assert result.row_count == 2
    assert result.as_dicts() == [{"id": 1, "name": "Ada"}, {"id": 3, "name": "Cy"}]


def test_repeated_column_names_get_suffixes(shop_db: Path) -> None:
    with Database(f"sqlite:///{shop_db}") as db:
        result = db.fetch(
            "SELECT c.id, o.id FROM customers c JOIN orders o ON o.customer_id = c.id"
        )
    assert list(result.as_dicts()[0]) == ["id", "id_2"]


def test_writes_are_refused_before_execution(shop_db: Path) -> None:
    with Database(f"sqlite:///{shop_db}") as db, pytest.raises(SQLGuardError):
        db.fetch("DELETE FROM customers")
    assert _count(shop_db) == 3


def test_connection_is_read_only_even_if_the_guard_is_bypassed(shop_db: Path) -> None:
    db = Database(f"sqlite:///{shop_db}")
    raw = db.engine.raw_connection()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            raw.cursor().execute("DELETE FROM customers")
    finally:
        raw.close()
        db.close()
    assert _count(shop_db) == 3


def test_missing_sqlite_file_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(DatabaseError, match="not found"):
        Database(f"sqlite:///{missing}")
    assert not missing.exists()


def test_query_errors_are_wrapped(shop_db: Path) -> None:
    with (
        Database(f"sqlite:///{shop_db}") as db,
        pytest.raises(DatabaseError, match="no such table"),
    ):
        db.fetch("SELECT * FROM nope")


def test_statement_timeout(shop_db: Path) -> None:
    slow = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 50000000) "
        "SELECT count(*) FROM n"
    )
    with Database(f"sqlite:///{shop_db}", statement_timeout_s=0.2) as db:
        with pytest.raises(QueryTimeoutError):
            db.fetch(slow)
        # The connection still works after a timeout.
        assert db.fetch("SELECT 1").rows == [(1,)]


def test_unsupported_backend_is_reported() -> None:
    pytest.importorskip("sqlalchemy")
    with pytest.raises(DatabaseError):
        Database("notadb://x")


def test_resolve_sqlite_url(tmp_path: Path) -> None:
    resolved = resolve_sqlite_url("sqlite:///data/x.db", tmp_path)
    assert resolved == "sqlite:///" + (tmp_path / "data" / "x.db").resolve().as_posix()
    assert resolve_sqlite_url("sqlite:////abs/x.db", tmp_path) == "sqlite:////abs/x.db"
    assert resolve_sqlite_url("postgresql://h/db", tmp_path) == "postgresql://h/db"


@pytest.mark.skipif(
    not os.environ.get("MINDZEVAL_TEST_POSTGRES_URL"),
    reason="set MINDZEVAL_TEST_POSTGRES_URL to run Postgres tests",
)
def test_postgres_read_only_and_timeout() -> None:
    url = os.environ["MINDZEVAL_TEST_POSTGRES_URL"]
    with Database(url, statement_timeout_s=0.5) as db:
        assert db.fetch("SELECT 1 AS one").rows == [(1,)]
        assert db.fetch("SELECT %(x)s::int AS x, 'a%%b' LIKE 'a%%' AS y", {"x": 2}).rows[0][0] == 2
        assert db.fetch("SELECT 'x' LIKE '%x' AS y").rows == [(True,)]
        with pytest.raises(QueryTimeoutError):
            db.fetch("SELECT count(*) FROM generate_series(1, 1000000000)")
        raw = db.engine.raw_connection()
        try:
            cursor = raw.cursor()
            cursor.execute("SET TRANSACTION READ ONLY")
            with pytest.raises(Exception, match="read-only"):
                cursor.execute("CREATE TEMP TABLE should_fail (a int)")
        finally:
            raw.rollback()
            raw.close()


def _count(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT count(*) FROM customers").fetchone()[0])
    finally:
        conn.close()
