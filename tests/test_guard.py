from __future__ import annotations

import pytest

from mindzeval.errors import SQLGuardError
from mindzeval.sql import check_read_only, dialect_from_url


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select id from customers where updated_at > '2024-01-01'",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "SELECT a FROM t UNION SELECT b FROM u",
        "SELECT id FROM t WHERE name LIKE '%B' AND x = %(p)s LIMIT 51",
        "SELECT CASE WHEN y ~ '^[0-9]{4}$' THEN CAST(y AS INTEGER) END FROM o",
        "SELECT * FROM (SELECT a FROM t LIMIT 5) AS sub;",
        "SELECT count(*), max(total) FROM orders",
    ],
)
def test_read_queries_pass(sql: str) -> None:
    check_read_only(sql, dialect="postgres")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("", "empty"),
        ("   ;  ", "empty"),
        ("INSERT INTO t VALUES (1)", "only SELECT"),
        ("UPDATE t SET a = 1", "only SELECT"),
        ("DELETE FROM t", "only SELECT"),
        ("DROP TABLE t", "only SELECT"),
        ("TRUNCATE t", "only SELECT"),
        ("CREATE TABLE x (a int)", "only SELECT"),
        ("SET search_path = evil", "only SELECT"),
        ("SELECT 1; SELECT 2", "one statement"),
        ("SELECT 1; DROP TABLE t", "one statement"),
        ("WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d", "write"),
        ("SELECT * INTO new_t FROM t", "SELECT ... INTO"),
        ("SELECT * FROM t FOR UPDATE", "row locks"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT nextval('seq')", "nextval"),
        ("SELEC id FROM", "could not be parsed"),
    ],
)
def test_unsafe_or_invalid_sql_is_refused(sql: str, reason: str) -> None:
    with pytest.raises(SQLGuardError, match=reason):
        check_read_only(sql, dialect="postgres")


def test_extra_forbidden_functions() -> None:
    check_read_only("SELECT my_udf(1)")
    with pytest.raises(SQLGuardError, match="my_udf"):
        check_read_only("SELECT my_udf(1)", forbidden_functions=["MY_UDF"])


def test_sqlite_dialect_and_qmark_params() -> None:
    check_read_only("SELECT * FROM t WHERE a = ?", dialect="sqlite")
    with pytest.raises(SQLGuardError):
        check_read_only("PRAGMA writable_schema = 1", dialect="sqlite")


@pytest.mark.parametrize(
    ("url", "dialect"),
    [
        ("postgresql://u@h/db", "postgres"),
        ("postgresql+psycopg2://u@h/db", "postgres"),
        ("sqlite:///x.db", "sqlite"),
        ("snowflake://acct", "snowflake"),
        ("mssql+pyodbc://dsn", "tsql"),
        ("unknown://x", None),
    ],
)
def test_dialect_from_url(url: str, dialect: str | None) -> None:
    assert dialect_from_url(url) == dialect
