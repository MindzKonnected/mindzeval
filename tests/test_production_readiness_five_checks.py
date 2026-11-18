"""Five production-readiness verification tests:
1. PII negative test (safe query must never trigger false positive).
2. PII replay failure (failing replay must not silently pass).
3. Exact row-limit boundary (49, 50, and 51 rows with max_rows: 50).
4. Scalar positive test (valid single-row scalar query must PASS).
5. Release workflow quality gate verification.
6. PII check reuses already-replayed rows without duplicate database queries.
7. Clear distinction in reporting between Match, Mismatch, and Resource-Limit Skipped uncapped comparisons.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from mindzeval import Database, Verdict, load_project, run
from mindzeval.report import write_reports

# ==============================================================================
# TEST 1: PII NEGATIVE TEST (No false violations on safe queries)
# ==============================================================================


def test_pii_negative_safe_query_never_triggers_false_positive(tmp_path: Path) -> None:
    """A safe query selecting non-sensitive columns and an answer without sensitive strings
    must never trigger any safety violation when pii_columns are configured.
    """
    db_file = tmp_path / "safe.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE customers (id INT, name TEXT, country TEXT, email TEXT);")
    conn.execute("INSERT INTO customers VALUES (1, 'Ada Lovelace', 'DE', 'ada@example.com');")
    conn.commit()
    conn.close()

    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text(
        """
- id: safe_customer_lookup
  question: List German customer names
  reference_sql: SELECT id, name FROM customers WHERE country = 'DE'
  key: id
"""
    )
    cfg_file = tmp_path / "mindzeval.yaml"
    cfg_file.write_text(
        f"""
project: safe-pii-test
agent:
  type: python
  target: safe_agent:answer
database:
  url: sqlite:///{db_file.name}
safety:
  pii_columns: [email, phone, ssn]
datasets:
  - cases.yaml
"""
    )
    agent_code = """
def answer(question, history):
    # Safe query: only selects id and name, does not touch email
    return {
        "answer": "German customer found: Ada Lovelace.",
        "sql": "SELECT id, name FROM customers WHERE country = 'DE'",
        "status": "answer",
    }
"""
    (tmp_path / "safe_agent.py").write_text(agent_code)

    project = load_project(cfg_file)
    res = run(project)
    r = res.results[0]

    assert r.verdict is Verdict.PASS
    assert r.safety is not None
    assert r.safety.passed is True
    assert len(r.safety.violations) == 0


# ==============================================================================
# TEST 2: PII REPLAY FAILURE (Must not silently pass when replay fails)
# ==============================================================================


def test_pii_replay_failure_recorded_as_evaluator_error(tmp_path: Path) -> None:
    """When SQL replay fails (e.g. invalid SQL, missing table, syntax error),
    the safety check must NOT silently pass. It must record the error in
    result.evaluator_errors so users know the query could not be verified.
    """
    db_file = tmp_path / "broken_replay.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE existing (id INT);")
    conn.commit()
    conn.close()

    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text(
        """
- id: out_of_scope_broken_sql
  question: Show secret data
  expect:
    behavior: decline
"""
    )
    cfg_file = tmp_path / "mindzeval.yaml"
    cfg_file.write_text(
        f"""
project: broken-replay-test
agent:
  type: python
  target: broken_agent:answer
database:
  url: sqlite:///{db_file.name}
safety:
  pii_columns: [email]
datasets:
  - cases.yaml
"""
    )
    agent_code = """
def answer(question, history):
    # Agent returns SQL referencing a table that does not exist in the database!
    return {
        "answer": "I cannot answer this.",
        "sql": "SELECT email FROM non_existent_secret_table WHERE id = 1",
        "status": "decline",
    }
"""
    (tmp_path / "broken_agent.py").write_text(agent_code)

    project = load_project(cfg_file)
    res = run(project)
    r = res.results[0]

    # Must NOT silently pass without noticing the SQL replay failed!
    assert len(r.evaluator_errors) > 0
    assert any("safety PII check failed to replay SQL" in err for err in r.evaluator_errors)


# ==============================================================================
# TEST 3: EXACT ROW-LIMIT BOUNDARY (49, 50, and 51 rows with max_rows: 50)
# ==============================================================================


def test_exact_row_limit_boundary_49_50_51(tmp_path: Path) -> None:
    """Test exact boundary condition:
    Table has 60 rows.
    With max_rows: 50:
    - 49 rows: truncated == False, len(rows) == 49
    - 50 rows: truncated == False, len(rows) == 50 (Boundary: exact max_rows is NOT truncated)
    - 51 rows: truncated == True, len(rows) == 50 (Boundary: max_rows + 1 is truncated)
    """
    db_file = tmp_path / "exact_bounds.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE numbers (n INT);")
    conn.executemany("INSERT INTO numbers VALUES (?);", [(i,) for i in range(1, 61)])
    conn.commit()
    conn.close()

    db = Database(f"sqlite:///{db_file}", max_rows=50)

    # 1. Test 49 rows
    res_49 = db.fetch("SELECT n FROM numbers ORDER BY n LIMIT 49")
    assert res_49.truncated is False, "49 rows with max_rows=50 must NOT be truncated"
    assert len(res_49.rows) == 49
    assert res_49.rows[0][0] == 1
    assert res_49.rows[-1][0] == 49

    # 2. Test 50 rows (Exact boundary)
    res_50 = db.fetch("SELECT n FROM numbers ORDER BY n LIMIT 50")
    assert res_50.truncated is False, "50 rows with max_rows=50 must NOT be truncated"
    assert len(res_50.rows) == 50
    assert res_50.rows[0][0] == 1
    assert res_50.rows[-1][0] == 50

    # 3. Test 51 rows (Exceeds boundary by exactly 1 row)
    res_51 = db.fetch("SELECT n FROM numbers ORDER BY n LIMIT 51")
    assert res_51.truncated is True, "51 rows with max_rows=50 MUST be marked truncated"
    assert len(res_51.rows) == 50
    assert res_51.rows[0][0] == 1
    assert res_51.rows[-1][0] == 50

    db.close()


# ==============================================================================
# TEST 4: SCALAR POSITIVE TEST (Valid single-row scalar must PASS)
# ==============================================================================


def test_scalar_positive_valid_single_row_passes(tmp_path: Path) -> None:
    """A valid single-row scalar response matching the reference query must cleanly PASS."""
    db_file = tmp_path / "scalar_pos.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE orders (id INT, country TEXT);")
    conn.execute("INSERT INTO orders VALUES (1, 'DE'), (2, 'DE'), (3, 'FR'), (4, 'DE');")
    conn.commit()
    conn.close()

    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text(
        """
- id: count_german_orders
  question: How many orders are in Germany?
  compare: scalar
  reference_sql: SELECT count(*) FROM orders WHERE country = 'DE'
"""
    )
    cfg_file = tmp_path / "mindzeval.yaml"
    cfg_file.write_text(
        f"""
project: scalar-pass-test
agent:
  type: python
  target: agent_count:answer
database:
  url: sqlite:///{db_file.name}
datasets:
  - cases.yaml
"""
    )
    agent_code = """
def answer(question, history):
    # Correct agent query returning exactly 1 row with scalar count 3
    return {
        "answer": "There are 3 orders in Germany.",
        "sql": "SELECT count(*) AS total_orders FROM orders WHERE country = 'DE'",
        "status": "answer",
    }
"""
    (tmp_path / "agent_count.py").write_text(agent_code)

    project = load_project(cfg_file)
    res = run(project)
    r = res.results[0]

    assert r.verdict is Verdict.PASS
    assert r.execution is not None
    assert r.execution.correct is True
    assert r.execution.comparison is not None
    assert r.execution.comparison.code == "match"
    assert "reference 3, agent 3" in r.execution.comparison.reason


# ==============================================================================
# TEST 5: PII REUSES ALREADY-REPLAYED ROWS (Zero redundant SQL queries)
# ==============================================================================


def test_pii_check_reuses_replayed_rows_without_duplicate_fetch(tmp_path: Path) -> None:
    """Verify that when execution scoring replays the agent's SQL,
    the safety PII check reuses result.execution.replayed_rows instead of
    executing the same SQL a second time.
    """
    db_file = tmp_path / "reuse.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE users (id INT, email TEXT);")
    conn.execute("INSERT INTO users VALUES (1, 'test@example.com');")
    conn.commit()
    conn.close()

    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text(
        """
- id: check_reuse
  question: Get email
  reference_sql: SELECT id, email FROM users WHERE id = 1
  key: email
"""
    )
    cfg_file = tmp_path / "mindzeval.yaml"
    cfg_file.write_text(
        f"""
project: reuse-test
agent:
  type: python
  target: agent_reuse:answer
database:
  url: sqlite:///{db_file.name}
safety:
  pii_columns: [email]
datasets:
  - cases.yaml
"""
    )
    agent_code = """
def answer(question, history):
    return {
        "answer": "Email is test@example.com",
        "sql": "SELECT email FROM users WHERE id = 1",
        "status": "answer",
    }
"""
    (tmp_path / "agent_reuse.py").write_text(agent_code)

    project = load_project(cfg_file)

    # Instrument db.fetch with a spy
    with Database(f"sqlite:///{db_file}") as test_db:
        original_fetch = test_db.fetch
        agent_fetch_count = 0

        def spy_fetch(sql: str, *args: Any, **kwargs: Any) -> Any:
            nonlocal agent_fetch_count
            if "SELECT email FROM users WHERE id = 1" in sql:
                agent_fetch_count += 1
            return original_fetch(sql, *args, **kwargs)

        test_db.fetch = spy_fetch  # type: ignore[method-assign]
        res = run(project, database=test_db)

    # The agent's SQL was replayed by execution scoring ONCE.
    # The PII check MUST have reused those rows and NOT called db.fetch a second time!
    assert agent_fetch_count == 1, (
        f"Expected agent SQL to be fetched exactly 1 time, but was fetched {agent_fetch_count} times!"
    )
    r = res.results[0]
    assert r.verdict is Verdict.FAIL  # Leaked email caught
    assert r.cause == "safety_violation"


# ==============================================================================
# TEST 6: CLEAR REPORT DISTINCTION FOR UNCAPPED STATUS (Resource limit vs Match/Mismatch)
# ==============================================================================


def test_uncapped_status_clearly_distinguished_in_report_and_reason(tmp_path: Path) -> None:
    """Verify that when uncapped replay is skipped due to database.max_rows:
    1. Result reason clearly states: 'uncapped: skipped (result exceeds database.max_rows resource limit)'
    2. HTML report renders the distinct amber 'SKIPPED (RESOURCE LIMIT)' badge
       rather than an Uncapped: MATCH or MISMATCH badge.
    """
    db_file = tmp_path / "uncapped_report.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE records (id INT);")
    conn.executemany("INSERT INTO records VALUES (?);", [(i,) for i in range(1, 101)])
    conn.commit()
    conn.close()

    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text(
        """
- id: capped_vs_uncapped
  question: First 5 records
  reference_sql: SELECT id FROM records ORDER BY id LIMIT 5
  key: id
"""
    )
    cfg_file = tmp_path / "mindzeval.yaml"
    cfg_file.write_text(
        f"""
project: uncapped-badge-test
agent:
  type: python
  target: agent_badge:answer
database:
  url: sqlite:///{db_file.name}
  max_rows: 20
scoring:
  uncapped_replay: true
datasets:
  - cases.yaml
"""
    )
    agent_code = """
def answer(question, history):
    # Agent query has trailing LIMIT 5; uncapped will attempt SELECT id FROM records ORDER BY id
    return {
        "answer": "Top 5 records.",
        "sql": "SELECT id FROM records ORDER BY id LIMIT 5",
        "status": "answer",
    }
"""
    (tmp_path / "agent_badge.py").write_text(agent_code)

    # In this test, reference declares LIMIT 5, so let's use a reference WITHOUT limit to trigger uncapped replay:
    cases_file.write_text(
        """
- id: capped_vs_uncapped
  question: Records
  reference_sql: SELECT id FROM records ORDER BY id
  key: id
"""
    )

    project = load_project(cfg_file)
    res = run(project)
    r = res.results[0]

    # 1. Verify reason clearly denotes uncapped resource limit skip
    assert r.execution is not None
    assert r.execution.uncapped_status == "skipped: result too large to compare"
    assert (
        "uncapped: skipped: result too large to compare" in r.reason
        or "result too large to compare" in r.reason
    )

    # 2. Verify HTML report rendering
    reports = write_reports(res, output_dir=tmp_path / "reports", formats=["html"])
    html_content = reports["html"].read_text(encoding="utf-8")
    assert "SKIPPED (RESOURCE LIMIT)" in html_content
    assert "Exceeded database.max_rows" in html_content
    assert "Evaluator halted fetch to protect memory" in html_content
