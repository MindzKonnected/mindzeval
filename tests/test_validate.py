from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mindzeval import Case, Database
from mindzeval.schema.config import ValidationConfig
from mindzeval.validate import validate_cases


@pytest.fixture
def linked_db(tmp_path: Path) -> Path:
    path = tmp_path / "linked.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, country TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), total REAL);
        CREATE VIEW v_orders AS SELECT * FROM orders;
        INSERT INTO customers VALUES (1, 'DE'), (2, 'FR'), (3, NULL);
        INSERT INTO orders VALUES (1, 1, 10), (2, 2, 20), (3, 99, 5);
        """
    )
    conn.commit()
    conn.close()
    return path


def _cases(*specs: dict) -> list[Case]:  # type: ignore[type-arg]
    return [Case.model_validate({"question": "q", **s}) for s in specs]


def _findings(
    path: Path, cases: list[Case], config: ValidationConfig | None = None
) -> dict[str, list[str]]:
    with Database(f"sqlite:///{path}") as db:
        report = validate_cases(cases, db, config or ValidationConfig())
    return {cid: [f.check for f in tv.findings] for (cid, _), tv in report.turns.items()}


def test_clean_reference_and_result_is_kept(linked_db: Path) -> None:
    cases = _cases(
        {"id": "ok", "reference_sql": "SELECT id FROM customers", "key": "id", "expected_rows": 3}
    )
    with Database(f"sqlite:///{linked_db}") as db:
        report = validate_cases(cases, db, ValidationConfig())
    check = report.get("ok", 0)
    assert (
        check is not None
        and not check.findings
        and check.result is not None
        and check.result.row_count == 3
    )


def test_errors_empty_and_drift(linked_db: Path) -> None:
    findings = _findings(
        linked_db,
        _cases(
            {"id": "broken", "reference_sql": "SELECT nope FROM customers"},
            {"id": "empty", "reference_sql": "SELECT id FROM customers WHERE id > 100"},
            {
                "id": "empty_ok",
                "reference_sql": "SELECT id FROM customers WHERE id > 100",
                "expect_empty": True,
            },
            {"id": "drift", "reference_sql": "SELECT id FROM customers", "expected_rows": 5},
            {
                "id": "scalar_drift",
                "reference_sql": "SELECT count(*) FROM customers",
                "compare": "scalar",
                "expected_rows": 4,
            },
        ),
    )
    assert findings["broken"] == ["reference_error"]
    assert findings["empty"] == ["empty"]
    assert findings["empty_ok"] == []
    assert findings["drift"] == ["drift"]
    assert findings["scalar_drift"] == ["drift"]


def test_empty_blocks_by_default_and_severity_is_configurable(linked_db: Path) -> None:
    cases = _cases({"id": "empty", "reference_sql": "SELECT id FROM customers WHERE id > 100"})
    with Database(f"sqlite:///{linked_db}") as db:
        assert validate_cases(cases, db, ValidationConfig()).get("empty", 0).blocked  # type: ignore[union-attr]
        relaxed = validate_cases(cases, db, ValidationConfig(empty="warn"))  # type: ignore[arg-type]
        assert not relaxed.get("empty", 0).blocked  # type: ignore[union-attr]
        off = validate_cases(cases, db, ValidationConfig(empty="off"))  # type: ignore[arg-type]
        assert off.get("empty", 0).findings == []  # type: ignore[union-attr]


def test_null_groups(linked_db: Path) -> None:
    findings = _findings(
        linked_db,
        _cases(
            {
                "id": "g",
                "reference_sql": "SELECT country, count(*) FROM customers GROUP BY country",
                "compare": "grouped",
            }
        ),
    )
    assert findings["g"] == ["null_groups"]


def test_orphans_detected_through_declared_foreign_key(linked_db: Path) -> None:
    findings = _findings(
        linked_db,
        _cases(
            {
                "id": "inflated",
                "reference_sql": "SELECT count(DISTINCT customer_id) FROM orders",
                "compare": "scalar",
            },
            {
                "id": "anchored",
                "reference_sql": "SELECT count(*) FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)",
                "compare": "scalar",
            },
            {
                "id": "unrelated",
                "reference_sql": "SELECT count(*) FROM orders",
                "compare": "scalar",
            },
        ),
    )
    assert findings["inflated"] == ["orphans"]
    assert findings["anchored"] == []
    assert findings["unrelated"] == []


def test_orphans_on_views_need_configured_relationships(linked_db: Path) -> None:
    cases = _cases(
        {
            "id": "view",
            "reference_sql": "SELECT count(DISTINCT customer_id) FROM v_orders",
            "compare": "scalar",
        }
    )
    assert _findings(linked_db, cases)["view"] == []
    config = ValidationConfig(relationships=["v_orders.customer_id -> customers.id"])
    assert _findings(linked_db, cases, config)["view"] == ["orphans"]
