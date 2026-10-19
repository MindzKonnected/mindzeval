from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def write(tmp_path: Path) -> Callable[[str, str], Path]:
    """Write a text file under the test's temp directory and return its path."""

    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def shop_db(tmp_path: Path) -> Path:
    """A small SQLite database with customers and orders."""
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);
        INSERT INTO customers VALUES (1, 'Ada', 'DE'), (2, 'Bo', 'FR'), (3, 'Cy', 'DE');
        INSERT INTO orders VALUES (1, 1, 10.5), (2, 1, 20.0), (3, 2, 7.25), (4, 9, 1.0);
        """
    )
    conn.commit()
    conn.close()
    return path
