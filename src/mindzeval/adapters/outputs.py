"""Score answers an agent produced earlier, without calling it.

The file is JSONL (one record per line) or a JSON list. Each record names the
case and turn, and holds the result record either under ``response`` or at the
top level::

    {"case_id": "customers_de", "turn": 0, "response": {"answer": "...", "sql": "..."}}
    {"case_id": "order_count", "answer": "4", "sql": "SELECT count(*) FROM orders"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindzeval.adapters.base import (
    Adapter,
    AgentError,
    AgentErrorKind,
    AgentRequest,
    coerce_response,
)
from mindzeval.errors import ConfigError, Issue
from mindzeval.schema.response import AgentResponse


class OutputsAdapter(Adapter):
    def __init__(self, path: Path) -> None:
        self.name = f"outputs:{path.name}"
        self._records = _load(path)

    async def call(self, request: AgentRequest) -> AgentResponse:
        record = self._records.get((request.case_id, request.turn))
        if record is None:
            raise AgentError(
                AgentErrorKind.MISSING_OUTPUT,
                f"no saved output for case '{request.case_id}' turn {request.turn}",
            )
        return coerce_response(record)


def _load(path: Path) -> dict[tuple[str, int], Any]:
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError([Issue(source, f"cannot read outputs file: {exc}")]) from None

    items: list[tuple[int | None, Any]] = []
    try:
        if path.suffix.lower() == ".json":
            loaded = json.loads(text)
            if not isinstance(loaded, list):
                raise ConfigError([Issue(source, "an outputs JSON file must contain a list")])
            items = [(None, item) for item in loaded]
        else:
            items = [
                (n, json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()
            ]
    except json.JSONDecodeError as exc:
        raise ConfigError([Issue(source, f"invalid JSON: {exc.msg}", line=exc.lineno)]) from None

    records: dict[tuple[str, int], Any] = {}
    issues: list[Issue] = []
    for line, item in items:
        if not isinstance(item, dict) or "case_id" not in item:
            issues.append(Issue(source, "each output needs a 'case_id'", line=line))
            continue
        key = (str(item["case_id"]), int(item.get("turn", 0)))
        if key in records:
            issues.append(
                Issue(source, f"duplicate output for case '{key[0]}' turn {key[1]}", line=line)
            )
            continue
        records[key] = item.get(
            "response", {k: v for k, v in item.items() if k not in ("case_id", "turn")}
        )
    if issues:
        raise ConfigError(issues)
    return records
