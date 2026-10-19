"""The agent result record: what an agent returns for one turn.

This is the contract between MindzEval and the agent under test, whatever
framework the agent is built with. Only the fields an agent can provide need to
be filled in; MindzEval scores what it is given and reports which checks were
skipped because data was missing.

Minimum useful record: ``answer`` plus ``sql`` (or ``rows``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Behavior(StrEnum):
    """What the agent did with the question."""

    ANSWER = "answer"
    """Answered the question, normally by running a query."""

    CLARIFY = "clarify"
    """Asked the user a clarifying question instead of answering."""

    DECLINE = "decline"
    """Declined: out of scope, not supported by the data, or not allowed."""


class ToolCall(BaseModel):
    """One tool invocation the agent made, if it reports them."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    error: str | None = None


class Usage(BaseModel):
    """Resource use for the turn, if the agent reports it."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    llm_calls: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class AgentResponse(BaseModel):
    """Everything the agent returned for one turn.

    ``rows`` may be given as a list of dicts, or as a list of lists together with
    ``columns``. A plain string is accepted as shorthand for ``{"answer": ...}``.
    Unknown fields are rejected so that a typo (``"SQL"``, ``"row"``) fails
    loudly instead of silently looking like a missing value; put anything extra
    in ``metadata``.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str | None = None
    """The text shown to the user."""

    sql: str | None = None
    """The final SQL the agent executed to produce the answer."""

    params: dict[str, Any] | list[Any] | None = None
    """Bind parameters for ``sql``, in the database driver's own style."""

    rows: list[dict[str, Any]] | None = None
    """The rows the agent's query returned (or the rows it showed the user)."""

    columns: list[str] | None = None
    """Column names, in order. Inferred from ``rows`` when not given."""

    truncated: bool | None = None
    """True when ``rows`` is only part of the result (e.g. a display limit was applied)."""

    status: Behavior | None = None
    """What the agent did. Needed for behaviour scoring; skipped when absent."""

    reason: str | None = None
    """Why the agent asked or declined, in the agent's own vocabulary."""

    tool_calls: list[ToolCall] | None = None
    trace: dict[str, Any] | None = None
    """Free-form internal steps, used by trace assertions (``path.to.field``)."""

    usage: Usage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"answer": data}
        if not isinstance(data, dict):
            return data

        rows = data.get("rows")
        if isinstance(rows, list) and rows and all(isinstance(r, (list, tuple)) for r in rows):
            columns = data.get("columns")
            if not columns:
                raise ValueError("rows given as lists need a 'columns' list naming each value")
            _check_unique(columns)
            width = len(columns)
            for i, row in enumerate(rows):
                if len(row) != width:
                    raise ValueError(
                        f"row {i} has {len(row)} values but {width} columns were given"
                    )
            data = {**data, "rows": [dict(zip(columns, row, strict=True)) for row in rows]}
        return data

    @model_validator(mode="after")
    def _infer_columns(self) -> AgentResponse:
        if self.columns is not None:
            _check_unique(self.columns)
        elif self.rows:
            seen: dict[str, None] = {}
            for row in self.rows:
                for key in row:
                    seen.setdefault(key, None)
            self.columns = list(seen)
        return self


def _check_unique(columns: list[Any]) -> None:
    dupes = sorted({str(c) for c in columns if columns.count(c) > 1})
    if dupes:
        raise ValueError(
            f"duplicate column names {dupes}; alias them so each column has a unique name"
        )
