"""The golden case format: one test question and what a correct agent does with it.

A case is either single-turn (``question`` at the top level) or multi-turn
(``turns``). Each turn says how a correct agent behaves and, when it should
answer, gives a reference query whose *results* the agent's results are compared
against. SQL text is never compared.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from mindzeval.schema.response import Behavior

CASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:/\-]*$"


class CompareMode(StrEnum):
    """How the agent's result is compared with the reference result."""

    SET = "set"
    """Same set of records, identified by ``key``. Row order and extra columns ignored."""

    ORDERED = "ordered"
    """Same records, identified by ``key``, in the same order (rankings, sorted lists)."""

    SCALAR = "scalar"
    """One value, e.g. a count or a total."""

    GROUPED = "grouped"
    """(group, value) pairs, e.g. a count per category."""

    TABLE = "table"
    """Whole result compared by value, for results with no identifying column."""

    NONE = "none"
    """No result comparison: the case only checks behaviour or trace."""


KEYED_MODES = frozenset({CompareMode.SET, CompareMode.ORDERED})


class Tolerance(BaseModel):
    """Resolved comparison policy. Every field has a value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_order: Literal["ignore", "strict"] = "ignore"
    """Whether row order matters in ``table`` mode. ``ordered`` mode is always strict."""

    relative: float = Field(default=1e-6, ge=0)
    """Relative tolerance for numeric values."""

    absolute: float = Field(default=1e-9, ge=0)
    """Absolute tolerance for numeric values."""

    extra_columns: Literal["allow", "forbid"] = "allow"
    """Whether the agent may return columns the reference does not have (``table`` mode)."""

    case_sensitive: bool = True
    """Whether text values must match case."""

    strip_whitespace: bool = True
    """Whether leading and trailing whitespace in text is ignored."""

    null_equals_empty: bool = False
    """Whether NULL and the empty string count as equal."""

    duplicates: Literal["ignore", "count"] = "ignore"
    """Whether repeated rows are significant (``count``) or collapsed (``ignore``)."""


class ToleranceOverride(BaseModel):
    """A partial :class:`Tolerance`. Only the fields that are set override."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_order: Literal["ignore", "strict"] | None = None
    relative: float | None = Field(default=None, ge=0)
    absolute: float | None = Field(default=None, ge=0)
    extra_columns: Literal["allow", "forbid"] | None = None
    case_sensitive: bool | None = None
    strip_whitespace: bool | None = None
    null_equals_empty: bool | None = None
    duplicates: Literal["ignore", "count"] | None = None

    def apply(self, base: Tolerance) -> Tolerance:
        """Return ``base`` with this override's set fields replaced."""
        changes = self.model_dump(exclude_none=True)
        return base.model_copy(update=changes) if changes else base


class Expectation(BaseModel):
    """What a correct agent does on this turn, beyond returning the right data."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    behavior: Behavior | None = Field(
        default=None, validation_alias=AliasChoices("behavior", "behaviour")
    )
    """Answer, ask for clarification, or decline. Scored in both directions."""

    reason: str | None = None
    """The reason the agent should report when it asks or declines."""

    trace: dict[str, Any] = Field(default_factory=dict)
    """Trace assertions: ``{"dotted.path": expected_value}`` against the agent's trace."""


class Turn(BaseModel):
    """One question in a case, with everything needed to score the answer."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    question: str = Field(min_length=1)
    reference_sql: str | None = Field(
        default=None, validation_alias=AliasChoices("reference_sql", "golden_sql")
    )
    """A query that returns the correct answer. Executed live at evaluation time."""

    compare: CompareMode | None = None
    """Comparison mode. Inferred when omitted: ``set`` if ``key`` is given, else ``table``."""

    key: list[str] | None = None
    """Column(s) that identify a record, for ``set`` and ``ordered`` modes."""

    expect: Expectation = Field(default_factory=Expectation)
    tolerance: ToleranceOverride | None = None
    expected_rows: int | None = Field(default=None, ge=0)
    """Row count the reference returned when the case was written. Used for drift checks."""

    expect_empty: bool = False
    """Set when an empty reference result is the correct answer."""

    @field_validator("key", mode="before")
    @classmethod
    def _key_as_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("key")
    @classmethod
    def _key_not_blank(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            if not value:
                raise ValueError("key must name at least one column")
            if any(not k.strip() for k in value):
                raise ValueError("key column names must not be blank")
        return value

    @field_validator("reference_sql")
    @classmethod
    def _sql_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("reference_sql must not be blank; omit it instead")
        return value

    @model_validator(mode="after")
    def _resolve_and_check(self) -> Turn:
        if self.compare is None:
            if self.reference_sql is None:
                self.compare = CompareMode.NONE
            elif self.key:
                self.compare = CompareMode.SET
            else:
                self.compare = CompareMode.TABLE

        behavior = self.expect.behavior
        if self.compare is not CompareMode.NONE and self.reference_sql is None:
            raise ValueError(f"compare '{self.compare}' needs a reference_sql")
        if self.compare in KEYED_MODES and not self.key:
            raise ValueError(
                f"compare '{self.compare}' needs a key: the column(s) that identify a record"
            )
        if self.reference_sql is not None and behavior in (Behavior.CLARIFY, Behavior.DECLINE):
            raise ValueError(
                f"a turn that expects the agent to {behavior} cannot also have a reference_sql"
            )
        if self.compare is CompareMode.NONE and behavior is None and not self.expect.trace:
            raise ValueError(
                "this turn checks nothing: add a reference_sql, an expected behavior, "
                "or trace assertions"
            )
        if self.expect_empty and self.expected_rows not in (None, 0):
            raise ValueError("expect_empty is set but expected_rows is not 0")
        return self

    @property
    def expected_behavior(self) -> Behavior | None:
        """The behaviour to score: explicit, or ``answer`` when a reference is given."""
        if self.expect.behavior is not None:
            return self.expect.behavior
        return Behavior.ANSWER if self.reference_sql is not None else None


# Fields that describe a single turn. With ``turns`` they are not allowed at the
# case level, except the three below that act as defaults for every turn.
_TURN_FIELDS = frozenset(Turn.model_fields) | {"golden_sql"}
_INHERITED = ("key", "compare", "tolerance")


class Case(BaseModel):
    """A golden case: a single question, or a conversation of several turns."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(pattern=CASE_ID_PATTERN, max_length=200)
    turns: list[Turn] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
    difficulty: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _source: str | None = PrivateAttr(default=None)

    @property
    def source(self) -> str | None:
        """Where the case was defined, as ``file:line``, when loaded from a file."""
        return self._source

    @model_validator(mode="before")
    @classmethod
    def _gather_turns(cls, data: Any) -> Any:
        """Fold a single-turn case into ``turns`` and push defaults down into each turn."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        turn_fields = {k: data.pop(k) for k in list(data) if k in _TURN_FIELDS}

        if "turns" not in data:
            if "question" not in turn_fields:
                raise ValueError("a case needs either a 'question' or a list of 'turns'")
            data["turns"] = [turn_fields]
            return data

        stray = sorted(k for k in turn_fields if k not in _INHERITED)
        if stray:
            raise ValueError(
                f"{stray} belong inside each turn when a case has 'turns' "
                f"(only {list(_INHERITED)} may be set at case level as defaults)"
            )
        turns = data["turns"]
        if isinstance(turns, list):
            data["turns"] = [{**turn_fields, **t} if isinstance(t, dict) else t for t in turns]
        return data

    @model_validator(mode="after")
    def _no_unknown_fields(self) -> Case:
        extra = sorted(self.model_extra or {})
        if extra:
            raise ValueError(f"unknown field(s) {extra}; put custom data in 'metadata'")
        return self

    @field_validator("tags", mode="before")
    @classmethod
    def _tags_as_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [t.strip() for t in value.replace(";", ",").split(",") if t.strip()]
        return value

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) > 1
