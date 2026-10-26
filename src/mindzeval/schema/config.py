"""The project config (``mindzeval.yaml``).

One file describes how to reach the agent and the database, which datasets to
run, how strictly to compare results, which checks to apply and where to write
reports. Every section except ``agent`` and ``datasets`` has defaults.

Secrets never belong in the file itself: reference environment variables with
``${NAME}`` (or ``${NAME:-default}``) and MindzEval substitutes them at load time.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from mindzeval.schema.case import ToleranceOverride
from mindzeval.schema.response import Behavior

_TARGET_PATTERN = r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$"
MAX_MATRIX_COMBINATIONS = 64


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class _AgentBase(_Strict):
    timeout_s: float = Field(default=120.0, gt=0)
    """Time allowed for one agent call."""

    retries: int = Field(default=2, ge=0, le=10)
    """Retries after a timeout, connection error or rate limit."""

    concurrency: int = Field(default=1, ge=1, le=64)
    """Cases run in parallel. Turns of one multi-turn case always run in order."""

    requests_per_minute: float | None = Field(default=None, gt=0)
    """Upper bound on agent calls per minute, to stay under provider rate limits."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Passed to the agent on every call (e.g. ``model``, ``prompt_version``)."""


class PythonAgentConfig(_AgentBase):
    """An agent called as a Python function: ``module.path:function``."""

    type: Literal["python"]
    target: str = Field(pattern=_TARGET_PATTERN)
    """Import path of the callable, e.g. ``my_project.agent:answer``."""


class ResponseMapping(_Strict):
    """Where each result field sits in the agent's JSON response (dotted paths)."""

    answer: str | None = "answer"
    sql: str | None = "sql"
    params: str | None = None
    rows: str | None = "rows"
    columns: str | None = None
    status: str | None = None
    reason: str | None = None
    tool_calls: str | None = None
    trace: str | None = None
    usage: str | None = None

    status_map: dict[str, Behavior] = Field(default_factory=dict)
    """Translate the agent's own status values, e.g. ``{"NEEDS_INPUT": "clarify"}``."""


class HttpAgentConfig(_AgentBase):
    """An agent reached over HTTP. Works for agents written in any language."""

    type: Literal["http"]
    url: str = Field(pattern=r"^https?://")
    method: Literal["POST", "PUT", "GET"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    """Request headers. Use ``${ENV_VAR}`` for tokens; values are never printed."""

    body: Any = None
    """Request body template. ``{question}``, ``{history}``, ``{session_id}`` and
    ``{params.<name>}`` are filled in per call. Defaults to a JSON body with
    ``question``, ``history``, ``session_id`` and ``params``."""

    response: ResponseMapping = Field(default_factory=ResponseMapping)


class OutputsAgentConfig(_Strict):
    """Pre-generated answers read from a file instead of calling an agent."""

    type: Literal["outputs"]
    path: str
    """JSONL or JSON file of ``{case_id, turn, response}`` records."""


AgentConfig = Annotated[
    PythonAgentConfig | HttpAgentConfig | OutputsAgentConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Database, scoring, checks
# ---------------------------------------------------------------------------


class DatabaseConfig(_Strict):
    url: SecretStr
    """SQLAlchemy URL, e.g. ``postgresql://${DB_USER}:${DB_PASSWORD}@host/db`` or
    ``sqlite:///data/sample.db``. Use a read-only login."""

    statement_timeout_s: float = Field(default=30.0, gt=0, le=3600)
    dialect: str | None = None
    """SQL dialect for parsing. Inferred from the URL when omitted."""


class ScoringConfig(_Strict):
    key: list[str] | None = None
    """Default identity column(s) for ``set`` / ``ordered`` cases that don't set one."""

    tolerance: ToleranceOverride = Field(default_factory=ToleranceOverride)
    behavior: bool = True
    """Score answer / clarify / decline behaviour when the agent reports ``status``."""

    uncapped_replay: bool = True
    """Re-run the agent's SQL without its display LIMIT to separate truncation from
    logic errors."""

    custom: list[str] = Field(default_factory=list)
    """Custom scorers as ``module.path:callable``. See the custom scorer guide."""

    @field_validator("custom")
    @classmethod
    def _custom_targets(cls, value: list[str]) -> list[str]:
        bad = [v for v in value if not re.match(_TARGET_PATTERN, v)]
        if bad:
            raise ValueError(f"custom scorers must look like 'module.path:callable', not {bad}")
        return value

    @field_validator("key", mode="before")
    @classmethod
    def _key_as_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class Severity(StrEnum):
    BLOCK = "block"
    """Exclude the case from scoring and report it."""
    WARN = "warn"
    """Score the case and report the warning."""
    OFF = "off"
    """Skip the check."""


class ValidationConfig(_Strict):
    """Checks applied to reference queries before any agent is scored."""

    empty: Severity = Severity.BLOCK
    drift: Severity = Severity.WARN
    orphans: Severity = Severity.WARN
    stability: Severity = Severity.WARN
    null_groups: Severity = Severity.WARN

    relationships: list[str] = Field(default_factory=list)
    """Extra references for the orphan check, as ``child.column -> parent.column``
    (schema-qualified names allowed). Foreign keys declared in the database are used
    automatically; list relationships here for views or undeclared keys."""

    @field_validator("relationships")
    @classmethod
    def _relationships(cls, value: list[str]) -> list[str]:
        for item in value:
            parse_relationship(item)
        return value


def parse_relationship(
    text: str,
) -> tuple[tuple[str | None, str, str], tuple[str | None, str, str]]:
    """Parse ``[schema.]table.column -> [schema.]table.column``."""

    def side(part: str) -> tuple[str | None, str, str]:
        pieces = [p.strip() for p in part.strip().split(".")]
        if len(pieces) not in (2, 3) or not all(pieces):
            raise ValueError(f"'{text}': each side must be table.column or schema.table.column")
        if len(pieces) == 2:
            return None, pieces[0], pieces[1]
        return pieces[0], pieces[1], pieces[2]

    if text.count("->") != 1:
        raise ValueError(f"'{text}': expected 'child.column -> parent.column'")
    left, right = text.split("->")
    return side(left), side(right)


class SafetyConfig(_Strict):
    """Deterministic safety checks on the agent's SQL and output."""

    block_writes: bool = True
    forbidden_tables: list[str] = Field(default_factory=list)
    pii_columns: list[str] = Field(default_factory=list)
    """Columns that must not appear in results or answers, as ``column`` or
    ``table.column``."""


class JudgeConfig(_Strict):
    """Model used by LLM-based checks. Reserved: no check in this release uses it."""

    provider: str
    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)


ReportFormat = Literal["csv", "json", "html"]


def _default_formats() -> list[ReportFormat]:
    return ["csv", "json", "html"]


class OutputConfig(_Strict):
    dir: str = "mindzeval-results"
    formats: list[ReportFormat] = Field(default_factory=_default_formats)


class Thresholds(_Strict):
    """Pass/fail gates for CI. A run fails if any set threshold is not met."""

    execution_accuracy: float | None = Field(default=None, ge=0, le=1)
    uncapped_accuracy: float | None = Field(default=None, ge=0, le=1)
    behavior_accuracy: float | None = Field(default=None, ge=0, le=1)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    max_evaluator_errors: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


class Config(_Strict):
    version: Literal[1] = 1
    project: str | None = None
    agent: AgentConfig
    database: DatabaseConfig | None = None
    datasets: list[str] = Field(min_length=1)
    """Dataset files or glob patterns, relative to the config file."""

    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    matrix: dict[str, list[Any]] = Field(default_factory=dict)
    """Run every combination of these agent params and compare them side by side."""

    judge: JudgeConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)
    """Directory relative paths are resolved against. Set by the loader."""

    @field_validator("datasets", mode="before")
    @classmethod
    def _datasets_as_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @field_validator("matrix")
    @classmethod
    def _matrix_values(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for name, options in value.items():
            if not options:
                raise ValueError(f"matrix '{name}' needs at least one value")
        return value

    @model_validator(mode="after")
    def _check_matrix(self) -> Config:
        if not self.matrix:
            return self
        if isinstance(self.agent, OutputsAgentConfig):
            raise ValueError("matrix runs need a live agent; they can't be used with 'outputs'")
        combos = 1
        for options in self.matrix.values():
            combos *= len(options)
        if combos > MAX_MATRIX_COMBINATIONS:
            raise ValueError(
                f"matrix has {combos} combinations; the limit is {MAX_MATRIX_COMBINATIONS}"
            )
        return self

    @property
    def matrix_combinations(self) -> list[dict[str, Any]]:
        """Every combination of matrix values, in a stable order. ``[{}]`` without a matrix."""
        combos: list[dict[str, Any]] = [{}]
        for name, options in self.matrix.items():
            combos = [{**c, name: option} for c in combos for option in options]
        return combos

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` against the config file's directory."""
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p)
