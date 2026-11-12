"""Exceptions raised by MindzEval.

Every exception MindzEval raises on purpose derives from :class:`MindzEvalError`,
so callers can tell MindzEval's own problems (bad config, unsafe SQL, database
errors) apart from failures inside the agent under test.
"""

from __future__ import annotations

from dataclasses import dataclass


class MindzEvalError(Exception):
    """Base class for all MindzEval errors."""


@dataclass(frozen=True)
class Issue:
    """One problem found in a config or dataset file.

    ``line`` and ``column`` are 1-based and may be ``None`` when the position is
    unknown (for example, a problem that spans several files).
    """

    source: str
    message: str
    path: str = ""
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        where = self.source
        if self.line is not None:
            where += f":{self.line}"
            if self.column is not None:
                where += f":{self.column}"
        field = f" {self.path}:" if self.path else ""
        return f"{where}:{field} {self.message}"


class ConfigError(MindzEvalError):
    """A config or dataset file is invalid. Carries every issue found, not just the first."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = list(issues)
        super().__init__(self._format())

    def _format(self) -> str:
        count = len(self.issues)
        header = f"{count} problem{'s' if count != 1 else ''} found:"
        return "\n".join([header, *(f"  {issue}" for issue in self.issues)])


class SQLGuardError(MindzEvalError):
    """A SQL statement was refused because it is not a single read-only query.

    ``kind`` says why: ``empty``, ``parse``, ``multiple``, ``not_query``, ``write``,
    ``lock`` or ``function``. The last four are safety issues; the first three
    mean the SQL is malformed.
    """

    UNSAFE_KINDS = frozenset({"not_query", "write", "lock", "function"})

    def __init__(self, message: str, kind: str = "write") -> None:
        super().__init__(message)
        self.kind = kind

    @property
    def unsafe(self) -> bool:
        return self.kind in self.UNSAFE_KINDS


class DatabaseError(MindzEvalError):
    """A query failed while MindzEval was executing it."""


class QueryTimeoutError(DatabaseError):
    """A query ran longer than the configured statement timeout."""
