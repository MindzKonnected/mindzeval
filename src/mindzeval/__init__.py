"""MindzEval: evaluation framework for SQL agents.

Grades the data an agent returns rather than the SQL it writes, validates the
reference answers before trusting them, and scores behaviour (answer / clarify /
decline) in both directions.
"""

from mindzeval._version import __version__
from mindzeval.db import Database, QueryResult
from mindzeval.errors import (
    ConfigError,
    DatabaseError,
    Issue,
    MindzEvalError,
    QueryTimeoutError,
    SQLGuardError,
)
from mindzeval.loading import Project, find_config, load_cases, load_config, load_project
from mindzeval.results import TurnResult, Verdict
from mindzeval.runner import RunResult, run, run_project, select_cases
from mindzeval.schema import (
    AgentResponse,
    Behavior,
    Case,
    CompareMode,
    Config,
    Expectation,
    Tolerance,
    ToleranceOverride,
    Turn,
)

__all__ = [
    "AgentResponse",
    "Behavior",
    "Case",
    "CompareMode",
    "Config",
    "ConfigError",
    "Database",
    "DatabaseError",
    "Expectation",
    "Issue",
    "MindzEvalError",
    "Project",
    "QueryResult",
    "QueryTimeoutError",
    "RunResult",
    "SQLGuardError",
    "Tolerance",
    "ToleranceOverride",
    "Turn",
    "TurnResult",
    "Verdict",
    "__version__",
    "find_config",
    "load_cases",
    "load_config",
    "load_project",
    "run",
    "run_project",
    "select_cases",
]
