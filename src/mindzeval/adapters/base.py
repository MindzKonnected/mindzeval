"""The adapter interface: how MindzEval calls an agent, whatever it is built with."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from mindzeval.schema.response import AgentResponse


@dataclass(frozen=True)
class Message:
    """One message of the conversation so far."""

    role: str  # "user" or "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class AgentRequest:
    """Everything an adapter passes to the agent for one turn."""

    question: str
    history: list[Message] = field(default_factory=list)
    session_id: str = ""
    """Stable across the turns of one case, unique per case and run."""
    case_id: str = ""
    turn: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    """Config ``params`` merged with the current matrix combination."""


class AgentErrorKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CONNECTION = "connection"
    CRASH = "crash"
    """The agent raised an exception or returned a server error."""
    INVALID_RESPONSE = "invalid_response"
    """The agent replied, but not with a valid result record."""
    MISSING_OUTPUT = "missing_output"
    """Pre-generated outputs have no entry for this turn."""


RETRYABLE = frozenset(
    {AgentErrorKind.TIMEOUT, AgentErrorKind.RATE_LIMITED, AgentErrorKind.CONNECTION}
)


class AgentError(Exception):
    """The agent call failed. These are agent-side failures, never MindzEval bugs."""

    def __init__(self, kind: AgentErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE


class Adapter(ABC):
    """Calls one agent. Implementations must be safe to call concurrently."""

    name: str = "agent"

    @abstractmethod
    async def call(self, request: AgentRequest) -> AgentResponse:
        """Return the agent's result for one turn, or raise :class:`AgentError`."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release resources (connections, clients)."""


def coerce_response(value: Any) -> AgentResponse:
    """Turn whatever an agent returned into an :class:`AgentResponse`, or raise AgentError."""
    if isinstance(value, AgentResponse):
        return value
    if value is None:
        raise AgentError(AgentErrorKind.INVALID_RESPONSE, "the agent returned nothing (None)")
    if hasattr(value, "model_dump") and not isinstance(value, dict):
        value = value.model_dump()
    try:
        return AgentResponse.model_validate(value)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'response'}: "
            f"{str(e['msg']).removeprefix('Value error, ')}"
            for e in exc.errors(include_url=False)
        )
        raise AgentError(
            AgentErrorKind.INVALID_RESPONSE, f"invalid result record: {problems}"
        ) from None
