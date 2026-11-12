"""Agent adapters: Python function, HTTP endpoint, or pre-generated outputs."""

from __future__ import annotations

from mindzeval.adapters.base import (
    Adapter,
    AgentError,
    AgentErrorKind,
    AgentRequest,
    Message,
    coerce_response,
)
from mindzeval.adapters.http import HttpAdapter
from mindzeval.adapters.outputs import OutputsAdapter
from mindzeval.adapters.python import PythonAdapter, load_target
from mindzeval.adapters.resilient import ResilientAdapter
from mindzeval.schema.config import (
    Config,
    HttpAgentConfig,
    OutputsAgentConfig,
    PythonAgentConfig,
)


def build_adapter(config: Config) -> Adapter:
    """Create the adapter described by ``config.agent``, with retries and rate limiting."""
    agent = config.agent
    if isinstance(agent, OutputsAgentConfig):
        return OutputsAdapter(config.resolve_path(agent.path))
    inner: Adapter
    if isinstance(agent, PythonAgentConfig):
        inner = PythonAdapter(load_target(agent.target, config.base_dir), name=agent.target)
    elif isinstance(agent, HttpAgentConfig):
        inner = HttpAdapter(agent)
    else:  # pragma: no cover - the union is exhaustive
        raise TypeError(f"unsupported agent config {type(agent).__name__}")
    return ResilientAdapter(
        inner,
        timeout_s=agent.timeout_s,
        retries=agent.retries,
        requests_per_minute=agent.requests_per_minute,
    )


__all__ = [
    "Adapter",
    "AgentError",
    "AgentErrorKind",
    "AgentRequest",
    "HttpAdapter",
    "Message",
    "OutputsAdapter",
    "PythonAdapter",
    "ResilientAdapter",
    "build_adapter",
    "coerce_response",
    "load_target",
]
