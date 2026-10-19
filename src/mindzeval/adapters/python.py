"""Call an agent that is a Python function (LangChain, LangGraph, CrewAI, plain code...).

The function can take any subset of these keyword arguments, and MindzEval
passes only the ones it declares:

    question: str            the question for this turn
    history: list[dict]      earlier turns as {"role": "user"|"assistant", "content": str}
    session_id: str          stable across the turns of one case
    case_id: str, turn: int  identity of the turn being evaluated
    params: dict             config params plus the current matrix combination

Alternatively it can take a single ``request`` argument (an
:class:`~mindzeval.adapters.AgentRequest`). It may be sync or async, and may
return an :class:`~mindzeval.AgentResponse`, a dict, a pydantic model or a string.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from collections.abc import Callable
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

_ARGUMENTS = ("question", "history", "session_id", "case_id", "turn", "params")


def load_target(target: str, base_dir: Path | None = None) -> Callable[..., Any]:
    """Import ``module.path:attribute``. ``base_dir`` is put on ``sys.path`` first."""
    module_name, _, attribute = target.partition(":")
    if base_dir is not None and str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError([Issue("agent.target", f"cannot import '{module_name}': {exc}")]) from exc
    for part in attribute.split("."):
        if not hasattr(obj, part):
            raise ConfigError([Issue("agent.target", f"'{module_name}' has no '{attribute}'")])
        obj = getattr(obj, part)
    if not callable(obj):
        raise ConfigError([Issue("agent.target", f"'{target}' is not callable")])
    return obj  # type: ignore[no-any-return]


class PythonAdapter(Adapter):
    def __init__(self, func: Callable[..., Any], *, name: str | None = None) -> None:
        self._func = func
        self.name = name or str(getattr(func, "__qualname__", "python-agent"))
        self._is_async = inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
            getattr(func, "__call__", None)  # noqa: B004 - callable objects
        )
        self._mode, self._accepted = _inspect_signature(func)

    def _arguments(self, request: AgentRequest) -> dict[str, Any]:
        if self._mode == "request":
            return {"request": request}
        values: dict[str, Any] = {
            "question": request.question,
            "history": [m.as_dict() for m in request.history],
            "session_id": request.session_id,
            "case_id": request.case_id,
            "turn": request.turn,
            "params": dict(request.params),
        }
        if self._mode == "kwargs":
            return values
        return {k: v for k, v in values.items() if k in self._accepted}

    async def call(self, request: AgentRequest) -> AgentResponse:
        kwargs = self._arguments(request)
        try:
            if self._is_async:
                value = await self._func(**kwargs)
            else:
                value = await asyncio.to_thread(self._func, **kwargs)
                if inspect.isawaitable(value):
                    value = await value
        except AgentError:
            raise
        except Exception as exc:  # the agent's own failure
            raise AgentError(_classify(exc), f"{type(exc).__name__}: {exc}") from exc
        return coerce_response(value)


def _inspect_signature(func: Callable[..., Any]) -> tuple[str, frozenset[str]]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return "kwargs", frozenset()
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return "kwargs", frozenset()
    names = [n for n, p in params.items() if p.kind is not inspect.Parameter.VAR_POSITIONAL]
    if names == ["request"]:
        return "request", frozenset()
    accepted = frozenset(n for n in names if n in _ARGUMENTS)
    required = [
        n
        for n, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    unknown = [n for n in required if n not in _ARGUMENTS]
    if unknown or "question" not in accepted:
        raise ConfigError(
            [
                Issue(
                    "agent.target",
                    f"the agent function must accept 'question' (and optionally "
                    f"{', '.join(_ARGUMENTS[1:])}); it has required parameters {required}",
                )
            ]
        )
    return "named", accepted


def _classify(exc: BaseException) -> AgentErrorKind:
    if isinstance(exc, TimeoutError):
        return AgentErrorKind.TIMEOUT
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status == 429 or "ratelimit" in type(exc).__name__.lower():
        return AgentErrorKind.RATE_LIMITED
    if isinstance(exc, ConnectionError):
        return AgentErrorKind.CONNECTION
    return AgentErrorKind.CRASH
