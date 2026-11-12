"""Call an agent over HTTP. Works for agents written in any language.

Request body: by default ``{"question", "history", "session_id", "params"}`` as
JSON. A custom ``body`` template is filled per call: a string that is exactly a
placeholder (``"{history}"``) is replaced by the value itself, keeping its type;
placeholders inside longer strings are substituted as text.

Placeholders: ``{question}``, ``{history}``, ``{session_id}``, ``{case_id}``,
``{turn}``, ``{params}`` and ``{params.<name>}``.

Response: each result field is read from the JSON response with a dotted path
(``data.rows``, ``choices.0.text``). Agent status values can be translated with
``status_map``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from mindzeval.adapters.base import (
    Adapter,
    AgentError,
    AgentErrorKind,
    AgentRequest,
    coerce_response,
)
from mindzeval.schema.config import HttpAgentConfig, ResponseMapping
from mindzeval.schema.response import AgentResponse

_PLACEHOLDER = re.compile(r"\{(question|history|session_id|case_id|turn|params(?:\.[\w.]+)?)\}")
_MISSING = object()


class HttpAdapter(Adapter):
    def __init__(self, config: HttpAgentConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self.name = config.url
        self._client = client or httpx.AsyncClient(timeout=config.timeout_s, follow_redirects=False)

    async def call(self, request: AgentRequest) -> AgentResponse:
        config = self._config
        values = _template_values(request)
        body = (
            render_template(config.body, values)
            if config.body is not None
            else {
                "question": request.question,
                "history": values["history"],
                "session_id": request.session_id,
                "params": request.params,
            }
        )
        headers = {"Accept": "application/json", **config.headers}
        try:
            if config.method == "GET":
                params = body if isinstance(body, dict) else None
                response = await self._client.get(
                    config.url, params=_flat(params), headers=headers, timeout=config.timeout_s
                )
            else:
                response = await self._client.request(
                    config.method, config.url, json=body, headers=headers, timeout=config.timeout_s
                )
        except httpx.TimeoutException as exc:
            raise AgentError(
                AgentErrorKind.TIMEOUT, f"no response within {config.timeout_s:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise AgentError(AgentErrorKind.CONNECTION, f"{type(exc).__name__}: {exc}") from exc

        if response.status_code == 429:
            raise AgentError(AgentErrorKind.RATE_LIMITED, "the agent returned HTTP 429")
        if response.status_code >= 400:
            kind = (
                AgentErrorKind.CRASH
                if response.status_code >= 500
                else AgentErrorKind.INVALID_RESPONSE
            )
            raise AgentError(kind, f"HTTP {response.status_code}: {response.text[:300]}")
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            raise AgentError(
                AgentErrorKind.INVALID_RESPONSE,
                f"the response is not JSON: {response.text[:200]!r}",
            ) from None
        return coerce_response(map_response(payload, config.response))

    async def aclose(self) -> None:
        await self._client.aclose()


def map_response(payload: Any, mapping: ResponseMapping) -> dict[str, Any]:
    """Pick each result field out of a JSON payload using the mapping's dotted paths."""
    out: dict[str, Any] = {}
    for field in (
        "answer",
        "sql",
        "params",
        "rows",
        "columns",
        "status",
        "reason",
        "tool_calls",
        "trace",
        "usage",
    ):
        path = getattr(mapping, field)
        if not path:
            continue
        value = get_path(payload, path, _MISSING)
        if value is _MISSING or value is None:
            continue
        out[field] = value

    status = out.get("status")
    if status is not None:
        mapped = mapping.status_map.get(str(status))
        if mapped is not None:
            out["status"] = mapped.value
        elif str(status).lower() in ("answer", "clarify", "decline"):
            out["status"] = str(status).lower()
        else:
            raise AgentError(
                AgentErrorKind.INVALID_RESPONSE,
                f"unknown status {status!r}; map it with agent.response.status_map",
            )
    return out


def get_path(data: Any, path: str, default: Any = None) -> Any:
    """Follow a dotted path through dicts and lists (``a.b.0.c``)."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif (
            isinstance(current, list)
            and part.lstrip("-").isdigit()
            and -len(current) <= int(part) < len(current)
        ):
            current = current[int(part)]
        else:
            return default
    return current


def render_template(template: Any, values: dict[str, Any]) -> Any:
    if isinstance(template, dict):
        return {k: render_template(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(v, values) for v in template]
    if not isinstance(template, str):
        return template
    whole = _PLACEHOLDER.fullmatch(template)
    if whole:
        return _lookup(values, whole.group(1))

    def text(match: re.Match[str]) -> str:
        value = _lookup(values, match.group(1))
        return value if isinstance(value, str) else json.dumps(value, default=str)

    return _PLACEHOLDER.sub(text, template)


def _template_values(request: AgentRequest) -> dict[str, Any]:
    return {
        "question": request.question,
        "history": [m.as_dict() for m in request.history],
        "session_id": request.session_id,
        "case_id": request.case_id,
        "turn": request.turn,
        "params": dict(request.params),
    }


def _lookup(values: dict[str, Any], name: str) -> Any:
    if name.startswith("params."):
        return get_path(values["params"], name[len("params.") :])
    return values[name]


def _flat(params: dict[str, Any] | None) -> dict[str, str] | None:
    if params is None:
        return None
    return {k: v if isinstance(v, str) else json.dumps(v, default=str) for k, v in params.items()}
