from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from mindzeval import AgentResponse, Behavior, ConfigError
from mindzeval.adapters import (
    AgentError,
    AgentErrorKind,
    AgentRequest,
    HttpAdapter,
    Message,
    OutputsAdapter,
    PythonAdapter,
    ResilientAdapter,
)
from mindzeval.adapters.http import get_path, render_template
from mindzeval.schema.config import HttpAgentConfig

REQUEST = AgentRequest(
    question="Which customers?",
    history=[Message("user", "hi"), Message("assistant", "hello")],
    session_id="s1",
    case_id="c1",
    turn=1,
    params={"model": "m1"},
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# --- python -----------------------------------------------------------------


def test_python_passes_only_declared_arguments() -> None:
    seen: dict[str, Any] = {}

    def agent(question: str, params: dict[str, Any]) -> dict[str, Any]:
        seen.update(question=question, params=params)
        return {"answer": "ok", "sql": "SELECT 1"}

    response = run(PythonAdapter(agent).call(REQUEST))
    assert response.sql == "SELECT 1"
    assert seen == {"question": "Which customers?", "params": {"model": "m1"}}


def test_python_async_and_kwargs_and_request_styles() -> None:
    async def a(question: str, **kwargs: Any) -> str:
        return f"{question}|{kwargs['history'][0]['content']}|{kwargs['session_id']}"

    def b(request: AgentRequest) -> AgentResponse:
        return AgentResponse(answer=request.case_id, status=Behavior.CLARIFY)

    assert run(PythonAdapter(a).call(REQUEST)).answer == "Which customers?|hi|s1"
    assert run(PythonAdapter(b).call(REQUEST)).status is Behavior.CLARIFY


def test_python_rejects_unusable_signature() -> None:
    def bad(prompt: str) -> str:
        return prompt

    with pytest.raises(ConfigError, match="must accept 'question'"):
        PythonAdapter(bad)


def test_python_agent_exceptions_become_agent_errors() -> None:
    class RateLimitError(Exception):
        pass

    def crashes(question: str) -> None:
        raise ValueError("boom")

    def limited(question: str) -> None:
        raise RateLimitError("slow down")

    with pytest.raises(AgentError) as crash:
        run(PythonAdapter(crashes).call(REQUEST))
    assert crash.value.kind is AgentErrorKind.CRASH and "boom" in crash.value.message
    with pytest.raises(AgentError) as limit:
        run(PythonAdapter(limited).call(REQUEST))
    assert limit.value.kind is AgentErrorKind.RATE_LIMITED


def test_invalid_return_value() -> None:
    def typo(question: str) -> dict[str, str]:
        return {"answer": "x", "SQL": "SELECT 1"}

    with pytest.raises(AgentError) as exc:
        run(PythonAdapter(typo).call(REQUEST))
    assert exc.value.kind is AgentErrorKind.INVALID_RESPONSE and "SQL" in exc.value.message


# --- http -------------------------------------------------------------------


def _http(handler: Any, **config: Any) -> HttpAdapter:
    cfg = HttpAgentConfig.model_validate(
        {"type": "http", "url": "https://agent.test/ask", **config}
    )
    return HttpAdapter(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_http_default_body_and_mapping() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content), auth=request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={
                "data": {
                    "text": "2 found",
                    "query": "SELECT id FROM t",
                    "cols": ["id"],
                    "rows": [[1], [2]],
                    "state": "DONE",
                }
            },
        )

    adapter = _http(
        handler,
        headers={"Authorization": "Bearer t"},
        response={
            "answer": "data.text",
            "sql": "data.query",
            "columns": "data.cols",
            "rows": "data.rows",
            "status": "data.state",
            "status_map": {"DONE": "answer"},
        },
    )
    response = run(adapter.call(REQUEST))
    assert response.rows == [{"id": 1}, {"id": 2}] and response.status is Behavior.ANSWER
    assert captured["question"] == "Which customers?" and captured["params"] == {"model": "m1"}
    assert captured["history"][1] == {"role": "assistant", "content": "hello"}
    assert captured["auth"] == "Bearer t"


def test_http_custom_body_template() -> None:
    body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    adapter = _http(
        handler,
        body={
            "input": {"q": "{question}", "messages": "{history}"},
            "model": "{params.model}",
            "tag": "case {case_id}",
        },
    )
    run(adapter.call(REQUEST))
    assert body["input"]["messages"][0]["content"] == "hi"
    assert body["model"] == "m1" and body["tag"] == "case c1"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (429, AgentErrorKind.RATE_LIMITED),
        (500, AgentErrorKind.CRASH),
        (401, AgentErrorKind.INVALID_RESPONSE),
    ],
)
def test_http_error_statuses(status: int, kind: AgentErrorKind) -> None:
    adapter = _http(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(AgentError) as exc:
        run(adapter.call(REQUEST))
    assert exc.value.kind is kind


def test_http_unknown_status_value_needs_mapping() -> None:
    adapter = _http(
        lambda r: httpx.Response(200, json={"answer": "?", "status": "HITL"}),
        response={"status": "status"},
    )
    with pytest.raises(AgentError, match="status_map"):
        run(adapter.call(REQUEST))


def test_get_path_and_templates() -> None:
    data = {"a": [{"b": 1}, {"b": 2}]}
    assert get_path(data, "a.1.b") == 2 and get_path(data, "a.5.b", "x") == "x"
    assert render_template("{turn}", {"turn": 3}) == 3
    assert render_template("t{turn}", {"turn": 3}) == "t3"


# --- outputs ----------------------------------------------------------------


def test_outputs_adapter(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"case_id": "c1", "turn": 1, "response": {"answer": "a", "sql": "SELECT 1"}}\n'
        '{"case_id": "c2", "answer": "flat"}\n',
        encoding="utf-8",
    )
    adapter = OutputsAdapter(path)
    assert run(adapter.call(REQUEST)).sql == "SELECT 1"
    with pytest.raises(AgentError) as exc:
        run(adapter.call(AgentRequest(question="q", case_id="missing")))
    assert exc.value.kind is AgentErrorKind.MISSING_OUTPUT


def test_outputs_duplicates_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text('{"case_id": "a"}\n{"case_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        OutputsAdapter(path)


# --- resilience -------------------------------------------------------------


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def flaky(question: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("down")
        return "ok"

    adapter = ResilientAdapter(PythonAdapter(flaky), retries=2, backoff_s=0.001)
    assert run(adapter.call(REQUEST)).answer == "ok" and adapter.attempts == 3


def test_crashes_are_not_retried_and_timeouts_are() -> None:
    def crash(question: str) -> str:
        raise ValueError("x")

    adapter = ResilientAdapter(PythonAdapter(crash), retries=3, backoff_s=0.001)
    with pytest.raises(AgentError):
        run(adapter.call(REQUEST))
    assert adapter.attempts == 1

    async def slow(question: str) -> str:
        await asyncio.sleep(1)
        return "late"

    timed = ResilientAdapter(PythonAdapter(slow), timeout_s=0.05, retries=1, backoff_s=0.001)
    with pytest.raises(AgentError) as exc:
        run(timed.call(REQUEST))
    assert exc.value.kind is AgentErrorKind.TIMEOUT and timed.attempts == 2
