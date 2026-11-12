# Connecting your agent

MindzEval calls your agent once per turn and needs back a [result record](../reference/agent-response.md): at least the **answer** and the **SQL** it ran (or the **rows**). Returning `status` (`answer` / `clarify` / `decline`) turns on behaviour scoring.

Pick the option that matches how your agent runs.

| Your agent is… | Use | Effort |
|---|---|---|
| Python code you can import | `type: python` | ~10 lines |
| A service with an HTTP API (any language) | `type: http` | config only |
| Something you can't call during the eval | `type: outputs` | export a file |

## Python function

```yaml
agent:
  type: python
  target: my_project.eval_adapter:answer   # module:function, relative to mindzeval.yaml
```

The function may accept any of `question`, `history`, `session_id`, `case_id`, `turn`, `params`; MindzEval passes only the ones it declares. It can be sync or async. `history` is a list of `{"role": "user" | "assistant", "content": str}` for earlier turns of the same case. `params` holds `agent.params` plus the current matrix values (for example `{"model": "gpt-5"}`).

### LangGraph

```python
# my_project/eval_adapter.py
from my_project.graph import build_graph

graph = build_graph()  # build once, reuse across cases

async def answer(question, history, session_id, params):
    messages = [(m["role"], m["content"]) for m in history] + [("user", question)]
    state = await graph.ainvoke(
        {"messages": messages},
        config={"configurable": {"thread_id": session_id, "model": params.get("model")}},
    )
    return {
        "answer": state["messages"][-1].content,
        "sql": state.get("sql"),            # wherever your graph keeps the final SQL
        "rows": state.get("rows"),          # optional: MindzEval re-runs the SQL if omitted
        "status": state.get("status", "answer"),
        "trace": {"plan": state.get("plan")},  # optional, for trace assertions
    }
```

If your graph doesn't store the SQL in its state, read it from the last call of your SQL tool in `state["messages"]` (the `ToolMessage` or the `AIMessage.tool_calls` arguments).

### LangChain SQL agent

```python
from langchain_community.agent_toolkits import create_sql_agent

agent = create_sql_agent(llm, db=db, agent_type="tool-calling", return_intermediate_steps=True)

def answer(question):
    result = agent.invoke({"input": question})
    sql = None
    for action, _observation in result["intermediate_steps"]:
        if action.tool == "sql_db_query":
            sql = action.tool_input["query"] if isinstance(action.tool_input, dict) else action.tool_input
    return {"answer": result["output"], "sql": sql, "status": "answer"}
```

### CrewAI

```python
from my_project.crew import build_crew

def answer(question):
    output = build_crew().kickoff(inputs={"question": question})
    data = output.json_dict or {}          # if your task uses output_json / output_pydantic
    return {"answer": output.raw, "sql": data.get("sql"), "rows": data.get("rows")}
```

### Reporting clarifications and refusals

Behaviour scoring needs `status`. Your agent usually knows when it asked a question or declined; return it:

```python
if needs_clarification:
    return {"answer": clarifying_question, "status": "clarify", "reason": "AMBIGUOUS_TERM"}
if out_of_scope:
    return {"answer": refusal_text, "status": "decline", "reason": "OUT_OF_DOMAIN"}
```

Without `status`, behaviour checks are skipped and the report says so. MindzEval never guesses from the answer text.

## HTTP endpoint

```yaml
agent:
  type: http
  url: https://agent.internal/v1/ask
  headers:
    Authorization: "Bearer ${AGENT_TOKEN}"
  body:                         # optional; this is a custom shape
    input: "{question}"
    messages: "{history}"
    thread_id: "{session_id}"
    model: "{params.model}"
  response:                     # dotted paths into the JSON response
    answer: output.text
    sql: output.sql
    rows: output.rows
    columns: output.columns     # only if rows are lists
    status: output.state
    status_map: {NEEDS_INPUT: clarify, REFUSED: decline, OK: answer}
```

Without `body`, MindzEval sends `{"question", "history", "session_id", "params"}`. A string that is exactly one placeholder keeps its type (`"{history}"` becomes a list); placeholders inside longer strings are substituted as text.

HTTP 429 and timeouts are retried with backoff (`retries`). Use `requests_per_minute` to stay under your LLM provider's rate limit and `concurrency` to run cases in parallel.

## Pre-generated outputs

Score answers produced elsewhere (a batch job, another team's system, a past run):

```yaml
agent:
  type: outputs
  path: outputs.jsonl
```

```json
{"case_id": "customers_in_germany", "turn": 0, "response": {"answer": "...", "sql": "SELECT ..."}}
```

## Comparing models or prompts

```yaml
agent:
  type: python
  target: my_project.eval_adapter:answer
matrix:
  model: [gpt-5, claude-sonnet]
  prompt_version: [v1, v2]
```

Every combination runs as its own pass and the report compares them side by side, including the turns whose verdict differs. `mindzeval run --list` shows how many agent calls that means before anything runs.
