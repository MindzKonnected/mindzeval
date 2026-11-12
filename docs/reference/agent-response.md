# Agent result record

For each turn, the agent returns a result record. MindzEval scores what it is given and reports which checks were skipped because a field was missing. The model is `mindzeval.AgentResponse`.

**Minimum:** `answer` plus `sql`, or `answer` plus `rows`.

| Field | Type | Needed for |
|---|---|---|
| `answer` | string | The text shown to the user. Answer checks and reports |
| `sql` | string | The final SQL the agent ran. Replay, uncapped scoring, SQL safety checks |
| `params` | object or list | Bind parameters for `sql`, in the driver's own style (`%(name)s`, `?`, `:name`) |
| `rows` | list of objects, or list of lists with `columns` | Result comparison. If absent, MindzEval re-runs `sql` to get them |
| `columns` | list of strings | Column order. Required when `rows` are lists; inferred otherwise |
| `truncated` | boolean | Set `true` when `rows` are only part of the result, for example when you cap rows in code instead of in the SQL. Without it, MindzEval infers truncation only from a trailing `LIMIT n` in `sql` |
| `status` | `answer` \| `clarify` \| `decline` | Behaviour scoring. Skipped (and reported as skipped) when absent |
| `reason` | string | Why the agent asked or declined, in the agent's own vocabulary |
| `tool_calls` | list of `{name, arguments, output, error}` | Tool-use reporting |
| `trace` | object | Trace assertions (`plan.operation`, `steps.0.tool`, …) |
| `usage` | `{input_tokens, output_tokens, llm_calls, cost_usd}` | Cost and usage reporting |
| `metadata` | object | Anything else, kept in reports |

Rules:
- A plain string is accepted as shorthand for `{"answer": "..."}`.
- Column names must be unique. Alias repeated names (`c.id AS customer_id`).
- Unknown fields are rejected, so a typo like `"SQL"` fails loudly instead of looking like a missing value. Put extra data in `metadata`.

Example:

```json
{
  "answer": "3 customers are based in Germany.",
  "sql": "SELECT id, name FROM customers WHERE country = %(c)s",
  "params": {"c": "DE"},
  "columns": ["id", "name"],
  "rows": [[1, "Ada"], [3, "Cy"], [7, "Ed"]],
  "status": "answer",
  "usage": {"input_tokens": 1820, "output_tokens": 96, "llm_calls": 3}
}
```
