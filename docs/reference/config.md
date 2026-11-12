# Config file

MindzEval reads `mindzeval.yaml` from the working directory, or the file given with `-c`. Relative paths are resolved against the config file's directory. Only `agent` and `datasets` are required.

## Secrets

Never write credentials into the file. Reference environment variables:

- `${NAME}` is replaced with the variable's value. A missing variable is an error, with its line number.
- `${NAME:-default}` uses `default` when the variable is unset or empty.
- `$$` writes a literal `$`.

A literal database password, `Authorization` / API-key header, or judge API key is refused. `mindzeval lint` doesn't substitute variables, so the config can be checked without secrets.

## Full example

```yaml
version: 1
project: shop-agent

agent:
  type: http                       # python | http | outputs
  url: https://agent.internal/ask
  method: POST
  headers:
    Authorization: "Bearer ${AGENT_TOKEN}"
  body:                            # optional; default sends question, history, session_id, params
    query: "{question}"
    conversation: "{history}"
  response:                        # where each field is in the JSON response
    answer: data.text
    sql: data.sql
    rows: data.rows
    status: data.state
    status_map: {NEEDS_INPUT: clarify, REFUSED: decline, DONE: answer}
  timeout_s: 120
  retries: 2
  concurrency: 4
  requests_per_minute: 30
  params:                          # passed to the agent on every call
    model: gpt-5

database:
  url: postgresql://reader:${DB_PASSWORD}@db.internal/shop   # use a read-only login
  statement_timeout_s: 30

datasets:
  - cases/*.yaml
  - cases/regression.csv

scoring:
  key: id                          # default for set/ordered cases without a key
  behavior: true
  uncapped_replay: true
  custom: [my_checks:mentions_currency]   # see guides/custom-scorers.md
  tolerance:
    row_order: ignore              # ignore | strict (table mode)
    relative: 1.0e-6
    absolute: 1.0e-9
    extra_columns: allow           # allow | forbid (table mode)
    case_sensitive: true
    strip_whitespace: true
    null_equals_empty: false
    duplicates: ignore             # ignore | count

validation:                        # block | warn | off
  empty: block
  drift: warn
  orphans: warn
  stability: warn
  null_groups: warn
  relationships:                   # extra links for the orphan check (views, undeclared keys)
    - order_items.order_id -> orders.id

safety:
  block_writes: true
  forbidden_tables: [payroll, audit_log]
  pii_columns: [email, customers.phone]

matrix:                            # run every combination, compare side by side
  model: [gpt-5, claude-sonnet]
  prompt_version: [v1, v2]

judge:                             # reserved for LLM-based checks; unused in this release
  provider: openai
  model: gpt-5-mini
  api_key: ${OPENAI_API_KEY}

output:
  dir: mindzeval-results
  formats: [csv, json, html]

thresholds:                        # CI fails when a set threshold isn't met
  execution_accuracy: 0.8
  behavior_accuracy: 0.9
  max_evaluator_errors: 0
```

## Agent types

| Type | Required | Notes |
|---|---|---|
| `python` | `target: module.path:function` | The function receives the question, history and params, and returns a result record |
| `http` | `url` | Any language. Map response fields with `response`; translate status values with `status_map` |
| `outputs` | `path` | Score answers saved earlier (JSONL or JSON). Can't be combined with `matrix` |

Shared settings for `python` and `http`: `timeout_s`, `retries`, `concurrency`, `requests_per_minute`, `params`.

## Matrix runs

Every combination of the `matrix` values runs as a separate pass, and the passes are compared side by side. Values are passed to the agent alongside `params`. The limit is 64 combinations. `mindzeval lint` shows how many agent calls a run will make.

## Golden validation

`mindzeval validate` (and every `run`) checks each reference query before grading:

| Check | Flags | Default |
|---|---|---|
| reference error | the query fails or isn't a single read | always blocks |
| `empty` | no rows, unless the case sets `expect_empty` | block |
| `drift` | row count (scalar cases: the value) differs from `expected_rows` | warn |
| `stability` | two runs return different results | warn |
| `null_groups` | a grouped reference has a NULL group | warn |
| `orphans` | the answer changes when rows pointing at missing records are excluded | warn |

The orphan check uses foreign keys declared in the database, plus any `relationships` you list (needed for views).

## Databases

Postgres and SQLite in this release. For Postgres, install `mindzeval[postgres]`. SQLite files are opened read-only and never created. Every query runs inside a read-only transaction with the statement timeout, and is refused unless it is a single read statement.
