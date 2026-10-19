# Case format

A **case** is a test question and what a correct agent does with it. Cases live in dataset files referenced from `mindzeval.yaml`: YAML, JSON, JSONL or CSV.

## Single-turn case

```yaml
- id: customers_in_germany            # unique across all datasets
  question: Which customers are in Germany?
  reference_sql: SELECT id FROM customers WHERE country = 'DE'
  key: id                             # column(s) identifying a record
  expected_rows: 42                   # optional: count when the case was written (drift check)
  tags: [smoke, location]
  category: filter
  difficulty: easy
  notes: Country is stored as an ISO code.
```

## Fields

| Field | Description |
|---|---|
| `id` | Letters, digits, `_ . : / -`. Unique across all dataset files |
| `question` | The question sent to the agent |
| `reference_sql` | A query returning the correct answer, run live at evaluation time. `golden_sql` is accepted too |
| `compare` | `set`, `ordered`, `scalar`, `grouped`, `table` or `none` (see below) |
| `key` | Column name, or list of names, identifying a record. Needed for `set` and `ordered` |
| `expect.behavior` | `answer`, `clarify` or `decline`. `behaviour` is accepted too |
| `expect.reason` | The reason the agent should report when it asks or declines |
| `expect.trace` | `{"dotted.path": value}` assertions against the agent's trace |
| `tolerance` | Per-case overrides of the comparison policy (see the config reference) |
| `expected_rows` | Reference row count when the case was written (for `scalar` cases: the value), for drift checks |
| `expect_empty` | `true` when an empty reference result is correct |
| `tags`, `category`, `difficulty`, `notes` | For filtering and report slices |
| `metadata` | Anything else |

## Comparison modes

| Mode | Use for | Compares |
|---|---|---|
| `set` | "Which customers…" | The set of `key` values. Row order and extra columns are ignored |
| `ordered` | "Top 10…", "sorted by…" | The `key` values in order. Set match is reported separately |
| `scalar` | "How many…", "What is the total…" | One value, with numeric tolerance |
| `grouped` | "How many per country…" | (group, value) pairs |
| `table` | Results with no identifying column | Rows by value, under the tolerance policy |
| `none` | Behaviour-only cases | Nothing; only behaviour and trace are checked |

When `compare` is omitted: `none` without a `reference_sql`, `set` when `key` is given, otherwise `table`.

## Behaviour cases

Some questions should not be answered. These cases have no `reference_sql`:

```yaml
- id: too_vague
  question: Show me the data
  expect: {behavior: clarify}

- id: out_of_scope
  question: What's the weather today?
  expect: {behavior: decline}
```

A case with a `reference_sql` expects `answer` unless stated otherwise. A case can't expect `clarify` or `decline` and also give a `reference_sql`.

Include **negative controls**: clear, answerable questions. Asking for clarification on those counts as a failure, so an agent that asks about everything can't score well.

## Multi-turn case

```yaml
- id: narrow_then_count
  key: id                             # default for every turn
  turns:
    - question: Which customers ordered in 2025?
      reference_sql: SELECT DISTINCT customer_id AS id FROM orders WHERE year = 2025
    - question: Just the ones in Germany
      reference_sql: |
        SELECT DISTINCT o.customer_id AS id
        FROM orders o JOIN customers c ON c.id = o.customer_id
        WHERE o.year = 2025 AND c.country = 'DE'
    - question: How many is that?
      reference_sql: SELECT count(DISTINCT ...) FROM ...
      compare: scalar
```

Turns share one conversation. `key`, `compare` and `tolerance` at case level apply to every turn that doesn't set its own.

## File-level defaults

```yaml
defaults:
  key: customer_id
  tags: [regression]
  category: filter
cases:
  - id: ...
```

`defaults` may set `key`, `compare`, `tolerance`, `tags` (merged with each case's tags), `category` and `difficulty`.

## CSV

One row per single-turn case. For multi-turn cases, repeat the `id` and add a `turn` column.

```csv
id,turn,question,reference_sql,compare,key,behavior,tags,meta.owner
customers_de,,Which customers are in Germany?,SELECT id FROM customers WHERE country='DE',,id,,smoke;location,ana
narrow,1,Customers who ordered in 2025,SELECT ...,,id,,,
narrow,2,Just the ones in Germany,SELECT ...,,id,,,
vague,,Show me the data,,,,clarify,,
```

Columns: `id`, `turn`, `question`, `reference_sql`, `compare`, `key` (comma-separated for composite keys), `behavior`, `reason`, `expected_rows`, `expect_empty`, `tags` (`,` or `;` separated), `category`, `difficulty`, `notes`. Custom columns must start with `meta.` and go into `metadata`.
