# Reading results

`mindzeval run` prints a summary and writes reports to `output.dir`:

| File | Use it for |
|---|---|
| `<name>-<time>.csv` | One row per turn: why did this case fail? |
| `<name>-<time>.json` | Rollup, run metadata and every turn. For trends, CI and tooling |
| `<name>-<time>.html` | A readable summary to share (when `html` is in `output.formats`) |

## Verdicts

| Verdict | Meaning |
|---|---|
| `PASS` | Every applicable check passed |
| `FAIL` | At least one check failed; `cause` names the most important one |
| `NOT_SCORED` | Nothing could be checked (e.g. a behaviour-only case and the agent reports no `status`) |
| `BLOCKED` | The reference query failed validation, so the agent wasn't graded. Fix the case |

## Headline numbers

| Number | Meaning |
|---|---|
| pass rate | PASS ÷ (PASS + FAIL) |
| execution accuracy | Turns with a reference query whose result matched, as shown to the user |
| uncapped accuracy | The agent's SQL re-run without its display `LIMIT`. "Was the query right?" |
| correct but capped | Wrong as shown, exact match uncapped: a display limit, not a logic error |
| behavior accuracy | Answer / clarify / decline checks that passed |
| precision / recall | Of what the agent returned, how much was right / of the right answer, how much it found |

High precision with low recall means the agent's filter is too strict (or truncated); low precision with high recall means too loose.

## Failure causes

Each failed turn has exactly one cause, chosen in this order:

| Cause | Meaning |
|---|---|
| `agent_timeout`, `agent_rate_limited`, `agent_unreachable`, `agent_crash` | The agent call failed |
| `invalid_response`, `missing_output` | The agent replied, but not with a valid result record / no saved output |
| `safety_violation` | Write attempt, forbidden table, or PII in the result or answer |
| `asked_instead_of_answering`, `declined_instead_of_answering` | It should have answered |
| `answered_instead_of_clarifying`, `answered_instead_of_declining`, `wrong_behavior` | It should have asked / declined |
| `no_answer` | No SQL and no rows came back |
| `answered_without_query` | An answer with no SQL or rows behind it: a confident number with no source |
| `sql_error` | The agent's SQL failed when MindzEval ran it |
| `missing_columns`, `extra_columns` | The identifying column is missing / extra columns not allowed |
| `truncated` | Cut short, and every row shown was correct |
| `wrong_order` | Right records, wrong order |
| `missing_rows`, `extra_rows`, `wrong_rows` | Too strict / too loose / both |
| `no_single_value`, `wrong_value` | Count or total cases |
| `wrong_groups`, `wrong_values` | Grouped cases: different groups / same groups, different values |
| `wrong_reason`, `trace_mismatch`, `custom_check_failed` | Reason, trace or custom checks |

## Evaluator health

Shown separately because these are **not** the agent's fault:

- **blocked turns**: reference queries that failed validation
- **validation warnings**: references that ran but look suspicious (drift, unstable results, NULL groups, orphan-inflated answers). Check these before blaming the agent: sometimes the reference is wrong and the agent is right
- **evaluator errors**: problems inside MindzEval or a custom scorer

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success, thresholds met |
| 1 | Thresholds not met, or `validate` found blocking problems |
| 2 | Config, dataset or usage error: nothing was evaluated |

`thresholds.max_evaluator_errors` (default 0) also counts blocked turns, so a broken answer key fails CI rather than producing a misleading number.
