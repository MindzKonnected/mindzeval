# Custom scorers

Add your own checks in plain Python. A scorer receives a `ScoreContext` for each turn and returns:

- `True` / `False`: passed or failed
- `None`: not applicable to this turn
- a `CustomResult(name, passed, score=None, reason="")`

```python
# my_checks.py
from mindzeval.scoring.custom import CustomResult, ScoreContext

def mentions_currency(ctx: ScoreContext):
    if ctx.case.category != "revenue":
        return None
    ok = "€" in (ctx.response.answer or "") or "EUR" in (ctx.response.answer or "")
    return CustomResult("mentions_currency", ok, reason="revenue answers must state the currency")

def no_select_star(ctx: ScoreContext):
    sql = (ctx.response.sql or "").upper()
    return None if not sql else "SELECT *" not in sql
```

```yaml
scoring:
  custom:
    - my_checks:mentions_currency
    - my_checks:no_select_star
```

`ScoreContext` fields: `case`, `turn`, `turn_index`, `response` (the agent's result record), `reference` (the reference query's result, if any), `params` (including matrix values) and `execution` (the built-in execution score, if any).

A failed custom check fails the turn with cause `custom_check_failed`. If a scorer raises an exception, that is recorded as an **evaluator error** (a bug in the scorer), not as an agent failure.
