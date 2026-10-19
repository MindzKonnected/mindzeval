# MindzEval

MindzEval proves your SQL agent is accurate and safe before it reaches users, and keeps proving it on every change.

> **Status: internal alpha (`0.1.0.dev0`).** Interfaces may change before `0.1.0`. Internal testers: start with [docs/alpha-testing.md](docs/alpha-testing.md).

## What it does

- **Grades the data, not the SQL.** The agent's results are compared with the results of a reference query, so correct SQL written a different way still passes.
- **Checks the answer key first.** Reference queries are validated against the live database (empty, drift, unstable, NULL groups, orphan-inflated) before any agent is graded.
- **Scores behaviour both ways.** Asking for clarification or declining is rewarded when it's right and penalised when it isn't.
- **Separates display limits from logic errors.** The agent's SQL is re-run without its display `LIMIT`, so a UI row cap isn't mistaken for a wrong query.
- **Checks safety.** Write attempts, forbidden tables and PII in results or answers.
- **Works with any agent.** LangChain, LangGraph, CrewAI, LlamaIndex, the OpenAI Agents SDK, plain Python, or any HTTP service.
- **Compares models and prompts.** Matrix runs evaluate every combination side by side.
- **Read-only by design.** Every query MindzEval runs is parsed and refused unless it is a single read, then executed in a read-only transaction with a timeout.

## Quickstart

```bash
pip install "mindzeval[postgres] @ git+https://github.com/MindzKonnected/mindzeval.git@development"
mindzeval quickstart demo
cd demo
mindzeval run
```

The example agent fails some cases on purpose so you can see what each verdict means.

## Your own agent

`mindzeval.yaml`:

```yaml
project: shop-agent
agent:
  type: python
  target: my_project.eval_adapter:answer     # or type: http with a URL
database:
  url: postgresql://reader:${DB_PASSWORD}@db.internal/shop
datasets: cases/*.yaml
```

`cases/customers.yaml`:

```yaml
- id: customers_in_germany
  question: Which customers are in Germany?
  reference_sql: SELECT id FROM customers WHERE country = 'DE'
  key: id

- id: vague_request
  question: Show me the data
  expect: {behavior: clarify}
```

```bash
mindzeval lint       # check config and cases (no connections, no secrets)
mindzeval validate   # check reference queries against the database
mindzeval run        # evaluate; exits 1 if thresholds aren't met
```

## Documentation

- [Connecting your agent](docs/guides/connecting-agents.md): Python, LangGraph, LangChain, CrewAI, HTTP, saved outputs, model comparison
- [Case format](docs/reference/case-format.md)
- [Config file](docs/reference/config.md)
- [Agent result record](docs/reference/agent-response.md)
- [Reading results](docs/reference/results.md): verdicts, numbers, failure causes, exit codes
- [Custom scorers](docs/guides/custom-scorers.md)

## Development

```bash
git clone https://github.com/MindzKonnected/mindzeval.git && cd mindzeval
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,postgres]"
pytest              # set MINDZEVAL_TEST_POSTGRES_URL to include Postgres
ruff check src tests && ruff format --check src tests && mypy
```

Work happens on `development` through pull requests. `production` holds released versions.
