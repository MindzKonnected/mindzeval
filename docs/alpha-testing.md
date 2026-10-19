# Internal alpha: how to test MindzEval on your agent

Thanks for trying MindzEval. This round is about finding what's missing or confusing before anything is public, so rough edges are expected and every report helps.

## 1. Install (about 2 minutes)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install "mindzeval[postgres] @ git+https://github.com/MindzKonnected/mindzeval.git@development"
mindzeval --version
```

## 2. Try the example (about 3 minutes)

```bash
mindzeval quickstart demo
cd demo
mindzeval run
```

Open the HTML report in `demo/mindzeval-results/`. The sample agent fails some cases on purpose.

## 3. Connect your own agent

1. Copy `demo/mindzeval.yaml` next to your project and point `agent` at your agent: [connecting agents](guides/connecting-agents.md).
2. Point `database.url` at a **read-only** login, using `${ENV_VAR}` for the password.
3. Write 10–20 cases from questions your users really ask: [case format](reference/case-format.md). Include a few vague and out-of-scope questions, plus clear ones that must *not* trigger a clarifying question.
4. Run:

```bash
mindzeval lint       # config and cases
mindzeval validate   # reference queries against the database
mindzeval run
```

## 4. Tell us what happened

Open an issue in the `mindzeval` repo with the label `alpha-feedback` (or use the issue template). Most useful:

- how long it took to get a first score, and where you got stuck
- which adapter you used, and whether your agent returns SQL, rows and status
- any verdict you think is **wrong**, with the case and the CSV row
- what you wished it checked or showed
- whether you'd run it in CI

Please don't paste credentials, customer data or full result rows into issues.
