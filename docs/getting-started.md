# Getting started: test MindzEval on your own agent

This guide takes you from install to a first score on your own SQL agent.

## 1. Install

MindzEval needs Python 3.11 or newer. It isn't on PyPI yet, so install it from GitHub:

```bash
git clone https://github.com/MindzKonnected/mindzeval.git
cd mindzeval
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install ".[postgres]"      # leave out [postgres] if you only use SQLite
mindzeval --version
```

Or install straight from GitHub without cloning:

```bash
pip install "mindzeval[postgres] @ git+https://github.com/MindzKonnected/mindzeval.git@main"
```

`pip install mindzeval` will work once it is published to PyPI.

## 2. Try the example

```bash
mindzeval quickstart demo
cd demo
mindzeval run
```

Open the HTML report in `demo/mindzeval-results/`. The sample agent fails some cases on purpose, so you can see what each verdict means.

## 3. Connect your own agent

1. Copy `demo/mindzeval.yaml` next to your project and point `agent` at your agent: [connecting agents](guides/connecting-agents.md). Runnable examples for specific frameworks are in [mindzeval-integration-examples](https://github.com/MindzKonnected/mindzeval-integration-examples).
2. Point `database.url` at a **read-only** login, using `${ENV_VAR}` for the password.
3. Write 10–20 cases from questions your users really ask: [case format](reference/case-format.md). Include a few vague and out-of-scope questions, plus clear ones that must *not* trigger a clarifying question.
4. Run:

```bash
mindzeval lint       # config and cases
mindzeval validate   # reference queries against the database
mindzeval run
```

## 4. Tell us what happened

MindzEval is early (`0.1.0.dev0`) and interfaces may change before `0.1.0`, so feedback helps. Open an issue in the [mindzeval repo](https://github.com/MindzKonnected/mindzeval/issues) using the feedback template. Most useful:

- how long it took to get a first score, and where you got stuck
- which adapter you used, and whether your agent returns SQL, rows and status
- any verdict you think is **wrong**, with the case and the CSV row
- what you wished it checked or showed
- whether you'd run it in CI

Please don't paste credentials, customer data or full result rows into issues.
