# Changelog

All notable changes to MindzEval are recorded here. The project follows
[semantic versioning](https://semver.org/); until `1.0` the public interfaces may change
between minor versions.

## Unreleased

### Added
- **Contracts:** agent result record, case format (single- and multi-turn, six comparison
  modes, composite keys, expected behaviour, trace assertions, tolerance overrides) and the
  `mindzeval.yaml` config.
- **Loading:** YAML / JSON / JSONL / CSV datasets with per-file defaults; every problem
  reported with file, line and field; `${ENV}` secrets; literal credentials refused.
- **Agent adapters:** Python function (sync or async, any framework), HTTP endpoint (body
  templates, response mapping, status translation) and pre-generated outputs; timeouts,
  retries with backoff, rate limiting and concurrency.
- **Database layer:** Postgres and SQLite, read-only transactions, statement timeouts, and
  a SQL guard that refuses anything but a single read query.
- **Golden validator** (`mindzeval validate`): reference errors, empty results, drift,
  stability, NULL groups and orphan-inflated answers (declared foreign keys or configured
  relationships).
- **Scoring:** set, ordered, scalar, grouped and value-based table comparison with a
  configurable tolerance policy; precision / recall / F1; replay of the agent's SQL when it
  returns no rows; uncapped replay to separate display limits from logic errors; behaviour
  scored in both directions; trace assertions; safety checks (write attempts, forbidden
  tables, PII); custom scorers; one failure cause per failed turn.
- **Runs** (`mindzeval run`): filters by tag, case and limit; dry run; matrix runs across
  models / prompts / any parameter; CSV, JSON and HTML reports; console summary; CI
  thresholds and exit codes.
- `mindzeval quickstart`: a runnable example project.
- Docs: connecting agents, case format, config, result record, reading results, custom
  scorers, getting-started guide.

### Changed
- Prepared the repository for public use: the internal alpha guide is now
  `docs/getting-started.md`, the alpha feedback issue template is now a general feedback
  template, install instructions use the default branch, and CI also runs on `main`.
- Documented the `truncated` field of the agent result record.
