# Changelog

All notable changes to MindzEval are recorded here. The project follows
[semantic versioning](https://semver.org/); until `1.0` the public interfaces may change
between minor versions.

## [0.1.1] - 2026-11-18

### Fixed
- **Safety / PII detection for SQL-only agents:** The PII check now inspects replayed query results and answer text when the agent returns only `sql` without pre-computed rows (the minimal contract). Replayed rows from execution scoring are reused directly with zero redundant database queries.
- **Resource limit and memory protection on replay (`max_rows`):** Added `database.max_rows` (default 25,000) to protect against memory exhaustion during uncapped query replay on large warehouse tables. The database layer reads bounded batches using `cursor.fetchmany(max_rows + 1)`. When a query exceeds the limit, uncapped replay is safely skipped and reported as `skipped: result too large to compare`, preventing out-of-memory crashes.
- **Scalar comparison single-row validation:** `_scalar_of` now strictly requires exactly one row (`len(table.rows) == 1`). Multi-row results (e.g. lists returned for count questions) are correctly flagged with failure code `no_single_value` rather than passing coincidentally when row 0 matches. Broken reference queries returning multiple rows are flagged as `reference_no_single_value` and blocked.
- **Release workflow quality gate:** Added a mandatory `check` prerequisite job to `.github/workflows/release.yml` that runs linting, formatting, strict type checking, full pytest execution, and tag-versus-version verification before uploading to PyPI.
- **Reporting clarity:** The HTML and text reports now clearly distinguish between an uncapped match, an uncapped mismatch, and an uncapped comparison skipped due to resource limits (`SKIPPED (RESOURCE LIMIT)`).
- **Saved-output file error reporting:** Malformed JSON lines in JSONL output files now report their exact 1-based line number instead of defaulting to line 1. Non-numeric `turn` values are caught and reported as clean config errors (`Issue(..., "'turn' must be an integer, got ...", line=...)`) instead of crashing with unhandled `ValueError`.
- **Per-turn retry recording:** `TurnResult.attempts` is now accurately recorded for every turn. `ResilientAdapter` tracks attempts per turn and passes them through to `TurnResult.attempts`, enabling detection of flaky agents and rate limits in CSV, JSON, and HTML reports.
- **HTTP error body sanitization:** HTTP adapter error messages are now sanitized to `HTTP <status> (<reason>)` by default, preventing agent stack traces, internal endpoints, and sensitive credentials from leaking into shared evaluation reports. Added `include_error_body` (default `false`) for opt-in local debugging.
- **Documentation and static expected results:** Fully documented static dataset evaluation fields (`expected_value`, `expected_result`, and list form of `expected_rows`) allowing benchmarking without a live database. Added stateful vs. stateless conversation memory guidance for LangGraph to prevent message duplication, and verified robust extraction patterns for LangChain and CrewAI agents with dedicated tests.

## [0.1.0] - 2026-11-12

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
