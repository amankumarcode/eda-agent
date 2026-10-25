# Data Validation Agent

Checks a customer's raw data files against an internal data model and reports what is
present, missing, or wrong on day one, instead of weeks into integration. It maps the
customer's columns to the data model, runs deterministic plausibility checks, runs
exploratory analysis by writing and executing Python in a sandbox, then writes an issue
report and drafts a customer email and a Jira ticket. **Drafts go to a review queue and
are never sent.**

The included data model, `retail_sales`, covers daily sales transactions, a product
master and a store master for demand-forecasting onboarding. Data models are
configuration, so other domains plug in without code changes.

## How it works

```
                    ┌──────────────────────────────────────────────┐
  files / message ─►│ orchestrator (LLM chooses the next tool)     │◄──────────┐
                    └──────────────┬───────────────────────────────┘           │
                                   │ code checks before following:             │
                                   │ budgets · tool-call cap · no progress ·   │
                                   │ coverage gate (+1 EDA run in batch)       │
        ┌──────────────┬───────────┼──────────────┬────────────────┐           │
        ▼              ▼           ▼              ▼                ▼           │
   run_parsing      run_eda   run_synthesis  reject_synthesis  stop_partial    │
   loaders,         LLM code  report from    (says why)        partial report  │
   mapping,         in the    findings, LLM       │            without an LLM  │
   type checks,     sandbox   writes wording      │                 │          │
   plausibility                + drafts           │                 ▼          │
        │              │           │              │                END         │
        └──────────────┴───────────┴──────────────┴────────────────────────────┘
```

- **One graph, two modes.** Batch (`dva run`) is one invocation: files in, report and
  drafts out. Chat (`dva chat`) is one invocation per message, with state kept across
  turns in SQLite.
- **The LLM only chooses; code enforces every limit.** Tools read state and write state
  updates themselves; the orchestrator never passes data between tools.
- **Plausibility checks are deterministic** and run inside `run_parsing`: ranges,
  cross-field rules, units (cases instead of units), category codes, duplicate rows and
  joins. They never depend on an LLM choosing to run them.
- **State holds references and summaries, never data rows.** Cleaned tables are written to
  Parquet under the session workspace; prompts contain paths, column names, dtypes and
  findings, not rows.
- **The report is built from structured findings in code.** The LLM writes only the summary
  and the wording around the issue lists in the drafts, so no finding can be dropped or
  misstated.

## Setup

Requires Python 3.11+ and an Anthropic API key.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then set ANTHROPIC_API_KEY in .env
```

`.env` is git-ignored; never commit keys. The `dva` command is installed by the package.

## Configuration

Everything lives in `config/settings.yaml` (or the file named by `DVA_CONFIG`). Relative
paths resolve against the project root.

| Section | What it controls |
|---|---|
| `models` | Model per role (`orchestrator`, `column_mapper`, `eda_coder`, `synthesis`, `eval_matcher`), with `max_tokens` and `effort`. Created with LangChain's `init_chat_model`. |
| `llm` | Structured outputs (`json_schema`) and their retry count. No `temperature` is sent: the Anthropic API rejects sampling parameters. |
| `limits` | Per batch run and per chat turn: tool-call cap (6 / 4), wall clock (300 s / 120 s), token budget (60k / 30k). |
| `sandbox` | Timeout (60 s), memory limit (1 GB), stdout limit (4,000 chars), import allowlist, blocked names, fix attempts (1). |
| `plausibility` | Unit-detection settings (grouping field, pack-size field, ratio tolerance), evidence-row cap. |
| `data_model` | Provider (`yaml` or `mcp`), default model name, MCP server settings. |
| `paths` | Workspace, data models, SQLite checkpoint file, fixtures, eval results. |

Default models: Claude Sonnet 5.5 for the orchestrator and synthesis, Claude Haiku 5.5 for
the column mapper and eval matcher, Claude Opus 5.5 for the EDA coder.

### Data models

A data model is a YAML file in `config/data_model/` with tables, fields (type, required,
synonyms, and optionally range, units, categories, join) and cross-field rules. Required
checks are derived from it: `<field>:present` for each required field, plus `:range`,
`:units`, `:categories` and `:join` where the field defines them. Cross-field rules support
`a <= b` style comparisons and `expr between low and high` over `+ - * /`; anything else is
rejected when the model loads.

To use an MCP server instead of YAML files, set `data_model.provider: mcp` and point
`data_model.mcp` at a server exposing `get_data_model` and `lookup_field` (a `url`, or a
`command` and `args` to launch over stdio). The default settings point at the mock server in
`tests/mcp_mock_server.py`, which serves the YAML files.

## Batch mode

```bash
dva run --files sales.xlsx products.csv stores.csv --data-model retail_sales --session cust-418
```

Accepts CSV, Excel (`.xlsx`, any sheet with a recognizable header row, even below title
rows) and email (`.eml` with CSV attachments or HTML tables). Prints the report path, the
summary, the stop reason, the tools run, tokens and time. Exit code 0 means the run
completed; 1 means it stopped early with a partial report; 2 is a usage error.

Output for a session goes to `workspace/<session>/`:

```
workspace/cust-418/
  report.md        the issue report, grouped by severity
  data/            cleaned tables as Parquet, columns named after model fields
  artifacts/       evidence CSVs referenced by findings
  review_queue/    email and Jira drafts (pending_review)
  sandbox/         the last EDA run's code and outputs
```

Stop reasons: `task_complete`, or a hard stop (`tool_call_cap`, `no_progress`,
`time_budget`, `token_budget`, `ended_without_report`) that writes a partial report from
the findings so far. A batch run always parses, runs EDA at least once, then synthesizes;
synthesis is rejected while required checks remain.

## Chat mode

```bash
dva chat --session cust-418 --data-model retail_sales
```

An example session (agent replies vary):

```
you> /upload sales.csv products.csv stores.csv
3 file(s) ready; they are sent with your next message.
you> Please check these files
agent> Parsed: 6 issues found ...
you> How many prices look wrong?
agent> 37 unit prices look like they were entered in cents ...
you> /quit
```

Uploaded files go with your next message. Parsing runs only when files are pending, so
follow-up questions are answered from the findings or by a focused EDA run. State is
saved in `workspace/checkpoints.sqlite` under the session id; run the same command again to
continue a conversation. Without uploaded data the agent asks for files instead of
calling tools.

## Review queue

```bash
dva review list [--session cust-418]
dva review show <draft_id>
dva review mark <draft_id> approved     # or rejected
```

Marking changes the draft's status only. **Nothing is ever sent**: there is no email or
Jira integration, and the queue and synthesis modules import nothing that can send.

## Sandbox limitations

EDA code written by the LLM runs in a separate Python process with the session folder as
working directory, a 60-second timeout, a minimal environment without API keys, and an AST
check before it runs (import allowlist; no `eval`, `exec`, `open`, `getattr`, dunder
attributes, or access to `os`, `sys`, `subprocess`, `socket`, `shutil`).

**This is a guardrail against mistakes, not a security boundary.** Code that passes the
check can still read and write files through pandas, and nothing isolates the process from
the machine. The 1 GB memory limit is enforced on Linux only: macOS rejects the limit and
Windows has no `resource` module, so runs continue without it and a warning is logged.
Container isolation is out of scope for this build; do not run the agent on untrusted
input where that matters.

## Tests

```bash
pytest            # unit and trajectory tests; no network, no real LLM
ruff check . && ruff format --check .
```

Tests use a scripted fake chat model (`tests/fakes.py`) that returns queued replies, tool
calls and structured outputs. Trajectory tests (`tests/trajectory/`) run the whole graph
and assert the exact sequence of tool calls and the stop reason: normal batch run,
synthesis rejected when called too early, tool-call cap, no progress, chat turns after
parsing, chat without data.

Fixtures in `tests/fixtures/` are generated with a fixed seed and committed. Each has a
`ground_truth.json` with the expected field mapping, missing fields and issues. Regenerate
them only when the data model changes:

```bash
python scripts/generate_fixtures.py
```

## Evals

```bash
python evals/run_evals.py [--fixtures kitchen_sink clean ...] [--workers 3]
```

Runs the agent with the real model on every fixture (about 10k tokens per fixture), prints
each finding-to-ground-truth match decision, a table per fixture and overall, and saves
`evals/results/<timestamp>.json` (git-ignored). Findings are matched in code first (same
field and issue type, count within tolerance); leftovers go to an LLM judge.

Latest run (2026-10-25, all 12 fixtures, default models; every run ended `task_complete`):

| Metric | Target | Latest |
|---|---|---|
| Field-mapping accuracy | ≥ 95% | 100% |
| Issue recall | ≥ 90% | 100% |
| Issue precision | ≥ 80% | 100% |
| EDA code execution success | ≥ 95% | 100% |
| First-pass schema validation | report only | 100% |
| Tool-call cap hit rate | ≤ 10% | 0% |
| Tool calls / latency / tokens per run | report only | 3 calls; mean 24.9 s (max 39.4); mean 10.1k tokens (max 12.0k) |

On these fixtures every injected issue is caught by the deterministic checks, so EDA has
not yet added findings of its own; the evals confirm EDA code runs cleanly and does not
report false problems.

## Project layout

```
config/settings.yaml              models, limits, sandbox, paths
config/data_model/retail_sales.yaml
src/dva/
  config.py  state.py  workspace.py  llm.py  review_queue.py  cli.py
  data_model/   models, YAML and MCP providers
  agents/       loaders, coercion, parsing, plausibility, eda, synthesis
  sandbox/      policy (AST check), runner (subprocess, emit helper)
  orchestrator/ prompts, tools, routing rules, graph
scripts/generate_fixtures.py
tests/          fakes, fixtures, unit and trajectory tests, mock MCP server
evals/          run_evals.py, metrics.py
```

## Known limitations

- Budgets are checked between graph steps, so a run can overshoot slightly while an LLM
  call or sandbox run is in progress.
- Ambiguous dates such as `05/02/2023` are read month-first.
- The column mapper sees column names and a coarse kind (numeric/text), never values; odd
  column names may stay unmapped.
- Only the `retail_sales` data model is built and tested.
