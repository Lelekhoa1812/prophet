# Prophet API

Prophet is a local forecasting and spreadsheet-analysis agent. This document is for **other agents and scripts** that want to use it as a tool: ChatGPT, Claude, Cursor, or your own code acts as the brain, and Prophet does the data work (profiling files, running sandboxed Python, forecasting with TimesFM 3, producing charts and reports).

Everything here is plain HTTP and JSON on your own machine. Nothing is sent anywhere except the planner's calls to NVIDIA NIM.

## Start the server

```bash
PYTHONPATH=src .venv/bin/python -m prophet_agent.api      # http://127.0.0.1:8010
```

Requirements: a `.env` with NVIDIA keys (`/api/health` reports `nim_keys_configured`). The server binds to loopback only, accepts loopback host names only, and refuses browser writes whose `Origin` is not loopback. Scripts and agent tools that send no `Origin` header are fine.

## The idea in one minute

1. Create a **notebook**. It is a durable workspace that keeps files, kept tables and a thread of questions.
2. **Upload files** to it (any number, any spreadsheet type). Prophet profiles each one in the background.
3. Send a **turn**: a plain-language goal such as "total sales by region, then forecast the next 8 weeks". Prophet plans, runs code, forecasts, and writes an answer.
4. **Poll** the turn until it finishes. If it needs a decision it pauses with a structured question; answer it and it continues.
5. **Download** the files it produced (charts, workbooks, reports, kept tables).

Only one turn runs at a time per notebook (a second one returns `409`). A turn can be stopped at any point.

## Conventions

- Base URL `http://127.0.0.1:8010`, JSON bodies, UTF-8.
- Ids are opaque strings. Do not parse them.
- Errors are `{"detail": "<message in words>"}` with a meaningful status: `404` unknown id, `409` busy or conflicting, `413` file too large (50 MB), `422` invalid input, `503` no model key.
- Timestamps are Unix seconds.

## Notebooks

### Create and list

```bash
curl -s -X POST localhost:8010/api/notebooks -H 'content-type: application/json' -d '{"title":"Q3 sales"}'
curl -s localhost:8010/api/notebooks
```

`POST` returns the notebook; `GET` returns rows with `id`, `title`, `sources`, `turns`, `status` (of the latest turn) and `updated`.

### Read everything about a notebook

```bash
curl -s localhost:8010/api/notebooks/$NID
```

Returns the notebook plus `sources`, `tables` (kept tables) and `turns`.

### Rename, set standing instructions, archive

```bash
curl -s -X PATCH localhost:8010/api/notebooks/$NID -H 'content-type: application/json' \
  -d '{"title":"Q3 sales","instructions":"Amounts are in AUD. Fiscal year starts 1 July.","archived":false}'
```

`instructions` (up to 2000 characters) are shown to the planner on every turn.

## Sources (files)

### Upload

Send the **raw file bytes** as the request body. Give the file name in the `X-File-Name` header (percent-encode non-ASCII), or `?name=` for quick scripts.

```bash
curl -s -X POST localhost:8010/api/notebooks/$NID/sources \
  -H 'X-File-Name: orders%20Q3.xlsx' --data-binary @"orders Q3.xlsx"
```

Returns the source immediately (`status: "profiling"`). Identical bytes already in the notebook return the existing source with `duplicate: true`.

Supported inputs include delimited text (CSV, TSV and other delimiters, any common encoding, header row anywhere), Excel (`xlsx`, `xlsm`, `xlsb`, `xls`), OpenDocument (`ods`), Parquet, Feather, and gzip, bzip2 or zip archives that contain these. Hidden sheets are listed but not read. Several tables on one sheet are flagged. Damaged or password-protected files are reported, never "repaired" silently.

### Status and what Prophet found

`GET /api/notebooks/{nid}` lists each source with:

| field | meaning |
|---|---|
| `status` | `profiling`, `ready`, `needs_attention`, `failed` |
| `included` | whether questions can read it |
| `card.sheets[]` | per sheet or table: `sheet`, `rows`, `header_row`, `columns`, `issues` |
| `card.hidden_sheets` | sheets that were not read |
| `error` | the reason, in words, when `failed` or `needs_attention` |

Poll until no source is `profiling`.

### Include, exclude, remove

```bash
curl -s -X PATCH localhost:8010/api/notebooks/$NID/sources/$SID -H 'content-type: application/json' -d '{"included":false}'
curl -s -X PATCH localhost:8010/api/notebooks/$NID/sources/$SID -H 'content-type: application/json' -d '{"removed":true}'
```

Excluded files are not mounted into the sandbox and are left out of the planner's context.

## Turns (questions and work)

### Send a turn

```bash
curl -s -X POST localhost:8010/api/notebooks/$NID/turns -H 'content-type: application/json' \
  -d '{"goal":"Total revenue by region, then forecast the next 8 weeks with a range."}'
# {"id":"<turn id>","seq":1}
```

`goal` is 3 to 2000 characters. The planner sees the notebook's instructions, a card per included file, the kept tables and a summary of the last turns, so follow-ups like "now split that by product" work.

### Poll a turn

```bash
curl -s "localhost:8010/api/missions/$TURN?after=0"
```

A turn is a mission. Fields:

| field | meaning |
|---|---|
| `status` | see below |
| `answer` | when finished: `{"status":"met","answer":"<markdown>"}`, or the stored answer object when it could not finish. Its `reason` is for developers; show `outcome` instead |
| `outcome` | `{"status", "reasons": [...], "developer": {...}}`. Each reason has `code`, `user_reason` (plain words), `next_step`, and where relevant `checks` (`[{name, plain}]`), `figures` (untraced numbers) or `which` (the allowance that ran out). `developer` holds the planner's own text and the failed check names |
| `progress` | the latest status line: `checklist_done`, `checklist_total`, `next` and `elapsed_s`; `null` before the first one |
| `events` | ordered records; pass `after=<last seq>` to read only new ones |
| `steps` | the ledger of plan, act, ask and evaluate steps with their outputs |
| `questions` | open questions awaiting you |

Reason codes: `blocked_by_check` (a required check did not pass), `unverified_figures` (figures not traced to a tool result), `stalled` (the same step kept repeating), `answered_now` (you asked for the answer now), `budget_spent` (a checkpoint), `not_settled`, `impossible`, `failed`, `outcome_unknown`, `cancelled`. A `met` or `completed` turn has no reasons.

Statuses: `created`, `running`, `waiting_user`, `completed`, `met`, `failed`, `stalled`, `impossible`, `iteration_budget`, `budget_exhausted`, `outcome_unknown`, `cancelled`. Working states are `created` and `running`. `waiting_user` needs an answer. Everything else is final.

A good poll loop waits 2 seconds between calls and stops on any final status. Latency is measured, not promised: across 58 stored missions on one machine, the median run took about 9 minutes (557 s) and the longest about 29 minutes (1,756 s, a run that continued into a second part). Short questions can finish in well under that. Use `progress` to show how far a run has got.

### Continue or answer now at a checkpoint

A part of the work ends at a checkpoint when its time or tool-call allowance is spent. Nothing is lost: every step is kept.

```bash
curl -s -X POST localhost:8010/api/missions/$TURN/continue       # 202: next part with a fresh allowance
curl -s -X POST localhost:8010/api/missions/$TURN/answer-now    # 202: write the answer now, labelled partial
```

Both return `409` unless the turn is at a checkpoint (`checkpoint`, or the older `budget_exhausted` and `iteration_budget`). Poll the turn again afterwards.

### Answer a question

When `status` is `waiting_user`, `questions` holds items in the AskUserQuestion shape:

```json
{"id":"<question id>","questions":[
  {"question":"Which column is the date?","header":"Date column","multiSelect":false,
   "options":[{"label":"order_date","description":"Looks like day-level dates"},
              {"label":"ship_date","description":"Only filled for shipped orders"}]}]}
```

Answer by header:

```bash
curl -s -X POST localhost:8010/api/questions/$QID/answer -H 'content-type: application/json' \
  -d '{"answers":{"Date column":"order_date"}}'
```

Keys are the question `header`. A single-choice answer is a string and a multi-select answer is a list of strings. The value may be one of the offered labels or your own free text (the UI's "Other"). The turn resumes.

### Stop

```bash
curl -s -X POST localhost:8010/api/missions/$TURN/cancel     # {"status":"cancelling"} or the final status
```

Safe to call on a finished turn.

### List the thread

```bash
curl -s localhost:8010/api/notebooks/$NID/turns
```

## Produced files and kept tables

Charts, workbooks and reports from a forecast run:

```bash
curl -s localhost:8010/api/missions/$TURN/artifacts     # [{"run_id","files":[{"path","sha256","bytes"}],"forecast":{...},"history":[...]}]
curl -OJ localhost:8010/api/artifacts/$RUN_ID/out/forecast.xlsx
```

Each file carries a `sha256` so a calling agent can verify what it downloaded.

Tables the analysis chose to keep (cleaned data, joins, features) are versioned and immutable:

```bash
curl -s localhost:8010/api/notebooks/$NID/tables
curl -s localhost:8010/api/notebooks/$NID/tables/$NAME/versions
curl -OJ "localhost:8010/api/notebooks/$NID/tables/$NAME/download?version=2"   # omit version for the latest
```

Kept tables are available to later turns as `derived/<name>.csv`.

## One-off quick forecasts (no notebook)

```bash
curl -s -X POST localhost:8010/api/datasets?name=weekly.csv --data-binary @weekly.csv     # {"id": ...}
curl -s -X POST localhost:8010/api/missions -H 'content-type: application/json' \
  -d '{"goal":"Forecast the next 8 weeks","dataset_ids":["<id>"]}'
curl -s localhost:8010/api/missions
```

Polling, answering, stopping and artifacts are the same as for turns.

## Introspection

| route | purpose |
|---|---|
| `GET /api/health` | `{"ok":true,"nim_keys_configured":bool}` |
| `GET /api/tools` | what the agent can do: `name`, `title`, `activity`, `description` |

## Driving Prophet from another agent

Give your agent this brief and the routes above:

> You can use Prophet to analyse and forecast spreadsheets. Create a notebook, upload the user's files, wait until none is `profiling`, then send one turn per question in plain language. Poll the turn until it reaches a final status. If it is `waiting_user`, either answer from what the user already told you or relay the question to the user, then post the answer. Quote Prophet's `answer` faithfully, including its row counts and caveats, and offer the produced files. Never invent figures that are not in the answer. If a turn ends `stalled`, `impossible` or `iteration_budget`, tell the user why in plain words and suggest what to add (more history, a clearer column, a narrower question).

Practical advice for calling agents:

- **One question per turn** gives the cleanest answers and traceable steps. Chain turns for multi-part work; the notebook remembers.
- **Tell Prophet what the data means** (units, currency, fiscal calendar, holidays, known events) in the notebook `instructions` or in the turn. It does not guess business facts.
- **Do not paste data into the goal.** Upload it; the goal is for intent.
- **Treat numbers as grounded only when the answer cites them** (file, sheet, column, row counts). Ask a follow-up when it does not.
- Run long work asynchronously: send the turn, do other things, poll.

## MCP server (Claude, Cursor and other MCP clients)

The same capabilities are available as MCP tools, so a client can use Prophet without writing HTTP calls. The bridge holds no state and no keys; it talks to the running server.

```bash
PYTHONPATH=src .venv/bin/python -m prophet_agent.api          # terminal 1: Prophet itself
```

Add it to the client's MCP settings (Claude Desktop, Cursor and others use the same shape):

```json
{
  "mcpServers": {
    "prophet": {
      "command": "/absolute/path/to/prophet/.venv/bin/python",
      "args": ["-m", "prophet_agent.mcp_server"],
      "env": { "PYTHONPATH": "/absolute/path/to/prophet/src", "PROPHET_URL": "http://127.0.0.1:8010" }
    }
  }
}
```

Claude Code: `claude mcp add prophet -e PYTHONPATH=/absolute/path/to/prophet/src -- /absolute/path/to/prophet/.venv/bin/python -m prophet_agent.mcp_server`

| tool | what it does |
|---|---|
| `prophet_create_notebook` | New notebook, optional standing instructions |
| `prophet_add_file` | Upload a local file by path and wait until it has been checked |
| `prophet_notebook` | Files with status, kept tables, past questions |
| `prophet_ask` | Ask a question and wait for the answer (or for a question back) |
| `prophet_status` | Keep waiting on a turn that was still working |
| `prophet_answer` | Answer questions Prophet asked |
| `prophet_cancel` | Stop a turn |
| `prophet_continue` | Continue a turn paused at a checkpoint (`POST .../continue`) and wait for the next stop |
| `prophet_answer_now` | Write the answer for a turn paused at a checkpoint (`POST .../answer-now`) |
| `prophet_files` | Charts, workbooks and reports a turn produced, with URLs and sha256 |
| `prophet_quick_forecast` | One-off forecast, no notebook |
| `prophet_list_notebooks` | List notebooks |

`prophet_ask` waits up to `wait_seconds` (default 240) and returns either the answer, a `waiting_user` payload with the questions, or a note to call `prophet_status` to keep waiting. A partial answer comes back with `reasons` (plain words, a `next` step each) and the figures that could not be traced; the planner's own text is not returned. At a checkpoint it says to call `prophet_continue` or `prophet_answer_now`. Transport is stdio (newline-delimited JSON-RPC, protocol 2024-11-05, standard library only).

## Limits and guarantees

- 50 MB per file, up to 10 files per quick forecast, 2000 characters per goal.
- One active turn per notebook. Starting another returns `409`.
- A tool call that finished is never re-run, and a turn that was interrupted by a restart resumes from its ledger without asking the model again for completed work.
- Analytics code runs in a restricted child process (read-only inputs, no network, no writes outside its run folder, time limit). It is a guardrail, not a security boundary: only point Prophet at files you trust.
- Prophet is a single-user local tool. There is no authentication; do not expose the port beyond loopback.
- Licensing notes for the forecasting model are in `docs/prophet/ARCHITECTURE.md`.
