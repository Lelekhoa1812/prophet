# Offline mission evals

Offline evals replay recorded research missions through the real goal loop (`src/prophet_agent/runtime/goal.py`, `runner.py`,
`store.py`). The planner and eval decisions come from the recording, in the order they were asked. The tool results come from
the recording, keyed by tool and arguments. No model is called and no network is used, so the suite runs in the offline gate
(`scripts/check.sh`) and a harness change can be checked in seconds.

The live counterpart is `scripts/workflows/research_panel.py`, which runs the open missions through the API with live models
and the network. See "Relation to the live panel" below.

## Layout

| Path | What it is |
| --- | --- |
| `scripts/build_eval_fixture.py` | Builds a fixture from a stored mission. Opens `data/prophet.sqlite` read-only (`mode=ro`). |
| `tests/prophet_agent/evals/fixtures/*.json` | One recorded mission each: goal, decisions, results, answers, checks. |
| `tests/prophet_agent/evals/checks/*.json` | The expected-outcome checks for one mission, as the builder takes them. |
| `tests/prophet_agent/evals/replay.py` | The replay helper: scripted brain, recorded tools, the run loop around `goal_program`. |
| `tests/prophet_agent/evals/test_mission_evals.py` | One test per check, plus a read-only check on the builder. |

Run only the evals with `PYTHONPATH=src .venv/bin/python -m pytest -q tests/prophet_agent/evals`. The full offline gate is
`scripts/check.sh`.

## Fixture format

```
id, about, source        mission id and its stored status (provenance only)
goal                     the mission goal, as the person wrote it
config                   GoalConfig fields the replay uses (auto_delegate, stall_window)
decisions.plan           planner decisions in the order the brain was asked (depth 0 and sub-agents)
decisions.eval           eval verdicts in the same order (eval, evalstall, repair)
answers                  answers to ask_user, in order
results                  tool calls: {tool, args, ok, result | error}; a repeated call has repeated entries, in run order
recorded                 what the live run did (status, stop reason, steps). For comparison only; no check reads it
checks                   the expected-outcome checks (below)
```

Results are trimmed to what the checks need: sandbox stdout is kept up to 4,000 characters, other text is cut to 240
characters, and other lists keep their first three items. Lists the guards read (`columns`, `outputs`, `first_rows`) are kept
whole. Trimming is the builder's job; do not edit a fixture by hand.

Two details matter when reading a replay:

- The loop builds its carry list (files passed into a sandbox cell) from a set. Carry lists are compared as sets in the
  replay, so the order of that list does not cause a miss.
- Sub-agents run inside the lead's step numbering, so the stored step order is the order in which the brain was asked. The
  replay uses that order for both the lead and the sub-agents.

## Adding a fixture

1. Find the mission id in `data/prophet.sqlite` (the `missions` table).
2. Write `tests/prophet_agent/evals/checks/<name>.json`: a JSON list of checks (below). Write the `why` of each check, because
   the test failure prints it.
3. Build the fixture from the main checkout's database (a worktree has no `data/`):

   ```
   PYTHONPATH=src .venv/bin/python scripts/build_eval_fixture.py \
       --db /Users/liamle/Downloads/prophet/data/prophet.sqlite \
       --mission <mission id> --id <name> \
       --checks tests/prophet_agent/evals/checks/<name>.json
   ```

4. Run the evals. A failing check is a finding: fix the code, or record the failure and keep the check. Do not loosen a check
   to make it pass.

## Checks

Every fixture also runs `no_unrecorded_calls`.

| Check | Arguments | Passes when |
| --- | --- | --- |
| `no_unrecorded_calls` | optional `allowed` | every call the replayed loop made had a recorded result, or at most `allowed` did. A miss means the replay diverged from the recording; a fixture that allows one says why (a harness change that moves a stop changes what a later, recorded decision meets). |
| `final_status_in` | `values` | the result status is one of `values` (for example `partial`, `stalled`, `met`). |
| `stop_reason_in` | `values` | the reason code (`stop_reason`) is one of `values`. |
| `runs_before_step` | `step` | the loop finished in fewer than `step` steps. Use it to show a run did not go to the old step count. |
| `stall_within_repeats` | `tool`, `contains`, `max_repeats` | the loop stopped before more than `max_repeats` executed calls to `tool` printed every string in `contains`, after the first one. The loop must stop: running out of recorded decisions fails the check. |
| `first_stall_within_repeats` | `tool`, `contains`, `max_repeats` | in the lead's evidence before the first stall rescue (or the stop), at most `max_repeats` calls to `tool` after the first printed every string in `contains`. This is the harness's part: what a rescue analyst does next comes from the recording. |
| `rescues_at_most` | `count` | at most `count` stall rescues ran; a second stall stops the run. |
| `file_named` | `name`, optional `present` | some executed call wrote a sandbox output whose path ends in `name` (`present: false` inverts it). |
| `download_refused` | `tool`, `name`, `reason_contains` | a call to `tool` under the file name `name` failed with an error containing every string in `reason_contains`, and none succeeded. |
| `no_download_accepted` | `name` | no download under `name` succeeded. |
| `max_eval_calls` | `value` | the evaluator was asked at most `value` times. |
| `no_unverified` | none | the result has no `unverified` figures (figures the analysis output and the pages read do not support). |

A replay is "stopped" when the loop ends on its own (a finish, a stall, a budget). If the loop asks the brain for a decision the
recording does not hold, the replay records `exhausted` and the stall check fails with that message.

## Replay fidelity

The replay is faithful for the planner's decisions, the order of calls, the stall rule, the tool-result history and the guards
it re-runs. It is not faithful for these:

- Free-text composition (`write`) and memory compaction are not replayed. The answer is the verdict's draft.
- Time budgets, checkpoints and auto-continue are not replayed: a fixture runs as one segment with no time allowance. The
  recorded EPL mission was split by a checkpoint and an auto-continue, which may be the cause of its one divergence (not
  established; see the status note).
- Tool results are the recorded ones. The only guard that is re-run is the name guard of `download_web_data`
  (`name_entity_refusal`), on the header and first rows of the recorded table. Other guards and parsers see the recorded output.
- A call whose arguments are not recorded is answered with an error and reported by `no_unrecorded_calls`.

## Relation to the live panel

| | Offline evals (this doc) | Live panel (`scripts/workflows/research_panel.py`) |
| --- | --- | --- |
| Model | recorded decisions, no model | live NIM models |
| Network and data | recorded results only | live endpoints and search |
| Tests | harness logic: stall rule, guards, repairs, runtime | end-to-end behaviour, data found, figures, answer quality |
| Speed and cost | seconds, offline | minutes per mission, API keys |
| Judges | the checks in the fixture | the M9 criteria in `docs/prophet/GOAL.md` |

Use the evals to catch harness regressions before a live run. Use the panel for M9-8 and for anything about what a model or
the web returns today. A passing eval does not show that a live mission now finds its data.

## Status at creation (2026-11-26)

Two fixtures were built from missions on 2026-11-26: EPL `56465803` and Nvidia `9e3408a4`.

- EPL: the replay ends as `stalled` after 14 repeats of the standings re-parse. The M9-7 limit is 4. The live run also did
  not stop within 4 repeats (it stopped at step 63 of 196). The replay has one divergence: a sub-agent sandbox cell whose
  arguments match no recorded call (`no_unrecorded_calls` fails). The likely cause is a planner decision that the replay
  consumes at a different point from the live run; this is not confirmed.
- Nvidia: the Apple file saved as `NVDA_daily.csv` is refused by the name guard (the reason names AAPL). The evaluator is not
  asked about the refused data.
