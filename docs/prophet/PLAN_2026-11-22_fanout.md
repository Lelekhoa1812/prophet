# Plan, fan-out round (2026-11-22)

Locked after the user's answers to the AskUserQuestion round on this date. This plan binds every subagent and the main
thread. Change it only with a new AskUserQuestion answer, recorded here.

## Approved decisions

| # | Decision | Answer |
|---|---|---|
| D1 | Offline slices to build | Item 10 horizon check; MinT reconciliation; equal-weight blend as a centre rule; clean_text boundary space |
| D2 | Live `panel_backtest` | Run now, with repeats. NIM calls are treated as free and TimesFM runs locally, so the budget is not a constraint. Test long, edge-case and pressure inputs |
| D3 | Merge | The main thread merges slice branches autonomously once each passes the gate and these rules. No further questions unless the user is asked to verify something |
| D4 | Gate policy | Find the root cause of every live-run gate block before changing a gate. Judge each gate's fitness on evidence, then keep, fix the planner, or downgrade. The main thread decides, not the user |

## Slices and owners

Each slice runs in its own git worktree on its own branch, created from the commit that holds this plan.

| Slice | Branch | Worktree | Files it may change | Done when |
|---|---|---|---|---|
| S1 horizon | `fx-horizon-check` | `/Users/liamle/Downloads/prophet-worktrees/horizon` | `src/prophet_agent/runtime/goal.py`, one new test file | Plan horizon compared with the requested horizon, mismatch detected, regression test fails before and passes after |
| S2 MinT | `fx-mint-reconcile` | `/Users/liamle/Downloads/prophet-worktrees/mint` | new `src/prophet_agent/forecast/reconcile.py`, new test file | MinT (shrinkage or OLS-free trace-minimising) reconciles a hierarchy so sums hold, tested on synthetic hierarchies with a known answer |
| S3 blend | `fx-equal-blend` | `/Users/liamle/Downloads/prophet-worktrees/blend` | `src/prophet_agent/forecast/combine.py`, `src/prophet_agent/forecast/rolling.py` (only if needed), `src/prophet_agent/tools/report_tool.py` (centre option only), tests | Equal-weight blend is a centre option, chosen only when the backtest says it wins, never fitted on the errors it is scored on |
| S4 clean | `fx-clean-text` | `/Users/liamle/Downloads/prophet-worktrees/clean` | `src/prophet_agent/nim/transport.py`, one test file in `tests/prophet_agent/nim/` | A continuation cut keeps its boundary space; regression test fails before and passes after |

The main thread owns, and no slice may touch: `src/prophet_agent/sandbox/prophet_*.py`, `src/prophet_agent/prompts/*`,
`src/prophet_agent/runtime/contracts.py`, `src/prophet_agent/tools/panel_tool.py`, `docs/prophet/*`.

## Main-thread work

- M1 Root-cause the live gate blocks (see the live evidence below), then decide each gate (D4). Fixes to the planner or
  the prompts are main-thread work, after a root-cause write-up in `docs/prophet/GOAL.md`.
- M2 After the merge: restart the API on the merged code, then run the live `panel_backtest` on a real panel with repeats,
  plus edge cases and a pressure run. Record every result as measured, real NIM and TimesFM calls.
- M3 Re-run retail and WTI once more after any gate or prompt change, to confirm the change.
- M4 Merge slices, run `scripts/check.sh` on main, push `origin/main`, remove worktrees and branches.

## Live evidence that drives M1 (2026-11-22, real API, TimesFM 3, real public series)

| Run | Status | Blocking check | Root-cause question to answer |
|---|---|---|---|
| retail rep1 | partial, 571 s | `scan_decisions_recorded`: `stationarity.verdict` has no ledger decision | Did the planner ever call the ledger? Is the id text it wrote different from the scan's id? |
| retail rep2 | completed, done_verified, 1532 s | none | What did this run do differently from rep1? |
| WTI rep1 | partial, 691 s | `baseline_compared` (keeps TimesFM centre; mean_last_season MASE 6.663 vs 7.278); `controls_tested` (context_points used, control_lab never ran) | Why did the planner not switch the centre, and not call control_lab? Budget? |
| WTI rep2 | partial, 510 s | `baseline_compared`; `scan_decisions_recorded` (`changepoints.level_shift`, `stationarity.verdict`) | Same centre question; and why the ledger ids were not recorded |

Also seen: `sandbox_python` rejected an unknown `rationale` argument (retail rep1) and `timesfm_forecast` rejected `season`
(WTI rep2). Both are planner argument errors and are in scope for M1.

## Development rules (apply to every slice and to the main thread)

1. Test first. Write the regression test, run it and see it fail for the stated reason, then fix. Report the failing output.
2. Gate before reporting. Run `scripts/check.sh` in the worktree (the symlinked `.venv` is in place). It must report all
   tests passing, with the skip count unchanged. A red gate is not reported as done.
3. Scope. Touch only the files in the table for your slice. If the fix needs another file, stop and report it.
4. Commit on your branch with a message ending `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`. Do not push,
   do not merge, do not change `main`, do not remove worktrees. The main thread does those.
5. Never put licence or research-use wording in code, prompts, skills, UI, tools or API. Markdown docs only (locked rule).
6. Label numbers. Synthetic replay, real staging or measured. Never present synthetic numbers as observed.
7. Follow the repo's docstring style: a `Motivation vs Logic` section at the top of each new module.
8. No live calls from a slice. The live API and its runs stay with the main thread.
9. Report back: branch, commit hash, files changed, test counts before and after, the failing-then-passing evidence, and
   anything you could not verify.
10. Research: if a slice needs literature (MinT, for example), use a short search and cite it; do not invent references.

## Merge and cleanup order

1. Merge S4 (clean), then S1 (horizon), then S3 (blend), then S2 (MinT), each with `git merge --no-ff` into `main`, running
   `scripts/check.sh` after each merge. Resolve any conflict surgically and re-run the gate.
2. Push `origin/main` once all four are merged and green.
3. Remove each worktree with `git worktree remove`, delete its local branch with `git branch -d`, and confirm
   `git worktree list` and `git branch --list` show only `main`.
