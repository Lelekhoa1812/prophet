"""Mission 9 stall rule: cells that only re-print a table already seen are stale, even when they save it under a new name.
Replay of EPL 56465803 (tests/prophet_agent/evals) printed the same 11-club table six cells in a row."""
from prophet_agent.runtime import goal as G

TABLE = "\n".join(f"{i} Club{i} 13 {30 - i} {i % 5} {i % 3}" for i in range(11)) + "\n(11, 9)\n"


def cell(stdout: str, outputs=(), ok=True) -> dict:
    return {"kind": "tool", "tool": "sandbox_python", "args": {"code": "print(df)"},
            "result": {"ok": ok, "stdout": stdout, "outputs": [{"path": p, "sha256": d} for p, d in outputs]}}


def test_a_new_name_for_the_same_content_is_not_progress():
    history = [cell(TABLE, [("out/table.csv", "aaa")])]
    history += [cell(TABLE, [(f"out/table_{k}.csv", "aaa")]) for k in range(3)]
    assert G._stalled(history, 3)


def test_reprinting_a_seen_table_while_rewriting_the_same_file_is_stale():
    history = [cell(TABLE, [("out/table.csv", "aaa")])]
    history += [cell(TABLE, [("out/assessment.json", f"j{k}")]) for k in range(4)]   # the first write is new
    assert len(G._stale_run(history)) == 3 and G._stalled(history, 3)


def test_a_file_with_a_new_name_and_new_content_is_progress():
    history = [cell(TABLE)] + [cell(TABLE, [(f"out/team_{k}.csv", f"t{k}")]) for k in range(3)]
    assert G._stale_run(history) == []


def test_new_rows_or_a_new_chart_are_progress():
    history = [cell(TABLE)] + [cell(TABLE) for _ in range(2)]
    fresh = "\n".join(f"{i} Club{i} 14 {31 - i} {i % 4} {i % 2}" for i in range(11)) + "\n(11, 9)\n"
    assert G._stale_run(history + [cell(fresh)]) == []
    assert G._stale_run(history + [cell(TABLE, [("out/points_path.png", "p1")])]) == []


def test_a_cell_with_no_table_neither_starts_nor_breaks_a_run():
    history = [cell(TABLE), cell(TABLE), cell("saved notes\n"), cell(TABLE), cell(TABLE)]
    assert len(G._stale_run(history)) == 3
