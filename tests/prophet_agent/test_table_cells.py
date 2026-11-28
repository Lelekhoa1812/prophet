"""Column-aware table check (runtime/table_cells.py). Live EPL: round 20261128-063203 wrote Nottingham Forest on 18 points
(printed: 9); round 20261128-091739 typed a standings table of 28, 27, 26... with Brighton first (printed: second on 22)."""
from __future__ import annotations

from prophet_agent.runtime import prediction_quality as pq
from prophet_agent.runtime import table_cells as tc
from prophet_agent.runtime.grounding import unverified

def _pandas_print() -> str:
    import pandas as pd
    return pd.DataFrame({"Pos": [1, 2, 17], "Team": ["Arsenal", "Brighton & Hove Albion", "Nottingham Forest"],
                         "Pl": [11, 11, 11], "W": [9, 7, 2], "D": [1, 1, 3], "L": [1, 3, 6], "GF": [25, 24, 9],
                         "GA": [8, 7, 15], "GD": [17, 17, -6], "Pts": [28, 22, 9]}).to_string()


PANDAS = _pandas_print()


def _sandbox(stdout):
    return {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True, "stdout": stdout}}


ANSWER = """Observed table:

| Club | Points |
|------|--------|
| Arsenal | 28 |
| Nottingham Forest | 18 |

| Club | Current pts | Expected final pts | Title probability |
|------|-------------|--------------------|-------------------|
| Arsenal | 28 | 82.7 | 75.4% |
| Brighton | 22 | 70.1 | 9.0% |
"""


def test_a_wrong_cell_is_found_by_its_column_and_the_club_alias():
    wrong = tc.mismatches(ANSWER, [_sandbox(PANDAS)])
    assert [(w["entity"], w["label"], w["stated"], w["printed"]) for w in wrong] == \
        [("Nottingham Forest", "Points", "18", [9.0])], "forecast columns and correct cells pass"
    assert "18" in unverified(ANSWER, [_sandbox(PANDAS)], "EPL")
    c = pq.check_table_cells(ANSWER, [_sandbox(PANDAS)])
    assert not c["ok"] and c["severity"] == "block" and "the tools printed 9" in c["detail"]


def test_pipe_csv_row_lists_and_objects_are_read_by_column():
    pipe = "| Pos | Team | Pts |\n|---|---|---|\n| 17 | Nott'm Forest | 9 |"
    csv = "Pos,Team,Played,Pts\n17,Nott'm Forest,11,9"
    rows = {"columns": ["team", "points"], "rows": [["Nott'm Forest", 9]]}
    obj = {"team": "Nottingham Forest", "points": 9}
    for evidence in (pipe, csv):
        assert tc.mismatches(ANSWER, [_sandbox(evidence)])[0]["printed"] == [9.0]
    for res in (rows, obj):
        h = [{"kind": "tool", "tool": "simulate_season", "args": {}, "result": {"ok": True, "table": [res]}}]
        assert tc.mismatches(ANSWER, h)[0]["printed"] == [9.0]


def test_nothing_is_judged_without_a_printed_value_for_that_column():
    assert tc.mismatches(ANSWER, [_sandbox("Nottingham Forest drew 1-1 on 18 November")]) == []
    assert tc.mismatches("No table here.", [_sandbox(PANDAS)]) == []
    assert pq.check_table_cells(ANSWER, []) is None


def test_a_club_in_no_printed_table_is_an_invented_row_once_a_full_table_was_printed():
    """Round 20261128-125459: West Ham, Wolves, Leicester and Southampton were typed into a standings table though the
    printed table held this season's 20 clubs and none of them."""
    import pandas as pd
    clubs = ["Arsenal", "Liverpool", "Chelsea", "Everton", "Fulham", "Brentford", "Burnley", "Sunderland", "Leeds",
             "Newcastle"]
    printed = pd.DataFrame({"Team": clubs, "Pl": [11] * 10, "Pts": list(range(28, 18, -1))}).to_string()
    answer = "| Club | Pl | Pts |\n|---|---|---|\n" + "".join(f"| {c} | 11 | {28 - k} |\n" for k, c in enumerate(clubs)) \
        + "| West Ham | 11 | 17 |\n"
    wrong = tc.mismatches(answer, [_sandbox(printed)])
    assert [(w["entity"], w.get("absent")) for w in wrong] == [("West Ham", True)]
    assert "West Ham: in no table the tools printed" in pq.check_table_cells(answer, [_sandbox(printed)])["detail"]
    few = pd.DataFrame({"Team": clubs[:3], "Pl": [11] * 3, "Pts": [28, 27, 26]}).to_string()
    assert tc.mismatches(answer, [_sandbox(few)]) == []          # a short printed list does not prove a row invented
