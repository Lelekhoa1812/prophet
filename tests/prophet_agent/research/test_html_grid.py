"""Grid placement for markup tables: rowspan, colspan, nested tables, multi-row headers and the sparse-cell gate.
Offline and synthetic: the clubs and figures are invented."""
import csv
import io

from prophet_agent.research.tables import html_tables


def page(body: str) -> str:
    return f"<!doctype html><html><body>{body}</body></html>"


def rows_of(candidate: dict) -> list[list[str]]:
    return list(csv.reader(io.StringIO(candidate["csv"])))


def test_a_rowspan_cell_fills_the_rows_below_without_shifting_them():
    body = ("<table><tr><th>Rank</th><th>Club</th><th>Pts</th></tr>"
            "<tr><td rowspan='2'>1</td><td>Alpha</td><td>30</td></tr>"
            "<tr><td>Beta</td><td>28</td></tr>"
            "<tr><td>2</td><td>Gamma</td><td>27</td></tr></table>")
    cand = html_tables(page(body))["candidates"][0]
    assert rows_of(cand) == [["Rank", "Club", "Pts"], ["1", "Alpha", "30"], ["1", "Beta", "28"], ["2", "Gamma", "27"]]


def test_a_colspan_cell_repeats_its_text_across_the_columns_it_covers():
    body = ("<table><tr><th>Club</th><th colspan='2'>Record</th></tr>"
            "<tr><td>Alpha</td><td>3</td><td>1</td></tr>"
            "<tr><td>Beta</td><td>4</td><td>0</td></tr></table>")
    cand = html_tables(page(body))["candidates"][0]
    assert rows_of(cand)[0] == ["Club", "Record", "Record"]
    assert rows_of(cand)[1] == ["Alpha", "3", "1"]


def test_a_rowspan_and_a_colspan_in_the_same_table_keep_every_later_cell_in_place():
    body = ("<table><tr><th>No</th><th>Club</th><th>Won</th><th>Lost</th></tr>"
            "<tr><td rowspan='2'>7</td><td colspan='2'>Delta</td><td>2</td></tr>"
            "<tr><td>5</td><td>Epsilon</td><td>3</td></tr></table>")
    cand = html_tables(page(body))["candidates"][0]
    assert rows_of(cand)[1] == ["7", "Delta", "Delta", "2"]
    assert rows_of(cand)[2] == ["7", "5", "Epsilon", "3"]


def test_a_table_that_holds_another_table_is_layout_and_its_nested_text_stays_out():
    body = ("<table><tr><td><table><tr><th>Inner</th><th>Score</th></tr><tr><td>x</td><td>9</td></tr></table></td>"
            "<td>outer</td></tr><tr><td>y</td><td>8</td></tr></table>")
    result = html_tables(page(body))
    assert result["skipped_layout"] >= 1
    assert [cand["columns"] for cand in result["candidates"]] == [["Inner", "Score"]]
    assert "outer" not in result["candidates"][0]["csv"]


def test_a_thead_with_two_rows_is_one_header_joined_per_column():
    body = ("<table><thead>"
            "<tr><th rowspan='2'>Club</th><th colspan='2'>Goals</th></tr>"
            "<tr><th scope='col'>For</th><th scope='col'>Against</th></tr>"
            "</thead><tbody><tr><td>Alpha</td><td>3</td><td>1</td></tr>"
            "<tr><td>Beta</td><td>2</td><td>2</td></tr></tbody></table>")
    cand = html_tables(page(body))["candidates"][0]
    assert cand["columns"] == ["Club", "Goals / For", "Goals / Against"]
    assert cand["row_count"] == 2


def test_leading_th_rows_without_a_thead_are_the_header_but_row_headers_are_data():
    body = ("<table><tr><th scope='col'>Club</th><th scope='col'>P</th></tr>"
            "<tr><th scope='row'>Alpha</th><td>12</td></tr>"
            "<tr><th scope='row'>Beta</th><td>11</td></tr></table>")
    cand = html_tables(page(body))["candidates"][0]
    assert cand["columns"] == ["Club", "P"]
    assert rows_of(cand)[1] == ["Alpha", "12"] and cand["row_count"] == 2


def test_a_body_that_is_more_than_ninety_percent_empty_is_layout_and_refused():
    body = ("<table><tr><th>a</th><th>b</th><th>c</th></tr>"
            "<tr><td>3</td><td></td><td></td></tr>"
            "<tr><td></td><td></td><td></td></tr>"
            "<tr><td></td><td></td><td></td></tr>"
            "<tr><td></td><td></td><td></td></tr></table>")
    result = html_tables(page(body))
    assert result["candidates"] == [] and result["skipped_layout"] == 1


def test_a_body_at_exactly_ninety_percent_empty_is_kept_and_its_empty_rows_are_dropped():
    # 10 body rows x 3 columns = 30 cells; 3 hold a value, so 27 (exactly 90%) are empty: kept, not more than 90%.
    spacer = "<tr><td></td><td></td><td></td></tr>"
    body = ("<table><tr><th>a</th><th>b</th><th>c</th></tr>"
            "<tr><td>3</td><td>4</td><td>5</td></tr>" + spacer * 9 + "</table>")
    result = html_tables(page(body))
    assert len(result["candidates"]) == 1 and result["candidates"][0]["row_count"] == 1

