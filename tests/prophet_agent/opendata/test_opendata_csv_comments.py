"""A '#' line is a comment only when it is not the table's header. A league table's header is "#,Club,P,...": its first cell
is the position column, and it must survive. Real comment blocks before a header still go. The NOAA text files keep their
'#' comment lines. The club names and figures are invented."""
from prophet_agent.opendata import fetch, store
from prophet_agent.opendata.csvlines import comment_flags, table_lines

LEAGUE = (b"#,Club,P,W,D,L,GD,Pts\n"
          b"1,Harbour FC,10,7,2,1,9,23\n"
          b"2,Riverside,10,6,3,1,8,21\n"
          b"3,Northgate,10,5,2,3,2,17\n")


def test_a_league_header_that_starts_with_hash_keeps_its_header_row():
    preview = store.preview(LEAGUE, "csv")
    assert preview["columns"] == ["#", "Club", "P", "W", "D", "L", "GD", "Pts"]
    assert preview["row_count"] == 3
    assert preview["rows"][0] == ["1", "Harbour FC", "10", "7", "2", "1", "9", "23"]


def test_the_split_of_a_league_table_keeps_the_hash_header_in_the_table():
    preamble, table = store.split_csv(LEAGUE.decode())
    assert preamble == "" and table.startswith("#,Club,P,W,D,L,GD,Pts\n")


def test_a_comment_block_before_a_header_is_dropped():
    body = (b"# Source: Example League office\n# Retrieved 2026-11-01\n"
            b"Pos,Club,P,Pts\n1,Harbour FC,10,23\n2,Riverside,10,21\n")
    preview = store.preview(body, "csv")
    assert preview["columns"] == ["Pos", "Club", "P", "Pts"] and preview["row_count"] == 2
    assert store.split_csv(body.decode())[1].startswith("Pos,Club,P,Pts\n")


def test_a_comment_with_commas_before_a_text_header_is_still_a_comment():
    body = b"# Table 3, season 2026, provisional\nPos,Club,P,Pts\n1,Harbour FC,10,23\n2,Riverside,10,21\n"
    assert store.preview(body, "csv")["columns"] == ["Pos", "Club", "P", "Pts"]


def test_a_comment_line_before_a_hash_header_is_dropped():
    body = b"# Season 2026, provisional\n#,Club,P,Pts\n1,Harbour FC,10,23\n2,Riverside,10,21\n"
    preview = store.preview(body, "csv")
    assert preview["columns"] == ["#", "Club", "P", "Pts"] and preview["row_count"] == 2


def test_a_title_line_before_a_hash_header_is_skipped_as_before():
    body = b"League table\n#,Club,P,Pts\n1,Harbour FC,10,23\n"
    assert store.preview(body, "csv")["columns"] == ["#", "Club", "P", "Pts"]


def test_a_hash_line_inside_the_data_is_a_comment():
    body = b"Pos,Club,P,Pts\n1,Harbour FC,10,23\n# a note, midseason\n2,Riverside,10,21\n"
    preview = store.preview(body, "csv")
    assert preview["columns"] == ["Pos", "Club", "P", "Pts"] and preview["row_count"] == 2


def test_the_fetch_content_check_reads_a_hash_header_as_a_table():
    assert fetch._csv_like(LEAGUE.decode()) is True
    assert table_lines(LEAGUE.decode())[0] == "#,Club,P,W,D,L,GD,Pts"


def test_noaa_style_hash_comments_are_still_dropped_before_the_data():
    # The NOAA daily CO2 file: a block of '#' lines, then whitespace-separated numbers with no commas.
    text = "# year month day decimal co2\n# more comment\n1974 5 19 1974.3781 333.46\n1974 5 20 1974.3836 333.28\n"
    assert comment_flags(text.splitlines()) == [True, True, False, False]
    columns, count, rows, _ = store._whitespace_table(text)
    assert count == 2 and rows[0] == ["1974", "5", "19", "1974.3781", "333.46"]


def test_noaa_style_comments_before_a_comma_table_are_dropped():
    text = "# CO2 daily\n# units: ppm\n1974,5,19,333.46\n1974,5,20,333.28\n"
    assert comment_flags(text.splitlines()) == [True, True, False, False]


def test_a_hash_line_that_is_the_only_line_is_a_comment():
    assert comment_flags(["# only a comment"]) == [True]
