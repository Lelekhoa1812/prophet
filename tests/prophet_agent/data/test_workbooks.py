"""Workbook ingestion: sheets, hidden sheets, title rows, two tables on a sheet, totals, archives, damaged and protected files."""
from __future__ import annotations

import gzip
import json
import zipfile

import openpyxl
import pytest

from prophet_agent.sandbox.runner import run_python

CODE = """
import json, prophet_lib as pl
print(json.dumps(pl.profile_workbook('in/{name}')))
"""


def _profile(tmp_path, name):
    res = run_python(CODE.format(name=name), inputs={name: tmp_path / name}, timeout_s=60)
    assert res.ok, res.stderr
    return json.loads(res.stdout)


def _book(tmp_path, build, name="b.xlsx"):
    wb = openpyxl.Workbook()
    build(wb)
    wb.save(tmp_path / name)
    return name


def test_title_rows_hidden_sheet_and_totals(tmp_path):
    def build(wb):
        ws = wb.active
        ws.title = "Sales"
        ws.append(["Quarterly sales report"])
        ws.append(["prepared by finance"])
        ws.append([])
        ws.append(["Month", "Units", "Revenue"])
        for i in range(1, 10):
            ws.append([f"2024-{i:02d}-01", 10 * i, 99.5 * i])
        ws.append(["Total", 450, 4477.5])
        hid = wb.create_sheet("Scratch")
        hid.append(["x"])
        hid.sheet_state = "hidden"

    p = _profile(tmp_path, _book(tmp_path, build))
    assert [s["name"] for s in p["sheets"]] == ["Sales"] and p["hidden_sheets"] == ["Scratch"]
    sheet = p["sheets"][0]
    assert sheet["read"]["header_row"] == 4 and sheet["read"]["skipped_preamble_lines"] >= 2
    assert sheet["rows"] == 10
    assert any("look like totals" in i for i in sheet["issues"])
    assert {c["name"] for c in sheet["columns"]} == {"Month", "Units", "Revenue"}


def test_two_tables_on_one_sheet_are_reported_not_merged(tmp_path):
    def build(wb):
        ws = wb.active
        ws.append(["Item", "Qty"])
        for i in range(5):
            ws.append([f"a{i}", i])
        ws.append([])
        ws.append(["Region", "Sales"])
        for i in range(5):
            ws.append([f"r{i}", 100 + i])

    sheet = _profile(tmp_path, _book(tmp_path, build))["sheets"][0]
    assert len(sheet["read"]["tables_on_sheet"]) == 2 and any("separate tables" in i for i in sheet["issues"])
    assert sheet["rows"] == 5 and {c["name"] for c in sheet["columns"]} == {"Item", "Qty"}


def test_dates_and_numbers_stored_as_real_cells(tmp_path):
    import datetime

    def build(wb):
        ws = wb.active
        ws.append(["Day", "Sales"])
        for i in range(30):
            ws.append([datetime.date(2024, 1, 1) + datetime.timedelta(days=i), 100.0 + i])

    cols = {c["name"]: c for c in _profile(tmp_path, _book(tmp_path, build))["sheets"][0]["columns"]}
    assert cols["Day"]["kind"] == "time" and cols["Day"]["typical_step"].startswith("1 days")
    assert cols["Sales"]["kind"] == "number" and cols["Sales"]["min"] == 100.0


def test_gzip_zip_and_damaged_and_wrong_extension(tmp_path):
    csv = b"d,v\n" + b"\n".join(b"2024-01-%02d,%d" % (i, i) for i in range(1, 25)) + b"\n"
    (tmp_path / "a.csv.gz").write_bytes(gzip.compress(csv))
    with zipfile.ZipFile(tmp_path / "z.zip", "w") as z:
        z.writestr("data/one.csv", csv)
        z.writestr("two.csv", csv)
    (tmp_path / "bad.xlsx").write_bytes(b"PK\x03\x04 this is not really a workbook")
    (tmp_path / "fake.xlsx").write_bytes(csv)          # a csv wearing an xlsx name
    for name, expect_rows in (("a.csv.gz", 24), ("z.zip", 24), ("fake.xlsx", 24)):
        sheet = _profile(tmp_path, name)["sheets"][0]
        assert sheet["rows"] == expect_rows, (name, sheet["issues"])
    assert "other_members_not_read" in _profile(tmp_path, "z.zip")["sheets"][0]["read"]
    bad = _profile(tmp_path, "bad.xlsx")["sheets"][0]
    assert bad["rows"] == 0 and bad["issues"], "a damaged file is reported, never read partially"


def _run(tmp_path, code, name, content):
    (tmp_path / name).write_text(content, encoding="utf-8")
    res = run_python(code.replace("{name}", name), inputs={name: tmp_path / name}, timeout_s=60)
    assert res.ok, res.stderr
    return json.loads(res.stdout)


def test_load_returns_one_typed_frame_and_keeps_codes_as_text(tmp_path):
    out = _run(tmp_path, """
import json, prophet_lib as pl
df = pl.load('in/{name}')
rep = df.attrs['report']
print(json.dumps({'dtypes': {c: str(t) for c, t in df.dtypes.items()}, 'sales': [None if v != v else v for v in df['Sales']],
                  'converted': sorted(rep['converted']), 'text': rep['kept_as_text'], 'month': int(df['Date'].dt.month.iloc[1])}))
""", "t.csv", 'Date,Sales,Code,Name\n15/11/2025,180,00123,a\n03/07/2025,"1,204",00456,b\n20/12/2025,,00789,c\n')
    assert out["dtypes"]["Sales"] == "float64" and out["dtypes"]["Date"].startswith("datetime64")
    assert out["sales"] == [180.0, 1204.0, None], "thousands separators parse and a blank stays missing, not zero"
    assert out["converted"] == ["Date", "Sales"] and out["text"] == ["Code", "Name"]
    assert out["month"] == 7, "the 15 and 20 force day-first, so 03/07 is 3 July"


def test_a_stated_date_order_settles_an_ambiguous_column_and_the_helpers_return_one_value(tmp_path):
    out = _run(tmp_path, """
import json, pandas as pd, prophet_lib as pl
s = pd.Series(['03/04/2025', '05/06/2025'])
print(json.dumps({'day': [d.month for d in pl.as_dates(s, dayfirst=True)], 'mon': [d.month for d in pl.as_dates(s, dayfirst=False)],
                  'num': list(pl.as_number(pd.Series(['1,5', '2'])))}))
""", "unused.csv", "x\n")
    assert out["day"] == [4, 6] and out["mon"] == [3, 5]
    assert len(out["num"]) == 2


def test_to_number_keeps_us_thousands_and_refuses_ids():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "prophet_agent" / "sandbox"))
    import prophet_lib as pl
    import pandas as pd
    text = ["$23,940.43", "1,234,567.89", "1.234,56", "12,5", "INV-1000", "555-1234", "(1,200.50)", "12kg", "AUD 100", "abc"]
    got, info = pl.to_number(pd.Series(text))
    assert got.iloc[0] == 23940.43 and got.iloc[1] == 1234567.89     # "1,234.56" is not European
    assert got.iloc[2] == 1234.56 and got.iloc[3] == 12.5            # "1.234,56" and "12,5" are
    assert got.iloc[4:6].isna().all() and got.iloc[9] != got.iloc[9]  # ids and phone numbers are not numbers
    assert got.iloc[6] == -1200.5 and got.iloc[7] == 12 and got.iloc[8] == 100
    assert info["decimal_comma_values"] == 2


def test_iso_dates_are_never_ambiguous():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "prophet_agent" / "sandbox"))
    import prophet_lib as pl
    import pandas as pd
    s = pd.Series(["2023-03-01", "2023-04-01", "2023-05-01", "2023-06-02"])
    got, info = pl.to_datetime(s)
    assert info["ambiguous_day_month"] is False and got.iloc[0] == pd.Timestamp("2023-03-01") and got.iloc[3] == pd.Timestamp("2023-06-02")
    got, _ = pl.to_datetime(s, dayfirst=True)             # even a stated day-first order cannot swap a year-first date
    assert got.iloc[0] == pd.Timestamp("2023-03-01")
    day, _ = pl.to_datetime(pd.Series(["03/04/2023", "05/06/2023"]))
    assert _["ambiguous_day_month"] is True


def test_a_two_row_merged_header_becomes_one_name_per_column(tmp_path):
    import csv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "prophet_agent" / "sandbox"))
    import prophet_lib as pl
    rows = [["Half year report"], [""], ["Region", "Q1 2026", "", "Q2 2026", ""], ["", "Units", "Revenue", "Units", "Revenue"]]
    rows += [[n, 100 + i, 1000.5 + i, 200 + i, 2000.5 + i] for i, n in enumerate(["Alpha", "Bay", "Cedar", "Dune", "Eagle"])]
    rows += [["TOTAL", 1, 2, 3, 4]]
    f = tmp_path / "hy.csv"
    with f.open("w", newline="") as h:
        csv.writer(h).writerows(rows)
    df = pl.load(f)
    assert list(df.columns) == ["Region", "Q1 2026 Units", "Q1 2026 Revenue", "Q2 2026 Units", "Q2 2026 Revenue"]
    assert df.attrs["report"]["read"]["header_rows"] == 2 and df["Q2 2026 Revenue"].iloc[0] == 2000.5
    plain = tmp_path / "plain.csv"
    plain.write_text("a,b,c\nx,1,2\ny,3,4\nz,5,6\nw,7,8\n")
    assert list(pl.load(plain).columns) == ["a", "b", "c"]
    text_body = tmp_path / "t.csv"                      # a text first data row must not be mistaken for a second header
    text_body.write_text("name,qty,price\nUnit A,1,2\nUnit B,3,4\nUnit C,5,6\nUnit D,7,8\nUnit E,9,10\n")
    assert list(pl.load(text_body).columns) == ["name", "qty", "price"]


def test_tidy_text_merges_case_and_spacing_variants_and_reports_it():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "prophet_agent" / "sandbox"))
    import prophet_lib as pl
    import pandas as pd
    s = pd.Series(["Sable Logistics", "sable logistics ", "SABLE  LOGISTICS", "Yarra Packaging", "Yarra Packaging.", None, ""])
    t = pl.tidy_text(s)
    assert t.iloc[:3].tolist() == ["Sable Logistics"] * 3 and t.iloc[3] == t.iloc[4] == "Yarra Packaging"
    assert t.iloc[5] is None and t.iloc[6] is None
    assert t.attrs["report"]["unique_before"] == 6 and t.attrs["report"]["unique_after"] == 2
    assert t.attrs["report"]["examples_merged"]


def test_profile_flags_names_typed_more_than_one_way(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "prophet_agent" / "sandbox"))
    import prophet_lib as pl
    names = ["Northgate Auto", "northgate auto", "NORTHGATE  AUTO ", "Elm & Oak Co", "elm & oak co", "Brightwater"] * 5
    f = tmp_path / "s.csv"
    f.write_text("Customer,Revenue\n" + "\n".join(f'"{n}",{100 + i}' for i, n in enumerate(names)) + "\n")
    prof = pl.profile(f)
    col = next(c for c in prof["columns"] if c["name"] == "Customer")
    assert col["spelling_variants"] == {"unique": 6, "after_tidy": 3}
    assert any("tidy_text" in i for i in prof["issues"])
