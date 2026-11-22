"""Polars names on pandas objects (prophet_lib, "polars names"): null_count, is_null, is_not_null and sort.

Motivation vs Logic
Motivation: the alias ``pl`` makes models write polars code. null_count was called 38 times and is_null 13 times in the
live ledger, and each call failed.
Logic: each name is added to the pandas class only where pandas has no method of that name, and it returns what the
polars call would mean in pandas terms. Tests check the values in-process and inside a real sandbox cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prophet_agent.sandbox import prophet_lib as pl
from prophet_agent.sandbox.runner import run_python


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"units": [1.0, np.nan, 3.0, np.nan], "sku": ["a", None, "c", "d"]})


def test_dataframe_null_count_is_one_row_of_missing_counts_per_column():
    out = _frame().null_count()
    assert isinstance(out, pd.DataFrame) and len(out) == 1
    assert out.iloc[0].to_dict() == {"units": 2, "sku": 1}


def test_series_null_count_is_an_int():
    n = _frame()["units"].null_count()
    assert n == 2 and isinstance(n, int)


def test_is_null_and_is_not_null_map_to_isna_and_notna():
    s = _frame()["units"]
    assert s.is_null().tolist() == s.isna().tolist()
    assert s.is_not_null().tolist() == s.notna().tolist()
    assert _frame().is_null().equals(_frame().isna())


def test_sort_maps_to_sort_values_and_descending_is_reversed():
    df = pd.DataFrame({"v": [2, 3, 1]})
    assert df.sort("v")["v"].tolist() == [1, 2, 3]
    assert df.sort("v", descending=True)["v"].tolist() == [3, 2, 1]
    assert df["v"].sort(descending=True).tolist() == [3, 2, 1]


def test_pandas_methods_that_already_exist_are_not_replaced():
    s = pd.Series([1, 2, 3])
    assert s.sort_values(ascending=False).tolist() == [3, 2, 1]
    assert pd.DataFrame({"a": [1]}).isna().shape == (1, 1)


def test_installing_the_names_twice_changes_nothing():
    before = pd.DataFrame.null_count
    pl._install_polars_names()
    assert pd.DataFrame.null_count is before


def test_the_polars_names_work_inside_a_sandbox_cell():
    code = (
        "import pandas as pd, numpy as np\n"
        "import prophet_lib as pl\n"
        "df = pd.DataFrame({'a': [1.0, None, 3.0], 'b': ['x', None, 'z']})\n"
        "print(df.null_count().iloc[0].to_dict())\n"
        "print(df['a'].null_count())\n"
        "print(df['a'].is_null().tolist(), int(df['a'].is_not_null().sum()))\n"
        "print(df.sort('a', descending=True)['a'].tolist())\n"
    )
    r = run_python(code)
    assert r.ok, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "{'a': 1, 'b': 1}"
    assert lines[1] == "1"
    assert lines[2] == "[False, True, False] 2"
    assert lines[3] == "[3.0, 1.0, nan]"
