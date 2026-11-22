"""A2-10: reconcile_forecasts tool. A hierarchy and base forecasts from a saved run go in; coherent reconciled forecasts per
node come out, with the method used, the warnings and a coherence check. Stub data only; no model is loaded."""
import json

import numpy as np
import pytest

from prophet_agent.tools import series_io
from prophet_agent.tools import reconcile_tool as rt

RUN = "b" * 32
H = 4
BASE = {  # base forecasts that are not coherent: Total is not the sum of A, B and C
    "Total": [30.0, 31.0, 33.0, 34.0],
    "AB": [19.0, 20.0, 21.0, 22.0],
    "A": [10.0, 10.5, 11.0, 11.0],
    "B": [9.5, 10.0, 9.8, 11.5],
    "C": [9.0, 9.5, 10.2, 10.0],
}
COHERENT = {"Total": [a + b + c for a, b, c in zip(BASE["A"], BASE["B"], BASE["C"])],
            "AB": [a + b for a, b in zip(BASE["A"], BASE["B"])], "A": BASE["A"], "B": BASE["B"], "C": BASE["C"]}
HIER = {"bottom": ["A", "B", "C"], "groups": {"Total": ["A", "B", "C"], "AB": ["A", "B"]}}


def _residuals(seed=0, T=30):
    rng = np.random.default_rng(seed)
    return {k: [float(v) for v in rng.normal(0, s, T)] for k, s in
            {"Total": 1.5, "AB": 1.0, "A": 0.8, "B": 0.9, "C": 0.7}.items()}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    out = tmp_path / RUN / "out"
    out.mkdir(parents=True)
    (out / "forecasts.json").write_text(json.dumps({"series": BASE, "dates": {}}))
    (out / "residuals.json").write_text(json.dumps({"series": _residuals(), "dates": {}}))
    (out / "coherent.json").write_text(json.dumps({"series": COHERENT, "dates": {}}))
    return tmp_path


def _args(**kw):
    base = {"source": {"run_id": RUN, "file": "out/forecasts.json"}, "hierarchy": HIER}
    base.update(kw)
    return base


def test_reconciled_output_is_coherent_for_every_node(sandbox):
    """Regression: the tool did not exist (A2-10). Every parent must equal the sum of its children in the output."""
    res = rt._run(_args(), "k")
    nodes = {k: np.asarray(v) for k, v in res["nodes"].items()}
    assert set(nodes) == {"Total", "AB", "A", "B", "C"}
    np.testing.assert_allclose(nodes["Total"], nodes["A"] + nodes["B"] + nodes["C"], atol=1e-9)
    np.testing.assert_allclose(nodes["AB"], nodes["A"] + nodes["B"], atol=1e-9)
    assert res["coherence"]["coherent_after"] is True and res["coherence"]["max_gap_after"] < 1e-9
    assert res["coherence"]["base_coherent"] is False and res["coherence"]["base_gap"] > 0.5


def test_a_coherent_base_comes_back_unchanged_for_every_method(sandbox):
    for method in ("mint_shrink", "wls_structural", "ols", "bottom_up"):
        res = rt._run(_args(source={"run_id": RUN, "file": "out/coherent.json"}, method=method,
                            residuals={"run_id": RUN, "file": "out/residuals.json"}), "k")
        for k, v in COHERENT.items():
            np.testing.assert_allclose(res["nodes"][k], v, atol=1e-6, err_msg=f"{method} changed {k}")


def test_mint_uses_residuals_and_says_so_and_without_them_falls_back_with_a_warning(sandbox):
    with_res = rt._run(_args(residuals={"run_id": RUN, "file": "out/residuals.json"}), "k")
    assert with_res["method_used"] == "mint_shrink" and with_res["shrinkage"] is not None
    without = rt._run(_args(), "k")
    assert without["method_used"] == "wls_structural" and without["shrinkage"] is None
    assert any("WLS-structural" in w for w in without["warnings"])


def test_the_summing_matrix_form_gives_the_same_answer_as_the_groups_form(sandbox):
    by_groups = rt._run(_args(method="ols"), "k")
    matrix = {"nodes": ["Total", "AB", "A", "B", "C"], "bottom": ["A", "B", "C"],
              "matrix": [[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    by_matrix = rt._run(_args(hierarchy=matrix, method="ols"), "k")
    for k in by_groups["nodes"]:
        np.testing.assert_allclose(by_matrix["nodes"][k], by_groups["nodes"][k], atol=1e-9)


def test_the_same_call_gives_the_same_result(sandbox):
    a = rt._run(_args(method="ols"), "k")
    b = rt._run(_args(method="ols"), "k")
    assert a == b
    tool = rt.reconcile_forecasts_tool()
    assert tool.name == "reconcile_forecasts" and tool.idempotent is True and tool.observe is rt.observe


@pytest.mark.parametrize("bad, message", [
    ({"method": "shrink"}, "method must be"),
    ({"hierarchy": {"bottom": ["A", "B"], "groups": {"Total": ["A", "Z"]}}}, "unknown"),
    ({"hierarchy": {"bottom": ["A", "B", "C"], "groups": {"A": ["A"]}}}, "same name"),
    ({"hierarchy": {"bottom": ["A", "B", "C"]}}, "groups or a summing matrix"),
    ({"hierarchy": {"bottom": ["A", "B", "C"], "groups": {"Total": ["A", "B", "C"]}, "matrix": [[1]]}}, "one of"),
    ({"hierarchy": {"bottom": ["A", "B", "C"], "groups": {"Total": ["A", "B", "C"]}, "extra": 1}}, "unknown keys"),
    ({"source": {"run_id": RUN, "file": "out/nope.json"}}, "no file"),
    ({"values": [1.0]}, "reconcile_forecasts needs source"),
    ({"bogus": 1}, "unknown arguments"),
])
def test_bad_arguments_are_refused_with_the_reason(sandbox, bad, message):
    with pytest.raises(ValueError, match=message):
        rt._run({**_args(), **bad}, "k")


def test_a_node_missing_from_the_base_file_is_refused(sandbox):
    (sandbox / RUN / "out" / "partial.json").write_text(json.dumps({"series": {"A": [1.0] * H, "B": [1.0] * H}}))
    with pytest.raises(ValueError, match="no base forecast for"):
        rt._run(_args(source={"run_id": RUN, "file": "out/partial.json"}), "k")


def test_observe_is_short_and_keeps_the_coherence_check(sandbox):
    view = rt.observe(rt._run(_args(method="ols"), "k"))
    assert {"method_used", "warnings", "coherence", "nodes"} <= set(view)
    assert view["coherence"]["coherent_after"] is True


def test_the_tool_is_registered_with_its_title_and_spec():
    from prophet_agent.tools import default_tools
    from prophet_agent.tools.reconcile_tool import RECONCILE_FORECASTS_SPEC
    assert RECONCILE_FORECASTS_SPEC["title"] and RECONCILE_FORECASTS_SPEC["activity"]
    registered = default_tools()["reconcile_forecasts"]
    assert registered.title == RECONCILE_FORECASTS_SPEC["title"] and "hierarchy" in registered.doc
    assert len(RECONCILE_FORECASTS_SPEC["description"]) < 700
