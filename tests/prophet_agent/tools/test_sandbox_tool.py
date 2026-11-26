

def test_a_later_run_reads_an_earlier_runs_output_through_carry():
    from prophet_agent.tools.sandbox_tool import make_run
    run = make_run({})
    first = run({"code": "open('out/scan.json','w').write('{\"a\": 1}')"}, "k1")
    assert first["ok"] and first["outputs"][0]["path"] == "out/scan.json"
    second = run({"code": "print(open('in/carried/scan.json').read())",
                  "carry": [{"run_id": first["run_id"], "file": "out/scan.json"}]}, "k2")
    assert second["ok"] and '"a": 1' in second["stdout"]
    import pytest
    with pytest.raises(ValueError, match="no file"):
        run({"code": "print(1)", "carry": [{"run_id": first["run_id"], "file": "out/missing.json"}]}, "k3")
    with pytest.raises(ValueError, match="run_id"):
        run({"code": "print(1)", "carry": [{"run_id": "../..", "file": "out/scan.json"}]}, "k4")
    with pytest.raises(ValueError, match="unknown arguments"):
        run({"code": "print(1)", "extra": 1}, "k5")


def test_pair_helpers_work_inside_the_sandbox_and_a_dict_read_is_explained():
    from prophet_agent.tools.sandbox_tool import make_run
    code = ("import numpy as np, prophet_stats as ps\n"
            "x = np.r_[np.zeros(60), np.ones(60) * 5] + np.random.default_rng(1).normal(0, .1, 120)\n"
            "ch, rep = ps.changepoints(x)\nprint(len(ch))\n"
            "try:\n    ps.changepoints(x)['changepoints']\nexcept TypeError as e:\n    print(str(e)[:60])\n")
    res = make_run({})({"code": code}, "k")
    assert res["ok"], res["stderr"]
    assert "returns a pair" in res["stdout"]


def test_a_carried_download_with_a_step_number_for_a_run_id_reads_the_mounted_file(tmp_path):
    """Panel round 3 NEM (2026-11-26): carry [{file: 'PRICE_AND_DEMAND_202610_NSW1.csv', run_id: '18'}], read from in/carried/."""
    import pytest
    from prophet_agent.tools.sandbox_tool import make_run
    landed = tmp_path / "PRICE_AND_DEMAND_202610_NSW1.csv"
    landed.write_text("SETTLEMENTDATE,TOTALDEMAND\n2026/10/01 00:05:00,7000\n")
    run = make_run({"PRICE_AND_DEMAND_202610_NSW1.csv": landed})
    res = run({"code": "print(open('in/carried/PRICE_AND_DEMAND_202610_NSW1.csv').read())",
               "carry": [{"file": "PRICE_AND_DEMAND_202610_NSW1.csv", "run_id": "18"}]}, "k1")
    assert res["ok"] and "7000" in res["stdout"]
    with pytest.raises(ValueError, match="run_id"):
        run({"code": "print(1)", "carry": [{"file": "out/not_a_download.csv", "run_id": "18"}]}, "k2")
