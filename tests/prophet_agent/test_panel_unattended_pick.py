"""The unattended panel's answers never narrow the work (round 20261128-063203, AURATE: "Univariate forecast only (current
path)" was picked and the drivers were dropped)."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _panel():
    path = Path(__file__).resolve().parents[2] / "scripts" / "workflows" / "research_panel.py"
    spec = importlib.util.spec_from_file_location("research_panel_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_univariate_shortcut_is_not_picked_even_when_marked():
    panel = _panel()
    opts = [{"label": "Univariate forecast only (current path) (Recommended)", "description": ""},
            {"label": "Download the yield and test it", "description": "score it with control_lab"}]
    assert panel._pick(opts) == "Download the yield and test it"
    assert panel._pick([{"label": "Deliver now", "description": ""}, {"label": "Keep testing (Recommended)",
                                                                       "description": ""}]) == "Keep testing (Recommended)"
