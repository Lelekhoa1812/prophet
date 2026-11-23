"""Round 6m: the mention helpers that the endpoint tests did not reach. Offline: pure functions only."""
from __future__ import annotations

from prophet_agent import mentions


def no_resolver(_id: str):
    return None


def test_a_pick_whose_label_is_only_brackets_still_reads_back_as_a_link():
    stored = mentions.link("skill", "choose-model-by-backtest", "  [ ]  ")
    assert [(m.kind, m.id) for m in mentions.parse(stored)] == [("skill", "choose-model-by-backtest")]
    assert mentions.plain(stored) == "Choose model by backtest"


def test_a_file_pick_with_no_label_says_it_is_a_file():
    stored = mentions.link("file", "0123456789abcdef0123456789abcdef", "")
    assert mentions.plain(stored) == "Attached file"


def test_a_playbook_cut_at_no_line_break_says_it_was_cut():
    body = "x" * 7000
    block = mentions.context_block("[/Plan](prophet:skill/plan)", skill_body=lambda _id: ("Plan", body),
                                   tool_info=no_resolver, file_name=no_resolver)
    assert "...[playbook cut]" in block
    assert len(block) < len(body)


def test_a_playbook_cut_at_a_line_break_keeps_whole_lines_and_says_it_was_cut():
    body = "\n".join(f"line {i}" for i in range(2000))
    block = mentions.context_block("[/Plan](prophet:skill/plan)", skill_body=lambda _id: ("Plan", body),
                                   tool_info=no_resolver, file_name=no_resolver)
    assert "...[playbook cut]" in block
    kept = block.split("\n...[playbook cut]")[0].split("\n")
    assert all(line.startswith("line ") or line.startswith("The person") or line == "" for line in kept)


def test_a_short_playbook_is_passed_whole_without_a_cut_note():
    block = mentions.context_block("[/Plan](prophet:skill/plan)", skill_body=lambda _id: ("Plan", "Step one.\nStep two."),
                                   tool_info=no_resolver, file_name=no_resolver)
    assert "Step one.\nStep two." in block
    assert "playbook cut" not in block
