"""A skill, tool or file picked with / or @ is a structured pick, not pasted words: the planner receives the playbook, the
tool request and the file, the file is attached, the sent message keeps its links, and anything that does not exist is
left as plain text. A scripted brain stands in for the planner, so nothing here calls a real provider."""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent import mentions
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.skills import list_skills, read_skill
from prophet_agent.tools import default_tools

SKILL = "choose-model-by-backtest"


class Recorder:
    def __init__(self):
        self.messages: list[list[dict]] = []

    def decide(self, messages, schema, name):
        self.messages.append(messages)
        if name == "plan":
            return {"action": "finish", "rationale": "done"}
        return {"verdict": "met", "reason": "ok", "answer": "all good"}


def _client(tmp_path):
    brain = Recorder()
    service = MissionService(tmp_path / "m.sqlite", lambda: brain, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service, brain


def _seen(brain: Recorder) -> str:
    return "\n".join(m["content"] for msgs in brain.messages for m in msgs if isinstance(m.get("content"), str))


def test_links_parse_and_the_plain_text_is_what_the_person_read():
    text = (f"Check [/Choose model by backtest](prophet:skill/{SKILL}) on [@sales.csv](prophet:file/abc123) "
            f"with [/Forecast](prophet:tool/timesfm_forecast), then [/Choose model by backtest](prophet:skill/{SKILL}) again")
    got = mentions.parse(text)
    assert [(m.kind, m.id) for m in got] == [("skill", SKILL), ("file", "abc123"), ("tool", "timesfm_forecast")], "once per pick"
    assert mentions.plain(text) == ("Check /Choose model by backtest on @sales.csv with /Forecast, "
                                    "then /Choose model by backtest again")
    assert mentions.parse("[x](prophet:other/1) [y](https://example.com) [z](prophet:file/a b)") == []


def test_link_drops_brackets_and_line_breaks_from_a_label():
    out = mentions.link("file", "id1", "we[ir]d\nname.csv")
    assert out == "[we ir d name.csv](prophet:file/id1)"
    assert mentions.parse(out)[0].label == "we ir d name.csv"


def test_a_picked_playbook_is_loaded_into_the_planner_context_and_the_goal_reads_as_plain_words(tmp_path):
    client, service, brain = _client(tmp_path)
    goal = f"Use [/Choose model by backtest](prophet:skill/{SKILL}) for the daily series"
    r = client.post("/api/missions", json={"goal": goal})
    assert r.status_code == 201, r.text
    service.wait(r.json()["id"], timeout=10)
    seen = _seen(brain)
    body = read_skill(SKILL)
    assert "The person invoked the playbook \"Choose model by backtest\"" in seen
    assert body[:120] in seen, "the playbook text itself reaches the planner"
    assert "prophet:skill" not in seen, "link syntax never reaches the model"
    assert "Use /Choose model by backtest for the daily series" in seen
    # the stored message keeps the link, so the sent bubble, retry and edit all show the chip
    assert client.get(f"/api/missions/{r.json()['id']}").json()["goal"] == goal


def test_a_tool_pick_is_stated_as_a_request(tmp_path):
    client, service, brain = _client(tmp_path)
    name = next(iter(service.tools))
    goal = f"run [/Tool](prophet:tool/{name}) now"
    r = client.post("/api/missions", json={"goal": goal})
    service.wait(r.json()["id"], timeout=10)
    seen = _seen(brain)
    assert "The person asked for these tools by name" in seen and f"(tool {name})" in seen


def test_a_picked_file_is_attached_and_named_as_the_subject(tmp_path):
    client, service, brain = _client(tmp_path)
    ds = client.post("/api/datasets?name=sales.csv", content=b"date,units\n2024-01-01,3\n2024-01-02,4\n").json()
    goal = f"What is in [@sales.csv](prophet:file/{ds['id']})?"
    r = client.post("/api/missions", json={"goal": goal})   # no dataset_ids sent: the pick alone attaches it
    assert r.status_code == 201, r.text
    mission = service.store.mission(r.json()["id"])
    assert mission["budget"]["dataset_ids"] == [ds["id"]]
    service.wait(r.json()["id"], timeout=10)
    seen = _seen(brain)
    assert "pointed at these files with @" in seen and "- sales.csv" in seen
    assert "Files the person attached" in seen, "the usual file listing still follows"


def test_a_pick_that_does_not_exist_stays_plain_text_and_is_not_an_error(tmp_path):
    client, service, brain = _client(tmp_path)
    goal = ("look at [/Gone](prophet:skill/no-such-playbook) and [/Nope](prophet:tool/no_such_tool) "
            "and [@x.csv](prophet:file/deadbeef)")
    r = client.post("/api/missions", json={"goal": goal})
    assert r.status_code == 201, r.text
    service.wait(r.json()["id"], timeout=10)
    seen = _seen(brain)
    assert "look at /Gone and /Nope and @x.csv" in seen
    assert "invoked the playbook" not in seen and "asked for these tools" not in seen and "pointed at these files" not in seen
    assert "dataset_ids" not in service.store.mission(r.json()["id"])["budget"]


def test_the_title_and_the_length_limit_use_the_words_not_the_links(tmp_path):
    client, service, _ = _client(tmp_path)
    long_link = mentions.link("file", "a" * 32, "report.csv")
    ok = client.post("/api/missions", json={"goal": long_link + " " + "x" * 1980})
    assert ok.status_code == 201, "the limit counts what the person sees, not the link syntax"
    mid = ok.json()["id"]
    service.wait(mid, timeout=10)
    assert "prophet:" not in (service.store.mission(mid).get("title") or "")
    too_long = client.post("/api/missions", json={"goal": long_link + " " + "x" * 2100})
    assert too_long.status_code == 422
    short = client.post("/api/missions", json={"goal": "[ab](prophet:tool/forecast)" + ""})
    assert short.status_code == 422, "two visible characters is still too short"


def test_a_notebook_file_pick_resolves_to_the_source_and_earlier_questions_are_shown_as_words(tmp_path):
    client, service, brain = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "NB"}).json()["id"]
    src = client.post(f"/api/notebooks/{nid}/sources?name=plan.csv", content=b"a,b\n1,2\n3,4\n").json()
    goal = f"Summarise [@plan.csv](prophet:file/{src['id']})"
    first = client.post(f"/api/notebooks/{nid}/turns", json={"goal": goal})
    assert first.status_code == 201, first.text
    service.wait(first.json()["id"], timeout=10)
    assert "pointed at these files with @" in _seen(brain) and "- plan.csv" in _seen(brain)
    brain.messages.clear()
    second = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "and now the totals"})
    service.wait(second.json()["id"], timeout=10)
    seen = _seen(brain)
    assert "Q1: Summarise @plan.csv" in seen and "prophet:file" not in seen


def test_the_skill_endpoints_list_and_read_the_playbooks(tmp_path):
    client, _, _ = _client(tmp_path)
    rows = client.get("/api/skills").json()
    assert {r["name"] for r in rows} == {s["name"] for s in list_skills()}
    one = next(r for r in rows if r["name"] == SKILL)
    assert one["title"] == "Choose model by backtest" and one["summary"]
    got = client.get(f"/api/skills/{SKILL}").json()
    assert got["body"] == read_skill(SKILL)
    assert client.get("/api/skills/not-a-skill").status_code == 404
