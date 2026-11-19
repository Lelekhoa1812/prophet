"""Optional Bedrock Haiku narration. Hard-capped by a lab-only ledger; refuses to publish invented numbers."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from ..config import BEDROCK_HARD_CAP_USD, BEDROCK_MODEL, BEDROCK_REGION, read_env_value
from ..store import open_lab

# Assumed list prices (USD per 1M tokens). The ledger is an estimate; the hard cap is set low enough to absorb error.
PRICE_IN, PRICE_OUT = 1.0, 5.0

SYSTEM = ("You write one short, plain-English explanation (max 70 words) of a business alert for a food wholesaler's manager. "
          "Use ONLY the facts provided. Do not introduce any number, name or claim that is not in the facts. No jargon, no hedging, "
          "no markdown. Say what is happening, why it matters in dollars, and the suggested action.")

_client = None


def spent() -> float:
    lab = open_lab()
    v = lab.execute("select coalesce(sum(cost_usd),0) from bedrock_ledger").fetchone()[0]
    lab.close()
    return float(v)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def allowed_numbers(p: dict) -> set[str]:
    blob = json.dumps({"title": p["title"], "evidence": p["evidence"], "action": p["action"], "d": p["dollars_at_stake"]}, default=str)
    base = _numbers(blob)
    out = set(base)
    for n in list(base):                       # allow rounded forms of any figure in the facts
        try:
            f = float(n)
            out |= {str(int(round(f))), f"{f:.1f}", f"{f:.0f}", str(int(f))}
        except ValueError:
            pass
    return out | {"1", "2", "3", "4", "5", "7", "14", "28", "30", "100"}


def validate(text: str, p: dict) -> tuple[bool, list[str]]:
    bad = [n for n in _numbers(text) if n not in allowed_numbers(p)]
    return (not bad, bad)


def _bedrock():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client(
            "bedrock-runtime", region_name=BEDROCK_REGION,
            aws_access_key_id=read_env_value("AWS_ACCESS_KEY_ID"), aws_secret_access_key=read_env_value("AWS_SECRET_ACCESS_KEY"))
    return _client


def narrate(p: dict) -> dict:
    """Returns {text, model, ok, rejected_numbers, cost_usd}. Never raises past the cap; falls back to None text."""
    if spent() >= BEDROCK_HARD_CAP_USD:
        return {"text": None, "model": BEDROCK_MODEL, "ok": False, "error": "hard cap reached", "cost_usd": 0.0}
    facts = {"title": p["title"], "why": p["evidence"].get("why"), "facts": p["evidence"].get("facts"),
             "action": p["action"], "dollars_at_stake": p["dollars_at_stake"]}
    resp = _bedrock().converse(
        modelId=BEDROCK_MODEL, system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": json.dumps(facts, default=str)}]}],
        inferenceConfig={"maxTokens": 1500})
    u = resp.get("usage", {})
    cost = u.get("inputTokens", 0) / 1e6 * PRICE_IN + u.get("outputTokens", 0) / 1e6 * PRICE_OUT
    lab = open_lab()                           # log spend first: the call is billed even if parsing fails
    lab.execute("insert into bedrock_ledger values (?,?,?,?,?,?,?)",
                [uuid.uuid4().hex, datetime.now(timezone.utc), BEDROCK_MODEL, u.get("inputTokens", 0), u.get("outputTokens", 0), cost, p.get("proposal_id")])
    lab.close()
    blocks = resp["output"]["message"]["content"]          # some models return a reasoning block before the text
    text = " ".join(b["text"] for b in blocks if "text" in b).strip()
    if not text:
        return {"text": None, "model": BEDROCK_MODEL, "ok": False, "error": "no text block", "cost_usd": cost}
    ok, bad = validate(text, p)
    return {"text": text if ok else None, "raw": text, "model": BEDROCK_MODEL, "ok": ok, "rejected_numbers": bad, "cost_usd": cost}
