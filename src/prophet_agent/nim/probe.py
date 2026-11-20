"""Live NIM probes (NIM.md §8). Prints statuses, timings and validity only: never keys, never generated text.

    PYTHONPATH=src .venv/bin/python -m prophet_agent.nim.probe keys              # one 1-token call per key
    PYTHONPATH=src .venv/bin/python -m prophet_agent.nim.probe models            # claim schema, thinking off
    PYTHONPATH=src .venv/bin/python -m prophet_agent.nim.probe models --thinking  # also thinking on (slow, 25-60 s each)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .errors import NimError
from .transport import DEFAULT_CHAIN, NimClient
from .schema import validate

CLAIM_PROMPT = [
    {"role": "system", "content": "Extract factual claims from the text. Use only what the text says."},
    {"role": "user", "content": "The warehouse ships 40 pallets a day. Stock nearing expiry is cleared in 14 days."},
]
CLAIM_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
        "type": "object",
        "properties": {"claim": {"type": "string"},
                       "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
        "required": ["claim", "confidence"],
    }}},
    "required": ["claims"],
}


def probe_keys(client: NimClient) -> list[dict]:
    return [client.probe(slot) for slot in range(1, len(client.keys) + 1)]


def probe_models(client: NimClient, *, thinking: bool) -> list[dict]:
    rows = []
    for model in DEFAULT_CHAIN:
        started = time.monotonic()
        row: dict = {"model": model, "thinking": "on" if thinking else "off"}
        try:
            reply = client.complete([model], CLAIM_PROMPT, thinking=thinking, schema=CLAIM_SCHEMA,
                                    schema_name="claim_probe_v1", max_tokens=8192 if thinking else 1024,
                                    budget_s=240.0)
            row.update(ok=True, finish=reply.finish_reason, key_slot=reply.key_slot,
                       schema_valid=validate(json.loads(reply.text), CLAIM_SCHEMA) is None,
                       text_chars=len(reply.text), usage=reply.usage)
        except NimError as exc:
            row.update(ok=False, error=type(exc).__name__, detail=str(exc)[:200])
        row["elapsed_s"] = round(time.monotonic() - started, 2)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prophet_agent.nim.probe")
    parser.add_argument("what", choices=["keys", "models"])
    parser.add_argument("--thinking", action="store_true", help="also run thinking-on for each model")
    args = parser.parse_args(argv)
    try:
        client = NimClient.from_env()
    except NimError as exc:
        print(f"not configured: {exc}")
        return 2
    with client:
        print(f"{len(client.keys)} key slot(s) configured (values not shown)")
        if args.what == "keys":
            for row in probe_keys(client):
                print(json.dumps(row))
            return 0
        runs = [False, True] if args.thinking else [False]
        for thinking in runs:
            for row in probe_models(client, thinking=thinking):
                print(json.dumps(row, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
