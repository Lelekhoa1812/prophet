"""Run the Prophet API on loopback: ``PYTHONPATH=src .venv/bin/python -m prophet_agent.api``.

Motivation vs Logic
Motivation: the console needs one command that starts the API against the local event store, with no personal paths.
Logic: builds the service on ``DB_PATH`` with the default tools and the NIM-backed planner, then serves on loopback.
The port is ``PROPHET_API_PORT`` (default 8010, chosen to stay clear of older local servers). The Vite dev server
(web/vite.config.ts) proxies /api to it.
"""
from __future__ import annotations

import os

import uvicorn

from prophet_agent.config import DB_PATH, ensure_dirs
from prophet_agent.tools import default_tools, tool_catalogue

from .app import create_app, default_brain_factory
from .service import MissionService


def build_app():
    ensure_dirs()
    service = MissionService(DB_PATH, default_brain_factory, tool_catalogue(), tools_factory=default_tools)
    return create_app(service)


def main() -> None:
    port = int(os.environ.get("PROPHET_API_PORT", "8010"))
    uvicorn.run(build_app(), host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
