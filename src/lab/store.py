"""Canonical DuckDB schema shared by the real snapshot and the synthetic twin."""
from __future__ import annotations

import duckdb

from .config import DATASETS, LAB_DB

DATASET_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
  product_id VARCHAR PRIMARY KEY, code VARCHAR, display_name VARCHAR, category VARCHAR,
  uom VARCHAR, is_catch_weight BOOLEAN, shelf_life_days INTEGER,
  unit_cost DOUBLE, unit_price DOUBLE
);
CREATE TABLE IF NOT EXISTS customers (
  customer_id VARCHAR PRIMARY KEY, code VARCHAR, display_name VARCHAR,
  segment VARCHAR, zone VARCHAR, created_on DATE
);
CREATE TABLE IF NOT EXISTS suppliers (
  supplier_id VARCHAR PRIMARY KEY, code VARCHAR, display_name VARCHAR
);
CREATE TABLE IF NOT EXISTS order_lines (
  order_id VARCHAR, line_no INTEGER, delivery_date DATE, customer_id VARCHAR, product_id VARCHAR,
  qty DOUBLE, packed_qty DOUBLE, short_qty DOUBLE, unit_price DOUBLE, unit_cost DOUBLE,
  cancelled BOOLEAN
);
CREATE TABLE IF NOT EXISTS lots (
  lot_id VARCHAR PRIMARY KEY, product_id VARCHAR, supplier_id VARCHAR, received_date DATE,
  use_by_date DATE, qty_received DOUBLE, unit_cost DOUBLE
);
CREATE TABLE IF NOT EXISTS stock_movements (
  lot_id VARCHAR, product_id VARCHAR, moved_on DATE, movement_type VARCHAR, qty_change DOUBLE
);
CREATE TABLE IF NOT EXISTS purchase_lines (
  po_id VARCHAR, supplier_id VARCHAR, product_id VARCHAR, created_on DATE, expected_on DATE,
  qty_ordered DOUBLE, qty_received DOUBLE, unit_cost DOUBLE
);
CREATE TABLE IF NOT EXISTS credits (
  customer_id VARCHAR, credit_date DATE, amount DOUBLE, reason VARCHAR
);
-- Ground truth exists only in the twin: injected shocks and their labels.
CREATE TABLE IF NOT EXISTS truth_events (
  event_id VARCHAR PRIMARY KEY, kind VARCHAR, entity_type VARCHAR, entity_id VARCHAR,
  start_date DATE, end_date DATE, magnitude DOUBLE, note VARCHAR
);
-- Daily outcomes the twin knows: stockouts and waste, so agents can be scored.
CREATE TABLE IF NOT EXISTS daily_outcomes (
  day DATE, product_id VARCHAR, demand DOUBLE, fulfilled DOUBLE, on_hand_end DOUBLE,
  expired_qty DOUBLE, expired_value DOUBLE, stockout BOOLEAN
);
-- True underlying expected demand per product and day (twin only): the noise floor for scoring.
CREATE TABLE IF NOT EXISTS truth_expected (
  day DATE, product_id VARCHAR, expected_qty DOUBLE
);
CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR);
"""

LAB_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id VARCHAR PRIMARY KEY, kind VARCHAR, dataset VARCHAR, agent VARCHAR, as_of DATE,
  started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR, code_version VARCHAR,
  model VARCHAR, params JSON, summary JSON
);
CREATE TABLE IF NOT EXISTS proposals (
  proposal_id VARCHAR PRIMARY KEY, run_id VARCHAR, dataset VARCHAR, agent VARCHAR, as_of DATE,
  entity_type VARCHAR, entity_id VARCHAR, title VARCHAR, severity VARCHAR,
  dollars_at_stake DOUBLE, confidence DOUBLE, autonomy_level VARCHAR, owner VARCHAR,
  evidence JSON, action JSON, status VARCHAR, dedupe_key VARCHAR,
  narration_templated VARCHAR, narration_model VARCHAR, created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id VARCHAR PRIMARY KEY, proposal_id VARCHAR, decided_at TIMESTAMP, decided_by VARCHAR,
  decision VARCHAR, reason VARCHAR, note VARCHAR
);
CREATE TABLE IF NOT EXISTS lessons (
  lesson_id VARCHAR PRIMARY KEY, dataset VARCHAR, agent VARCHAR, scope_type VARCHAR, scope_id VARCHAR,
  effect JSON, source_decision_id VARCHAR, active BOOLEAN, created_at TIMESTAMP, undone_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS agent_config (
  dataset VARCHAR, agent VARCHAR, enabled BOOLEAN, owner VARCHAR, quiet_start VARCHAR,
  quiet_end VARCHAR, cooldown_days INTEGER, params JSON, autonomy_level VARCHAR,
  PRIMARY KEY (dataset, agent)
);
CREATE TABLE IF NOT EXISTS forecast_cache (
  cache_key VARCHAR PRIMARY KEY, dataset VARCHAR, series_id VARCHAR, as_of DATE, horizon INTEGER,
  payload JSON
);
CREATE TABLE IF NOT EXISTS bedrock_ledger (
  call_id VARCHAR PRIMARY KEY, called_at TIMESTAMP, model VARCHAR, input_tokens INTEGER,
  output_tokens INTEGER, cost_usd DOUBLE, proposal_id VARCHAR
);
CREATE TABLE IF NOT EXISTS backtests (
  backtest_id VARCHAR PRIMARY KEY, dataset VARCHAR, created_at TIMESTAMP, config JSON, results JSON
);
CREATE TABLE IF NOT EXISTS replay_state (
  dataset VARCHAR PRIMARY KEY, cursor_date DATE, speed DOUBLE, running BOOLEAN
);
"""


def open_dataset(name: str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DATASETS[name]), read_only=read_only)
    if not read_only:
        con.execute(DATASET_SCHEMA)
    return con


def open_lab() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(LAB_DB))
    con.execute(LAB_SCHEMA)
    return con
