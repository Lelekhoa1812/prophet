"""Turn fetched pages and files into one checked table.

The goal loop stays the only loop. This package checks a claim, reads tables a fetch already stored, and binds a
series name only after the entity and the columns match.
"""
from prophet_agent.source_table.calibrate import calibrate_overlap
from prophet_agent.source_table.choose import choose_columns
from prophet_agent.source_table.claim import build_claim
from prophet_agent.source_table.diagnose import diagnose_failure
from prophet_agent.source_table.evaluate import process_score
from prophet_agent.source_table.extract import extract_html
from prophet_agent.source_table.fuse import fuse_sources
from prophet_agent.source_table.publish import bind_series
from prophet_agent.source_table.validate import validate_table

__all__ = ["bind_series", "build_claim", "calibrate_overlap", "choose_columns", "diagnose_failure", "extract_html",
           "fuse_sources", "process_score", "validate_table"]
