"""Synthetic demand data with known ground truth, for offline forecast and agent tests. See generator.py for the shape."""
from prophet_agent.synth.censor import StockRun, censor_series, simulate_stock
from prophet_agent.synth.generator import SyntheticDataset, generate

__all__ = ["StockRun", "SyntheticDataset", "censor_series", "generate", "simulate_stock"]
