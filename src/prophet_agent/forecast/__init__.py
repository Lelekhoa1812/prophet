"""Domain-free forecasting: accuracy metrics, honest baselines, TSB, distributions, backtest and the TimesFM 3 wrapper.

Importing this package does not load TimesFM. Use `prophet_agent.forecast.timesfm` for the model itself.
"""
from .types import Forecast

__all__ = ["Forecast"]
