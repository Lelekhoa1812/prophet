import numpy as np
import pandas as pd

from lab.twin.censor import censoring_mask


def outcomes(stock):                                      # stock: (products, days) bool
    P, D = stock.shape
    return pd.DataFrame({"day": np.tile(pd.date_range("2026-01-01", periods=D), P), "product_id": np.repeat([f"p{i}" for i in range(P)], D),
                         "stockout": stock.ravel()})


def test_orders_drop_only_in_the_days_after_a_stockout():
    stock = np.zeros((2, 10), bool); stock[0, 3] = True
    pi = np.repeat([0, 1], 10); c = np.tile(np.arange(10), 2)
    keep = censoring_mask(outcomes(stock), [f"p{i}" for i in range(2)], pd.date_range("2026-01-01", periods=10), pi, c, p_drop=1.0, lag_days=2, seed=1)
    k = keep.reshape(2, 10)
    assert k[1].all()                                     # other product untouched
    assert list(k[0]) == [True, True, True, True, False, False, True, True, True, True]   # days 4 and 5 lose their orders, not the stockout day itself


def test_drop_rate_and_determinism():
    stock = np.ones((1, 400), bool)
    pi = np.zeros(400, int); c = np.arange(400)
    args = (outcomes(stock), ["p0"], pd.date_range("2026-01-01", periods=400), pi, c)
    a = censoring_mask(*args, p_drop=0.4, lag_days=1, seed=3); b = censoring_mask(*args, p_drop=0.4, lag_days=1, seed=3)
    assert (a == b).all()
    assert abs((~a[1:]).mean() - 0.4) < 0.08
    assert a[0]                                            # nothing precedes the first day


def test_zero_strength_changes_nothing():
    stock = np.ones((1, 30), bool)
    keep = censoring_mask(outcomes(stock), ["p0"], pd.date_range("2026-01-01", periods=30), np.zeros(30, int), np.arange(30), p_drop=0.0, lag_days=2, seed=1)
    assert keep.all()


def test_censored_twin_drops_orders_but_keeps_latent_demand(tmp_path, monkeypatch):
    from lab import store
    from lab.twin import build
    from lab.twin.generator import TwinConfig
    monkeypatch.setattr(store, "DATASETS", {"a": tmp_path / "a.duckdb", "b": tmp_path / "b.duckdb"})
    cfg = TwinConfig(seed=5, start="2025-01-01", end="2025-06-30", n_products=25, n_customers=20, n_suppliers=4)
    build.main(name="a", cfg=cfg)
    build.main(name="b", cfg=cfg, censor={"p_drop": 0.6, "lag_days": 2})
    q = lambda n, sql: store.open_dataset(n, read_only=True).execute(sql).fetchone()[0]
    assert q("b", "select count(*) from order_lines") < q("a", "select count(*) from order_lines")
    assert q("a", "select sum(demand) from daily_outcomes") == q("b", "select sum(demand) from daily_outcomes")   # latent demand identical
    assert q("b", "select sum(qty) from order_lines") < q("b", "select sum(demand) from daily_outcomes")
    assert "censored" in q("b", "select value from meta where key='source'")
