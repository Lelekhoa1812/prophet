"""Build the twin dataset: world + BAU supply history, written into twin.duckdb."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..store import open_dataset
from .generator import TwinConfig, TwinWorld, build_world
from .supply import SimResult, simulate


def order_lines_frame(world: TwinWorld, fill: np.ndarray) -> pd.DataFrame:
    r, c = np.nonzero(world.demand)
    p = world.pairs
    pi = p.pi.values[r]; ci = p.ci.values[r]
    qty = world.demand[r, c]
    f = fill[pi, c]
    df = pd.DataFrame({
        "delivery_date": world.days.values[c].astype("datetime64[D]"),
        "customer_id": world.customers.customer_id.values[ci],
        "product_id": world.products.product_id.values[pi],
        "qty": qty, "packed_qty": np.round(qty * f, 3),
        "unit_price": np.round(world.price[r, c], 2),
        "unit_cost": np.round(world.products.unit_cost.values[pi] * world.cost_mult[pi, c], 2),
        "cancelled": False,
    })
    df["short_qty"] = df.qty - df.packed_qty
    key = df.customer_id + "|" + df.delivery_date.astype(str)
    df["order_id"] = "O-" + pd.factorize(key)[0].astype(str)
    df["order_id"] = df.order_id.str.zfill(8)
    df = df.sort_values(["delivery_date", "order_id"]).reset_index(drop=True)
    df["line_no"] = df.groupby("order_id").cumcount() + 1
    return df[["order_id", "line_no", "delivery_date", "customer_id", "product_id", "qty", "packed_qty",
               "short_qty", "unit_price", "unit_cost", "cancelled"]]


def write_twin(world: TwinWorld, res: SimResult, name: str = "twin") -> dict:
    con = open_dataset(name)
    for t in ["products", "customers", "suppliers", "order_lines", "lots", "stock_movements",
              "purchase_lines", "credits", "truth_events", "daily_outcomes", "truth_expected", "meta"]:
        con.execute(f"DELETE FROM {t}")
    P = world.products.copy()
    P = P[["product_id", "code", "display_name", "category", "uom", "is_catch_weight", "shelf_life_days", "unit_cost", "unit_price"]]
    C = world.customers[["customer_id", "code", "display_name", "segment", "zone", "created_on"]]
    S = world.suppliers[["supplier_id", "code", "display_name"]]
    ol = order_lines_frame(world, res.fill)
    T = pd.DataFrame(world.truth).rename(columns={"start": "start_date", "end": "end_date"})
    T = T[["event_id", "kind", "entity_type", "entity_id", "start_date", "end_date", "magnitude", "note"]]
    frames = {"products": P, "customers": C, "suppliers": S, "order_lines": ol, "lots": res.lots,
              "stock_movements": res.movements, "purchase_lines": res.purchases, "truth_events": T,
              "daily_outcomes": res.outcomes}
    exp = np.zeros((len(world.products), len(world.days)))
    np.add.at(exp, world.pairs.pi.values, world.expected)
    ex_df = pd.DataFrame({"day": np.tile(world.days.values.astype("datetime64[D]"), len(world.products)),
                          "product_id": np.repeat(world.products.product_id.values, len(world.days)),
                          "expected_qty": exp.ravel()})
    frames["truth_expected"] = ex_df
    for tname, df in frames.items():
        con.register("df_in", df)
        con.execute(f"INSERT INTO {tname} SELECT * FROM df_in")
        con.unregister("df_in")
    con.execute("INSERT INTO meta VALUES ('source','Synthetic twin (calibrated on staging profile)'),"
                f"('seed','{world.cfg.seed}'),('first_date','{world.days[0].date()}'),('last_date','{world.days[-1].date()}')")
    counts = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in frames}
    con.close()
    return counts


def main(seed: int = 7) -> dict:
    world = build_world(TwinConfig(seed=seed))
    res = simulate(world)
    counts = write_twin(world, res)
    counts["waste_value"] = round(res.waste_value)
    counts["stockout_value"] = round(res.stockout_value)
    return counts


if __name__ == "__main__":
    print(main())
