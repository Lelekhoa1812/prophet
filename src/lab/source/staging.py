"""Read-only extractor: FoodFlow staging tenant -> pseudonymised local DuckDB.

Guards: session is read-only, every statement must start with SELECT, and only the
curated axis_semantic views plus products are read. Names are replaced by codes; the
code-to-name map lives in a separate local file used only when the console's reveal is on.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import duckdb
import pandas as pd
import psycopg

from ..config import DATASETS, REVEAL_DB, read_env_value
from ..store import open_dataset
from .pseudo import CodeBook, stable_hash


def short(kind: str, real_id) -> str:
    return stable_hash(kind, real_id)[:8].upper()

SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.I)
_WRITE_WORDS = re.compile(r"\b(insert|update|delete|merge|truncate|create|alter|drop|grant|revoke|copy|call|do|set|reset|vacuum|lock)\b|\binto\b", re.I)


def check_select(sql: str) -> None:
    """Refuse anything that is not one plain read. A second line of defence only: the database role is the real limit.

    Bugs vs Fixes
    Bug: the guard was an `assert`, which Python removes under `-O`, and it only checked the first word, so
    'select 1; drop table x' or 'select ... into t' would have passed.
    Fix: raise PermissionError, require a single statement, and reject write keywords outside string literals.
    """
    body = re.sub(r"'(?:[^']|'')*'", "''", sql).strip().rstrip(";")
    if not SELECT_ONLY.match(body) or ";" in body or _WRITE_WORDS.search(body):
        raise PermissionError(f"non-read query refused: {sql[:60]!r}")


def role_audit(cur) -> dict:
    """What the connected database role could do, from catalog reads only. Printed without any connection detail.

    Motivation vs Logic
    Motivation: read-only was enforced by a client-side session setting. A role that can write plus a client that
    behaves is protection by convention. 19 Nov 2026 audit: the role was a member of neon_superuser, owned 1,463
    relations and could write to all 209 visible relations.
    Logic: record superuser-like flags, role memberships and how many relations the role may INSERT/UPDATE/DELETE.
    """
    cur.execute("select rolsuper, rolcreatedb, rolcreaterole, rolbypassrls, rolreplication from pg_roles where rolname=current_user")
    flags = cur.fetchone()
    cur.execute("select r.rolname from pg_auth_members m join pg_roles r on r.oid=m.roleid join pg_roles u on u.oid=m.member where u.rolname=current_user")
    member_of = sorted(x[0] for x in cur.fetchall())
    cur.execute("""select count(*) filter (where has_table_privilege(current_user, c.oid,'INSERT') or has_table_privilege(current_user, c.oid,'UPDATE')
                                              or has_table_privilege(current_user, c.oid,'DELETE') or has_table_privilege(current_user, c.oid,'TRUNCATE')), count(*)
                   from pg_class c join pg_namespace n on n.oid=c.relnamespace
                   where c.relkind in ('r','p','v','m') and n.nspname not in ('pg_catalog','information_schema')""")
    writable, total = cur.fetchone()
    return {"superuser_like": bool(any(flags) or "neon_superuser" in member_of), "member_of": member_of, "writable_relations": int(writable),
            "total_relations": int(total), "can_write": int(writable) > 0,
            "note": "Read-only is enforced by the client session only." if int(writable) > 0 else "Role has no write privileges."}

QUERIES = {
    "lines": """
        select l.sales_order_id as order_id, l.line_number as line_no, l.delivery_date,
               l.customer_id, l.product_id, l.ordered_quantity as qty, l.packed_quantity as packed_qty,
               l.not_available_quantity as short_qty, l.unit_sell_price as unit_price,
               (l.is_order_cancelled or l.order_status in ('CANCELLED','Cancelled')) as cancelled,
               l.product_category_name as category, l.ordered_uom_code as uom
        from axis_semantic.sales_order_lines l where l.delivery_date is not null""",
    "orders": """
        select sales_order_id as order_id, subtotal_ex_gst, total_cost, margin_percent
        from axis_semantic.sales_orders""",
    "customers": """
        select customer_id, account_code, customer_group_name as segment, delivery_zone_name as zone,
               created_at::date as created_on
        from axis_semantic.customers""",
    "products": """
        select id as product_id, sku, name, "isCatchWeight" as is_catch_weight,
               "receivingShelfLifeDays" as shelf_life_days, "lastUnitPurchaseCost" as unit_cost,
               "lastUnitTargetSale" as unit_price
        from public.products""",
    "lots": """
        select lot_id, product_id, received_date, use_by_date, unit_landed_cost as unit_cost
        from axis_semantic.stock_lots""",
    "moves": """
        select lot_id, product_id, movement_local_date as moved_on, movement_type, quantity_change as qty_change
        from axis_semantic.stock_movements""",
    "po": """
        select pl.purchase_order_id as po_id, po.supplier_id, pl.product_id, po.created_at::date as created_on,
               po.expected_delivery_date as expected_on, pl.ordered_quantity as qty_ordered,
               pl.received_quantity as qty_received, pl.unit_purchase_cost as unit_cost
        from axis_semantic.purchase_order_lines pl
        join axis_semantic.purchase_orders po on po.purchase_order_id = pl.purchase_order_id""",
    "credits": """
        select customer_id, credit_date, subtotal_ex_gst as amount, credit_reasons as reason
        from axis_semantic.sales_credit_notes where voided_at is null""",
}


def _fetch(cur, name: str) -> pd.DataFrame:
    sql = QUERIES[name]
    check_select(sql)
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in df.columns:  # Decimal -> float
        if df[c].dtype == object and len(df) and df[c].map(lambda v: hasattr(v, "as_tuple")).any():
            df[c] = df[c].astype(float)
    return df


def extract() -> dict:
    url = read_env_value("STAGING_TENANT_DATABASE_URL")
    conn = psycopg.connect(url, connect_timeout=20, options="-c default_transaction_read_only=on")
    conn.read_only = True                          # every transaction starts READ ONLY, besides the session option
    cur = conn.cursor()
    audit = role_audit(cur)
    raw = {k: _fetch(cur, k) for k in QUERIES}
    conn.rollback()
    conn.close()

    lines, prods, custs = raw["lines"], raw["products"], raw["customers"]
    pcode = CodeBook("product", "P", list(prods.product_id) + list(lines.product_id))
    ccode = CodeBook("customer", "C", list(custs.customer_id) + list(lines.customer_id))
    scode = CodeBook("supplier", "S", [s for s in raw["po"].supplier_id.dropna()] or ["none"])

    cat_by_product = lines.dropna(subset=["category"]).groupby("product_id").category.first()
    uom_by_product = lines.dropna(subset=["uom"]).groupby("product_id").uom.first()

    con = open_dataset("staging")
    for t in ["products", "customers", "suppliers", "order_lines", "lots", "stock_movements",
              "purchase_lines", "credits", "truth_events", "daily_outcomes", "truth_expected", "meta"]:
        con.execute(f"DELETE FROM {t}")

    p = prods.copy()
    p["category"] = p.product_id.map(cat_by_product).fillna("Uncategorised")
    p["uom"] = p.product_id.map(uom_by_product).fillna("EA")
    p["code"] = p.product_id.map(pcode)
    p["display_name"] = p.category + " · " + p.code
    p = p[["product_id", "code", "display_name", "category", "uom", "is_catch_weight", "shelf_life_days",
           "unit_cost", "unit_price"]].copy()
    p["product_id"] = p.product_id.map(pcode)
    p["shelf_life_days"] = p.shelf_life_days.fillna(0).astype(int)
    p["is_catch_weight"] = p.is_catch_weight.fillna(False).astype(bool)
    p = p.drop_duplicates("product_id")

    c = custs.copy()
    c["code"] = c.customer_id.map(ccode)
    c["display_name"] = c.segment.fillna("Customer") + " account " + c.code
    c = c.rename(columns={"customer_id": "_real"})
    c["customer_id"] = c.code
    c = c[["customer_id", "code", "display_name", "segment", "zone", "created_on"]].drop_duplicates("customer_id")

    s = pd.DataFrame({"supplier_id": sorted(scode.codes.values())})
    s["code"], s["display_name"] = s.supplier_id, "Supplier " + s.supplier_id

    ol = lines.copy()
    ol["customer_id"] = ol.customer_id.map(ccode)
    ol["product_id"] = ol.product_id.map(pcode)
    ol["order_id"] = "O-" + ol.order_id.map(lambda x: short("order", x))
    ol["unit_cost"] = ol.product_id.map(p.set_index("product_id").unit_cost)
    ol["cancelled"] = ol.cancelled.fillna(False).astype(bool)
    ol = ol[["order_id", "line_no", "delivery_date", "customer_id", "product_id", "qty", "packed_qty",
             "short_qty", "unit_price", "unit_cost", "cancelled"]]

    lots = raw["lots"].copy()
    lots["lot_id"] = "L-" + lots.lot_id.map(lambda x: short("lot", x))
    lots["product_id"] = lots.product_id.map(pcode)
    lots["supplier_id"] = None
    lots["qty_received"] = None
    lots = lots[["lot_id", "product_id", "supplier_id", "received_date", "use_by_date", "qty_received", "unit_cost"]]

    mv = raw["moves"].copy()
    mv["lot_id"] = "L-" + mv.lot_id.map(lambda x: short("lot", x))
    mv["product_id"] = mv.product_id.map(pcode)

    po = raw["po"].copy()
    po["po_id"] = "PO-" + po.po_id.map(lambda x: short("po", x))
    po["supplier_id"] = po.supplier_id.map(lambda x: scode(x) if pd.notna(x) else None)
    po["product_id"] = po.product_id.map(pcode)

    cr = raw["credits"].copy()
    cr["customer_id"] = cr.customer_id.map(lambda x: ccode(x) if pd.notna(x) and str(x) in ccode.codes else None)

    for name, df in [("products", p), ("customers", c), ("suppliers", s), ("order_lines", ol), ("lots", lots),
                     ("stock_movements", mv), ("purchase_lines", po), ("credits", cr)]:
        con.register("df_in", df)
        con.execute(f"INSERT INTO {name} SELECT * FROM df_in")
        con.unregister("df_in")

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute("INSERT INTO meta VALUES ('source','FoodFlow staging tenant (read-only session, pseudonymised)'),"
                f"('extracted_at','{stamp}'),('role_audit','{json.dumps(audit)}')")
    lo, hi = con.execute("select min(delivery_date), max(delivery_date) from order_lines").fetchone()
    con.execute(f"INSERT INTO meta VALUES ('first_date','{lo}'),('last_date','{hi}')")

    # Local-only reveal map (real names), separate file, never read by default.
    rv = duckdb.connect(str(REVEAL_DB))
    rv.execute("CREATE OR REPLACE TABLE reveal_map (kind VARCHAR, code VARCHAR, real_name VARCHAR)")
    rows = [("product", pcode(r.product_id), r.name) for r in prods.itertuples()]
    rows += [("customer", ccode(r.customer_id), r.account_code) for r in custs.itertuples()]
    rv.executemany("INSERT INTO reveal_map VALUES (?,?,?)", rows)
    rv.close()

    counts = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in
              ["products", "customers", "order_lines", "lots", "stock_movements", "purchase_lines", "credits"]}
    con.close()
    return counts


if __name__ == "__main__":
    print(extract())
