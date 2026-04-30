from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# Deterministic date-derived columns (never forecast as exogenous drivers).
CALENDAR_BASE_COLS = [
    "dow",
    "day_of_week",
    "is_weekend",
    "day",
    "day_of_month",
    "month",
    "quarter",
    "weekofyear",
    "dayofyear",
    "day_of_year",
    "is_month_start",
    "is_month_end",
    "days_since_start",
    "days_to_tet",
]

# Tet (Lunar New Year) Gregorian dates needed for this competition window.
# Keep one year ahead (2025) so days after Tet-2024 still map to the next Tet.
TET_LUNAR_NEW_YEAR = {
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
    2025: "2025-01-29",
}

# Core business drivers kept regardless of auto ranking.
CORE_DRIVER_FEATURES = [
    "order_count",
    "units_sold",
    "avg_unit_price",
    "discount_rate_avg",
    "promo_used_rate",
    "active_promo_count",
    "promo_depth_proxy",
    "sessions",
    "unique_visitors",
    "page_views",
    "inv_stock_on_hand",
    "inv_units_sold",
    "inv_days_of_supply",
    "inv_fill_rate",
    "return_qty",
    "refund_amount",
    "review_count",
    "new_signup_count",
    # Derived interactions
    "conversion_rate_proxy",
    "items_per_order",
    "views_per_session",
    "orders_per_visitor",
    "traffic_quality_proxy",
    "promo_pressure_index",
    "effective_promo_pressure",
    "inventory_pressure_index",
    "availability_index",
    "stock_cover_ratio",
    "return_rate_qty",
    "revenue_proxy",
    "net_demand_proxy",
]

# Share-level features below are often noisy in long-horizon forecasts.
NOISY_DRIVER_PREFIXES = [
    "order_status_share_",
    "order_source_share_",
    "device_share_",
    "payment_method_share_",
    "return_reason_share_",
    "traffic_source_share_",
    "acq_channel_share_",
]

# Conservative auto-pruning of exogenous drivers:
# 1) rank by model importance, 2) validate by quick CV, 3) only prune when metrics do not degrade.
AUTO_DRIVER_MIN_KEEP = 12
AUTO_DRIVER_KEEP_RATIO = 0.45
AUTO_DRIVER_CV_TOL = 0.0015
PIPELINE_CACHE_VERSION = "v6_lgb_native_feature_refresh"

# -----------------------------
# Utility
# -----------------------------


def safe_div(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)



def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))



def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den == 0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / den)



def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def mean_rank_3metrics(metric_map: dict[str, dict[str, float]]) -> dict[str, float]:
    names = list(metric_map.keys())
    mean_rank = {n: 0.0 for n in names}
    specs = [("mae", True), ("rmse", True), ("r2", False)]  # True => lower is better

    for metric_name, lower_is_better in specs:
        vals = np.array([metric_map[n][metric_name] for n in names], dtype=float)
        order = np.argsort(vals if lower_is_better else -vals)
        sorted_vals = vals[order]
        ranks = np.zeros(len(names), dtype=float)

        i = 0
        rank_pos = 1
        while i < len(names):
            j = i + 1
            while j < len(names) and np.isclose(sorted_vals[j], sorted_vals[i]):
                j += 1
            avg_rank = (rank_pos + (rank_pos + (j - i) - 1)) / 2.0
            for k in range(i, j):
                ranks[k] = avg_rank
            rank_pos += (j - i)
            i = j

        for idx_in_sorted, idx_in_names in enumerate(order):
            mean_rank[names[idx_in_names]] += ranks[idx_in_sorted]

    return {n: mean_rank[n] / 3.0 for n in names}


def log(msg: str) -> None:
    ts = pd.Timestamp.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def score_rank(metrics_a: dict[str, float], metrics_b: dict[str, float]) -> tuple[float, float]:
    # Lower is better for MAE/RMSE, higher is better for R2.
    ranks_a = 0
    ranks_b = 0
    if metrics_a["mae"] <= metrics_b["mae"]:
        ranks_a += 1
    else:
        ranks_b += 1
    if metrics_a["rmse"] <= metrics_b["rmse"]:
        ranks_a += 1
    else:
        ranks_b += 1
    if metrics_a["r2"] >= metrics_b["r2"]:
        ranks_a += 1
    else:
        ranks_b += 1
    return float(ranks_a), float(ranks_b)



def date_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.date_range(start, end, freq="D")})



def days_to_next_tet(ts: pd.Timestamp) -> int:
    d = pd.Timestamp(ts).normalize()
    for y in [d.year - 1, d.year, d.year + 1, d.year + 2]:
        if y in TET_LUNAR_NEW_YEAR:
            tet_day = pd.Timestamp(TET_LUNAR_NEW_YEAR[y])
            if tet_day >= d:
                return int((tet_day - d).days)

    future_tet = [pd.Timestamp(v) for v in TET_LUNAR_NEW_YEAR.values() if pd.Timestamp(v) >= d]
    if future_tet:
        return int((min(future_tet) - d).days)
    return 0



def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    if start_date is None:
        start_ts = pd.Timestamp(d.min()).normalize()
    else:
        start_ts = pd.Timestamp(start_date).normalize()

    out["dow"] = d.dt.dayofweek
    out["day_of_week"] = out["dow"]
    out["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    out["day"] = d.dt.day
    out["day_of_month"] = out["day"]
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["weekofyear"] = d.dt.isocalendar().week.astype(int)
    out["dayofyear"] = d.dt.dayofyear
    out["day_of_year"] = out["dayofyear"]
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["days_since_start"] = (d.dt.normalize() - start_ts).dt.days.astype(int)
    out["days_to_tet"] = d.apply(days_to_next_tet).astype(int)
    return out



def add_fourier(df: pd.DataFrame, period_col: str = "dayofyear", max_k: int = 5) -> pd.DataFrame:
    out = df.copy()
    x = out[period_col].astype(float).values
    for k in range(1, max_k + 1):
        out[f"sin_{k}"] = np.sin(2.0 * np.pi * k * x / 365.25)
        out[f"cos_{k}"] = np.cos(2.0 * np.pi * k * x / 365.25)
    return out


# -----------------------------
# Data loading
# -----------------------------


@dataclass
class DataBundle:
    products: pd.DataFrame
    customers: pd.DataFrame
    promotions: pd.DataFrame
    geography: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    payments: pd.DataFrame
    shipments: pd.DataFrame
    returns: pd.DataFrame
    reviews: pd.DataFrame
    sales: pd.DataFrame
    inventory: pd.DataFrame
    web_traffic: pd.DataFrame
    sample_submission: pd.DataFrame



def load_data(base_dir: Path) -> DataBundle:
    return DataBundle(
        products=pd.read_csv(base_dir / "products.csv"),
        customers=pd.read_csv(base_dir / "customers.csv", parse_dates=["signup_date"]),
        promotions=pd.read_csv(base_dir / "promotions.csv", parse_dates=["start_date", "end_date"]),
        geography=pd.read_csv(base_dir / "geography.csv"),
        orders=pd.read_csv(base_dir / "orders.csv", parse_dates=["order_date"]),
        order_items=pd.read_csv(base_dir / "order_items.csv", dtype={"promo_id": "string", "promo_id_2": "string"}),
        payments=pd.read_csv(base_dir / "payments.csv"),
        shipments=pd.read_csv(base_dir / "shipments.csv", parse_dates=["ship_date", "delivery_date"]),
        returns=pd.read_csv(base_dir / "returns.csv", parse_dates=["return_date"]),
        reviews=pd.read_csv(base_dir / "reviews.csv", parse_dates=["review_date"]),
        sales=pd.read_csv(base_dir / "sales.csv", parse_dates=["Date"]).sort_values("Date"),
        inventory=pd.read_csv(base_dir / "inventory.csv", parse_dates=["snapshot_date"]),
        web_traffic=pd.read_csv(base_dir / "web_traffic.csv", parse_dates=["date"]),
        sample_submission=pd.read_csv(base_dir / "sample_submission.csv", parse_dates=["Date"]),
    )


# -----------------------------
# Feature mart from all CSV files
# -----------------------------


def aggregate_orders_daily(orders: pd.DataFrame) -> pd.DataFrame:
    base = orders.copy()
    base["Date"] = pd.to_datetime(base["order_date"])

    g = base.groupby("Date", as_index=False).agg(order_count=("order_id", "count"))

    status_share = (
        base.pivot_table(index="Date", columns="order_status", values="order_id", aggfunc="count", fill_value=0)
        .reset_index()
    )
    status_share_cols = [c for c in status_share.columns if c != "Date"]
    for c in status_share_cols:
        status_share[f"order_status_share_{c}"] = safe_div(status_share[c], status_share[status_share_cols].sum(axis=1))
    status_share = status_share[["Date"] + [f"order_status_share_{c}" for c in status_share_cols]]

    top_sources = base["order_source"].value_counts().head(4).index.tolist()
    src = base.assign(order_source=np.where(base["order_source"].isin(top_sources), base["order_source"], "other"))
    src_p = src.pivot_table(index="Date", columns="order_source", values="order_id", aggfunc="count", fill_value=0).reset_index()
    src_cols = [c for c in src_p.columns if c != "Date"]
    for c in src_cols:
        src_p[f"order_source_share_{c}"] = safe_div(src_p[c], src_p[src_cols].sum(axis=1))
    src_p = src_p[["Date"] + [f"order_source_share_{c}" for c in src_cols]]

    top_device = base["device_type"].value_counts().head(4).index.tolist()
    dev = base.assign(device_type=np.where(base["device_type"].isin(top_device), base["device_type"], "other"))
    dev_p = dev.pivot_table(index="Date", columns="device_type", values="order_id", aggfunc="count", fill_value=0).reset_index()
    dev_cols = [c for c in dev_p.columns if c != "Date"]
    for c in dev_cols:
        dev_p[f"device_share_{c}"] = safe_div(dev_p[c], dev_p[dev_cols].sum(axis=1))
    dev_p = dev_p[["Date"] + [f"device_share_{c}" for c in dev_cols]]

    out = g.merge(status_share, on="Date", how="left").merge(src_p, on="Date", how="left").merge(dev_p, on="Date", how="left")
    return out



def aggregate_order_items_daily(order_items: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    line = order_items.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    line = line.merge(products[["product_id", "category", "segment"]], on="product_id", how="left")
    line["Date"] = pd.to_datetime(line["order_date"])
    line["line_amount"] = line["quantity"] * line["unit_price"]
    line["discount_rate"] = safe_div(line["discount_amount"], line["line_amount"])  # line level
    line["promo_used"] = (line["promo_id"].notna() | line["promo_id_2"].notna()).astype(int)

    core = line.groupby("Date", as_index=False).agg(
        units_sold=("quantity", "sum"),
        avg_unit_price=("unit_price", "mean"),
        discount_rate_avg=("discount_rate", "mean"),
        promo_used_rate=("promo_used", "mean"),
    )

    top_cat = products["category"].value_counts().head(4).index.tolist()
    cat = line.assign(category=np.where(line["category"].isin(top_cat), line["category"], "other"))
    cat_g = cat.groupby(["Date", "category"], as_index=False)["line_amount"].sum()
    cat_p = cat_g.pivot_table(index="Date", columns="category", values="line_amount", fill_value=0).reset_index()
    cat_cols = [c for c in cat_p.columns if c != "Date"]
    for c in cat_cols:
        cat_p[f"category_mix_{c}"] = safe_div(cat_p[c], cat_p[cat_cols].sum(axis=1))
    cat_p = cat_p[["Date"] + [f"category_mix_{c}" for c in cat_cols]]

    top_seg = products["segment"].value_counts().head(4).index.tolist()
    seg = line.assign(segment=np.where(line["segment"].isin(top_seg), line["segment"], "other"))
    seg_g = seg.groupby(["Date", "segment"], as_index=False)["line_amount"].sum()
    seg_p = seg_g.pivot_table(index="Date", columns="segment", values="line_amount", fill_value=0).reset_index()
    seg_cols = [c for c in seg_p.columns if c != "Date"]
    for c in seg_cols:
        seg_p[f"segment_mix_{c}"] = safe_div(seg_p[c], seg_p[seg_cols].sum(axis=1))
    seg_p = seg_p[["Date"] + [f"segment_mix_{c}" for c in seg_cols]]

    return core.merge(cat_p, on="Date", how="left").merge(seg_p, on="Date", how="left")



def aggregate_payments_daily(payments: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    p = payments.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    p["Date"] = pd.to_datetime(p["order_date"])
    core = p.groupby("Date", as_index=False).agg(
        avg_payment_value=("payment_value", "mean"),
        installments_mean=("installments", "mean"),
    )
    top_pm = p["payment_method"].value_counts().head(4).index.tolist()
    p2 = p.assign(payment_method=np.where(p["payment_method"].isin(top_pm), p["payment_method"], "other"))
    piv = p2.pivot_table(index="Date", columns="payment_method", values="order_id", aggfunc="count", fill_value=0).reset_index()
    pm_cols = [c for c in piv.columns if c != "Date"]
    for c in pm_cols:
        piv[f"payment_method_share_{c}"] = safe_div(piv[c], piv[pm_cols].sum(axis=1))
    piv = piv[["Date"] + [f"payment_method_share_{c}" for c in pm_cols]]
    return core.merge(piv, on="Date", how="left")



def aggregate_shipments_daily(shipments: pd.DataFrame) -> pd.DataFrame:
    sh = shipments.copy()
    sh["Date"] = pd.to_datetime(sh["ship_date"])
    sh["delivery_lead_days"] = (pd.to_datetime(sh["delivery_date"]) - pd.to_datetime(sh["ship_date"])).dt.days
    out = sh.groupby("Date", as_index=False).agg(
        avg_shipping_fee=("shipping_fee", "mean"),
        delivery_lead_time=("delivery_lead_days", "mean"),
        shipped_count=("order_id", "count"),
    )
    return out



def aggregate_returns_daily(returns: pd.DataFrame, order_items: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    ret = returns.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    ret["Date"] = pd.to_datetime(ret["return_date"])

    # refund rate proxy using order-item line amount
    oi = order_items.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    oi["line_amount"] = oi["quantity"] * oi["unit_price"]
    oi["Date"] = pd.to_datetime(oi["order_date"])
    oi_daily = (
        oi.dropna(subset=["Date"])
        .groupby("Date", as_index=False)
        .agg(item_amount=("line_amount", "sum"))
    )

    core = ret.groupby("Date", as_index=False).agg(
        return_count=("return_id", "count"),
        return_qty=("return_quantity", "sum"),
        refund_amount=("refund_amount", "sum"),
    )

    reason_top = ret["return_reason"].value_counts().head(4).index.tolist()
    r2 = ret.assign(return_reason=np.where(ret["return_reason"].isin(reason_top), ret["return_reason"], "other"))
    piv = r2.pivot_table(index="Date", columns="return_reason", values="return_id", aggfunc="count", fill_value=0).reset_index()
    r_cols = [c for c in piv.columns if c != "Date"]
    for c in r_cols:
        piv[f"return_reason_share_{c}"] = safe_div(piv[c], piv[r_cols].sum(axis=1))
    piv = piv[["Date"] + [f"return_reason_share_{c}" for c in r_cols]]

    out = core.merge(oi_daily, on="Date", how="left")
    out["refund_rate"] = safe_div(out["refund_amount"], out["item_amount"])  # proxy
    out = out.drop(columns=["item_amount"])
    return out.merge(piv, on="Date", how="left")



def aggregate_reviews_daily(reviews: pd.DataFrame) -> pd.DataFrame:
    rv = reviews.copy()
    rv["Date"] = pd.to_datetime(rv["review_date"])
    rv["low_rating"] = (rv["rating"] <= 2).astype(int)
    return rv.groupby("Date", as_index=False).agg(
        review_count=("review_id", "count"),
        rating_mean=("rating", "mean"),
        low_rating_share=("low_rating", "mean"),
    )



def aggregate_customers_daily(customers: pd.DataFrame) -> pd.DataFrame:
    c = customers.copy()
    c["Date"] = pd.to_datetime(c["signup_date"])
    out = c.groupby("Date", as_index=False).agg(new_signup_count=("customer_id", "count"))

    top_ch = c["acquisition_channel"].fillna("unknown").value_counts().head(4).index.tolist()
    c2 = c.assign(acquisition_channel=np.where(c["acquisition_channel"].fillna("unknown").isin(top_ch), c["acquisition_channel"].fillna("unknown"), "other"))
    piv = c2.pivot_table(index="Date", columns="acquisition_channel", values="customer_id", aggfunc="count", fill_value=0).reset_index()
    ch_cols = [c for c in piv.columns if c != "Date"]
    for c in ch_cols:
        piv[f"acq_channel_share_{c}"] = safe_div(piv[c], piv[ch_cols].sum(axis=1))
    piv = piv[["Date"] + [f"acq_channel_share_{c}" for c in ch_cols]]
    return out.merge(piv, on="Date", how="left")



def aggregate_geography_orders_daily(orders: pd.DataFrame, geography: pd.DataFrame) -> pd.DataFrame:
    o = orders.merge(geography[["zip", "region"]], on="zip", how="left")
    o["Date"] = pd.to_datetime(o["order_date"])

    top_reg = o["region"].fillna("unknown").value_counts().head(4).index.tolist()
    o2 = o.assign(region=np.where(o["region"].fillna("unknown").isin(top_reg), o["region"].fillna("unknown"), "other"))
    piv = o2.pivot_table(index="Date", columns="region", values="order_id", aggfunc="count", fill_value=0).reset_index()
    reg_cols = [c for c in piv.columns if c != "Date"]
    for c in reg_cols:
        piv[f"region_order_share_{c}"] = safe_div(piv[c], piv[reg_cols].sum(axis=1))
    return piv[["Date"] + [f"region_order_share_{c}" for c in reg_cols]]



def aggregate_promotions_daily(promotions: pd.DataFrame, all_dates: pd.Series) -> pd.DataFrame:
    # 50 rows only, safe to expand.
    date_df = pd.DataFrame({"Date": pd.to_datetime(all_dates)})
    date_df["k"] = 1
    p = promotions.copy()
    p["k"] = 1
    expanded = date_df.merge(p, on="k", how="left").drop(columns=["k"])
    active = expanded[(expanded["Date"] >= expanded["start_date"]) & (expanded["Date"] <= expanded["end_date"])].copy()
    if active.empty:
        out = date_df[["Date"]].copy()
        out["active_promo_count"] = 0
        out["stackable_share"] = 0.0
        out["promo_depth_proxy"] = 0.0
        return out

    active["depth"] = np.where(
        active["promo_type"].eq("percentage"),
        active["discount_value"].astype(float),
        np.minimum(active["discount_value"].astype(float), 50.0),
    )
    out = active.groupby("Date", as_index=False).agg(
        active_promo_count=("promo_id", "count"),
        stackable_share=("stackable_flag", "mean"),
        promo_depth_proxy=("depth", "mean"),
    )
    return out



def aggregate_inventory_monthly_to_daily(inventory: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    inv = inventory.copy()
    inv_month = inv.groupby("snapshot_date", as_index=False).agg(
        inv_stock_on_hand=("stock_on_hand", "sum"),
        inv_units_received=("units_received", "sum"),
        inv_units_sold=("units_sold", "sum"),
        inv_stockout_days=("stockout_days", "mean"),
        inv_days_of_supply=("days_of_supply", "mean"),
        inv_fill_rate=("fill_rate", "mean"),
        inv_stockout_flag=("stockout_flag", "mean"),
        inv_overstock_flag=("overstock_flag", "mean"),
        inv_reorder_flag=("reorder_flag", "mean"),
        inv_sell_through_rate=("sell_through_rate", "mean"),
    ).sort_values("snapshot_date")

    # Snapshot at month end becomes known from next month start.
    inv_month["effective_date"] = inv_month["snapshot_date"] + pd.offsets.MonthBegin(1)

    daily = date_grid(start, end)
    daily = daily.merge(inv_month.drop(columns=["snapshot_date"]), how="left", left_on="Date", right_on="effective_date")
    daily = daily.drop(columns=["effective_date"]) 
    daily = daily.sort_values("Date")
    feat_cols = [c for c in daily.columns if c != "Date"]
    daily[feat_cols] = daily[feat_cols].ffill().fillna(0)
    return daily



def aggregate_web_traffic_daily(web_traffic: pd.DataFrame) -> pd.DataFrame:
    wt = web_traffic.copy()
    wt["Date"] = pd.to_datetime(wt["date"])
    core = wt.groupby("Date", as_index=False).agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_duration_sec=("avg_session_duration_sec", "mean"),
    )
    top_src = wt["traffic_source"].value_counts().head(4).index.tolist()
    wt2 = wt.assign(traffic_source=np.where(wt["traffic_source"].isin(top_src), wt["traffic_source"], "other"))
    piv = wt2.pivot_table(index="Date", columns="traffic_source", values="sessions", aggfunc="sum", fill_value=0).reset_index()
    src_cols = [c for c in piv.columns if c != "Date"]
    for c in src_cols:
        piv[f"traffic_source_share_{c}"] = safe_div(piv[c], piv[src_cols].sum(axis=1))
    piv = piv[["Date"] + [f"traffic_source_share_{c}" for c in src_cols]]
    return core.merge(piv, on="Date", how="left")


def add_derived_business_features(mart: pd.DataFrame) -> pd.DataFrame:
    out = mart.copy()
    if "order_count" in out.columns and "sessions" in out.columns:
        out["conversion_rate_proxy"] = safe_div(out["order_count"], out["sessions"])
    if "units_sold" in out.columns and "order_count" in out.columns:
        out["items_per_order"] = safe_div(out["units_sold"], out["order_count"])
    if "page_views" in out.columns and "sessions" in out.columns:
        out["views_per_session"] = safe_div(out["page_views"], out["sessions"])
    if "order_count" in out.columns and "unique_visitors" in out.columns:
        out["orders_per_visitor"] = safe_div(out["order_count"], out["unique_visitors"])
    if "avg_session_duration_sec" in out.columns and "bounce_rate" in out.columns:
        out["traffic_quality_proxy"] = out["avg_session_duration_sec"].astype(float) * (1.0 - out["bounce_rate"].astype(float))

    if "active_promo_count" in out.columns and "promo_depth_proxy" in out.columns:
        out["promo_pressure_index"] = out["active_promo_count"].astype(float) * out["promo_depth_proxy"].astype(float)
    if "promo_pressure_index" in out.columns and "stackable_share" in out.columns:
        out["effective_promo_pressure"] = out["promo_pressure_index"].astype(float) * (1.0 + out["stackable_share"].astype(float))

    if "inv_stockout_days" in out.columns and "inv_fill_rate" in out.columns:
        out["inventory_pressure_index"] = out["inv_stockout_days"].astype(float) * (1.0 - out["inv_fill_rate"].astype(float))
    if "inv_fill_rate" in out.columns and "inv_stockout_flag" in out.columns:
        out["availability_index"] = out["inv_fill_rate"].astype(float) * (1.0 - out["inv_stockout_flag"].astype(float))
    if "inv_stock_on_hand" in out.columns and "inv_units_sold" in out.columns:
        out["stock_cover_ratio"] = safe_div(out["inv_stock_on_hand"], out["inv_units_sold"] + 1.0)

    if "return_qty" in out.columns and "units_sold" in out.columns:
        out["return_rate_qty"] = safe_div(out["return_qty"], out["units_sold"])
    if "units_sold" in out.columns and "avg_unit_price" in out.columns:
        out["revenue_proxy"] = out["units_sold"].astype(float) * out["avg_unit_price"].astype(float)
    if "units_sold" in out.columns and "return_rate_qty" in out.columns:
        out["net_demand_proxy"] = out["units_sold"].astype(float) * (1.0 - out["return_rate_qty"].astype(float))

    return out



def build_daily_feature_mart(bundle: DataBundle) -> pd.DataFrame:
    sales = bundle.sales[["Date", "Revenue", "COGS"]].copy().sort_values("Date")
    start = sales["Date"].min()
    end = sales["Date"].max()
    grid = date_grid(start, end)

    all_dates_for_promos = pd.concat([grid["Date"], bundle.sample_submission["Date"]], ignore_index=True)

    feats = [
        aggregate_orders_daily(bundle.orders),
        aggregate_order_items_daily(bundle.order_items, bundle.orders, bundle.products),
        aggregate_payments_daily(bundle.payments, bundle.orders),
        aggregate_shipments_daily(bundle.shipments),
        aggregate_returns_daily(bundle.returns, bundle.order_items, bundle.orders),
        aggregate_reviews_daily(bundle.reviews),
        aggregate_customers_daily(bundle.customers),
        aggregate_geography_orders_daily(bundle.orders, bundle.geography),
        aggregate_promotions_daily(bundle.promotions, all_dates_for_promos),
        aggregate_inventory_monthly_to_daily(bundle.inventory, start, end),
        aggregate_web_traffic_daily(bundle.web_traffic),
    ]

    mart = grid.merge(sales, on="Date", how="left")
    for f in feats:
        mart = mart.merge(f, on="Date", how="left")

    mart = add_derived_business_features(mart)

    # Fill missing feature values; keep targets untouched.
    feature_cols = [c for c in mart.columns if c not in ["Date", "Revenue", "COGS"]]
    mart[feature_cols] = mart[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calendar & Fourier
    mart = add_calendar_features(mart, "Date", start_date=start)
    mart = add_fourier(mart, "dayofyear", max_k=5)

    return mart.sort_values("Date").reset_index(drop=True)



def apply_time_safety_shifts(mart: pd.DataFrame) -> pd.DataFrame:
    out = mart.copy().sort_values("Date")
    # Keep deterministic date-based features aligned with the same timestamp.
    # Only shift exogenous transactional/operational features to avoid same-day leakage.
    shift_cols = [
        c
        for c in out.columns
        if c not in ["Date", "Revenue", "COGS"]
        and c not in CALENDAR_BASE_COLS
        and not c.startswith("sin_")
        and not c.startswith("cos_")
        and not c.startswith("lag_")
        and not c.startswith("roll_")
    ]
    out[shift_cols] = out[shift_cols].shift(1)
    out[shift_cols] = out[shift_cols].fillna(0)
    return out


# -----------------------------
# Driver forecasting
# -----------------------------


def build_calendar_matrix(dates: pd.Series, start_date: pd.Timestamp | None = None) -> np.ndarray:
    d = pd.to_datetime(dates)
    out = pd.DataFrame({"Date": d})
    out = add_calendar_features(out, "Date", start_date=start_date)
    out = add_fourier(out, "dayofyear", max_k=3)

    # One-hot minimal sets
    X = [np.ones(len(out))]
    for k in range(1, 7):
        X.append((out["dow"].values == k).astype(float))
    for m in range(2, 13):
        X.append((out["month"].values == m).astype(float))
    for col in ["day_of_month", "day_of_year", "day_of_week", "days_since_start", "days_to_tet"]:
        X.append(out[col].values.astype(float))
    for col in [c for c in out.columns if c.startswith("sin_") or c.startswith("cos_")]:
        X.append(out[col].values.astype(float))
    return np.column_stack(X)



def seasonal_naive_recursive(history: np.ndarray, history_dates: pd.Series, future_dates: pd.Series) -> np.ndarray:
    hist = list(history.astype(float))
    date_hist = list(pd.to_datetime(history_dates))

    md_map = pd.DataFrame({"Date": pd.to_datetime(history_dates), "y": history}).assign(
        month=lambda x: x["Date"].dt.month,
        day=lambda x: x["Date"].dt.day,
    ).groupby(["month", "day"], as_index=False)["y"].mean()
    md_lookup = {(int(r.month), int(r.day)): float(r.y) for _, r in md_map.iterrows()}

    preds: list[float] = []
    for d in pd.to_datetime(future_dates):
        cands = []
        for lag, w in [(7, 0.35), (28, 0.25), (365, 0.4)]:
            if len(hist) >= lag:
                cands.append((hist[-lag], w))
        if cands:
            val = sum(v * w for v, w in cands) / sum(w for _, w in cands)
        else:
            val = md_lookup.get((int(d.month), int(d.day)), float(np.mean(hist) if hist else 0.0))
        preds.append(max(0.0, float(val)))
        hist.append(float(preds[-1]))
        date_hist.append(d)
    return np.array(preds)



def ridge_closed_form(X: np.ndarray, y: np.ndarray, l2: float = 5.0) -> np.ndarray:
    return np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ y)



def ridge_recursive_with_lags(history: np.ndarray, history_dates: pd.Series, future_dates: pd.Series) -> np.ndarray:
    lags = [1, 7, 14, 28, 56, 364]
    max_lag = max(lags)
    if len(history) <= max_lag + 5:
        return seasonal_naive_recursive(history, history_dates, future_dates)

    d_hist = pd.to_datetime(history_dates).reset_index(drop=True)
    y = np.asarray(history, dtype=float)

    cal_X = build_calendar_matrix(d_hist, start_date=d_hist.iloc[0])
    rows = []
    target = []
    for i in range(max_lag, len(y)):
        row = list(cal_X[i])
        for lg in lags:
            row.append(y[i - lg])
        row.append(np.mean(y[i - 7 : i]))
        row.append(np.mean(y[i - 28 : i]))
        rows.append(row)
        target.append(y[i])

    X_train = np.asarray(rows, dtype=float)
    y_train = np.asarray(target, dtype=float)
    beta = ridge_closed_form(X_train, y_train, l2=10.0)

    hist = list(y)
    future_idx = pd.to_datetime(future_dates).reset_index(drop=True)
    future_cal = build_calendar_matrix(future_idx, start_date=d_hist.iloc[0])
    preds: list[float] = []

    for i, d in enumerate(future_idx):
        row_cal = future_cal[i]
        row = list(row_cal)
        for lg in lags:
            row.append(hist[-lg])
        row.append(float(np.mean(hist[-7:])))
        row.append(float(np.mean(hist[-28:])))
        pred = float(np.dot(np.asarray(row, dtype=float), beta))
        pred = max(0.0, pred)
        preds.append(pred)
        hist.append(pred)

    return np.asarray(preds)



def walk_forward_splits(n_rows: int, horizon: int, n_folds: int = 2) -> list[tuple[int, int, int]]:
    splits = []
    for fold in range(n_folds):
        valid_start = n_rows - horizon * (n_folds - fold)
        valid_end = valid_start + horizon
        if valid_start <= 0:
            continue
        splits.append((0, valid_start, valid_end))
    return splits



def seasonal_cv_target(
    train_df: pd.DataFrame,
    target_col: str,
    horizon: int,
    n_folds: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    train_df = train_df.sort_values("Date").reset_index(drop=True)
    splits = walk_forward_splits(len(train_df), horizon, n_folds=n_folds)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for _, tr_end, va_end in splits:
        tr = train_df.iloc[:tr_end].copy()
        va = train_df.iloc[tr_end:va_end].copy()
        p = seasonal_naive_recursive(
            history=tr[target_col].astype(float).values,
            history_dates=tr["Date"],
            future_dates=va["Date"],
        )
        all_true.append(va[target_col].astype(float).values)
        all_pred.append(np.asarray(p, dtype=float))

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    return metric_pack(y_true, y_pred), y_true, y_pred


def best_linear_blend(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> tuple[float, dict[str, float], np.ndarray]:
    best_w = 1.0
    best_metrics = metric_pack(y_true, pred_a)
    best_pred = np.asarray(pred_a, dtype=float)
    for w in np.linspace(0.0, 1.0, 21):
        pred = np.clip(w * pred_a + (1.0 - w) * pred_b, 0.0, None)
        m = metric_pack(y_true, pred)
        better = (m["rmse"] < best_metrics["rmse"]) or (
            np.isclose(m["rmse"], best_metrics["rmse"]) and m["mae"] < best_metrics["mae"]
        )
        if better:
            best_w = float(w)
            best_metrics = m
            best_pred = pred
    return best_w, best_metrics, best_pred


def evaluate_driver_models(y: np.ndarray, dates: pd.Series, horizon: int, n_folds: int = 2) -> tuple[str, dict[str, float], dict[str, float]]:
    splits = walk_forward_splits(len(y), horizon, n_folds=n_folds)
    if not splits:
        return "seasonal", {"mae": 1e18, "rmse": 1e18, "r2": -1e18}, {"mae": 1e18, "rmse": 1e18, "r2": -1e18}

    m1_all = []
    m2_all = []
    for _, tr_end, va_end in splits:
        y_tr = y[:tr_end]
        d_tr = dates.iloc[:tr_end]
        y_va = y[tr_end:va_end]
        d_va = dates.iloc[tr_end:va_end]

        p1 = seasonal_naive_recursive(y_tr, d_tr, d_va)
        p2 = ridge_recursive_with_lags(y_tr, d_tr, d_va)

        m1_all.append(metric_pack(y_va, p1))
        m2_all.append(metric_pack(y_va, p2))

    m1 = {k: float(np.mean([m[k] for m in m1_all])) for k in ["mae", "rmse", "r2"]}
    m2 = {k: float(np.mean([m[k] for m in m2_all])) for k in ["mae", "rmse", "r2"]}

    ra, rb = score_rank(m1, m2)
    best = "seasonal" if ra >= rb else "ridge"
    return best, m1, m2



def forecast_driver_series(daily_df: pd.DataFrame, driver_name: str, future_dates: pd.Series, horizon: int) -> tuple[pd.Series, dict[str, object]]:
    t0 = time.perf_counter()
    hist = daily_df[["Date", driver_name]].copy().sort_values("Date")
    y = hist[driver_name].astype(float).values
    d = hist["Date"]

    best, m1, m2 = evaluate_driver_models(y, d, horizon=horizon, n_folds=2)

    if best == "seasonal":
        pred = seasonal_naive_recursive(y, d, future_dates)
    else:
        pred = ridge_recursive_with_lags(y, d, future_dates)

    info = {
        "driver": driver_name,
        "selected_model": best,
        "seasonal_cv": m1,
        "ridge_cv": m2,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    return pd.Series(pred, index=pd.to_datetime(future_dates), name=driver_name), info


def forecast_exog_for_fold(
    train_fold_df: pd.DataFrame,
    valid_dates: pd.Series,
    driver_cols: list[str],
    horizon: int,
    tr_end: int,
    va_end: int,
    cache: dict[tuple[int, int, int, str], np.ndarray] | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame({"Date": pd.to_datetime(valid_dates).reset_index(drop=True)})
    for c in driver_cols:
        key = (tr_end, va_end, horizon, c)
        if cache is not None and key in cache:
            pred = cache[key]
        else:
            pred_series, _ = forecast_driver_series(
                daily_df=train_fold_df[["Date", c]],
                driver_name=c,
                future_dates=out["Date"],
                horizon=horizon,
            )
            pred = pred_series.values.astype(float)
            if cache is not None:
                cache[key] = pred
        out[c] = pred
    return out


# -----------------------------
# Target modeling
# -----------------------------


class RidgeFallbackModel:
    def __init__(self, l2: float = 5.0):
        self.l2 = l2
        self.beta: np.ndarray | None = None

    def fit(self, X, y) -> "RidgeFallbackModel":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self.beta = ridge_closed_form(X_arr, y_arr, self.l2)
        return self

    def predict(self, X) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("Model not fitted")
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.beta


class LightGBMNativeModel:
    def __init__(self, params: dict[str, object], n_estimators: int):
        self.params = params
        self.n_estimators = int(n_estimators)
        self.booster = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X, y) -> "LightGBMNativeModel":
        import lightgbm as lgb

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        train_set = lgb.Dataset(X_arr, label=y_arr, free_raw_data=False)
        self.booster = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=self.n_estimators,
        )
        self.feature_importances_ = np.asarray(
            self.booster.feature_importance(importance_type="gain"),
            dtype=float,
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model not fitted")
        X_arr = np.asarray(X, dtype=float)
        return np.asarray(self.booster.predict(X_arr), dtype=float)


_LGBM_FAILURE_LOGGED = False



def build_target_training_matrix(
    df: pd.DataFrame,
    target_col: str,
    driver_cols: list[str],
    lag_cols: list[int],
    roll_windows: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy().sort_values("Date")
    # Defensive: ensure deterministic date features exist even if caller passes a reduced column set.
    if not all(c in out.columns for c in CALENDAR_BASE_COLS):
        out = add_calendar_features(out, "Date")
    if not any(c.startswith("sin_") for c in out.columns):
        out = add_fourier(out, "dayofyear", max_k=5)

    for lg in lag_cols:
        out[f"lag_{target_col}_{lg}"] = out[target_col].shift(lg)
    for w in roll_windows:
        out[f"roll_{target_col}_{w}"] = out[target_col].shift(1).rolling(w).mean()
        out[f"roll_std_{target_col}_{w}"] = out[target_col].shift(1).rolling(w).std()

    for span in [7, 30, 90]:
        out[f"ewm_{target_col}_{span}"] = out[target_col].shift(1).ewm(span=span, adjust=False).mean()

    if f"lag_{target_col}_1" in out.columns and f"lag_{target_col}_7" in out.columns:
        out[f"mom_{target_col}_1_7"] = out[f"lag_{target_col}_1"] - out[f"lag_{target_col}_7"]
    if f"lag_{target_col}_7" in out.columns and f"lag_{target_col}_28" in out.columns:
        out[f"mom_{target_col}_7_28"] = out[f"lag_{target_col}_7"] - out[f"lag_{target_col}_28"]
        out[f"ratio_{target_col}_7_28"] = safe_div(out[f"lag_{target_col}_7"], out[f"lag_{target_col}_28"])
    if f"lag_{target_col}_28" in out.columns and f"lag_{target_col}_364" in out.columns:
        out[f"ratio_{target_col}_28_364"] = safe_div(out[f"lag_{target_col}_28"], out[f"lag_{target_col}_364"])

    use_cols = [
        *CALENDAR_BASE_COLS,
    ]
    use_cols += [c for c in out.columns if c.startswith("sin_") or c.startswith("cos_")]
    use_cols += driver_cols
    use_cols += [f"lag_{target_col}_{lg}" for lg in lag_cols]
    use_cols += [f"roll_{target_col}_{w}" for w in roll_windows]
    use_cols += [f"roll_std_{target_col}_{w}" for w in roll_windows]
    use_cols += [f"ewm_{target_col}_{span}" for span in [7, 30, 90]]
    use_cols += [c for c in out.columns if c.startswith(f"mom_{target_col}_") or c.startswith(f"ratio_{target_col}_")]

    model_df = out[use_cols + [target_col]].dropna().copy()
    X = model_df[use_cols].astype(float)
    y = model_df[target_col].astype(float)
    return X, y



def try_build_lgbm(params: dict[str, object], seed: int = 42):
    global _LGBM_FAILURE_LOGGED
    try:
        import lightgbm as _  # noqa: F401

        p = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": float(params.get("learning_rate", 0.03)),
            "num_leaves": int(params.get("num_leaves", 63)),
            "max_depth": int(params.get("max_depth", -1)),
            "min_data_in_leaf": int(params.get("min_child_samples", 20)),
            "feature_fraction": float(params.get("colsample_bytree", 0.9)),
            "bagging_fraction": float(params.get("subsample", 0.9)),
            "bagging_freq": 1,
            "lambda_l2": float(params.get("reg_lambda", 5.0)),
            "seed": seed,
            "verbosity": -1,
            "force_col_wise": True,
        }
        n_estimators = int(params.get("n_estimators", 700))
        return LightGBMNativeModel(params=p, n_estimators=n_estimators), "lightgbm_native"
    except Exception as ex:
        if not _LGBM_FAILURE_LOGGED:
            log(f"[Model] LightGBM unavailable, fallback to ridge: {type(ex).__name__}: {ex}")
            _LGBM_FAILURE_LOGGED = True
        return RidgeFallbackModel(l2=8.0), "ridge_fallback"



def random_param_samples(n_trials: int, rng: np.random.Generator) -> list[dict[str, object]]:
    params = []
    for _ in range(n_trials):
        params.append(
            {
                "n_estimators": int(rng.integers(400, 1200)),
                "learning_rate": float(rng.uniform(0.01, 0.08)),
                "num_leaves": int(rng.integers(31, 127)),
                "max_depth": int(rng.choice([-1, 6, 8, 10, 12])),
                "min_child_samples": int(rng.integers(10, 60)),
                "subsample": float(rng.uniform(0.7, 1.0)),
                "colsample_bytree": float(rng.uniform(0.7, 1.0)),
                "reg_lambda": float(rng.uniform(0.5, 20.0)),
            }
        )
    return params



def ewm_last(values: list[float], span: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    out = float(values[0])
    for v in values[1:]:
        out = alpha * float(v) + (1.0 - alpha) * out
    return float(out)


def recursive_target_predict(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target_col: str,
    model,
    driver_cols: list[str],
    lag_cols: list[int],
    roll_windows: list[int],
    feature_cols: list[str],
) -> np.ndarray:
    hist = history_df.sort_values("Date")
    fut = future_df.sort_values("Date")

    y_hist = list(hist[target_col].astype(float).values)
    hist_start_date = pd.to_datetime(hist["Date"].iloc[0]).normalize()
    preds: list[float] = []

    for _, row in fut.iterrows():
        d = pd.to_datetime(row["Date"])
        d_norm = d.normalize()

        feat = {
            "dow": int(d.dayofweek),
            "day_of_week": int(d.dayofweek),
            "is_weekend": int(d.dayofweek in [5, 6]),
            "day": int(d.day),
            "day_of_month": int(d.day),
            "month": int(d.month),
            "quarter": int((d.month - 1) // 3 + 1),
            "weekofyear": int(d.isocalendar().week),
            "dayofyear": int(d.dayofyear),
            "day_of_year": int(d.dayofyear),
            "is_month_start": int(d.is_month_start),
            "is_month_end": int(d.is_month_end),
            "days_since_start": int((d_norm - hist_start_date).days),
            "days_to_tet": int(days_to_next_tet(d)),
        }
        for k in range(1, 6):
            feat[f"sin_{k}"] = float(np.sin(2.0 * np.pi * k * d.dayofyear / 365.25))
            feat[f"cos_{k}"] = float(np.cos(2.0 * np.pi * k * d.dayofyear / 365.25))

        for c in driver_cols:
            feat[c] = float(row[c])

        for lg in lag_cols:
            feat[f"lag_{target_col}_{lg}"] = float(y_hist[-lg])
        for w in roll_windows:
            feat[f"roll_{target_col}_{w}"] = float(np.mean(y_hist[-w:]))
            feat[f"roll_std_{target_col}_{w}"] = float(np.std(y_hist[-w:], ddof=0))

        for span in [7, 30, 90]:
            feat[f"ewm_{target_col}_{span}"] = ewm_last(y_hist, span=span)

        feat[f"mom_{target_col}_1_7"] = float(y_hist[-1] - y_hist[-7])
        feat[f"mom_{target_col}_7_28"] = float(y_hist[-7] - y_hist[-28])
        feat[f"ratio_{target_col}_7_28"] = float(y_hist[-7] / y_hist[-28]) if y_hist[-28] != 0 else 0.0
        feat[f"ratio_{target_col}_28_364"] = float(y_hist[-28] / y_hist[-364]) if y_hist[-364] != 0 else 0.0

        X_row = pd.DataFrame([{c: feat[c] for c in feature_cols}], columns=feature_cols)
        pred = float(model.predict(X_row)[0])
        pred = max(0.0, pred)
        preds.append(pred)
        y_hist.append(pred)

    return np.asarray(preds)



def walk_forward_cv_target(
    train_df: pd.DataFrame,
    target_col: str,
    driver_cols: list[str],
    lag_cols: list[int],
    roll_windows: list[int],
    build_model_fn: Callable[[dict[str, object]], tuple[object, str]],
    params: dict[str, object],
    horizon: int,
    n_folds: int,
    exog_forecast_cache: dict[tuple[int, int, int, str], np.ndarray] | None = None,
    trial_label: str = "",
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    train_df = train_df.sort_values("Date").reset_index(drop=True)
    splits = walk_forward_splits(len(train_df), horizon, n_folds=n_folds)

    all_true = []
    all_pred = []

    for fold_idx, (_, tr_end, va_end) in enumerate(splits, start=1):
        t_fold = time.perf_counter()
        if trial_label:
            log(f"{trial_label} fold {fold_idx}/{len(splits)}: train_end={tr_end}, valid_end={va_end}")
        tr = train_df.iloc[:tr_end].copy()
        va = train_df.iloc[tr_end:va_end].copy()

        X_tr, y_tr = build_target_training_matrix(tr, target_col, driver_cols, lag_cols, roll_windows)
        model, _ = build_model_fn(params)

        model.fit(X_tr, y_tr)

        # Build realistic exogenous inputs for validation horizon:
        # forecast drivers from the training fold instead of using realized future values.
        exog_valid = forecast_exog_for_fold(
            train_fold_df=tr[["Date"] + driver_cols],
            valid_dates=va["Date"],
            driver_cols=driver_cols,
            horizon=horizon,
            tr_end=tr_end,
            va_end=va_end,
            cache=exog_forecast_cache,
        )

        p = recursive_target_predict(
            history_df=tr,
            future_df=exog_valid,
            target_col=target_col,
            model=model,
            driver_cols=driver_cols,
            lag_cols=lag_cols,
            roll_windows=roll_windows,
            feature_cols=X_tr.columns.tolist(),
        )

        all_true.append(va[target_col].values)
        all_pred.append(p)
        if trial_label:
            log(f"{trial_label} fold {fold_idx}/{len(splits)} done in {time.perf_counter() - t_fold:.1f}s")

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    return metric_pack(y_true, y_pred), y_true, y_pred



def tune_model(
    train_df: pd.DataFrame,
    target_col: str,
    driver_cols: list[str],
    lag_cols: list[int],
    roll_windows: list[int],
    horizon: int,
    n_folds: int,
    n_trials: int,
    seed: int,
    exog_forecast_cache: dict[tuple[int, int, int, str], np.ndarray] | None = None,
) -> tuple[dict[str, object], dict[str, float], np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    builder = lambda p: try_build_lgbm(p, seed=seed)

    best_params: dict[str, object] = {}
    best_metrics = {"mae": float("inf"), "rmse": float("inf"), "r2": -float("inf")}
    best_true = np.array([])
    best_pred = np.array([])
    backend = "unknown"
    shared_cache = exog_forecast_cache if exog_forecast_cache is not None else {}

    # Ensure at least one run with defaults
    candidates = [{}] + random_param_samples(n_trials=max(0, n_trials - 1), rng=rng)

    for i, p in enumerate(candidates, start=1):
        t_trial = time.perf_counter()
        log(f"[{target_col}] lgbm trial {i}/{len(candidates)} started")
        metrics, y_true, y_pred = walk_forward_cv_target(
            train_df=train_df,
            target_col=target_col,
            driver_cols=driver_cols,
            lag_cols=lag_cols,
            roll_windows=roll_windows,
            build_model_fn=builder,
            params=p,
            horizon=horizon,
            n_folds=n_folds,
            exog_forecast_cache=shared_cache,
            trial_label=f"[{target_col}] lgbm trial {i}",
        )

        log(
            f"[{target_col}] lgbm trial {i}/{len(candidates)} done "
            f"in {time.perf_counter() - t_trial:.1f}s | "
            f"rmse={metrics['rmse']:.2f}, mae={metrics['mae']:.2f}, r2={metrics['r2']:.4f}"
        )

        # Primary objective: rmse, secondary mae, then r2.
        better = (metrics["rmse"] < best_metrics["rmse"]) or (
            np.isclose(metrics["rmse"], best_metrics["rmse"]) and metrics["mae"] < best_metrics["mae"]
        )
        if better:
            best_params = p
            best_metrics = metrics
            best_true = y_true
            best_pred = y_pred
            _, backend = builder(p)

    return best_params, best_metrics, best_true, best_pred, backend



def get_model_importance_map(model, feature_cols: list[str]) -> dict[str, float]:
    if not hasattr(model, "feature_importances_"):
        return {}
    raw = np.asarray(getattr(model, "feature_importances_"), dtype=float).reshape(-1)
    if len(raw) != len(feature_cols):
        return {}
    return {c: float(v) for c, v in zip(feature_cols, raw)}


def auto_select_driver_cols(
    train_df: pd.DataFrame,
    target_col: str,
    driver_cols: list[str],
    lag_cols: list[int],
    roll_windows: list[int],
    horizon: int,
    seed: int,
    exog_forecast_cache: dict[tuple[int, int, int, str], np.ndarray] | None = None,
) -> tuple[list[str], dict[str, object]]:
    if len(driver_cols) <= AUTO_DRIVER_MIN_KEEP:
        return list(driver_cols), {
            "enabled": True,
            "accepted": False,
            "reason": "too_few_drivers",
            "selected_count": len(driver_cols),
            "original_count": len(driver_cols),
        }

    X_full, y_full = build_target_training_matrix(train_df, target_col, driver_cols, lag_cols, roll_windows)
    probe_model, probe_backend = try_build_lgbm({}, seed=seed + 997)
    if not probe_backend.startswith("lightgbm"):
        return list(driver_cols), {
            "enabled": True,
            "accepted": False,
            "reason": "lgbm_unavailable",
            "selected_count": len(driver_cols),
            "original_count": len(driver_cols),
        }

    probe_model.fit(X_full, y_full)
    importance_map = get_model_importance_map(probe_model, X_full.columns.tolist())
    if not importance_map:
        return list(driver_cols), {
            "enabled": True,
            "accepted": False,
            "reason": "importance_unavailable",
            "selected_count": len(driver_cols),
            "original_count": len(driver_cols),
        }

    driver_importance = {c: float(importance_map.get(c, 0.0)) for c in driver_cols}
    ranked = sorted(driver_cols, key=lambda c: (driver_importance[c], c), reverse=True)

    min_keep = max(AUTO_DRIVER_MIN_KEEP, int(np.ceil(len(driver_cols) * AUTO_DRIVER_KEEP_RATIO)))
    min_keep = min(min_keep, len(driver_cols))
    positive = [c for c in ranked if driver_importance[c] > 0]
    candidate = positive if len(positive) >= min_keep else ranked[:min_keep]

    if len(candidate) >= len(driver_cols):
        return list(driver_cols), {
            "enabled": True,
            "accepted": False,
            "reason": "no_prunable_drivers",
            "selected_count": len(driver_cols),
            "original_count": len(driver_cols),
        }

    builder = lambda p: try_build_lgbm({}, seed=seed + 131)
    shared_cache = exog_forecast_cache if exog_forecast_cache is not None else {}

    log(f"[{target_col}] Driver auto-select: quick CV baseline with {len(driver_cols)} drivers")
    base_metrics, _, _ = walk_forward_cv_target(
        train_df=train_df,
        target_col=target_col,
        driver_cols=driver_cols,
        lag_cols=lag_cols,
        roll_windows=roll_windows,
        build_model_fn=builder,
        params={},
        horizon=horizon,
        n_folds=1,
        exog_forecast_cache=shared_cache,
        trial_label=f"[{target_col}] fs-baseline",
    )

    log(f"[{target_col}] Driver auto-select: quick CV candidate with {len(candidate)} drivers")
    cand_metrics, _, _ = walk_forward_cv_target(
        train_df=train_df,
        target_col=target_col,
        driver_cols=candidate,
        lag_cols=lag_cols,
        roll_windows=roll_windows,
        build_model_fn=builder,
        params={},
        horizon=horizon,
        n_folds=1,
        exog_forecast_cache=shared_cache,
        trial_label=f"[{target_col}] fs-candidate",
    )

    non_degrade = (
        cand_metrics["rmse"] <= base_metrics["rmse"] * (1.0 + AUTO_DRIVER_CV_TOL)
        and cand_metrics["mae"] <= base_metrics["mae"] * (1.0 + AUTO_DRIVER_CV_TOL)
        and cand_metrics["r2"] >= base_metrics["r2"] - AUTO_DRIVER_CV_TOL
    )

    selected = candidate if non_degrade else list(driver_cols)
    report = {
        "enabled": True,
        "accepted": bool(non_degrade),
        "reason": "cv_non_degrade" if non_degrade else "cv_degrade",
        "original_count": len(driver_cols),
        "selected_count": len(selected),
        "dropped_count": int(len(driver_cols) - len(selected)),
        "base_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
        "driver_importance_top20": [
            {"feature": c, "importance": float(driver_importance[c])}
            for c in ranked[:20]
        ],
    }
    return selected, report


# -----------------------------
# Main pipeline
# -----------------------------


def build_driver_forecasts(
    train_df: pd.DataFrame,
    future_dates: pd.Series,
    driver_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    fourier_cols = [c for c in train_df.columns if c.startswith("sin_") or c.startswith("cos_")]
    blocked = set(CALENDAR_BASE_COLS + fourier_cols)
    all_driver_cols = [
        c
        for c in train_df.columns
        if c
        not in [
            "Date",
            "Revenue",
            "COGS",
        ]
        and c not in blocked
        and not c.startswith("lag_")
        and not c.startswith("roll_")
    ]
    if driver_cols is None:
        driver_cols = all_driver_cols
    else:
        driver_cols = [c for c in driver_cols if c in all_driver_cols]
    if not driver_cols:
        raise ValueError("No driver columns available for forecasting.")

    horizon = len(future_dates)
    pred_df = pd.DataFrame({"Date": pd.to_datetime(future_dates)})
    meta: list[dict[str, object]] = []

    for i, c in enumerate(driver_cols, start=1):
        log(f"[Drivers] {i}/{len(driver_cols)} forecasting '{c}'")
        s_pred, info = forecast_driver_series(train_df[["Date", c]], c, future_dates, horizon=horizon)
        pred_df[c] = s_pred.values
        meta.append(info)
        log(
            f"[Drivers] {i}/{len(driver_cols)} done '{c}' | "
            f"model={info['selected_model']} | {info['elapsed_sec']:.1f}s"
        )

    return pred_df, meta



def fit_and_forecast_target(
    train_df: pd.DataFrame,
    future_driver_df: pd.DataFrame,
    target_col: str,
    n_trials: int,
    horizon: int,
    seed: int,
    exog_forecast_cache: dict[tuple[int, int, int, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    t_all = time.perf_counter()
    log(f"[{target_col}] Start target pipeline")
    # Driver columns are only exogenous columns forecast for future horizon.
    driver_cols = [c for c in future_driver_df.columns if c != "Date"]
    lag_cols = [1, 2, 3, 7, 14, 21, 28, 35, 42, 56, 84, 112, 168, 364, 365, 366]
    roll_windows = [7, 14, 28, 56, 84]
    shared_cache = exog_forecast_cache if exog_forecast_cache is not None else {}

    driver_cols, driver_select_report = auto_select_driver_cols(
        train_df=train_df,
        target_col=target_col,
        driver_cols=driver_cols,
        lag_cols=lag_cols,
        roll_windows=roll_windows,
        horizon=horizon,
        seed=seed,
        exog_forecast_cache=shared_cache,
    )
    future_driver_df = future_driver_df[["Date"] + driver_cols].copy()
    log(
        f"[{target_col}] Driver auto-select result: {len(driver_cols)} drivers kept "
        f"(accepted={driver_select_report.get('accepted', False)})"
    )

    # Tune LightGBM and compare against seasonal baseline.
    log(f"[{target_col}] Tuning LightGBM")
    p_lgbm, m_lgbm, y_true_lgbm, y_pred_lgbm, backend_lgbm = tune_model(
        train_df=train_df,
        target_col=target_col,
        driver_cols=driver_cols,
        lag_cols=lag_cols,
        roll_windows=roll_windows,
        horizon=horizon,
        n_folds=2,
        n_trials=n_trials,
        seed=seed,
        exog_forecast_cache=shared_cache,
    )

    # Compare against seasonal baseline and pick best strategy by full CV ranking.
    m_seasonal, y_true_seasonal, y_pred_seasonal = seasonal_cv_target(
        train_df=train_df,
        target_col=target_col,
        horizon=horizon,
        n_folds=2,
    )
    blend_weight = 1.0
    if len(y_true_lgbm) == len(y_pred_lgbm) == len(y_pred_seasonal) and len(y_true_lgbm) > 0:
        blend_weight, m_blend, _ = best_linear_blend(y_true_lgbm, y_pred_lgbm, y_pred_seasonal)
    else:
        m_blend = m_lgbm

    cv_metric_map = {
        "lgbm": m_lgbm,
        "seasonal": m_seasonal,
        "blend": m_blend,
    }
    cv_rank = mean_rank_3metrics(cv_metric_map)
    best_key = min(cv_rank, key=lambda k: cv_rank[k])
    selected_strategy = {
        "lgbm": "lgbm_only",
        "seasonal": "seasonal_only",
        "blend": "blend_lgbm_seasonal",
    }[best_key]

    # Refit full model
    log(f"[{target_col}] Refit LightGBM and forecast horizon")
    X_full, y_full = build_target_training_matrix(train_df, target_col, driver_cols, lag_cols, roll_windows)
    feature_cols = X_full.columns.tolist()

    model_lgbm, _ = try_build_lgbm(p_lgbm, seed=seed)
    model_lgbm.fit(X_full, y_full)

    # Forecast horizon using recursive path.
    future_df = future_driver_df[["Date"] + driver_cols].copy().sort_values("Date")
    pred = recursive_target_predict(
        history_df=train_df[["Date", target_col] + driver_cols],
        future_df=future_df,
        target_col=target_col,
        model=model_lgbm,
        driver_cols=driver_cols,
        lag_cols=lag_cols,
        roll_windows=roll_windows,
        feature_cols=feature_cols,
    )
    pred_lgbm = np.clip(pred, 0, None)
    pred_seasonal = seasonal_naive_recursive(
        history=train_df[target_col].astype(float).values,
        history_dates=train_df["Date"],
        future_dates=future_df["Date"],
    )
    if selected_strategy == "seasonal_only":
        pred = np.clip(pred_seasonal, 0, None)
    elif selected_strategy == "blend_lgbm_seasonal":
        pred = np.clip(blend_weight * pred_lgbm + (1.0 - blend_weight) * pred_seasonal, 0, None)
    else:
        pred = pred_lgbm

    selected_cv = {
        "lgbm_only": m_lgbm,
        "seasonal_only": m_seasonal,
        "blend_lgbm_seasonal": m_blend,
    }[selected_strategy]

    info = {
        "target": target_col,
        "model_backend": backend_lgbm,
        "model_a_backend": backend_lgbm,
        "model_b_backend": "seasonal_naive_recursive",
        "driver_cols_selected": driver_cols,
        "driver_auto_select": driver_select_report,
        "model_params": p_lgbm,
        "model_cv": selected_cv,
        "model_a_params": p_lgbm,
        "model_b_params": {"type": "seasonal_naive_recursive"},
        "model_a_cv": m_lgbm,
        "model_b_cv": m_seasonal,
        "blend_weight_model_a": blend_weight,
        "blend_cv": m_blend,
        "cv_mean_rank": {
            "model_a": cv_rank["lgbm"],
            "model_b": cv_rank["seasonal"],
            "blend": cv_rank["blend"],
        },
        "selected_strategy": selected_strategy,
    }
    log(f"[{target_col}] Completed in {time.perf_counter() - t_all:.1f}s")
    return pred, info


def fit_and_forecast_cogs_from_ratio(
    train_df: pd.DataFrame,
    future_driver_df: pd.DataFrame,
    revenue_pred: np.ndarray,
    n_trials: int,
    horizon: int,
    seed: int,
    exog_forecast_cache: dict[tuple[int, int, int, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if len(revenue_pred) != len(future_driver_df):
        raise ValueError("Revenue prediction length must match COGS horizon length.")

    work_train = train_df.copy()
    work_future = future_driver_df.copy()

    ratio_raw = work_train["COGS"].astype(float) / work_train["Revenue"].astype(float).replace(0.0, np.nan)
    ratio_clean = ratio_raw.replace([np.inf, -np.inf], np.nan)
    ratio_median = float(ratio_clean.median()) if ratio_clean.notna().any() else 0.7
    work_train["COGS_ratio"] = ratio_clean.fillna(ratio_median).clip(lower=0.0, upper=3.0)

    # Inject revenue level as a known input for ratio modeling.
    work_train["revenue_level_input"] = work_train["Revenue"].astype(float)
    work_future["revenue_level_input"] = np.asarray(revenue_pred, dtype=float)

    ratio_pred, ratio_meta = fit_and_forecast_target(
        train_df=work_train,
        future_driver_df=work_future,
        target_col="COGS_ratio",
        n_trials=n_trials,
        horizon=horizon,
        seed=seed,
        exog_forecast_cache=exog_forecast_cache,
    )

    cogs_pred = np.asarray(ratio_pred, dtype=float) * np.asarray(revenue_pred, dtype=float)
    cogs_pred = np.clip(cogs_pred, 0.0, None)

    info = {
        **ratio_meta,
        "target": "COGS",
        "cogs_mode": "ratio_times_revenue_pred",
        "ratio_target_col": "COGS_ratio",
        "ratio_fill_median": ratio_median,
        "cv_metric_space": "ratio",
    }
    return cogs_pred, info



def load_cached_prediction(cache_path: Path, expected_len: int) -> np.ndarray | None:
    try:
        arr = np.asarray(np.load(cache_path), dtype=float).reshape(-1)
    except Exception:
        return None
    if len(arr) != expected_len:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def load_cached_meta_dict(cache_path: Path) -> dict[str, object] | None:
    try:
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def make_submission(sample_submission: pd.DataFrame, pred_revenue: np.ndarray, pred_cogs: np.ndarray) -> pd.DataFrame:
    # SAFETY CLAMP: COGS cannot exceed Revenue (spec: cogs < price for every product).
    # This monotonically improves any submission where COGS > Revenue on some days.
    pred_cogs = np.minimum(np.asarray(pred_cogs, dtype=float),
                            np.asarray(pred_revenue, dtype=float) * 0.95)
    sample = sample_submission[["Date"]].copy().reset_index(drop=True)
    sample = sample_submission[["Date"]].copy().reset_index(drop=True)
    date_sorted = pd.to_datetime(sample["Date"]).sort_values().reset_index(drop=True)
    if len(date_sorted) != len(pred_revenue) or len(date_sorted) != len(pred_cogs):
        raise ValueError("Prediction length does not match submission horizon.")

    pred_by_date = pd.DataFrame(
        {
            "Date": date_sorted,
            "Revenue": np.asarray(pred_revenue, dtype=float),
            "COGS": np.asarray(pred_cogs, dtype=float),
        }
    )
    pred_by_date = pred_by_date.drop_duplicates(subset=["Date"], keep="first").set_index("Date")

    out = sample.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Revenue"] = out["Date"].map(pred_by_date["Revenue"])
    out["COGS"] = out["Date"].map(pred_by_date["COGS"])
    if out[["Revenue", "COGS"]].isna().any().any():
        raise ValueError("Submission mapping failed: some dates are missing predicted values.")
    out["Revenue"] = out["Revenue"].clip(lower=0)
    out["COGS"] = out["COGS"].clip(lower=0)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    return out



def select_curated_drivers(
    train_df: pd.DataFrame,
    available_cols: list[str],
    target_col: str = "Revenue",
    max_drivers: int = 50,
) -> list[str]:
    clean_available = [c for c in available_cols if c in train_df.columns and c != "Date"]
    if not clean_available:
        return []

    # Keep strong business drivers, then rank the rest by absolute correlation.
    selected = [c for c in CORE_DRIVER_FEATURES if c in clean_available]
    residual = [c for c in clean_available if c not in selected]

    filtered_residual: list[str] = []
    for c in residual:
        if any(c.startswith(pfx) for pfx in NOISY_DRIVER_PREFIXES):
            continue
        s = train_df[c].astype(float)
        if s.nunique(dropna=False) <= 1:
            continue
        filtered_residual.append(c)

    y = train_df[target_col].astype(float)
    corr_map: dict[str, float] = {}
    for c in filtered_residual:
        x = train_df[c].astype(float)
        corr = x.corr(y)
        corr_map[c] = abs(float(corr)) if pd.notna(corr) else 0.0

    ranked = sorted(filtered_residual, key=lambda c: (corr_map.get(c, 0.0), c), reverse=True)
    # Keep at least a broad context for tree models, but avoid very weak noise.
    min_extra_keep = 12
    corr_floor = 0.01
    for c in ranked:
        if max_drivers > 0 and len(selected) >= max_drivers:
            break
        if corr_map.get(c, 0.0) >= corr_floor or len(selected) < len(CORE_DRIVER_FEATURES) + min_extra_keep:
            selected.append(c)

    # De-duplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for c in selected:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out[:max_drivers] if max_drivers > 0 else out


def run_pipeline(
    base_dir: Path,
    out_dir: Path,
    n_trials: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{PIPELINE_CACHE_VERSION}_t{n_trials}_s{seed}"
    driver_cache_path = out_dir / f"driver_future_forecasts_{run_tag}.csv"
    driver_meta_path = out_dir / f"driver_model_selection_{run_tag}.json"
    rev_pred_cache = out_dir / f"pred_revenue_{run_tag}.npy"
    cogs_pred_cache = out_dir / f"pred_cogs_{run_tag}.npy"
    rev_meta_cache = out_dir / f"meta_revenue_{run_tag}.json"
    cogs_meta_cache = out_dir / f"meta_cogs_{run_tag}.json"

    log("[Stage] Loading data...")
    bundle = load_data(base_dir)

    # 1) Build training feature mart (all csv)
    log("[Stage] Building feature mart...")
    mart = build_daily_feature_mart(bundle)
    mart_safe = apply_time_safety_shifts(mart)

    # 2) Select and forecast exogenous drivers for test horizon
    future_dates = bundle.sample_submission["Date"].sort_values().reset_index(drop=True)
    all_exog_cols = [
        c
        for c in mart_safe.columns
        if c not in ["Date", "Revenue", "COGS"]
        and c not in CALENDAR_BASE_COLS
        and not c.startswith("sin_")
        and not c.startswith("cos_")
        and not c.startswith("lag_")
        and not c.startswith("roll_")
    ]
    exog_driver_cols = select_curated_drivers(
        train_df=mart_safe,
        available_cols=all_exog_cols,
        target_col="Revenue",
        max_drivers=50,
    )
    log(f"[Stage] Selected {len(exog_driver_cols)} driver candidates before forecasting")

    cache_valid = False
    if driver_cache_path.exists():
        log("[Stage] Loading cached driver forecasts...")
        driver_future_df = pd.read_csv(driver_cache_path, parse_dates=["Date"])
        cache_valid = (
            "Date" in driver_future_df.columns
            and len(driver_future_df) == len(future_dates)
            and pd.to_datetime(driver_future_df["Date"]).reset_index(drop=True).equals(pd.to_datetime(future_dates))
            and all(c in driver_future_df.columns for c in exog_driver_cols)
        )
        if cache_valid:
            driver_future_df = driver_future_df[["Date"] + exog_driver_cols].copy()
            if driver_meta_path.exists():
                driver_meta = json.loads(driver_meta_path.read_text(encoding="utf-8"))
            else:
                driver_meta = []
        else:
            log("[Stage] Driver cache invalid for current horizon/date index. Rebuilding...")

    if not driver_cache_path.exists() or not cache_valid:
        log("[Stage] Forecasting drivers...")
        driver_future_df, driver_meta = build_driver_forecasts(
            train_df=mart_safe.drop(columns=["Revenue", "COGS"]).copy(),
            future_dates=future_dates,
            driver_cols=exog_driver_cols,
        )
        driver_future_df.to_csv(driver_cache_path, index=False)
        pd.DataFrame(driver_meta).to_json(driver_meta_path, orient="records", indent=2)
        log(f"[Checkpoint] Saved driver cache: {driver_cache_path}")

    # 3) Attach drivers to train set
    train_model_df = mart_safe.copy()
    driver_future_df = driver_future_df[["Date"] + exog_driver_cols].copy()

    revenue_driver_cols = list(exog_driver_cols)
    cogs_driver_cols = list(exog_driver_cols)
    log(
        f"[Stage] Driver counts by target: "
        f"Revenue={len(revenue_driver_cols)}, COGS={len(cogs_driver_cols)}"
    )

    # ensure same driver columns in train
    missing_driver_cols = [c for c in exog_driver_cols if c not in train_model_df.columns]
    for c in missing_driver_cols:
        train_model_df[c] = 0.0

    # 4) Target models
    horizon = len(future_dates)
    revenue_future_driver_df = driver_future_df[["Date"] + revenue_driver_cols].copy()
    cogs_future_driver_df = driver_future_df[["Date"] + cogs_driver_cols].copy()

    base_horizon_signature = {
        "len": int(horizon),
        "start": str(pd.to_datetime(future_dates).min().date()),
        "end": str(pd.to_datetime(future_dates).max().date()),
    }
    rev_horizon_signature = {**base_horizon_signature, "driver_cols": revenue_driver_cols}
    cogs_horizon_signature = {
        **base_horizon_signature,
        "driver_cols": cogs_driver_cols,
        "cogs_mode": "ratio_times_revenue_pred_v1",
    }
    shared_exog_forecast_cache: dict[tuple[int, int, int, str], np.ndarray] = {}
    if rev_pred_cache.exists() and rev_meta_cache.exists():
        cached = load_cached_prediction(rev_pred_cache, expected_len=horizon)
        rev_meta_cached = load_cached_meta_dict(rev_meta_cache)
        rev_meta_valid = (
            rev_meta_cached is not None and rev_meta_cached.get("horizon_signature") == rev_horizon_signature
        )
        if cached is not None and rev_meta_valid:
            log("[Stage] Loading cached Revenue prediction...")
            pred_rev = cached
            rev_meta = rev_meta_cached if rev_meta_cached is not None else {}
        else:
            log("[Stage] Revenue cache invalid (shape/NaN/Inf/horizon signature). Re-training...")
            pred_rev, rev_meta = fit_and_forecast_target(
                train_df=train_model_df,
                future_driver_df=revenue_future_driver_df,
                target_col="Revenue",
                n_trials=n_trials,
                horizon=horizon,
                seed=seed,
                exog_forecast_cache=shared_exog_forecast_cache,
            )
            rev_meta["horizon_signature"] = rev_horizon_signature
            np.save(rev_pred_cache, pred_rev)
            rev_meta_cache.write_text(json.dumps(rev_meta, indent=2), encoding="utf-8")
            log(f"[Checkpoint] Saved Revenue cache: {rev_pred_cache}")
    else:
        log(f"[Stage] Training Revenue models (trials={n_trials})...")
        pred_rev, rev_meta = fit_and_forecast_target(
            train_df=train_model_df,
            future_driver_df=revenue_future_driver_df,
            target_col="Revenue",
            n_trials=n_trials,
            horizon=horizon,
            seed=seed,
            exog_forecast_cache=shared_exog_forecast_cache,
        )
        rev_meta["horizon_signature"] = rev_horizon_signature
        np.save(rev_pred_cache, pred_rev)
        rev_meta_cache.write_text(json.dumps(rev_meta, indent=2), encoding="utf-8")
        log(f"[Checkpoint] Saved Revenue cache: {rev_pred_cache}")

    if cogs_pred_cache.exists() and cogs_meta_cache.exists():
        cached = load_cached_prediction(cogs_pred_cache, expected_len=horizon)
        cogs_meta_cached = load_cached_meta_dict(cogs_meta_cache)
        cogs_meta_valid = (
            cogs_meta_cached is not None and cogs_meta_cached.get("horizon_signature") == cogs_horizon_signature
        )
        if cached is not None and cogs_meta_valid:
            log("[Stage] Loading cached COGS prediction...")
            pred_cogs = cached
            cogs_meta = cogs_meta_cached if cogs_meta_cached is not None else {}
        else:
            log("[Stage] COGS cache invalid (shape/NaN/Inf/horizon signature). Re-training...")
            pred_cogs, cogs_meta = fit_and_forecast_cogs_from_ratio(
                train_df=train_model_df,
                future_driver_df=cogs_future_driver_df,
                revenue_pred=pred_rev,
                n_trials=n_trials,
                horizon=horizon,
                seed=seed + 100,
                exog_forecast_cache=shared_exog_forecast_cache,
            )
            cogs_meta["horizon_signature"] = cogs_horizon_signature
            np.save(cogs_pred_cache, pred_cogs)
            cogs_meta_cache.write_text(json.dumps(cogs_meta, indent=2), encoding="utf-8")
            log(f"[Checkpoint] Saved COGS cache: {cogs_pred_cache}")
    else:
        log(f"[Stage] Training COGS models (trials={n_trials})...")
        pred_cogs, cogs_meta = fit_and_forecast_cogs_from_ratio(
            train_df=train_model_df,
            future_driver_df=cogs_future_driver_df,
            revenue_pred=pred_rev,
            n_trials=n_trials,
            horizon=horizon,
            seed=seed + 100,
            exog_forecast_cache=shared_exog_forecast_cache,
        )
        cogs_meta["horizon_signature"] = cogs_horizon_signature
        np.save(cogs_pred_cache, pred_cogs)
        cogs_meta_cache.write_text(json.dumps(cogs_meta, indent=2), encoding="utf-8")
        log(f"[Checkpoint] Saved COGS cache: {cogs_pred_cache}")

    # 5) Submission
    submission = make_submission(bundle.sample_submission, pred_rev, pred_cogs)

    submission_path = out_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    # Artifacts
    pd.DataFrame(driver_meta).to_json(driver_meta_path, orient="records", indent=2)
    with open(out_dir / "target_model_report.json", "w", encoding="utf-8") as f:
        json.dump({"Revenue": rev_meta, "COGS": cogs_meta}, f, indent=2)

    # Sanity checks
    sample = bundle.sample_submission.reset_index(drop=True)
    ok_rows = len(sample) == len(submission)
    ok_dates = pd.to_datetime(sample["Date"]).dt.strftime("%Y-%m-%d").equals(submission["Date"])
    nulls = int(submission[["Revenue", "COGS"]].isna().sum().sum())
    infs = int(np.isinf(submission[["Revenue", "COGS"]].to_numpy(dtype=float)).sum())

    checks = {
        "rows_match_sample": ok_rows,
        "date_order_match_sample": ok_dates,
        "null_value_count": nulls,
        "inf_value_count": infs,
        "min_revenue": float(submission["Revenue"].min()),
        "min_cogs": float(submission["COGS"].min()),
    }
    with open(out_dir / "submission_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    log(f"Saved: {submission_path}")
    print(json.dumps(checks, indent=2))



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part 3 forecasting pipeline with all CSV sources.")
    p.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "part3")
    p.add_argument("--n-trials", type=int, default=20, help="Random-search trials per model/target.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        args.base_dir,
        args.out_dir,
        n_trials=args.n_trials,
        seed=args.seed,
    )
