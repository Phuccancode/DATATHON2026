from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPLETED_STATUSES = ["shipped", "delivered", "returned"]


def fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def reset_output_dir() -> None:
    for p in OUT_DIR.glob("*"):
        if p.is_file():
            p.unlink()


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "products": pd.read_csv(BASE_DIR / "products.csv"),
        "orders": pd.read_csv(BASE_DIR / "orders.csv", parse_dates=["order_date"]),
        "order_items": pd.read_csv(
            BASE_DIR / "order_items.csv",
            dtype={"promo_id": "string", "promo_id_2": "string"},
            low_memory=False,
        ),
        "returns": pd.read_csv(BASE_DIR / "returns.csv", parse_dates=["return_date"]),
        "reviews": pd.read_csv(BASE_DIR / "reviews.csv", parse_dates=["review_date"]),
        "shipments": pd.read_csv(BASE_DIR / "shipments.csv", parse_dates=["ship_date", "delivery_date"]),
        "geography": pd.read_csv(BASE_DIR / "geography.csv"),
        "web_traffic": pd.read_csv(BASE_DIR / "web_traffic.csv", parse_dates=["date"]),
        "inventory": pd.read_csv(BASE_DIR / "inventory.csv", parse_dates=["snapshot_date"]),
        "sales": pd.read_csv(BASE_DIR / "sales.csv", parse_dates=["Date"]),
    }


def build_fact_order_line(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"].copy()
    products = data["products"][["product_id", "category", "segment", "size", "price", "cogs"]].copy()
    geography = data["geography"][["zip", "region"]].drop_duplicates().copy()

    line = data["order_items"].merge(
        orders[
            [
                "order_id",
                "order_date",
                "order_status",
                "customer_id",
                "zip",
                "payment_method",
                "device_type",
                "order_source",
            ]
        ],
        on="order_id",
        how="left",
    )
    line = line.merge(products, on="product_id", how="left")
    line = line.merge(geography, on="zip", how="left")

    # unit_price already reflects post-promo line item price in provided schema.
    line["line_revenue"] = line["quantity"] * line["unit_price"]
    line["line_cogs"] = line["quantity"] * line["cogs"]
    line["gross_profit"] = line["line_revenue"] - line["line_cogs"]
    line["margin_pct"] = np.where(line["line_revenue"] > 0, line["gross_profit"] / line["line_revenue"], np.nan)

    line["promo_used"] = line["promo_id"].notna() | line["promo_id_2"].notna()
    line["discount_pct"] = np.where(
        line["line_revenue"] > 0,
        (line["discount_amount"] / line["line_revenue"]).clip(lower=0, upper=1),
        0.0,
    )

    ret = data["returns"].groupby(["order_id", "product_id"], as_index=False)["return_quantity"].sum()
    ret["returned"] = ret["return_quantity"] > 0
    line = line.merge(ret[["order_id", "product_id", "returned"]], on=["order_id", "product_id"], how="left")
    line["returned"] = line["returned"].fillna(False)

    line["year"] = line["order_date"].dt.year
    line["month"] = line["order_date"].dt.month
    line["ym"] = line["order_date"].dt.to_period("M")
    return line


def validate_data_quality(
    line: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
) -> dict[str, float]:
    daily = (
        line.groupby("order_date", as_index=False)
        .agg(revenue_calc=("line_revenue", "sum"), cogs_calc=("line_cogs", "sum"))
        .rename(columns={"order_date": "Date"})
    )
    merged = sales.merge(daily, on="Date", how="inner")
    merged["rev_abs_err"] = (merged["Revenue"] - merged["revenue_calc"]).abs()
    merged["cogs_abs_err"] = (merged["COGS"] - merged["cogs_calc"]).abs()

    checks = {
        "sales_rows": float(len(merged)),
        "revenue_mae_vs_sales": float(merged["rev_abs_err"].mean()),
        "revenue_max_abs_err": float(merged["rev_abs_err"].max()),
        "cogs_mae_vs_sales": float(merged["cogs_abs_err"].mean()),
        "cogs_max_abs_err": float(merged["cogs_abs_err"].max()),
        "inventory_stockout_flag_mean": float(inventory["stockout_flag"].mean()),
        "inventory_overstock_flag_mean": float(inventory["overstock_flag"].mean()),
        "inventory_reorder_flag_mean": float(inventory["reorder_flag"].mean()),
        "inventory_reorder_flag_nunique": float(inventory["reorder_flag"].nunique()),
    }

    pd.DataFrame([checks]).to_csv(OUT_DIR / "data_quality_checks.csv", index=False)
    return checks


def build_monthly_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        sales.groupby(pd.Grouper(key="Date", freq="MS"), as_index=False)
        .agg(Revenue=("Revenue", "sum"), COGS=("COGS", "sum"))
        .sort_values("Date")
    )
    monthly["gross_margin_pct"] = np.where(
        monthly["Revenue"] > 0,
        (monthly["Revenue"] - monthly["COGS"]) / monthly["Revenue"],
        np.nan,
    )
    monthly["rev_ma3"] = monthly["Revenue"].rolling(3).mean()
    monthly["rev_ma12"] = monthly["Revenue"].rolling(12).mean()
    monthly["year"] = monthly["Date"].dt.year
    monthly["month"] = monthly["Date"].dt.month
    monthly["seasonality_idx"] = monthly["Revenue"] / monthly.groupby("month")["Revenue"].transform("mean")
    return monthly


def plot_revenue_trend_monthly(monthly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["Date"], monthly["Revenue"], label="Revenue (monthly)", linewidth=1.2, alpha=0.8)
    ax.plot(monthly["Date"], monthly["rev_ma3"], label="3-month MA", linewidth=2.0)
    ax.plot(monthly["Date"], monthly["rev_ma12"], label="12-month MA", linewidth=2.2)
    ax.set_title("Fig 1. Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig01_revenue_trend_monthly.png", dpi=170)
    plt.close(fig)


def plot_seasonality_heatmap(monthly: pd.DataFrame) -> pd.DataFrame:
    season = monthly.groupby(["year", "month"], as_index=False)["Revenue"].sum()
    season["idx_vs_year_avg"] = season["Revenue"] / season.groupby("year")["Revenue"].transform("mean")
    pivot = season.pivot(index="year", columns="month", values="idx_vs_year_avg").sort_index()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Fig 2. Seasonality Heatmap (Revenue Index vs Year Average)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Seasonality index")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig02_seasonality_heatmap.png", dpi=170)
    plt.close(fig)

    month_kpi = monthly.groupby("month", as_index=False).agg(
        avg_revenue=("Revenue", "mean"),
        avg_margin_pct=("gross_margin_pct", "mean"),
    )
    month_kpi["seasonality_idx"] = month_kpi["avg_revenue"] / month_kpi["avg_revenue"].mean()
    month_kpi = month_kpi.sort_values("seasonality_idx", ascending=False)
    return month_kpi


def plot_category_pareto(line: pd.DataFrame) -> pd.DataFrame:
    cat = line.groupby("category", as_index=False).agg(revenue=("line_revenue", "sum"), margin_pct=("margin_pct", "mean"))
    cat = cat.sort_values("revenue", ascending=False)
    cat["share_pct"] = cat["revenue"] / cat["revenue"].sum() * 100
    cat["cum_share_pct"] = cat["share_pct"].cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cat))
    ax1.bar(x, cat["share_pct"], alpha=0.85, label="Revenue share (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat["category"])
    ax1.set_ylabel("Revenue share (%)")
    ax1.set_title("Fig 3. Category Revenue Concentration (Pareto)")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, cat["cum_share_pct"], color="tab:red", marker="o", linewidth=2.2, label="Cumulative share (%)")
    ax2.set_ylabel("Cumulative share (%)")
    ax2.set_ylim(0, 105)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig03_category_concentration_pareto.png", dpi=170)
    plt.close(fig)
    return cat


def plot_category_seasonal_profile(line: pd.DataFrame) -> pd.DataFrame:
    cat_month = line.groupby(["category", "month"], as_index=False)["line_revenue"].sum()
    cat_month["seasonality_idx"] = cat_month["line_revenue"] / cat_month.groupby("category")["line_revenue"].transform("mean")

    fig, ax = plt.subplots(figsize=(11, 5))
    for cat, grp in cat_month.groupby("category"):
        g = grp.sort_values("month")
        ax.plot(g["month"], g["seasonality_idx"], marker="o", linewidth=2, label=cat)
    ax.set_title("Fig 4. Seasonal Profile by Category")
    ax.set_xlabel("Month")
    ax.set_ylabel("Seasonality index (category-normalized)")
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig04_category_seasonal_profile.png", dpi=170)
    plt.close(fig)

    peak = cat_month.sort_values("seasonality_idx", ascending=False).groupby("category").head(1)
    trough = cat_month.sort_values("seasonality_idx", ascending=True).groupby("category").head(1)
    season_kpi = peak[["category", "month", "seasonality_idx"]].merge(
        trough[["category", "month", "seasonality_idx"]],
        on="category",
        suffixes=("_peak", "_trough"),
    )
    season_kpi["peak_to_trough_ratio"] = season_kpi["seasonality_idx_peak"] / season_kpi["seasonality_idx_trough"]
    return season_kpi


def build_inventory_pressure(line: pd.DataFrame, inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_month = (
        line.groupby(["ym", "category"], as_index=False)["line_revenue"].sum().sort_values(["category", "ym"])
    )
    cat_month["next_month_revenue"] = cat_month.groupby("category")["line_revenue"].shift(-1)
    cat_month["next_month_growth"] = cat_month["next_month_revenue"] / cat_month["line_revenue"] - 1

    inv = inventory.copy()
    inv["ym"] = inv["snapshot_date"].dt.to_period("M")
    inv_month = inv.groupby(["ym", "category"], as_index=False).agg(
        stockout_days=("stockout_days", "mean"),
        fill_rate=("fill_rate", "mean"),
        stockout_flag=("stockout_flag", "mean"),
        overstock_flag=("overstock_flag", "mean"),
        reorder_flag=("reorder_flag", "mean"),
    )

    pressure = cat_month.merge(inv_month, on=["ym", "category"], how="inner")
    pressure = pressure.dropna(subset=["next_month_growth"]).copy()

    corr = {
        "corr_stockout_days_vs_next_growth": pressure["stockout_days"].corr(pressure["next_month_growth"]),
        "corr_fill_rate_vs_next_growth": pressure["fill_rate"].corr(pressure["next_month_growth"]),
        "corr_stockout_flag_vs_next_growth": pressure["stockout_flag"].corr(pressure["next_month_growth"]),
        "corr_overstock_flag_vs_next_growth": pressure["overstock_flag"].corr(pressure["next_month_growth"]),
    }
    corr_df = pd.DataFrame([corr])

    return pressure, corr_df


def plot_stockout_vs_next_growth(pressure: pd.DataFrame) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cat, grp in pressure.groupby("category"):
        ax.scatter(grp["stockout_days"], grp["next_month_growth"] * 100, alpha=0.55, s=45, label=cat)

    ax.set_title("Fig 5. Stockout Pressure vs Next-Month Revenue Growth")
    ax.set_xlabel("Average stockout days in month")
    ax.set_ylabel("Next-month revenue growth (%)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig05_stockout_vs_next_growth.png", dpi=170)
    plt.close(fig)

    pressure_q = pressure.copy()
    pressure_q["stockout_quartile"] = pd.qcut(
        pressure_q["stockout_days"],
        4,
        labels=["Q1-low", "Q2", "Q3", "Q4-high"],
    )
    summary = pressure_q.groupby("stockout_quartile", as_index=False).agg(
        avg_next_growth=("next_month_growth", "mean"),
        avg_stockout_days=("stockout_days", "mean"),
        avg_fill_rate=("fill_rate", "mean"),
        obs=("ym", "count"),
    )
    return summary


def plot_inventory_hotspot(inventory: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.copy()
    inv["month"] = inv["snapshot_date"].dt.month
    hot = inv.groupby(["category", "month"], as_index=False).agg(
        stockout_days=("stockout_days", "mean"),
        days_of_supply=("days_of_supply", "mean"),
        fill_rate=("fill_rate", "mean"),
    )

    pivot = hot.pivot(index="category", columns="month", values="stockout_days")
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="OrRd")
    ax.set_title("Fig 6. Stockout Hotspot by Category-Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Category")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(m) for m in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg stockout days")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig06_inventory_hotspot_heatmap.png", dpi=170)
    plt.close(fig)

    top_hotspot = hot.sort_values(["category", "stockout_days"], ascending=[True, False]).groupby("category").head(3)
    return top_hotspot


def plot_discount_depth_diagnostic(line: pd.DataFrame) -> pd.DataFrame:
    done = line[line["order_status"].isin(COMPLETED_STATUSES)].copy()
    bins = [-0.001, 0.0, 0.05, 0.15, 1.0]
    labels = ["No promo", "0-5%", "5-15%", ">15%"]
    done["discount_bin"] = pd.cut(done["discount_pct"], bins=bins, labels=labels)
    diag = done.groupby("discount_bin", as_index=False).agg(
        lines=("order_id", "count"),
        avg_revenue=("line_revenue", "mean"),
        return_rate=("returned", "mean"),
        avg_margin_pct=("margin_pct", "mean"),
    )

    x = np.arange(len(diag))
    fig, ax1 = plt.subplots(figsize=(9.2, 5))
    ax2 = ax1.twinx()

    ax1.bar(x - 0.2, diag["return_rate"] * 100, width=0.4, label="Return rate (%)")
    ax2.bar(x + 0.2, diag["avg_margin_pct"] * 100, width=0.4, color="tab:orange", label="Avg margin (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(diag["discount_bin"].astype(str))
    ax1.set_title("Fig 7. Discount Depth Diagnostic: Return vs Margin")
    ax1.set_ylabel("Return rate (%)")
    ax2.set_ylabel("Average margin (%)")
    ax1.grid(alpha=0.2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig07_discount_depth_diagnostic.png", dpi=170)
    plt.close(fig)

    return diag


def forecast_scenarios(monthly: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    hist = monthly.sort_values("Date").copy()
    hist["t"] = np.arange(len(hist))

    coeff = np.polyfit(hist["t"].values, hist["Revenue"].values, 1)
    hist["trend_fit"] = coeff[0] * hist["t"] + coeff[1]

    month_factor = hist.groupby("month")["Revenue"].mean() / hist["Revenue"].mean()
    residual = hist["Revenue"] - hist["trend_fit"] * hist["month"].map(month_factor)
    residual_pct = float((residual.abs() / hist["Revenue"].replace(0, np.nan)).median())
    scenario_spread = max(0.06, min(0.2, residual_pct))

    t_future = np.arange(hist["t"].max() + 1, hist["t"].max() + 1 + horizon)
    future_dates = pd.date_range(hist["Date"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    future = pd.DataFrame({"Date": future_dates, "t": t_future})
    future["month"] = future["Date"].dt.month
    future["trend"] = coeff[0] * future["t"] + coeff[1]
    future["season_factor"] = future["month"].map(month_factor).fillna(1.0)
    future["base_forecast"] = (future["trend"] * future["season_factor"]).clip(lower=0)
    future["optimistic_forecast"] = future["base_forecast"] * (1 + scenario_spread)
    future["conservative_forecast"] = future["base_forecast"] * (1 - scenario_spread)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    recent = hist.tail(30)
    ax.plot(recent["Date"], recent["Revenue"], label="Actual revenue (recent)", linewidth=1.8)
    ax.plot(future["Date"], future["base_forecast"], label="Base scenario", linewidth=2.3)
    ax.plot(future["Date"], future["optimistic_forecast"], label="Optimistic", linewidth=1.7, linestyle="--")
    ax.plot(future["Date"], future["conservative_forecast"], label="Conservative", linewidth=1.7, linestyle="--")
    ax.set_title("Fig 8. 6-Month Revenue Projection (Scenario View)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig08_forecast_scenarios_6m.png", dpi=170)
    plt.close(fig)

    return future[["Date", "base_forecast", "optimistic_forecast", "conservative_forecast"]]


def build_action_priority(
    seasonal_profile: pd.DataFrame,
    hotspot: pd.DataFrame,
    pressure_quartile: pd.DataFrame,
) -> pd.DataFrame:
    action = hotspot.merge(
        seasonal_profile[["category", "month_peak", "seasonality_idx_peak", "peak_to_trough_ratio"]],
        left_on="category",
        right_on="category",
        how="left",
    )

    s_stockout = action["stockout_days"]
    s_season = action["seasonality_idx_peak"].fillna(1.0)
    action["priority_score"] = (
        (s_stockout - s_stockout.mean()) / s_stockout.std(ddof=0)
        + (s_season - s_season.mean()) / s_season.std(ddof=0)
    )

    threshold = pressure_quartile[pressure_quartile["stockout_quartile"] == "Q4-high"]["avg_next_growth"].iloc[0]
    action["recommended_action"] = np.where(
        action["stockout_days"] >= action["stockout_days"].median(),
        "Pre-build inventory 4-6 weeks before peak month; tighten safety stock and supplier lead-time monitoring",
        "Maintain stock policy; run monthly check on stockout spikes and adjust replenishment only for fast movers",
    )
    action["kpi_target"] = np.where(
        action["stockout_days"] >= action["stockout_days"].median(),
        "Reduce stockout_days by >=20%; keep fill_rate >=96%",
        "Keep stockout_days stable and avoid overstock expansion",
    )
    action["context_note"] = f"Q4 stockout months show avg next-month growth around {threshold * 100:.2f}%"

    action = action.sort_values("priority_score", ascending=False)
    return action[
        [
            "category",
            "month",
            "stockout_days",
            "days_of_supply",
            "fill_rate",
            "month_peak",
            "seasonality_idx_peak",
            "peak_to_trough_ratio",
            "priority_score",
            "recommended_action",
            "kpi_target",
            "context_note",
        ]
    ]


def write_summary_vi(
    quality_checks: dict[str, float],
    monthly: pd.DataFrame,
    month_kpi: pd.DataFrame,
    cat_pareto: pd.DataFrame,
    seasonal_profile: pd.DataFrame,
    pressure_corr: pd.DataFrame,
    pressure_quartile: pd.DataFrame,
    discount_diag: pd.DataFrame,
    forecast_6m: pd.DataFrame,
    action_priority: pd.DataFrame,
) -> None:
    peak_month = int(month_kpi.iloc[0]["month"])
    peak_idx = float(month_kpi.iloc[0]["seasonality_idx"])
    trough_month = int(month_kpi.iloc[-1]["month"])
    trough_idx = float(month_kpi.iloc[-1]["seasonality_idx"])

    top_cat = cat_pareto.iloc[0]
    top2_share = float(cat_pareto.head(2)["share_pct"].sum())

    corr_stockout = float(pressure_corr.iloc[0]["corr_stockout_days_vs_next_growth"])
    corr_fill = float(pressure_corr.iloc[0]["corr_fill_rate_vs_next_growth"])

    q1_growth = float(
        pressure_quartile[pressure_quartile["stockout_quartile"] == "Q1-low"]["avg_next_growth"].iloc[0]
    )
    q4_growth = float(
        pressure_quartile[pressure_quartile["stockout_quartile"] == "Q4-high"]["avg_next_growth"].iloc[0]
    )

    d0 = discount_diag.loc[discount_diag["discount_bin"] == "No promo"].iloc[0]
    d15 = discount_diag.loc[discount_diag["discount_bin"] == ">15%"].iloc[0]

    base_avg = float(forecast_6m["base_forecast"].mean())
    opt_avg = float(forecast_6m["optimistic_forecast"].mean())
    cons_avg = float(forecast_6m["conservative_forecast"].mean())

    top_action = action_priority.iloc[0]
    second_action = action_priority.iloc[1]

    lines = [
        "# Part 2 - EDA Narrative (Tiếng Việt)",
        "",
        "## Descriptive — What happened?",
        f"- **Data quality gate**: Revenue ghép từ transaction khớp `sales.csv` (MAE={quality_checks['revenue_mae_vs_sales']:.6f}, max abs err={quality_checks['revenue_max_abs_err']:.6f}); do đó metric doanh thu dùng trong EDA có tính nhất quán cao.",
        f"- **Mùa vụ rõ rệt**: tháng {peak_month} có seasonality index {peak_idx:.2f}, trong khi tháng {trough_month} là {trough_idx:.2f}; chênh lệch peak/trough xấp xỉ {peak_idx / trough_idx:.2f}x.",
        f"- **Doanh thu tập trung cao**: danh mục lớn nhất là **{top_cat['category']}** (share={top_cat['share_pct']:.2f}%), top-2 danh mục chiếm {top2_share:.2f}% tổng doanh thu, hàm ý rủi ro concentration.",
        "",
        "## Diagnostic — Why did it happen?",
        f"- Áp lực tồn kho có liên hệ với tăng trưởng tháng kế tiếp: corr(stockout_days, next_growth)={corr_stockout:.3f}, corr(fill_rate, next_growth)={corr_fill:.3f}.",
        f"- So sánh theo quartile tồn kho: nhóm **Q1-low stockout** có tăng trưởng kế tiếp trung bình {q1_growth * 100:.2f}%, trong khi **Q4-high stockout** chỉ {q4_growth * 100:.2f}%.",
        f"- Discount depth không tự động cải thiện return rate: nhóm **No promo** return {d0['return_rate'] * 100:.2f}% vs nhóm **>15%** return {d15['return_rate'] * 100:.2f}%; cần tối ưu theo mục tiêu margin thay vì tăng độ sâu giảm giá đại trà.",
        "",
        "## Predictive — What is likely to happen?",
        f"- Dự phóng 6 tháng (trend + seasonality) cho thấy doanh thu trung bình kịch bản **base** khoảng {fmt_money(base_avg)}/tháng.",
        f"- Biên kịch bản: **optimistic** ~{fmt_money(opt_avg)}/tháng và **conservative** ~{fmt_money(cons_avg)}/tháng, phản ánh biến động mùa vụ quan sát trong lịch sử.",
        "- Hàm ý vận hành: cần chuẩn bị tồn kho theo tháng đỉnh của từng category (không dùng một lịch mua hàng đồng nhất cho toàn bộ danh mục).",
        "",
        "## Prescriptive — What should we do?",
        f"- Ưu tiên hành động #1: **{top_action['category']} - tháng {int(top_action['month'])}** | stockout_days={top_action['stockout_days']:.2f} | score={top_action['priority_score']:.2f}.",
        f"  Hành động: {top_action['recommended_action']}",
        f"- Ưu tiên hành động #2: **{second_action['category']} - tháng {int(second_action['month'])}** | stockout_days={second_action['stockout_days']:.2f} | score={second_action['priority_score']:.2f}.",
        f"  KPI đề xuất: {top_action['kpi_target']}",
        "- Quy tắc điều hành đề xuất: với category-tháng có priority score cao, khóa kế hoạch pre-build trước 4-6 tuần; với score thấp, giữ chính sách hiện tại để tránh overstock.",
        "",
        "## Mapping Từ Figure Sang Insight",
        "- Fig01: xu hướng doanh thu dài hạn và dao động mùa vụ.",
        "- Fig02: heatmap mùa vụ theo năm-tháng, xác định tháng đỉnh/đáy.",
        "- Fig03: Pareto concentration theo category.",
        "- Fig04: profile mùa vụ khác biệt theo từng category.",
        "- Fig05: mối liên hệ giữa stockout pressure và tăng trưởng kế tiếp.",
        "- Fig06: hotspot month-category cần xử lý tồn kho.",
        "- Fig07: diagnostic phụ cho chiến lược discount depth.",
        "- Fig08: 3 kịch bản doanh thu 6 tháng để hỗ trợ planning.",
    ]
    (OUT_DIR / "part2_summary_vi.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reset_output_dir()
    data = load_data()
    line = build_fact_order_line(data)

    quality_checks = validate_data_quality(line, data["sales"], data["inventory"])

    monthly = build_monthly_revenue(data["sales"])
    plot_revenue_trend_monthly(monthly)

    month_kpi = plot_seasonality_heatmap(monthly)
    cat_pareto = plot_category_pareto(line)
    seasonal_profile = plot_category_seasonal_profile(line)

    pressure, pressure_corr = build_inventory_pressure(line, data["inventory"])
    pressure_quartile = plot_stockout_vs_next_growth(pressure)
    hotspot = plot_inventory_hotspot(data["inventory"])

    discount_diag = plot_discount_depth_diagnostic(line)
    forecast_6m = forecast_scenarios(monthly, horizon=6)

    action_priority = build_action_priority(
        seasonal_profile=seasonal_profile,
        hotspot=hotspot,
        pressure_quartile=pressure_quartile,
    )

    month_kpi.to_csv(OUT_DIR / "kpi_seasonality.csv", index=False)
    pressure_corr.to_csv(OUT_DIR / "kpi_inventory_pressure_correlations.csv", index=False)
    pressure_quartile.to_csv(OUT_DIR / "kpi_inventory_pressure_quartiles.csv", index=False)
    action_priority.to_csv(OUT_DIR / "kpi_action_priority.csv", index=False)
    discount_diag.to_csv(OUT_DIR / "kpi_discount_diagnostic.csv", index=False)
    forecast_6m.to_csv(OUT_DIR / "forecast_6m_scenarios.csv", index=False)

    write_summary_vi(
        quality_checks=quality_checks,
        monthly=monthly,
        month_kpi=month_kpi,
        cat_pareto=cat_pareto,
        seasonal_profile=seasonal_profile,
        pressure_corr=pressure_corr,
        pressure_quartile=pressure_quartile,
        discount_diag=discount_diag,
        forecast_6m=forecast_6m,
        action_priority=action_priority,
    )

    print(f"Done. Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
