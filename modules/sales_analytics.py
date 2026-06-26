#!/usr/bin/env python3
"""
modules/sales_analytics.py
Refactored sales analytics with integrated statistical tests.
Handles channel profitability, MRP dispersion, top SKUs, and regional revenue
with proper statistical validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SALES_ANALYTICS_DIR, TOPN_SKUS
from modules.visualization import plot_horizontal_bar, plot_boxplot


# ──────────────────────────────────────────────
#  Channel profitability with descriptive stats
# ──────────────────────────────────────────────
def compute_channel_profitability(sales_df):
    """
    Compute channel profitability statistics and save results.
    (Statistical significance tests are in statistical_tests.py)
    """
    print(f"\n{'='*60}")
    print("CHANNEL PROFITABILITY ANALYSIS")
    print(f"{'='*60}")

    SALES_ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    profit_cols = [c for c in sales_df.columns
                   if "shiprocket" in c.lower() or "increff" in c.lower()]

    if not profit_cols:
        print("[sales] No channel profitability columns detected.")
        return {}

    results = {}
    for col in profit_cols:
        vals = pd.to_numeric(
            sales_df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce"
        ).dropna()
        results[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(vals.median()),
            "n": len(vals),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
        print(f"  {col}: mean={vals.mean():.4f} ± {vals.std():.4f}, median={vals.median():.4f}, n={len(vals)}")

    # Save
    prof_df = pd.DataFrame(results).T
    prof_df.to_csv(SALES_ANALYTICS_DIR / "channel_profitability_descriptive.csv")

    # Plot
    if results:
        valid_results = {k: v for k, v in results.items() if v["n"] > 0}
        if valid_results:
            plot_data = pd.DataFrame({
                "channel": list(valid_results.keys()),
                "mean_profit": [v["mean"] for v in valid_results.values()],
            }).sort_values("mean_profit")
            plot_horizontal_bar(
                plot_data, "mean_profit", "channel",
                SALES_ANALYTICS_DIR,
                title="Channel Mean Profitability Comparison",
                filename="channel_profitability",
                color="#4CAF50",
            )

    return results


# ──────────────────────────────────────────────
#  MRP dispersion with descriptive stats
# ──────────────────────────────────────────────
def mrp_dispersion_analysis(sales_df):
    """
    Compute MRP dispersion across marketplaces with descriptive statistics.
    (Statistical significance tests are in statistical_tests.py)
    """
    print(f"\n{'='*60}")
    print("MRP DISPERSION ANALYSIS")
    print(f"{'='*60}")

    SALES_ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    mrp_cols = [c for c in sales_df.columns if
                "mrp" in c.lower() or "amazon mrp" in c.lower() or
                "ajio" in c.lower() or "flipkart" in c.lower() or
                "myntra" in c.lower() or "paytm" in c.lower() or
                "limeroad" in c.lower()]

    if not mrp_cols:
        print("[sales] No MRP-like columns detected.")
        return None

    print(f"  MRP columns: {mrp_cols}")

    tmp = sales_df[mrp_cols].copy()
    for c in mrp_cols:
        tmp[c] = pd.to_numeric(
            tmp[c].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce"
        )

    # Descriptive stats
    desc = tmp.describe().T
    desc.to_csv(SALES_ANALYTICS_DIR / "mrp_dispersion_stats.csv")
    print(f"\n{desc.to_string()}")

    # Boxplot
    melted = tmp.melt(var_name="store", value_name="mrp").dropna()
    plot_boxplot(
        melted, "store", "mrp",
        SALES_ANALYTICS_DIR,
        title="MRP Dispersion Across Marketplace Channels",
        filename="mrp_dispersion",
    )

    return mrp_cols


# ──────────────────────────────────────────────
#  Top SKUs & Regional Revenue
# ──────────────────────────────────────────────
def top_skus_and_region_analysis(sales_df, topn=None):
    """
    Compute top SKUs by revenue and regional revenue breakdown with statistics.
    """
    print(f"\n{'='*60}")
    print("TOP SKU & REGIONAL REVENUE ANALYSIS")
    print(f"{'='*60}")

    SALES_ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    topn = topn or TOPN_SKUS

    # Top SKUs
    if "SKU" in sales_df.columns and "Amount" in sales_df.columns:
        skuagg = (
            sales_df.groupby("SKU")
            .agg(total_revenue=("Amount", "sum"), total_qty=("Qty", "sum"), orders=("Amount", "count"))
            .reset_index()
            .sort_values("total_revenue", ascending=False)
        )
        skuagg.head(topn).to_csv(SALES_ANALYTICS_DIR / "top_skus.csv", index=False)

        # Per-order stats for top SKUs
        per_sku_stats = []
        for sku in skuagg.head(topn)["SKU"].tolist():
            g = sales_df[sales_df["SKU"] == sku]
            per_sku_stats.append({
                "SKU": sku,
                "total_revenue": float(skuagg[skuagg["SKU"] == sku]["total_revenue"].values[0]),
                "orders": int(skuagg[skuagg["SKU"] == sku]["orders"].values[0]),
                "revenue_mean": float(g["Amount"].mean()),
                "revenue_std": float(g["Amount"].std()),
                "qty_mean": float(g["Qty"].mean()) if "Qty" in g else np.nan,
                "qty_std": float(g["Qty"].std()) if "Qty" in g else np.nan,
            })
        pd.DataFrame(per_sku_stats).to_csv(SALES_ANALYTICS_DIR / "top_skus_stats.csv", index=False)

        plot_horizontal_bar(
            skuagg.head(topn), "total_revenue", "SKU",
            SALES_ANALYTICS_DIR,
            title=f"Top {topn} SKUs by Revenue",
            filename="top_skus",
        )
        print(f"  Top {topn} SKUs saved.")

    # Regional revenue
    if "region" in sales_df.columns and sales_df["region"].notna().sum() > 0:
        region_agg = (
            sales_df.groupby("region")
            .agg(revenue=("Amount", "sum"), count_orders=("Amount", "count"))
            .reset_index()
            .sort_values("revenue", ascending=False)
        )
        region_agg.to_csv(SALES_ANALYTICS_DIR / "sales_by_region.csv", index=False)

        # Per-region stats
        def per_region_stats(g):
            return pd.Series({
                "revenue_sum": g["Amount"].sum(),
                "revenue_mean": g["Amount"].mean(),
                "revenue_std": g["Amount"].std(),
                "qty_mean": g["Qty"].mean() if "Qty" in g else np.nan,
                "qty_std": g["Qty"].std() if "Qty" in g else np.nan,
                "n_orders": len(g),
            })

        reg_stats = (
            sales_df.groupby("region")
            .apply(per_region_stats)
            .reset_index()
            .sort_values("revenue_sum", ascending=False)
        )
        reg_stats.to_csv(SALES_ANALYTICS_DIR / "sales_region_stats.csv", index=False)

        plot_horizontal_bar(
            region_agg.head(20), "revenue", "region",
            SALES_ANALYTICS_DIR,
            title="Top 20 Regions by Revenue",
            filename="sales_by_region",
            color="#FF9800",
        )
        print(f"  Regional stats saved.")

    # Overall stats
    overall = {}
    if "Amount" in sales_df.columns:
        overall["amount_mean"] = float(sales_df["Amount"].mean())
        overall["amount_std"] = float(sales_df["Amount"].std())
        overall["amount_median"] = float(sales_df["Amount"].median())
    if "Qty" in sales_df.columns:
        overall["qty_mean"] = float(sales_df["Qty"].mean())
        overall["qty_std"] = float(sales_df["Qty"].std())
    pd.DataFrame([overall]).to_csv(SALES_ANALYTICS_DIR / "sales_overall_stats.csv", index=False)
    print(f"  Overall stats: {overall}")

    return overall


# ──────────────────────────────────────────────
#  Full sales analytics pipeline
# ──────────────────────────────────────────────
def run_full_sales_analytics(sales_df, topn=None):
    """Run the complete sales analytics pipeline."""
    results = {}
    results["channel_profitability"] = compute_channel_profitability(sales_df)
    results["mrp_cols"] = mrp_dispersion_analysis(sales_df)
    results["overall"] = top_skus_and_region_analysis(sales_df, topn=topn)
    return results
