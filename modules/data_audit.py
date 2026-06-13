#!/usr/bin/env python3
"""
modules/data_audit.py
ASIN matching audit and data consistency checks.
Addresses reviewer criticism about data linkage quality between
reviews and sales datasets.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_AUDIT_DIR
from modules.visualization import plot_asin_overlap


# ──────────────────────────────────────────────
#  ASIN Overlap Analysis
# ──────────────────────────────────────────────
def asin_overlap_analysis(reviews_df, sales_df):
    """
    Analyze the overlap of ASINs between review data and sales data.
    Returns a comprehensive audit report.
    """
    print(f"\n{'='*60}")
    print("ASIN MATCHING & DATA LINKAGE AUDIT")
    print(f"{'='*60}")

    DATA_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    report = {}

    # Get unique ASINs from each dataset
    review_asins = set()
    if "ProductId" in reviews_df.columns:
        review_asins = set(reviews_df["ProductId"].dropna().unique())
    report["review_unique_asins"] = len(review_asins)
    print(f"  Reviews — Unique ASINs (ProductId): {len(review_asins):,}")

    sales_asins = set()
    if "ASIN" in sales_df.columns:
        sales_asins = set(sales_df["ASIN"].dropna().unique())
    report["sales_unique_asins"] = len(sales_asins)
    print(f"  Sales — Unique ASINs: {len(sales_asins):,}")

    # Compute overlap
    matched = review_asins & sales_asins
    reviews_only = review_asins - sales_asins
    sales_only = sales_asins - review_asins

    report["matched_asins"] = len(matched)
    report["reviews_only_asins"] = len(reviews_only)
    report["sales_only_asins"] = len(sales_only)
    report["match_rate_reviews"] = len(matched) / len(review_asins) if review_asins else 0
    report["match_rate_sales"] = len(matched) / len(sales_asins) if sales_asins else 0

    print(f"\n  Matched ASINs: {len(matched):,}")
    print(f"  Reviews-only ASINs: {len(reviews_only):,}")
    print(f"  Sales-only ASINs: {len(sales_only):,}")
    print(f"  Match rate (reviews): {report['match_rate_reviews']:.2%}")
    print(f"  Match rate (sales): {report['match_rate_sales']:.2%}")

    # Plot overlap
    plot_asin_overlap(
        len(reviews_only), len(sales_only), len(matched),
        DATA_AUDIT_DIR, filename="asin_overlap",
    )

    # Volume analysis — how much revenue/reviews are covered by matched ASINs
    if matched:
        if "ProductId" in reviews_df.columns:
            matched_reviews = reviews_df[reviews_df["ProductId"].isin(matched)]
            report["matched_review_count"] = len(matched_reviews)
            report["matched_review_pct"] = len(matched_reviews) / len(reviews_df) if len(reviews_df) > 0 else 0
            print(f"\n  Reviews covered by matched ASINs: {len(matched_reviews):,} ({report['matched_review_pct']:.2%})")

        if "ASIN" in sales_df.columns and "Amount" in sales_df.columns:
            matched_sales = sales_df[sales_df["ASIN"].isin(matched)]
            total_revenue = sales_df["Amount"].sum()
            matched_revenue = matched_sales["Amount"].sum()
            report["matched_sales_rows"] = len(matched_sales)
            report["matched_revenue"] = float(matched_revenue)
            report["matched_revenue_pct"] = float(matched_revenue / total_revenue) if total_revenue > 0 else 0
            print(f"  Sales rows with matched ASINs: {len(matched_sales):,}")
            print(f"  Revenue covered by matched ASINs: {matched_revenue:,.2f} ({report['matched_revenue_pct']:.2%})")

    # Save detailed lists
    pd.DataFrame({"matched_asin": sorted(matched)}).to_csv(
        DATA_AUDIT_DIR / "matched_asins.csv", index=False
    )
    pd.DataFrame({"reviews_only_asin": sorted(reviews_only)}).to_csv(
        DATA_AUDIT_DIR / "reviews_only_asins.csv", index=False
    )
    pd.DataFrame({"sales_only_asin": sorted(sales_only)}).to_csv(
        DATA_AUDIT_DIR / "sales_only_asins.csv", index=False
    )

    return report


# ──────────────────────────────────────────────
#  SKU Coverage Analysis
# ──────────────────────────────────────────────
def sku_coverage_analysis(sales_df, reviews_df):
    """Analyze how many SKUs have matching review ASINs."""
    print(f"\n  SKU COVERAGE ANALYSIS")
    print(f"  {'─'*40}")

    DATA_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if "SKU" not in sales_df.columns or "ASIN" not in sales_df.columns:
        print("  [audit] SKU or ASIN column not found in sales data.")
        return {}

    review_asins = set()
    if "ProductId" in reviews_df.columns:
        review_asins = set(reviews_df["ProductId"].dropna().unique())

    # Get SKU-ASIN mapping
    sku_asin_map = sales_df[["SKU", "ASIN"]].dropna().drop_duplicates()
    total_skus = sku_asin_map["SKU"].nunique()
    skus_with_reviews = sku_asin_map[sku_asin_map["ASIN"].isin(review_asins)]["SKU"].nunique()

    report = {
        "total_unique_skus": total_skus,
        "skus_with_matching_reviews": skus_with_reviews,
        "sku_coverage_rate": skus_with_reviews / total_skus if total_skus > 0 else 0,
    }
    print(f"  Total unique SKUs: {total_skus}")
    print(f"  SKUs with matching reviews: {skus_with_reviews}")
    print(f"  SKU coverage rate: {report['sku_coverage_rate']:.2%}")

    sku_asin_map["has_reviews"] = sku_asin_map["ASIN"].isin(review_asins)
    sku_asin_map.to_csv(DATA_AUDIT_DIR / "sku_asin_review_coverage.csv", index=False)

    return report


# ──────────────────────────────────────────────
#  Temporal Overlap Analysis
# ──────────────────────────────────────────────
def temporal_overlap_analysis(reviews_df, sales_df):
    """Compare the date ranges of reviews and sales data."""
    print(f"\n  TEMPORAL OVERLAP ANALYSIS")
    print(f"  {'─'*40}")

    report = {}

    if "time_parsed" in reviews_df.columns:
        rev_dates = reviews_df["time_parsed"].dropna()
        if not rev_dates.empty:
            report["review_date_min"] = str(rev_dates.min())
            report["review_date_max"] = str(rev_dates.max())
            report["review_date_range_days"] = (rev_dates.max() - rev_dates.min()).days
            print(f"  Reviews date range: {rev_dates.min().date()} → {rev_dates.max().date()} ({report['review_date_range_days']} days)")

    if "date_parsed" in sales_df.columns:
        sales_dates = sales_df["date_parsed"].dropna()
        if not sales_dates.empty:
            report["sales_date_min"] = str(sales_dates.min())
            report["sales_date_max"] = str(sales_dates.max())
            report["sales_date_range_days"] = (sales_dates.max() - sales_dates.min()).days
            print(f"  Sales date range: {sales_dates.min().date()} → {sales_dates.max().date()} ({report['sales_date_range_days']} days)")

    # Overlap period
    if "review_date_min" in report and "sales_date_min" in report:
        rev_min = pd.Timestamp(report["review_date_min"])
        rev_max = pd.Timestamp(report["review_date_max"])
        sal_min = pd.Timestamp(report["sales_date_min"])
        sal_max = pd.Timestamp(report["sales_date_max"])
        overlap_start = max(rev_min, sal_min)
        overlap_end = min(rev_max, sal_max)
        if overlap_start < overlap_end:
            report["overlap_start"] = str(overlap_start)
            report["overlap_end"] = str(overlap_end)
            report["overlap_days"] = (overlap_end - overlap_start).days
            print(f"  Overlap period: {overlap_start.date()} → {overlap_end.date()} ({report['overlap_days']} days)")
        else:
            report["overlap_days"] = 0
            print(f"  ⚠ NO temporal overlap between reviews and sales!")

    return report


# ──────────────────────────────────────────────
#  Full data audit
# ──────────────────────────────────────────────
def run_full_data_audit(reviews_df, sales_df):
    """Run the complete data quality and linkage audit."""
    DATA_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    audit = {}
    audit["asin_overlap"] = asin_overlap_analysis(reviews_df, sales_df)
    audit["sku_coverage"] = sku_coverage_analysis(sales_df, reviews_df)
    audit["temporal_overlap"] = temporal_overlap_analysis(reviews_df, sales_df)

    # Write comprehensive audit report
    report_lines = [
        "=" * 60,
        "DATA LINKAGE AUDIT REPORT",
        "=" * 60,
        "",
        "1. ASIN OVERLAP",
        f"   Review ASINs:    {audit['asin_overlap'].get('review_unique_asins', 'N/A'):>10,}",
        f"   Sales ASINs:     {audit['asin_overlap'].get('sales_unique_asins', 'N/A'):>10,}",
        f"   Matched:         {audit['asin_overlap'].get('matched_asins', 'N/A'):>10,}",
        f"   Match Rate (R):  {audit['asin_overlap'].get('match_rate_reviews', 0):>10.2%}",
        f"   Match Rate (S):  {audit['asin_overlap'].get('match_rate_sales', 0):>10.2%}",
        "",
        "2. SKU COVERAGE",
        f"   Total SKUs:          {audit['sku_coverage'].get('total_unique_skus', 'N/A')}",
        f"   SKUs with reviews:   {audit['sku_coverage'].get('skus_with_matching_reviews', 'N/A')}",
        f"   Coverage rate:       {audit['sku_coverage'].get('sku_coverage_rate', 0):.2%}",
        "",
        "3. TEMPORAL OVERLAP",
    ]
    for k, v in audit.get("temporal_overlap", {}).items():
        report_lines.append(f"   {k}: {v}")

    report_lines.extend([
        "",
        "=" * 60,
        "NOTE: This audit should be reviewed to verify that the review",
        "products and sales products originate from the same commercial",
        "environment and product catalogue.",
        "=" * 60,
    ])

    report_text = "\n".join(report_lines)
    with open(DATA_AUDIT_DIR / "data_linkage_audit.txt", "w") as f:
        f.write(report_text)

    # Save audit as JSON
    # Convert non-serializable values
    audit_serializable = json.loads(json.dumps(audit, default=str))
    with open(DATA_AUDIT_DIR / "data_audit.json", "w") as f:
        json.dump(audit_serializable, f, indent=2)

    print(f"\n[audit] Full audit report saved to {DATA_AUDIT_DIR}")
    return audit


# Need json import at top
import json
