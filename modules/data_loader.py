#!/usr/bin/env python3
"""
modules/data_loader.py
Data loading, preprocessing, and standardization for reviews and sales data.
Refactored from the legacy amazon_ecom_analysis_integrated_full.py with improvements.
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RANDOM_SEED, RESULTS_ROOT


# ──────────────────────────────────────────────
#  Safe CSV reading
# ──────────────────────────────────────────────
def safe_read_csv(path, **kwargs):
    """Read a CSV file with error handling."""
    path = Path(path)
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception as e:
        print(f"[data_loader] Failed to read {path}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────
#  Date detection and parsing
# ──────────────────────────────────────────────
def detect_and_parse_date(df, candidates=None):
    """Detect date-like columns and parse them."""
    if candidates is None:
        candidates = ["Date", "DATE", "date", "date_parsed"]
    for c in candidates:
        if c in df.columns:
            try:
                df["date_parsed"] = pd.to_datetime(df[c], errors="coerce", dayfirst=False)
                if df["date_parsed"].isna().mean() > 0.3:
                    df["date_parsed"] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                return df
            except Exception:
                continue
    df["date_parsed"] = pd.NaT
    return df


# ──────────────────────────────────────────────
#  Amount cleaning
# ──────────────────────────────────────────────
def clean_amount_series(s):
    """
    Clean an amount-like column or multiple columns merged under 'Amount'.
    Handles both Series and DataFrame inputs, flattening nested columns if needed.
    """
    if isinstance(s, pd.DataFrame):
        all_cols = []
        for col in s.columns:
            col_data = s[col]
            if isinstance(col_data, pd.DataFrame):
                for subcol in col_data.columns:
                    sub_data = pd.to_numeric(
                        col_data[subcol].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
                        errors="coerce",
                    )
                    all_cols.append(sub_data)
            else:
                sub_data = pd.to_numeric(
                    col_data.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
                    errors="coerce",
                )
                all_cols.append(sub_data)
        if not all_cols:
            return pd.Series(dtype=float)
        combined = pd.concat(all_cols, axis=1)
        return combined.mean(axis=1, skipna=True)
    elif isinstance(s, pd.Series):
        if s.dtype == object:
            s = s.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
        return pd.to_numeric(s, errors="coerce")
    else:
        return pd.Series(dtype=float)


# ──────────────────────────────────────────────
#  Reviews loading
# ──────────────────────────────────────────────
def load_reviews(path):
    """Load Amazon review CSV and parse timestamps."""
    df = safe_read_csv(path)
    if df.empty:
        return df
    expected = [
        "Id", "ProductId", "UserId", "ProfileName",
        "HelpfulnessNumerator", "HelpfulnessDenominator",
        "Score", "Time", "Summary", "Text",
    ]
    df = df[[c for c in expected if c in df.columns]]
    if "Time" in df.columns and np.issubdtype(df["Time"].dtype, np.number):
        try:
            df["time_parsed"] = pd.to_datetime(df["Time"], unit="s", errors="coerce")
        except Exception:
            df["time_parsed"] = pd.to_datetime(df["Time"], errors="coerce")
    else:
        df["time_parsed"] = (
            pd.to_datetime(df["Time"], errors="coerce") if "Time" in df.columns else pd.NaT
        )
    print(f"[data_loader] Reviews loaded: {df.shape}")
    return df


# ──────────────────────────────────────────────
#  Reviews preprocessing
# ──────────────────────────────────────────────
def preprocess_reviews(df, drop_neutral=True, save_csv=True, output_dir=None):
    """
    Preprocess reviews: clean text, filter neutral, create binary labels.
    Returns DataFrame with 'text_clean' and 'sent_label' columns.
    """
    df = df.dropna(subset=["Text", "Score"]).copy()
    df["text_clean"] = (
        df["Text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    )
    if drop_neutral and "Score" in df.columns:
        df = df[df["Score"] != 3]
    df["sent_label"] = df["Score"].apply(
        lambda x: "positive" if x >= 4 else ("negative" if x <= 2 else "neutral")
    )
    df = df[df["sent_label"] != "neutral"].sample(frac=1.0, random_state=RANDOM_SEED).reset_index(
        drop=True
    )

    if save_csv:
        out_dir = output_dir or RESULTS_ROOT
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "reviews_preprocessed.csv", index=False)
        print(f"[data_loader] Preprocessed reviews saved: {df.shape}")

    # Report class distribution
    dist = df["sent_label"].value_counts()
    ratio = dist.max() / dist.min() if dist.min() > 0 else float("inf")
    print(f"[data_loader] Class distribution:\n{dist.to_string()}")
    print(f"[data_loader] Imbalance ratio: {ratio:.2f}:1")

    return df


# ──────────────────────────────────────────────
#  Sales loading and standardization
# ──────────────────────────────────────────────
def load_and_standardize_sales(sales_folder, save_csv=True, output_dir=None):
    """
    Load all sales CSVs from a folder, auto-map columns,
    standardize, and return a merged DataFrame.
    """
    p = Path(sales_folder)
    files = {f.name.lower(): str(f) for f in p.glob("*.csv")}
    print(f"[data_loader] Detected sales files: {list(files.keys())}")

    df_list = []
    for name, path in files.items():
        df = safe_read_csv(path)
        if df.empty:
            continue
        df["source_file"] = name
        df_list.append(df)
    if not df_list:
        print("[data_loader] No sales CSV files found!")
        return pd.DataFrame()

    big = pd.concat(df_list, ignore_index=True, sort=False)

    # Auto-map columns
    colmap = {}
    for c in big.columns:
        lc = c.lower().strip()
        if lc in ["sku", "sku code", "sku_code"] or "sku code" in lc or lc.startswith("sku "):
            colmap[c] = "SKU"
        if "asin" in lc:
            colmap[c] = "ASIN"
        if lc == "amount" or lc.startswith("amount") or "gross" in lc or "amt" in lc or "rate" in lc:
            colmap[c] = "Amount"
        if "qty" in lc or "pcs" in lc:
            colmap[c] = "Qty"
        if "ship-state" in lc or "ship_state" in lc or "ship state" in lc:
            colmap[c] = "ship-state"
        if "ship-country" in lc or lc == "country" or "ship country" in lc:
            colmap[c] = "ship-country"
        if "date" in lc or "time" in lc:
            colmap[c] = "Date"

    big = big.rename(columns=colmap)
    big = detect_and_parse_date(big, ["Date", "DATE", "date_parsed"])

    # Remove duplicate column names
    if big.columns.duplicated().any():
        big = big.loc[:, ~big.columns.duplicated()]

    if "Amount" in big.columns:
        big["Amount"] = clean_amount_series(big["Amount"])
    if "Qty" in big.columns:
        big["Qty"] = pd.to_numeric(big["Qty"], errors="coerce")

    # Derive region
    if "ship-state" in big.columns:
        big["region"] = big["ship-state"].astype(str)
    elif "ship-country" in big.columns:
        big["region"] = big["ship-country"].astype(str)
    else:
        big["region"] = None

    if save_csv:
        out_dir = output_dir or RESULTS_ROOT / "sales_analytics"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        big.to_csv(out_dir / "merged_sales.csv", index=False)

    print(f"[data_loader] Merged sales rows: {big.shape[0]}")
    return big


# ──────────────────────────────────────────────
#  Quick summary for inspection
# ──────────────────────────────────────────────
def summarize_datasets(reviews_df, sales_df):
    """Print a quick summary of both datasets."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    if reviews_df is not None and not reviews_df.empty:
        print(f"\nReviews: {reviews_df.shape[0]:,} rows × {reviews_df.shape[1]} cols")
        if "sent_label" in reviews_df.columns:
            print(f"  Labels: {reviews_df['sent_label'].value_counts().to_dict()}")
        if "ProductId" in reviews_df.columns:
            print(f"  Unique ASINs (ProductId): {reviews_df['ProductId'].nunique():,}")
        if "time_parsed" in reviews_df.columns:
            print(f"  Date range: {reviews_df['time_parsed'].min()} -> {reviews_df['time_parsed'].max()}")

    if sales_df is not None and not sales_df.empty:
        print(f"\nSales: {sales_df.shape[0]:,} rows × {sales_df.shape[1]} cols")
        if "ASIN" in sales_df.columns:
            print(f"  Unique ASINs: {sales_df['ASIN'].nunique():,}")
        if "SKU" in sales_df.columns:
            print(f"  Unique SKUs: {sales_df['SKU'].nunique():,}")
        if "Amount" in sales_df.columns:
            print(f"  Amount: mean={sales_df['Amount'].mean():.2f}, std={sales_df['Amount'].std():.2f}")
        if "region" in sales_df.columns:
            print(f"  Unique regions: {sales_df['region'].nunique()}")

    print("=" * 60 + "\n")
