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
#  State & City Standardization Dictionaries
# ──────────────────────────────────────────────
OFFICIAL_INDIAN_STATES = [
    "ANDAMAN & NICOBAR ISLANDS", "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM",
    "BIHAR", "CHANDIGARH", "CHHATTISGARH", "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DELHI", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU & KASHMIR",
    "JHARKHAND", "KARNATAKA", "KERALA", "LADAKH", "LAKSHADWEEP", "MADHYA PRADESH",
    "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA",
    "PUDUCHERRY", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TELANGANA",
    "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL"
]

STATE_MAPPING = {
    # Typos & Alternate Spellings
    "RAJSTHAN": "RAJASTHAN",
    "RAJSHTHAN": "RAJASTHAN",
    "RAJUSTHAN": "RAJASTHAN",
    "RAJHSTAN": "RAJASTHAN",
    "RAJASTAN": "RAJASTHAN",
    "RAJSHATHAN": "RAJASTHAN",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "NEW DELHI": "DELHI",
    "DELHI NCR": "DELHI",
    "DADRA AND NAGAR": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DADRA AND NAGAR HAVELI": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DAMAN & DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "ANDAMAN & NICOBAR": "ANDAMAN & NICOBAR ISLANDS",
    "ANDAMAN AND NICOBAR": "ANDAMAN & NICOBAR ISLANDS",

    # State Abbreviations
    "NL": "NAGALAND",
    "PB": "PUNJAB",
    "RJ": "RAJASTHAN",
    "AR": "ARUNACHAL PRADESH",
    "AP": "ANDHRA PRADESH",
    "TS": "TELANGANA",
    "TG": "TELANGANA",
    "UK": "UTTARAKHAND",
    "UA": "UTTARAKHAND",
    "UP": "UTTAR PRADESH",
    "WB": "WEST BENGAL",
    "MP": "MADHYA PRADESH",
    "MH": "MAHARASHTRA",
    "KA": "KARNATAKA",
    "TN": "TAMIL NADU",
    "KL": "KERALA",
    "GJ": "GUJARAT",
    "HR": "HARYANA",
    "HP": "HIMACHAL PRADESH",
    "JK": "JAMMU & KASHMIR",
    "JH": "JHARKHAND",
    "CT": "CHHATTISGARH",
    "CG": "CHHATTISGARH",
    "TR": "TRIPURA",
    "MN": "MANIPUR",
    "ML": "MEGHALAYA",
    "MZ": "MIZORAM",
    "SK": "SIKKIM",
    "AS": "ASSAM",
    "DL": "DELHI",
    "PY": "PUDUCHERRY",
    "LA": "LADAKH",
    "CH": "CHANDIGARH",
    "DN": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DD": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "AN": "ANDAMAN & NICOBAR ISLANDS",
    "LD": "LAKSHADWEEP"
}

CITY_TO_STATE = {
    # Rajasthan
    "JAIPUR": "RAJASTHAN", "JODHPUR": "RAJASTHAN", "UDAIPUR": "RAJASTHAN", "SURATGARH": "RAJASTHAN",
    "DHAULPUR": "RAJASTHAN", "NIMBAHERA": "RAJASTHAN", "GANGAPUR CITY": "RAJASTHAN", "BHEEM": "RAJASTHAN",
    "BARAN": "RAJASTHAN", "BANSWARA": "RAJASTHAN", "AJMER": "RAJASTHAN", "SRIGANGANAGAR": "RAJASTHAN",
    "KOTA": "RAJASTHAN", "BIKANER": "RAJASTHAN", "ALWAR": "RAJASTHAN", "BHILWARA": "RAJASTHAN",
    "MADANGANJ-KISHANGARH  AJMER": "RAJASTHAN", "KISHANGARH": "RAJASTHAN",

    # Punjab
    "ZIRAKPUR": "PUNJAB", "MOHALI": "PUNJAB", "ZIRA": "PUNJAB", "LUDHIANA": "PUNJAB",
    "AMRITSAR": "PUNJAB", "JALANDHAR": "PUNJAB", "PATIALA": "PUNJAB", "BATHINDA": "PUNJAB",
    "PATHANKOT": "PUNJAB",

    # Nagaland
    "DIMAPUR": "NAGALAND", "KOHIMA": "NAGALAND", "MOKOKCHUNG": "NAGALAND",

    # Goa
    "PANAJI": "GOA", "PONDA": "GOA", "MARGAO": "GOA", "OLD GOA": "GOA", "SOUTH GOA": "GOA",
    "NORTH GOA": "GOA", "TISWADI": "GOA", "CUNCOLIM": "GOA", "TIVIM": "GOA", "NERUL": "GOA",
    "VASCO DA GAMA": "GOA", "PORVORIM": "GOA", "MAPUSA": "GOA",

    # Odisha
    "BHUBANESWAR": "ODISHA", "BHUBANESHWAR": "ODISHA", "CUTTACK": "ODISHA", "ROURKELA": "ODISHA",
    "PURI": "ODISHA", "BALASORE": "ODISHA", "SAMBALPUR": "ODISHA",

    # Arunachal Pradesh
    "ITANAGAR": "ARUNACHAL PRADESH", "NAHARLAGUN": "ARUNACHAL PRADESH", "PASIGHAT": "ARUNACHAL PRADESH",

    # Delhi
    "NEW DELHI": "DELHI", "DELHI": "DELHI", "NORTH WEST DELHI": "DELHI", "SOUTH DELHI": "DELHI",
    "EAST DELHI": "DELHI", "WEST DELHI": "DELHI", "CENTRAL DELHI": "DELHI",

    # Maharashtra
    "MUMBAI": "MAHARASHTRA", "PUNE": "MAHARASHTRA", "NAGPUR": "MAHARASHTRA", "THANE": "MAHARASHTRA",
    "NAVI MUMBAI": "MAHARASHTRA", "NASIK": "MAHARASHTRA", "NASHIK": "MAHARASHTRA", "AURANGABAD": "MAHARASHTRA",
    "SOLAPUR": "MAHARASHTRA", "KOLHAPUR": "MAHARASHTRA",

    # Karnataka
    "BENGALURU": "KARNATAKA", "BANGALORE": "KARNATAKA", "MYSORE": "KARNATAKA", "MYSURU": "KARNATAKA",
    "HUBBALLI": "KARNATAKA", "HUBLI": "KARNATAKA", "MANGALORE": "KARNATAKA", "MANGALURU": "KARNATAKA",
    "BELAGAVI": "KARNATAKA", "BELGAUM": "KARNATAKA",

    # Tamil Nadu
    "CHENNAI": "TAMIL NADU", "COIMBATORE": "TAMIL NADU", "MADURAI": "TAMIL NADU", "TIRUCHIRAPPALLI": "TAMIL NADU",
    "SALEM": "TAMIL NADU", "TIRUPPUR": "TAMIL NADU", "ERODE": "TAMIL NADU", "VELLORE": "TAMIL NADU",

    # West Bengal
    "KOLKATA": "WEST BENGAL", "HOWRAH": "WEST BENGAL", "SILIGURI": "WEST BENGAL", "DURGAPUR": "WEST BENGAL",
    "ASANSOL": "WEST BENGAL", "KHARAGPUR": "WEST BENGAL",

    # Gujarat
    "AHMEDABAD": "GUJARAT", "SURAT": "GUJARAT", "VADODARA": "GUJARAT", "RAJKOT": "GUJARAT",
    "GANDHINAGAR": "GUJARAT", "BHAVNAGAR": "GUJARAT", "JAMNAGAR": "GUJARAT",

    # Telangana
    "HYDERABAD": "TELANGANA", "WARANGAL": "TELANGANA", "NIZAMABAD": "TELANGANA", "KARIMNAGAR": "TELANGANA",

    # Andhra Pradesh
    "VISAKHAPATNAM": "ANDHRA PRADESH", "VIJAYAWADA": "ANDHRA PRADESH", "GUNTUR": "ANDHRA PRADESH",
    "TIRUPATI": "ANDHRA PRADESH", "NELLORE": "ANDHRA PRADESH", "KADAPA": "ANDHRA PRADESH",

    # Uttar Pradesh
    "LUCKNOW": "UTTAR PRADESH", "KANPUR": "UTTAR PRADESH", "AGRA": "UTTAR PRADESH", "VARANASI": "UTTAR PRADESH",
    "NOIDA": "UTTAR PRADESH", "GREATER NOIDA": "UTTAR PRADESH", "GHAZIABAD": "UTTAR PRADESH",
    "MEERUT": "UTTAR PRADESH", "PRAYAGRAJ": "UTTAR PRADESH", "ALLAHABAD": "UTTAR PRADESH",

    # Kerala
    "THIRUVANANTHAPURAM": "KERALA", "TRIVANDRUM": "KERALA", "KOCHI": "KERALA", "COCHIN": "KERALA",
    "KOZHIKODE": "KERALA", "CALICUT": "KERALA", "THRISSUR": "KERALA", "KOLLAM": "KERALA",

    # Madhya Pradesh
    "BHOPAL": "MADHYA PRADESH", "INDORE": "MADHYA PRADESH", "GWALIOR": "MADHYA PRADESH",
    "JABALPUR": "MADHYA PRADESH", "UJJAIN": "MADHYA PRADESH",

    # Bihar
    "PATNA": "BIHAR", "GAYA": "BIHAR", "BHAGALPUR": "BIHAR", "MUZAFFARPUR": "BIHAR",

    # Assam
    "GUWAHATI": "ASSAM", "SILCHAR": "ASSAM", "DIBRUGARH": "ASSAM", "JORHAT": "ASSAM",

    # Haryana
    "GURUGRAM": "HARYANA", "GURGAON": "HARYANA", "FARIDABAD": "HARYANA", "PANIPAT": "HARYANA",
    "AMBALA": "HARYANA", "HISAR": "HARYANA", "ROHTAK": "HARYANA",

    # Uttarakhand
    "DEHRADUN": "UTTARAKHAND", "HARIDWAR": "UTTARAKHAND", "ROORKEE": "UTTARAKHAND", "HALDWANI": "UTTARAKHAND",

    # Himachal Pradesh
    "SHIMLA": "HIMACHAL PRADESH", "DHARAMSHALA": "HIMACHAL PRADESH", "SOLAN": "HIMACHAL PRADESH", "MANDI": "HIMACHAL PRADESH",

    # Jammu & Kashmir
    "SRINAGAR": "JAMMU & KASHMIR", "JAMMU": "JAMMU & KASHMIR", "ANANTNAG": "JAMMU & KASHMIR",

    # Jharkhand
    "RANCHI": "JHARKHAND", "JAMSHEDPUR": "JHARKHAND", "DHANBAD": "JHARKHAND", "BOKARO": "JHARKHAND",

    # Chhattisgarh
    "RAIPUR": "CHHATTISGARH", "BHILAI": "CHHATTISGARH", "BILASPUR": "CHHATTISGARH",

    # Chandigarh
    "CHANDIGARH": "CHANDIGARH",

    # Puducherry
    "PUDUCHERRY": "PUDUCHERRY", "PONDICHERRY": "PUDUCHERRY", "KARAIKAL": "PUDUCHERRY"
}


def clean_single_state(val, city_val=None):
    """Clean a single ship-state value using direct mapping, city lookup, and fuzzy matching."""
    import difflib

    if pd.isna(val) or val is None:
        raw_s = ""
    else:
        raw_s = str(val).strip()

    # Handle numeric placeholders ('0') or empty
    if raw_s in ["0", "nan", "NaN", "None", "", "APO"]:
        raw_s = ""

    # Try mapping composite/slash strings like 'Punjab/Mohali/Zirakpur'
    if "/" in raw_s:
        parts = [p.strip().upper() for p in raw_s.split("/")]
        for p in parts:
            if p in OFFICIAL_INDIAN_STATES or p in STATE_MAPPING:
                return STATE_MAPPING.get(p, p)

    clean_s = raw_s.upper()

    # Direct Mapping Lookup
    if clean_s in STATE_MAPPING:
        return STATE_MAPPING[clean_s]

    if clean_s in OFFICIAL_INDIAN_STATES:
        return clean_s

    # Impute via City Lookup if available
    if city_val and not pd.isna(city_val):
        city_str = str(city_val).strip().upper()
        if city_str in CITY_TO_STATE:
            return CITY_TO_STATE[city_str]
        for c, st in CITY_TO_STATE.items():
            if c in city_str:
                return st

    # Fuzzy String Match Fallback
    if clean_s and len(clean_s) > 3:
        matches = difflib.get_close_matches(clean_s, OFFICIAL_INDIAN_STATES, n=1, cutoff=0.7)
        if matches:
            return matches[0]

    return clean_s if clean_s else "UNKNOWN"


def clean_ship_state_series(state_series, city_series=None):
    """Clean a pandas Series of ship-state values."""
    cleaned = []
    cities = city_series if city_series is not None else [None] * len(state_series)
    for st, ct in zip(state_series, cities):
        cleaned.append(clean_single_state(st, ct))
    return pd.Series(cleaned, index=state_series.index)


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
        if "ship-city" in lc or "ship_city" in lc or "ship city" in lc:
            colmap[c] = "ship-city"
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

    # Clean ship-state and derive region
    if "ship-state" in big.columns:
        city_col = big["ship-city"] if "ship-city" in big.columns else None
        big["ship-state"] = clean_ship_state_series(big["ship-state"], city_col)
        big["region"] = big["ship-state"]
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
