#!/usr/bin/env python3
"""
modules/statistical_tests.py
Formal statistical tests for revenue concentration, marketplace pricing,
and channel profitability — addressing reviewer criticism about purely
descriptive analysis without significance testing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import STATISTICAL_TESTS_DIR
from modules.visualization import plot_lorenz_curve


# ──────────────────────────────────────────────
#  Revenue Concentration Measures
# ──────────────────────────────────────────────
def gini_coefficient(values):
    """Compute the Gini coefficient for a set of values."""
    values = np.sort(np.array(values, dtype=float))
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def herfindahl_hirschman_index(values):
    """
    Compute the Herfindahl-Hirschman Index (HHI).
    HHI = sum of squared market shares. Range: 1/N (perfect equality) to 1 (monopoly).
    """
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    shares = values / values.sum()
    return float(np.sum(shares ** 2))


def revenue_concentration_analysis(sales_df, group_col="SKU", value_col="Amount"):
    """
    Compute Gini coefficient, HHI, and generate Lorenz curve for revenue concentration.
    """
    print(f"\n{'='*60}")
    print("REVENUE CONCENTRATION ANALYSIS")
    print(f"{'='*60}")

    STATISTICAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    if group_col not in sales_df.columns or value_col not in sales_df.columns:
        print(f"[stat] Missing columns: {group_col} or {value_col}")
        return {}

    # Aggregate revenue per group
    agg = sales_df.groupby(group_col)[value_col].sum().dropna()
    agg = agg[agg > 0]

    if len(agg) == 0:
        print("[stat] No valid revenue data for concentration analysis.")
        return {}

    gini = gini_coefficient(agg.values)
    hhi = herfindahl_hirschman_index(agg.values)

    # Interpretation
    if hhi < 0.01:
        hhi_interp = "Highly competitive (low concentration)"
    elif hhi < 0.15:
        hhi_interp = "Moderately concentrated"
    elif hhi < 0.25:
        hhi_interp = "Highly concentrated"
    else:
        hhi_interp = "Very highly concentrated (near monopoly)"

    print(f"  Gini Coefficient: {gini:.4f}")
    print(f"  HHI: {hhi:.6f} — {hhi_interp}")
    print(f"  Number of {group_col}s: {len(agg)}")
    print(f"  Top 10% revenue share: {agg.nlargest(max(1, len(agg)//10)).sum() / agg.sum() * 100:.1f}%")

    # Save results
    results = {
        "gini_coefficient": gini,
        "hhi": hhi,
        "hhi_interpretation": hhi_interp,
        "num_groups": len(agg),
        "top_10pct_share": float(agg.nlargest(max(1, len(agg)//10)).sum() / agg.sum()),
    }
    pd.DataFrame([results]).to_csv(STATISTICAL_TESTS_DIR / "revenue_concentration.csv", index=False)

    # Lorenz curve
    plot_lorenz_curve(
        agg.values, STATISTICAL_TESTS_DIR,
        title=f"Lorenz Curve — {group_col} Revenue",
        filename=f"lorenz_curve_{group_col.lower()}",
        gini=gini,
    )

    return results


# ──────────────────────────────────────────────
#  Regional Concentration
# ──────────────────────────────────────────────
def regional_concentration_analysis(sales_df):
    """Run concentration analysis on regional revenue distribution."""
    if "region" not in sales_df.columns or "Amount" not in sales_df.columns:
        print("[stat] Missing region or Amount columns.")
        return {}
    return revenue_concentration_analysis(sales_df, group_col="region", value_col="Amount")


# ──────────────────────────────────────────────
#  Normality Testing
# ──────────────────────────────────────────────
def test_normality(data, name="data", max_sample=5000):
    """
    Shapiro-Wilk test for normality (on a subsample if data is large).
    Returns (W statistic, p-value, is_normal at alpha=0.05).
    """
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return None, None, False
    if len(data) > max_sample:
        rng = np.random.RandomState(42)
        data = rng.choice(data, max_sample, replace=False)
    w, p = stats.shapiro(data)
    is_normal = p > 0.05
    print(f"  Shapiro-Wilk ({name}): W={w:.6f}, p={p:.6f} → {'Normal' if is_normal else 'Non-normal'}")
    return w, p, is_normal


def test_homogeneity_of_variance(groups, names=None):
    """
    Levene's test for homogeneity of variance.
    groups: list of arrays.
    """
    clean_groups = [np.array(g, dtype=float)[~np.isnan(np.array(g, dtype=float))] for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 0]
    if len(clean_groups) < 2:
        return None, None, False
    f_stat, p = stats.levene(*clean_groups)
    equal_var = p > 0.05
    print(f"  Levene's test: F={f_stat:.4f}, p={p:.6f} → {'Equal' if equal_var else 'Unequal'} variance")
    return f_stat, p, equal_var


# ──────────────────────────────────────────────
#  MRP Dispersion — ANOVA / Kruskal-Wallis
# ──────────────────────────────────────────────
def mrp_dispersion_test(sales_df):
    """
    Test whether MRP values differ significantly across marketplaces.
    Uses ANOVA if data is normal, Kruskal-Wallis otherwise.
    """
    print(f"\n{'='*60}")
    print("MRP DISPERSION — STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*60}")

    STATISTICAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    # Detect MRP-like columns
    mrp_cols = [c for c in sales_df.columns if
                "mrp" in c.lower() or "amazon mrp" in c.lower() or
                "ajio" in c.lower() or "flipkart" in c.lower() or
                "myntra" in c.lower() or "paytm" in c.lower() or
                "limeroad" in c.lower()]

    if not mrp_cols:
        print("[stat] No MRP-like columns found.")
        return {}

    print(f"  MRP columns detected: {mrp_cols}")

    # Clean and prepare data
    groups = {}
    for col in mrp_cols:
        vals = pd.to_numeric(
            sales_df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce"
        ).dropna()
        if len(vals) > 0:
            groups[col] = vals.values

    if len(groups) < 2:
        print("[stat] Need at least 2 MRP columns for comparison.")
        return {}

    # Normality checks
    print("\n  Normality tests:")
    normality_results = {}
    all_normal = True
    for name, vals in groups.items():
        w, p, is_normal = test_normality(vals, name=name)
        normality_results[name] = {"shapiro_W": w, "shapiro_p": p, "is_normal": is_normal}
        if not is_normal:
            all_normal = False

    # Variance homogeneity
    print("\n  Variance homogeneity:")
    group_arrays = list(groups.values())
    lev_f, lev_p, equal_var = test_homogeneity_of_variance(group_arrays, list(groups.keys()))

    # Choose test
    results = {
        "normality": normality_results,
        "levene_F": lev_f, "levene_p": lev_p, "equal_variance": equal_var,
    }

    if all_normal and equal_var:
        # One-way ANOVA
        f_stat, p_val = stats.f_oneway(*group_arrays)
        test_name = "One-way ANOVA"
        results["test"] = test_name
        results["F_statistic"] = float(f_stat)
        results["p_value"] = float(p_val)
        print(f"\n  {test_name}: F={f_stat:.4f}, p={p_val:.6f}")
    else:
        # Kruskal-Wallis (non-parametric)
        h_stat, p_val = stats.kruskal(*group_arrays)
        test_name = "Kruskal-Wallis H-test"
        results["test"] = test_name
        results["H_statistic"] = float(h_stat)
        results["p_value"] = float(p_val)
        print(f"\n  {test_name}: H={h_stat:.4f}, p={p_val:.6f}")

    sig = "significant" if p_val < 0.05 else "not significant"
    print(f"  Result: MRP differences are {sig} at α=0.05")
    results["significant"] = p_val < 0.05

    # Effect size (eta-squared for ANOVA or epsilon-squared for KW)
    if "F_statistic" in results:
        # Eta-squared
        all_data = np.concatenate(group_arrays)
        ss_total = np.sum((all_data - all_data.mean()) ** 2)
        ss_between = sum(len(g) * (g.mean() - all_data.mean()) ** 2 for g in group_arrays)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        results["eta_squared"] = float(eta_sq)
        print(f"  Effect size (η²): {eta_sq:.4f}")
    else:
        # Epsilon-squared for Kruskal-Wallis
        n_total = sum(len(g) for g in group_arrays)
        epsilon_sq = (results["H_statistic"] - len(group_arrays) + 1) / (n_total - len(group_arrays))
        results["epsilon_squared"] = float(max(0, epsilon_sq))
        print(f"  Effect size (ε²): {max(0, epsilon_sq):.4f}")

    # Post-hoc pairwise comparisons (Dunn's test via Mann-Whitney with Bonferroni)
    if p_val < 0.05 and len(groups) > 2:
        print("\n  Post-hoc pairwise comparisons (Mann-Whitney with Bonferroni):")
        names = list(groups.keys())
        n_comparisons = len(names) * (len(names) - 1) // 2
        pairwise = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                u_stat, mw_p = stats.mannwhitneyu(groups[names[i]], groups[names[j]], alternative="two-sided")
                corrected_p = min(mw_p * n_comparisons, 1.0)  # Bonferroni
                sig_label = "***" if corrected_p < 0.001 else "**" if corrected_p < 0.01 else "*" if corrected_p < 0.05 else "ns"
                pairwise.append({
                    "group1": names[i], "group2": names[j],
                    "U_statistic": float(u_stat), "p_raw": float(mw_p),
                    "p_corrected": float(corrected_p), "significant": sig_label,
                })
                print(f"    {names[i]} vs {names[j]}: U={u_stat:.0f}, p_corr={corrected_p:.6f} ({sig_label})")
        results["pairwise_comparisons"] = pairwise
        pd.DataFrame(pairwise).to_csv(STATISTICAL_TESTS_DIR / "mrp_pairwise_comparisons.csv", index=False)

    # Save
    pd.DataFrame([{k: v for k, v in results.items() if not isinstance(v, (dict, list))}]).to_csv(
        STATISTICAL_TESTS_DIR / "mrp_dispersion_test.csv", index=False
    )

    return results


# ──────────────────────────────────────────────
#  Channel Profitability — t-test / Mann-Whitney
# ──────────────────────────────────────────────
def channel_profitability_test(sales_df):
    """
    Test whether Shiprocket and INCREFF profitability differ significantly.
    Uses Welch's t-test or Mann-Whitney U depending on normality.
    """
    print(f"\n{'='*60}")
    print("CHANNEL PROFITABILITY — STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*60}")

    STATISTICAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    profit_cols = [c for c in sales_df.columns
                   if "shiprocket" in c.lower() or "increff" in c.lower()]

    if len(profit_cols) < 2:
        print("[stat] Need at least 2 channel columns for comparison.")
        # Still compute descriptive stats if any exist
        if profit_cols:
            for col in profit_cols:
                vals = pd.to_numeric(
                    sales_df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                    errors="coerce"
                ).dropna()
                print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")
        return {}

    # Prepare the two groups
    groups = {}
    for col in profit_cols:
        vals = pd.to_numeric(
            sales_df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce"
        ).dropna()
        if len(vals) > 0:
            groups[col] = vals.values

    if len(groups) < 2:
        print("[stat] Insufficient data after cleaning.")
        return {}

    names = list(groups.keys())
    results = {"channels": names}

    # Descriptive stats
    for name, vals in groups.items():
        results[f"{name}_mean"] = float(np.mean(vals))
        results[f"{name}_std"] = float(np.std(vals))
        results[f"{name}_n"] = len(vals)
        print(f"  {name}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

    # Normality checks
    print("\n  Normality tests:")
    normal_flags = []
    for name, vals in groups.items():
        _, _, is_normal = test_normality(vals, name=name)
        normal_flags.append(is_normal)

    g1 = groups[names[0]]
    g2 = groups[names[1]]

    if all(normal_flags):
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        test_name = "Welch's t-test"
        results["test"] = test_name
        results["t_statistic"] = float(t_stat)
        results["p_value"] = float(p_val)
    else:
        # Mann-Whitney U
        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_name = "Mann-Whitney U test"
        results["test"] = test_name
        results["U_statistic"] = float(u_stat)
        results["p_value"] = float(p_val)

    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
    cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
    results["cohens_d"] = float(cohens_d)

    if abs(cohens_d) < 0.2:
        d_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        d_interp = "small"
    elif abs(cohens_d) < 0.8:
        d_interp = "medium"
    else:
        d_interp = "large"
    results["effect_size_interpretation"] = d_interp

    sig = "significant" if p_val < 0.05 else "not significant"
    results["significant"] = p_val < 0.05
    print(f"\n  {test_name}: p={p_val:.6f} → {sig}")
    print(f"  Cohen's d: {cohens_d:.4f} ({d_interp} effect)")

    pd.DataFrame([{k: v for k, v in results.items() if not isinstance(v, list)}]).to_csv(
        STATISTICAL_TESTS_DIR / "channel_profitability_test.csv", index=False
    )

    return results


# ──────────────────────────────────────────────
#  Run all statistical tests
# ──────────────────────────────────────────────
def run_all_statistical_tests(sales_df):
    """Run the complete suite of statistical tests on sales data."""
    results = {}

    # Revenue concentration — SKU level
    results["sku_concentration"] = revenue_concentration_analysis(
        sales_df, group_col="SKU", value_col="Amount"
    )

    # Revenue concentration — Regional level
    results["regional_concentration"] = regional_concentration_analysis(sales_df)

    # MRP dispersion significance
    results["mrp_dispersion"] = mrp_dispersion_test(sales_df)

    # Channel profitability test
    results["channel_profitability"] = channel_profitability_test(sales_df)

    print(f"\n[stat] All statistical test results saved to {STATISTICAL_TESTS_DIR}")
    return results
