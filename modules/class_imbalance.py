#!/usr/bin/env python3
"""
modules/class_imbalance.py
Systematic experiments comparing class imbalance handling strategies.
Strategies: Baseline, Class Weighting, SMOTE, Random Oversampling,
            Random Undersampling.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RANDOM_SEED, TEST_SIZE, COMPARATIVE_DIR,
)
from modules.visualization import plot_imbalance_comparison


# ──────────────────────────────────────────────
#  Imbalance strategies
# ──────────────────────────────────────────────
def _get_strategies():
    """
    Return a dict of strategy_name -> (resample_fn, model_kwargs).
    resample_fn takes (X_train, y_train) and returns (X_resampled, y_resampled).
    """
    strategies = {}

    # 1. Baseline — no resampling, no weighting
    strategies["Baseline (no handling)"] = {
        "resample_fn": lambda X, y: (X, y),
        "model_kwargs": {},
    }

    # 2. Class weighting
    strategies["Class Weighting (balanced)"] = {
        "resample_fn": lambda X, y: (X, y),
        "model_kwargs": {"class_weight": "balanced"},
    }

    # 3. SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        def smote_resample(X, y):
            sm = SMOTE(random_state=RANDOM_SEED)
            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res
        strategies["SMOTE"] = {
            "resample_fn": smote_resample,
            "model_kwargs": {},
        }
    except ImportError:
        print("[class_imbalance] imbalanced-learn not installed. SMOTE will be skipped.")

    # 4. Random oversampling
    try:
        from imblearn.over_sampling import RandomOverSampler
        def ros_resample(X, y):
            ros = RandomOverSampler(random_state=RANDOM_SEED)
            X_res, y_res = ros.fit_resample(X, y)
            return X_res, y_res
        strategies["Random Oversampling"] = {
            "resample_fn": ros_resample,
            "model_kwargs": {},
        }
    except ImportError:
        pass

    # 5. Random undersampling
    try:
        from imblearn.under_sampling import RandomUnderSampler
        def rus_resample(X, y):
            rus = RandomUnderSampler(random_state=RANDOM_SEED)
            X_res, y_res = rus.fit_resample(X, y)
            return X_res, y_res
        strategies["Random Undersampling"] = {
            "resample_fn": rus_resample,
            "model_kwargs": {},
        }
    except ImportError:
        pass

    return strategies


# ──────────────────────────────────────────────
#  Run imbalance experiments
# ──────────────────────────────────────────────
def run_imbalance_experiments(X_train_tfidf, y_train, X_test_tfidf, y_test, labels):
    """
    Run all imbalance strategies with Logistic Regression
    and compare their effect on negative-class recall.
    """
    print(f"\n{'='*60}")
    print("CLASS IMBALANCE EXPERIMENTS")
    print(f"{'='*60}")

    strategies = _get_strategies()
    results = []

    for strategy_name, config in strategies.items():
        print(f"\n[imbalance] Strategy: {strategy_name}")

        resample_fn = config["resample_fn"]
        model_kwargs = config["model_kwargs"]

        # Resample training data
        try:
            X_res, y_res = resample_fn(X_train_tfidf, y_train)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        if hasattr(y_res, "value_counts"):
            dist = y_res.value_counts()
        else:
            unique, counts = np.unique(y_res, return_counts=True)
            dist = dict(zip(unique, counts))
        print(f"  Resampled distribution: {dict(dist) if isinstance(dist, dict) else dist.to_dict()}")

        # Train model
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, **model_kwargs)
        clf.fit(X_res, y_res)
        preds = clf.predict(X_test_tfidf)

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )

        # Per-class
        per_class = {}
        for cls in labels:
            p, r, f, s = precision_recall_fscore_support(
                y_test, preds, labels=[cls], average=None, zero_division=0
            )
            per_class[cls] = {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0])}

        neg_recall = per_class.get("negative", {}).get("recall", 0.0)
        neg_f1 = per_class.get("negative", {}).get("f1", 0.0)

        print(f"  Accuracy: {acc:.4f} | F1(macro): {f1_m:.4f} | "
              f"Neg Recall: {neg_recall:.4f} | Neg F1: {neg_f1:.4f}")

        results.append({
            "strategy": strategy_name,
            "accuracy": acc,
            "precision_macro": prec_m,
            "recall_macro": rec_m,
            "f1_macro": f1_m,
            "negative_recall": neg_recall,
            "negative_f1": neg_f1,
            "negative_precision": per_class.get("negative", {}).get("precision", 0.0),
            "positive_recall": per_class.get("positive", {}).get("recall", 0.0),
            "positive_f1": per_class.get("positive", {}).get("f1", 0.0),
        })

    # Save results
    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(COMPARATIVE_DIR / "imbalance_experiment_table.csv", index=False)

    # Plot comparison
    if not results_df.empty:
        plot_imbalance_comparison(results_df, COMPARATIVE_DIR)

    print(f"\n[imbalance] Results saved to {COMPARATIVE_DIR / 'imbalance_experiment_table.csv'}")
    print("\n" + results_df.to_string(index=False))

    return results_df
