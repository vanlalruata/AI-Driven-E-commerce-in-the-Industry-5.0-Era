#!/usr/bin/env python3
"""
modules/sentiment_models.py
Unified interface for training multiple classical ML models on TF-IDF features.
Models: Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost.
Each model is trained and evaluated identically for fair comparison.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RANDOM_SEED, TEST_SIZE, MAX_TFIDF_FEATURES, TFIDF_NGRAM_RANGE,
    get_model_metrics_dir, get_model_figures_dir, get_model_artifacts_dir,
    MODEL_DISPLAY_NAMES, PREFER_GPU,
)
from modules.visualization import plot_confusion_matrix


# ──────────────────────────────────────────────
#  Model Registry
# ──────────────────────────────────────────────
def get_model_registry():
    """
    Return a dict of model_name -> (model_instance, supports_predict_proba).
    Models that don't natively support predict_proba are wrapped with CalibratedClassifierCV.
    """
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False
        print("[sentiment_models] XGBoost not installed. Skipping XGBoost model.")

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, C=10.0, class_weight="balanced", random_state=RANDOM_SEED),
            True,   # supports predict_proba
        ),
        "tfidf_lr": (
            LogisticRegression(max_iter=1000, C=10.0, random_state=RANDOM_SEED),
            True,
        ),
        "svm": (
            CalibratedClassifierCV(
                LinearSVC(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
                cv=3,
            ),
            True,   # CalibratedClassifierCV adds predict_proba
        ),
        "naive_bayes": (
            MultinomialNB(alpha=1.0),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1,
            ),
            True,
        ),
    }

    if xgb_available:
        # Note: XGBoost 3.2.0 on Windows has a known bug where training on a CPU-resident SciPy sparse TF-IDF matrix
        # with device='cuda' corrupts the booster's tree building, leading to extremely low accuracy (~23%).
        # The registry template defaults to CPU. Callers (e.g. cross-validation) may override to GPU
        # after converting sparse matrices to dense, which avoids the bug.
        print("[sentiment_models] XGBoost registry template: CPU (callers may override to GPU with dense matrices).")
        models["xgboost"] = (
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=-1,
            ),
            True,
        )

    return models


# ──────────────────────────────────────────────
#  TF-IDF Vectorizer (shared across all models)
# ──────────────────────────────────────────────
def build_tfidf_vectorizer(X_train, max_features=None, ngram_range=None):
    """Fit a TF-IDF vectorizer on training data."""
    mf = max_features or MAX_TFIDF_FEATURES
    ngr = ngram_range or TFIDF_NGRAM_RANGE
    vect = TfidfVectorizer(ngram_range=ngr, max_features=mf, sublinear_tf=True)
    X_train_tfidf = vect.fit_transform(X_train)
    print(f"[sentiment_models] TF-IDF matrix shape: {X_train_tfidf.shape}")
    return vect, X_train_tfidf


# ──────────────────────────────────────────────
#  Single model training
# ──────────────────────────────────────────────
def train_single_model(model_name, model, X_train_tfidf, y_train, X_test_tfidf, y_test, labels):
    """
    Train a single model and return a results dictionary.
    """
    display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    print(f"\n[sentiment_models] Training {display}...")

    # Handle XGBoost label encoding
    if model_name == "xgboost":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        model.fit(X_train_tfidf, y_train_enc)
        preds_enc = model.predict(X_test_tfidf)
        preds = le.inverse_transform(preds_enc)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_tfidf)
            # Map to original class order
            class_order = le.classes_
            prob_dict = {c: probs[:, i] for i, c in enumerate(class_order)}
        else:
            probs = None
            prob_dict = None
    else:
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_tfidf)
            class_order = model.classes_
            prob_dict = {c: probs[:, i] for i, c in enumerate(class_order)}
        else:
            probs = None
            prob_dict = None

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=labels)

    print(f"  Accuracy: {acc:.4f} | Precision(macro): {prec_macro:.4f} | "
          f"Recall(macro): {rec_macro:.4f} | F1(macro): {f1_macro:.4f}")

    # Per-class metrics
    per_class = {}
    for cls in labels:
        p, r, f, s = precision_recall_fscore_support(
            y_test, preds, labels=[cls], average=None, zero_division=0
        )
        per_class[cls] = {
            "precision": float(p[0]), "recall": float(r[0]),
            "f1": float(f[0]), "support": int(s[0]),
        }
        print(f"  {cls}: P={p[0]:.4f} R={r[0]:.4f} F1={f[0]:.4f} Support={s[0]}")

    return {
        "model_name": model_name,
        "model": model,
        "predictions": preds,
        "probabilities": probs,
        "prob_dict": prob_dict,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ──────────────────────────────────────────────
#  Save model results to its dedicated directory
# ──────────────────────────────────────────────
def save_model_results(result, vect, labels):
    """Save all artifacts for a single model to its result directory."""
    model_name = result["model_name"]

    metrics_dir = get_model_metrics_dir(model_name)
    figures_dir = get_model_figures_dir(model_name)
    artifacts_dir = get_model_artifacts_dir(model_name)

    # Classification report CSV
    pd.DataFrame(result["classification_report"]).transpose().to_csv(
        metrics_dir / "classification_report.csv"
    )

    # Per-class metrics
    pd.DataFrame(result["per_class"]).T.to_csv(metrics_dir / "per_class_metrics.csv")

    # Confusion matrix figure
    plot_confusion_matrix(result["confusion_matrix"], labels, figures_dir, model_name)

    # Save model and vectorizer
    joblib.dump(result["model"], artifacts_dir / "model.joblib")
    joblib.dump(vect, artifacts_dir / "vectorizer.joblib")

    # Save training config
    config = {
        "model_name": model_name,
        "accuracy": result["accuracy"],
        "precision_macro": result["precision_macro"],
        "recall_macro": result["recall_macro"],
        "f1_macro": result["f1_macro"],
        "per_class": result["per_class"],
    }
    with open(artifacts_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[sentiment_models] Results saved for {model_name}")


# ──────────────────────────────────────────────
#  Train all models
# ──────────────────────────────────────────────
def train_all_models(reviews_df, save_results=True):
    """
    Train all registered models on the preprocessed reviews.
    Returns a dict of {model_name: result_dict} and the shared vectorizer.
    """
    X = reviews_df["text_clean"]
    y = reviews_df["sent_label"]
    labels = sorted(y.unique().tolist())

    print(f"\n{'='*60}")
    print("TRAINING ALL CLASSICAL MODELS")
    print(f"{'='*60}")
    print(f"Dataset size: {len(X):,} samples")
    print(f"Labels: {labels}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Build shared TF-IDF vectorizer
    vect, X_train_tfidf = build_tfidf_vectorizer(X_train)
    X_test_tfidf = vect.transform(X_test)

    # Train each model
    registry = get_model_registry()
    all_results = {}

    for model_name, (model, _) in registry.items():
        result = train_single_model(
            model_name, model,
            X_train_tfidf, y_train,
            X_test_tfidf, y_test,
            labels,
        )
        # Store test data references for later evaluation
        result["X_test"] = X_test
        result["y_test"] = y_test
        result["X_test_tfidf"] = X_test_tfidf

        if save_results:
            save_model_results(result, vect, labels)
        all_results[model_name] = result

    # Build comparison table
    if save_results:
        _save_comparison_table(all_results, labels)

    return all_results, vect, X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, labels


def _save_comparison_table(all_results, labels):
    """Build and save a cross-model comparison table."""
    from config import COMPARATIVE_DIR
    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, res in all_results.items():
        row = {
            "model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "accuracy": res["accuracy"],
            "precision_macro": res["precision_macro"],
            "recall_macro": res["recall_macro"],
            "f1_macro": res["f1_macro"],
        }
        for cls in labels:
            if cls in res["per_class"]:
                for metric in ["precision", "recall", "f1"]:
                    row[f"{cls}_{metric}"] = res["per_class"][cls][metric]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(COMPARATIVE_DIR / "model_comparison_table.csv")
    print(f"\n[sentiment_models] Comparison table saved to {COMPARATIVE_DIR / 'model_comparison_table.csv'}")
    print("\n" + df.to_string())
    return df


def load_tfidf_lr(reviews_df):
    """
    Load a previously trained TF-IDF+LR model from disk.
    Returns the result dict if a saved model is found, or None otherwise.
    """
    from config import get_model_artifacts_dir, TEST_SIZE, RANDOM_SEED
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

    artifacts_dir = get_model_artifacts_dir("tfidf_lr")
    model_path = artifacts_dir / "model.joblib"
    vect_path = artifacts_dir / "vectorizer.joblib"
    config_path = artifacts_dir / "training_config.json"

    if not (model_path.exists() and vect_path.exists() and config_path.exists()):
        return None

    print(f"\n{'='*60}")
    print("LOADING PREVIOUSLY TRAINED TF-IDF+LR")
    print(f"{'='*60}")

    try:
        model = joblib.load(model_path)
        vect = joblib.load(vect_path)
    except Exception as e:
        print(f"  [tfidf_lr] Failed to load saved model/vectorizer: {e}")
        return None

    print(f"  ✓ Model loaded from {artifacts_dir}")

    # Reconstruct splits
    X = reviews_df["text_clean"]
    y = reviews_df["sent_label"]
    labels = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    X_test_tfidf = vect.transform(X_test)
    preds = model.predict(X_test_tfidf)
    probs = model.predict_proba(X_test_tfidf) if hasattr(model, "predict_proba") else None
    prob_dict = {c: probs[:, i] for i, c in enumerate(model.classes_)} if probs is not None else None

    acc = accuracy_score(y_test, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=labels)

    per_class = {}
    for cls in labels:
        p, r, f, s = precision_recall_fscore_support(
            y_test, preds, labels=[cls], average=None, zero_division=0
        )
        per_class[cls] = {
            "precision": float(p[0]), "recall": float(r[0]),
            "f1": float(f[0]), "support": int(s[0]),
        }

    print(f"  Accuracy: {acc:.4f} | Precision(macro): {prec_macro:.4f} | Recall(macro): {rec_macro:.4f} | F1(macro): {f1_macro:.4f}")

    return {
        "model_name": "tfidf_lr",
        "model": model,
        "predictions": preds,
        "probabilities": probs,
        "prob_dict": prob_dict,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_test": y_test,
        "X_test": X_test,
    }


def train_tfidf_lr(reviews_df, save_results=True):
    """
    Train a TF-IDF+LR model from scratch.
    """
    from config import RANDOM_SEED, TEST_SIZE
    from sklearn.model_selection import train_test_split

    print(f"\n{'='*60}")
    print("TRAINING TF-IDF+LR MODEL")
    print(f"{'='*60}")

    X = reviews_df["text_clean"]
    y = reviews_df["sent_label"]
    labels = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    vect, X_train_tfidf = build_tfidf_vectorizer(X_train)
    X_test_tfidf = vect.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    result = train_single_model(
        "tfidf_lr", model,
        X_train_tfidf, y_train,
        X_test_tfidf, y_test,
        labels,
    )

    result["X_test"] = X_test
    result["y_test"] = y_test
    result["X_test_tfidf"] = X_test_tfidf

    if save_results:
        save_model_results(result, vect, labels)

    return result, vect, X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, labels
