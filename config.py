#!/usr/bin/env python3
"""
config.py
Centralized configuration for the Industry 5.0 E-Commerce Sentiment Analysis Pipeline.
All paths, hyperparameters, feature flags, and output directory management live here.
"""

from pathlib import Path
import json

# ──────────────────────────────────────────────
#  Default Paths (override via CLI or set here)
# ──────────────────────────────────────────────
DEFAULT_REVIEWS_PATH = r"H:/Datasets/ecommerce/ecommerce_review/Reviews.csv"
DEFAULT_SALES_FOLDER = r"H:/Datasets/ecommerce/ecommerce_sale"

# ──────────────────────────────────────────────
#  Output Root
# ──────────────────────────────────────────────
RESULTS_ROOT = Path("results")

# Model-specific output directories
MODEL_NAMES = [
    "logistic_regression",
    "svm",
    "naive_bayes",
    "random_forest",
    "xgboost",
    "tfidf_lr",
    "distilbert",
]

# Sub-directories under each model
MODEL_SUBDIRS = ["metrics", "figures", "artifacts"]

# Other result directories
COMPARATIVE_DIR = RESULTS_ROOT / "comparative"
STATISTICAL_TESTS_DIR = RESULTS_ROOT / "statistical_tests"
DATA_AUDIT_DIR = RESULTS_ROOT / "data_audit"
SALES_ANALYTICS_DIR = RESULTS_ROOT / "sales_analytics"

# ──────────────────────────────────────────────
#  Hyperparameters
# ──────────────────────────────────────────────
RANDOM_SEED = 42
CV_FOLDS = 10
TEST_SIZE = 0.15
MAX_TFIDF_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
SENTIMENT_RUNS = 5  # legacy repeated hold-out runs
TOPN_SKUS = 20

# ──────────────────────────────────────────────
#  Feature Flags
# ──────────────────────────────────────────────
PREFER_GPU = True                # Prefer GPU (CUDA) for training models (XGBoost, DistilBERT)
ENABLE_DISTILBERT = False        # Toggle transformer model
DISTILBERT_MAX_SAMPLES = 50000   # Subsample for transformer training
DISTILBERT_EPOCHS = 3
DISTILBERT_BATCH_SIZE = 32
DISTILBERT_MAX_LEN = 128

ENABLE_SMOTE = True              # Enable SMOTE experiments
ENABLE_CALIBRATION = True        # Enable calibration analysis

# ──────────────────────────────────────────────
#  Figure settings
# ──────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMATS = ["pdf", "png", "eps"]  # Save in three formats
FIGURE_STYLE = "whitegrid"

# Publication-quality color palette
MODEL_COLORS = {
    "logistic_regression": "#2196F3",
    "svm": "#FF5722",
    "naive_bayes": "#4CAF50",
    "random_forest": "#9C27B0",
    "xgboost": "#FF9800",
    "tfidf_lr": "#3F51B5",
    "distilbert": "#E91E63",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (LinearSVC)",
    "naive_bayes": "Multinomial NB",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "tfidf_lr": "TF-IDF+LR",
    "distilbert": "DistilBERT",
}


# ──────────────────────────────────────────────
#  Directory management helpers
# ──────────────────────────────────────────────
def get_model_dir(model_name: str) -> Path:
    """Return the root directory for a specific model."""
    return RESULTS_ROOT / "models" / model_name


def get_model_metrics_dir(model_name: str) -> Path:
    d = get_model_dir(model_name) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_model_figures_dir(model_name: str) -> Path:
    d = get_model_dir(model_name) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_model_artifacts_dir(model_name: str) -> Path:
    d = get_model_dir(model_name) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_all_directories():
    """Create the full output directory tree."""
    for model_name in MODEL_NAMES:
        for subdir in MODEL_SUBDIRS:
            (RESULTS_ROOT / "models" / model_name / subdir).mkdir(parents=True, exist_ok=True)
    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)
    STATISTICAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    SALES_ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[config] Output directory tree created under {RESULTS_ROOT.resolve()}")


def save_run_config(output_path: Path = None):
    """Save the current configuration to a JSON file for reproducibility."""
    cfg = {
        "random_seed": RANDOM_SEED,
        "cv_folds": CV_FOLDS,
        "test_size": TEST_SIZE,
        "max_tfidf_features": MAX_TFIDF_FEATURES,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
        "sentiment_runs": SENTIMENT_RUNS,
        "topn_skus": TOPN_SKUS,
        "enable_distilbert": ENABLE_DISTILBERT,
        "prefer_gpu": PREFER_GPU,
        "enable_smote": ENABLE_SMOTE,
        "enable_calibration": ENABLE_CALIBRATION,
        "figure_dpi": FIGURE_DPI,
        "figure_formats": FIGURE_FORMATS,
    }
    out = output_path or (RESULTS_ROOT / "run_config.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] Run configuration saved to {out}")
