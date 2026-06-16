#!/usr/bin/env python3
"""
main.py
Menu-driven entry point for the Industry 5.0 E-Commerce Sentiment Analysis Pipeline.
Provides interactive CLI to run individual analysis components or the full pipeline.
Each component stores results in structured directories under results/.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    DEFAULT_REVIEWS_PATH, DEFAULT_SALES_FOLDER,
    RESULTS_ROOT, COMPARATIVE_DIR,
    ENABLE_DISTILBERT, RANDOM_SEED, TEST_SIZE,
    create_all_directories, save_run_config,
)


# ──────────────────────────────────────────────
#  State holder — caches loaded data across menu runs
# ──────────────────────────────────────────────
class PipelineState:
    """Holds shared state between menu operations."""
    def __init__(self):
        self.reviews_path = None
        self.sales_folder = None
        self.reviews_raw = None
        self.reviews_pp = None
        self.sales_df = None
        self.all_model_results = None
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.labels = None
        self.model_registry = None
        self.eval_results = None

state = PipelineState()


# ──────────────────────────────────────────────
#  Menu display
# ──────────────────────────────────────────────
def show_menu():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║       Industry 5.0 E-Commerce Sentiment Analysis Pipeline         ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   1.  Data Loading & Preprocessing                                ║
║   2.  Data Quality Audit (ASIN Matching & Consistency)            ║
║   3.  Sentiment Model Training (All Classical Models)             ║
║   4.  Sentiment Model Training (TF-IDF+LR)                        ║
║   5.  Sentiment Model Training (DistilBERT — Optional)            ║
║   6.  Model Evaluation & Comparison (CV, ROC, Calibration)        ║
║   7.  Class Imbalance Analysis (SMOTE, Weighting experiments)     ║
║   8.  Sales Analytics (Profitability, MRP, Top SKUs, Regions)     ║
║   9.  Statistical Tests (Gini, HHI, ANOVA, t-tests)               ║
║  10.  Integration: Reviews ↔ Sales Linkage                        ║
║  11.  Generate Comparative Model Report                           ║
║  12.  Run Full Pipeline (All of the above)                        ║
║   0.  Exit                                                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")


# ──────────────────────────────────────────────
#  Menu handlers
# ──────────────────────────────────────────────
def menu_1_load_data():
    """Load and preprocess both datasets."""
    from modules.data_loader import load_reviews, preprocess_reviews, load_and_standardize_sales, summarize_datasets

    print("\n[1] DATA LOADING & PREPROCESSING")
    state.reviews_raw = load_reviews(state.reviews_path)
    state.reviews_pp = preprocess_reviews(state.reviews_raw, save_csv=True, output_dir=RESULTS_ROOT)
    state.sales_df = load_and_standardize_sales(state.sales_folder, save_csv=True)
    summarize_datasets(state.reviews_pp, state.sales_df)
    print("[1] ✓ Data loading complete.\n")


def menu_2_data_audit():
    """Run ASIN matching and data consistency audit."""
    _ensure_data_loaded()
    from modules.data_audit import run_full_data_audit

    print("\n[2] DATA QUALITY AUDIT")
    audit = run_full_data_audit(state.reviews_pp, state.sales_df)
    print("[2] ✓ Data audit complete.\n")
    return audit


def menu_3_train_models():
    """Train all classical ML models."""
    _ensure_data_loaded()

    # Check if all classical models are already trained
    classical_names = ["logistic_regression", "svm", "naive_bayes", "random_forest", "xgboost"]
    all_trained = True
    for mname in classical_names:
        m_dir = RESULTS_ROOT / "models" / mname / "artifacts"
        if not (m_dir / "model.joblib").exists() or not (m_dir / "training_config.json").exists():
            all_trained = False
            break

    if all_trained:
        print("\n[3] SENTIMENT MODEL TRAINING (Classical)")
        print("[3] ✓ All classical models loaded from disk (no retraining needed).")
        _ensure_models_trained()
        print("[3] ✓ Figures regenerated.")
        return state.all_model_results

    from modules.sentiment_models import train_all_models

    print("\n[3] SENTIMENT MODEL TRAINING (Classical)")
    results = train_all_models(state.reviews_pp, save_results=True)
    (state.all_model_results, state.vectorizer,
     state.X_train, state.X_test, state.y_train, state.y_test,
     state.X_train_tfidf, state.X_test_tfidf, state.labels) = results

    from modules.sentiment_models import get_model_registry
    state.model_registry = get_model_registry()

    print("[3] ✓ All classical models trained.\n")
    return state.all_model_results


def menu_4_train_tfidf_lr():
    """Train proposed TF-IDF+LR model (unbalanced)."""
    _ensure_data_loaded()
    from modules.sentiment_models import load_tfidf_lr, train_tfidf_lr

    print("\n[4] SENTIMENT MODEL TRAINING (TF-IDF+LR)")

    # Try loading a previously trained model first
    result = load_tfidf_lr(state.reviews_pp)
    if result is not None:
        print("[4] ✓ TF-IDF+LR loaded from disk (no retraining needed).")
        # Ensure state variables are populated
        if state.vectorizer is None:
            import joblib
            from config import get_model_artifacts_dir
            state.vectorizer = joblib.load(get_model_artifacts_dir("tfidf_lr") / "vectorizer.joblib")
            
            from sklearn.model_selection import train_test_split
            from config import TEST_SIZE, RANDOM_SEED
            X = state.reviews_pp["text_clean"]
            y = state.reviews_pp["sent_label"]
            state.labels = sorted(y.unique().tolist())
            state.X_train, state.X_test, state.y_train, state.y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
            )
            state.X_train_tfidf = state.vectorizer.transform(state.X_train)
            state.X_test_tfidf = state.vectorizer.transform(state.X_test)
            from modules.sentiment_models import get_model_registry
            state.model_registry = get_model_registry()
    else:
        print("[4] No saved TF-IDF+LR model found — training from scratch...")
        result_tuple = train_tfidf_lr(state.reviews_pp, save_results=True)
        result = result_tuple[0]
        (state.vectorizer, state.X_train, state.X_test, state.y_train, state.y_test,
         state.X_train_tfidf, state.X_test_tfidf, state.labels) = result_tuple[1:]
        from modules.sentiment_models import get_model_registry
        state.model_registry = get_model_registry()
        print("[4] ✓ TF-IDF+LR training complete.")

    if result is not None:
        if state.all_model_results is None:
            state.all_model_results = {}
        state.all_model_results["tfidf_lr"] = result
    return result


def menu_5_train_distilbert():
    """Train DistilBERT (optional transformer model)."""
    _ensure_data_loaded()
    from modules.transformer_models import load_distilbert, train_distilbert

    print("\n[5] DISTILBERT TRAINING")

    # Try loading a previously trained model first
    result = load_distilbert(state.reviews_pp)
    if result is not None:
        print("[5] ✓ DistilBERT loaded from disk (no retraining needed).")
    else:
        print("[5] No saved DistilBERT model found — training from scratch...")
        result = train_distilbert(state.reviews_pp, save_results=True)
        print("[5] ✓ DistilBERT training complete.")

    if result is not None:
        if state.all_model_results is None:
            state.all_model_results = {}
        state.all_model_results["distilbert"] = result
    return result


def menu_6_evaluate_models():
    """Run comprehensive model evaluation."""
    _ensure_models_trained()
    from modules.model_evaluation import run_full_evaluation
    from modules.sentiment_models import get_model_registry

    print("\n[6] MODEL EVALUATION & COMPARISON")

    # Pass raw text — TF-IDF will be fitted inside each CV fold
    # to prevent data leakage (test fold vocabulary never seen during fit)
    X_text_full = state.reviews_pp["text_clean"]
    y_full = state.reviews_pp["sent_label"]

    model_registry = get_model_registry()

    state.eval_results = run_full_evaluation(
        state.all_model_results, state.reviews_pp,
        X_text_full, y_full, model_registry, state.labels,
    )
    print("[6] ✓ Model evaluation complete.\n")
    return state.eval_results


def menu_7_imbalance_analysis():
    """Run class imbalance experiments."""
    _ensure_models_trained()
    from modules.class_imbalance import run_imbalance_experiments
    import numpy as np

    print("\n[7] CLASS IMBALANCE ANALYSIS")
    y_train_arr = np.array(state.y_train)
    results = run_imbalance_experiments(
        state.X_train_tfidf, y_train_arr,
        state.X_test_tfidf, np.array(state.y_test),
        state.labels,
    )
    print("[7] ✓ Class imbalance analysis complete.\n")
    return results


def menu_8_sales_analytics():
    """Run sales analytics."""
    _ensure_data_loaded()
    from modules.sales_analytics import run_full_sales_analytics

    print("\n[8] SALES ANALYTICS")
    if state.sales_df is None or state.sales_df.empty:
        print("  No sales data loaded!")
        return
    results = run_full_sales_analytics(state.sales_df)
    print("[8] ✓ Sales analytics complete.\n")
    return results


def menu_9_statistical_tests():
    """Run statistical significance tests."""
    _ensure_data_loaded()
    from modules.statistical_tests import run_all_statistical_tests

    print("\n[9] STATISTICAL SIGNIFICANCE TESTS")
    if state.sales_df is None or state.sales_df.empty:
        print("  No sales data loaded!")
        return
    results = run_all_statistical_tests(state.sales_df)
    print("[9] ✓ Statistical tests complete.\n")
    return results


def menu_10_integration():
    """Run reviews-sales integration."""
    _ensure_data_loaded()

    print("\n[10] REVIEWS ↔ SALES INTEGRATION")
    if state.sales_df is None or "ASIN" not in state.sales_df.columns:
        print("  ASIN column not found in sales data.")
        return None

    from config import SALES_ANALYTICS_DIR
    SALES_ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    map_asin_sku = state.sales_df[["ASIN", "SKU"]].dropna().drop_duplicates()
    agg_rev = (
        state.reviews_pp.groupby("ProductId")
        .agg(review_count=("Id", "count") if "Id" in state.reviews_pp.columns else ("Score", "count"),
             mean_score=("Score", "mean"))
        .reset_index()
        .rename(columns={"ProductId": "ASIN"})
    )
    merged = map_asin_sku.merge(agg_rev, on="ASIN", how="left")
    merged.to_csv(SALES_ANALYTICS_DIR / "asin_sku_sentiment_summary.csv", index=False)
    print(f"  Integrated {len(merged)} ASIN-SKU-sentiment records.")
    print("[10] ✓ Integration complete.\n")
    return merged


def menu_11_comparative_report():
    """Generate the comparative model report."""
    print("\n[11] COMPARATIVE MODEL REPORT")
    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Gather all model metrics
    import pandas as pd
    model_dirs = list((RESULTS_ROOT / "models").iterdir()) if (RESULTS_ROOT / "models").exists() else []

    all_rows = []
    for model_dir in model_dirs:
        config_path = model_dir / "artifacts" / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            row = {
                "model": cfg.get("model_name", model_dir.name),
                "accuracy": cfg.get("accuracy"),
                "precision_macro": cfg.get("precision_macro"),
                "recall_macro": cfg.get("recall_macro"),
                "f1_macro": cfg.get("f1_macro"),
            }
            per_class = cfg.get("per_class", {})
            for cls, metrics in per_class.items():
                for m, v in metrics.items():
                    row[f"{cls}_{m}"] = v
            all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows).set_index("model")
        df.to_csv(COMPARATIVE_DIR / "final_model_comparison.csv")
        print(f"  Final comparison table:\n{df.to_string()}")

        # Also check for CV results
        cv_path = COMPARATIVE_DIR / "cv_comparison_table.csv"
        if cv_path.exists():
            cv_df = pd.read_csv(cv_path, index_col=0)
            print(f"\n  Cross-validation comparison:\n{cv_df.to_string()}")

        # Check for imbalance results
        imb_path = COMPARATIVE_DIR / "imbalance_experiment_table.csv"
        if imb_path.exists():
            imb_df = pd.read_csv(imb_path)
            print(f"\n  Imbalance experiment results:\n{imb_df.to_string(index=False)}")
    else:
        print("  No model results found. Train models first (option 3).")

    print("[11] ✓ Report generation complete.\n")


def menu_12_full_pipeline():
    """Run the entire pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("RUNNING FULL PIPELINE")
    print("=" * 60 + "\n")

    menu_1_load_data()
    menu_2_data_audit()
    menu_3_train_models()
    menu_4_train_tfidf_lr()

    if ENABLE_DISTILBERT:
        menu_5_train_distilbert()

    menu_6_evaluate_models()
    menu_7_imbalance_analysis()
    menu_8_sales_analytics()
    menu_9_statistical_tests()
    menu_10_integration()
    menu_11_comparative_report()

    # Save pipeline summary
    _save_pipeline_summary()

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print(f"All results saved to: {RESULTS_ROOT.resolve()}")
    print("=" * 60)


# ──────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────
def _ensure_data_loaded():
    """Load data if not already loaded."""
    if state.reviews_pp is None or state.sales_df is None:
        print("[info] Data not yet loaded. Loading now...")
        menu_1_load_data()


def _ensure_models_trained():
    """
    Load models from disk if previously trained, or train fresh.
    This prevents unnecessary retraining when the program restarts.
    """
    _ensure_data_loaded()

    if state.all_model_results is not None:
        return  # Already loaded in this session

    # Attempt to load previously saved models from results/models/
    import joblib
    from config import RESULTS_ROOT, get_model_artifacts_dir

    models_root = RESULTS_ROOT / "models"
    if models_root.exists():
        loaded = {}
        loaded_vect = None

        for model_dir in sorted(models_root.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            artifacts_dir = model_dir / "artifacts"
            model_path = artifacts_dir / "model.joblib"
            vect_path = artifacts_dir / "vectorizer.joblib"
            config_path = artifacts_dir / "training_config.json"

            if config_path.exists():
                # DistilBERT: use dedicated loader (PyTorch model, not joblib)
                if model_name == "distilbert":
                    from modules.transformer_models import load_distilbert
                    db_result = load_distilbert(state.reviews_pp)
                    if db_result is not None:
                        loaded[model_name] = db_result
                    else:
                        print(f"[info] DistilBERT config found but model could not be loaded — skipping.")
                    continue



                with open(config_path) as f:
                    cfg = json.load(f)

                result = {
                    "model_name": model_name,
                    "model": None,
                    "accuracy": cfg.get("accuracy"),
                    "precision_macro": cfg.get("precision_macro"),
                    "recall_macro": cfg.get("recall_macro"),
                    "f1_macro": cfg.get("f1_macro"),
                    "per_class": cfg.get("per_class", {}),
                    "predictions": None,
                    "probabilities": None,
                    "prob_dict": None,
                    "classification_report": None,
                    "confusion_matrix": None,
                }

                # Load the model object if available
                if model_path.exists():
                    try:
                        result["model"] = joblib.load(model_path)
                    except Exception as e:
                        print(f"[info] Could not load {model_name} model: {e}")

                # Load the shared TF-IDF vectorizer
                if vect_path.exists() and loaded_vect is None:
                    try:
                        loaded_vect = joblib.load(vect_path)
                    except Exception as e:
                        print(f"[info] Could not load vectorizer: {e}")

                loaded[model_name] = result

        if loaded:
            print(f"[info] Loaded {len(loaded)} previously trained model(s) from disk: {list(loaded.keys())}")
            state.all_model_results = loaded
            state.vectorizer = loaded_vect

            # Reconstruct train/test split and TF-IDF state for Options 5 & 6
            from sklearn.model_selection import train_test_split
            from config import TEST_SIZE, RANDOM_SEED

            X = state.reviews_pp["text_clean"]
            y = state.reviews_pp["sent_label"]
            state.labels = sorted(y.unique().tolist())
            state.X_train, state.X_test, state.y_train, state.y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
            )

            if loaded_vect is not None:
                state.X_train_tfidf = loaded_vect.transform(state.X_train)
                state.X_test_tfidf = loaded_vect.transform(state.X_test)
                print("[info] TF-IDF state reconstructed from saved vectorizer.")
            else:
                # Re-fit the vectorizer if it was not saved
                from modules.sentiment_models import build_tfidf_vectorizer
                state.vectorizer, state.X_train_tfidf = build_tfidf_vectorizer(state.X_train)
                state.X_test_tfidf = state.vectorizer.transform(state.X_test)
                print("[info] TF-IDF vectorizer re-fitted (saved vectorizer not found).")

            # Re-generate predictions for loaded classical models so that
            # error analysis, ROC-AUC, calibration, and McNemar's test work
            from sklearn.metrics import (
                classification_report, confusion_matrix, precision_recall_fscore_support,
            )
            for mname, mresult in loaded.items():
                if mname in ["distilbert"]:
                    continue  # Already fully populated by loaders
                model_obj = mresult.get("model")
                if model_obj is None:
                    print(f"[info] {mname}: no model object loaded — skipping prediction regeneration.")
                    continue

                print(f"[info] Regenerating predictions for {mname}...")
                if mname == "xgboost":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    le.fit(state.y_train)
                    preds_enc = model_obj.predict(state.X_test_tfidf)
                    preds = le.inverse_transform(preds_enc)
                    if hasattr(model_obj, "predict_proba"):
                        probs = model_obj.predict_proba(state.X_test_tfidf)
                        class_order = le.classes_
                        prob_dict = {c: probs[:, i] for i, c in enumerate(class_order)}
                    else:
                        probs, prob_dict = None, None
                else:
                    preds = model_obj.predict(state.X_test_tfidf)
                    if hasattr(model_obj, "predict_proba"):
                        probs = model_obj.predict_proba(state.X_test_tfidf)
                        class_order = model_obj.classes_
                        prob_dict = {c: probs[:, i] for i, c in enumerate(class_order)}
                    else:
                        probs, prob_dict = None, None

                mresult["predictions"] = preds
                mresult["probabilities"] = probs
                mresult["prob_dict"] = prob_dict
                mresult["y_test"] = state.y_test
                mresult["X_test"] = state.X_test
                mresult["classification_report"] = classification_report(
                    state.y_test, preds, output_dict=True, zero_division=0,
                )
                mresult["confusion_matrix"] = confusion_matrix(
                    state.y_test, preds, labels=state.labels,
                )
                # Regenerate confusion matrix plots
                from modules.visualization import plot_confusion_matrix
                from config import get_model_figures_dir
                plot_confusion_matrix(
                    mresult["confusion_matrix"], state.labels,
                    get_model_figures_dir(mname), mname
                )
            print("[info] All model predictions regenerated and confusion matrices plotted.")

            from modules.sentiment_models import get_model_registry
            state.model_registry = get_model_registry()
            return

    # No saved models found — train from scratch
    print("[info] No saved models found. Training now...")
    menu_3_train_models()


def _save_pipeline_summary():
    """Save a comprehensive pipeline summary JSON."""
    summary = {
        "reviews_preprocessed": int(state.reviews_pp.shape[0]) if state.reviews_pp is not None else 0,
        "merged_sales_rows": int(state.sales_df.shape[0]) if state.sales_df is not None else 0,
    }

    if state.all_model_results:
        model_summaries = {}
        for name, res in state.all_model_results.items():
            model_summaries[name] = {
                "accuracy": res.get("accuracy"),
                "f1_macro": res.get("f1_macro"),
                "per_class": res.get("per_class"),
            }
        summary["models"] = model_summaries

    with open(RESULTS_ROOT / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[summary] Pipeline summary saved to {RESULTS_ROOT / 'pipeline_summary.json'}")


# ──────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Industry 5.0 E-Commerce Sentiment Analysis Pipeline",
    )
    parser.add_argument(
        "--reviews", type=str, default=DEFAULT_REVIEWS_PATH,
        help=f"Path to Reviews.csv (default: {DEFAULT_REVIEWS_PATH})",
    )
    parser.add_argument(
        "--sales_folder", type=str, default=DEFAULT_SALES_FOLDER,
        help=f"Path to sales CSV folder (default: {DEFAULT_SALES_FOLDER})",
    )
    parser.add_argument(
        "--run", type=int, default=None,
        help="Run a specific menu option non-interactively (e.g., --run 12 for full pipeline)",
    )
    args = parser.parse_args()

    state.reviews_path = args.reviews
    state.sales_folder = args.sales_folder

    # Create output directory tree
    create_all_directories()
    save_run_config()

    # Non-interactive mode
    if args.run is not None:
        _dispatch(args.run)
        return

    # Interactive menu loop
    while True:
        show_menu()
        try:
            choice = input("Select an option [0-12]: ").strip()
            if not choice:
                continue
            choice = int(choice)
        except (ValueError, EOFError):
            print("Invalid input. Please enter a number 0-12.")
            continue

        if choice == 0:
            print("\nExiting. Goodbye!")
            break

        _dispatch(choice)

        input("\nPress Enter to continue...")


def _dispatch(choice):
    """Route menu choice to handler."""
    handlers = {
        1: menu_1_load_data,
        2: menu_2_data_audit,
        3: menu_3_train_models,
        4: menu_4_train_tfidf_lr,
        5: menu_5_train_distilbert,
        6: menu_6_evaluate_models,
        7: menu_7_imbalance_analysis,
        8: menu_8_sales_analytics,
        9: menu_9_statistical_tests,
        10: menu_10_integration,
        11: menu_11_comparative_report,
        12: menu_12_full_pipeline,
    }
    handler = handlers.get(choice)
    if handler:
        try:
            handler()
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Invalid option: {choice}")


if __name__ == "__main__":
    main()
