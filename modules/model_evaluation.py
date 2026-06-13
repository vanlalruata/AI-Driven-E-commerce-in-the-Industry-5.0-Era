#!/usr/bin/env python3
"""
modules/model_evaluation.py
Comprehensive model evaluation: K-Fold CV, ROC-AUC, PR-AUC,
calibration analysis, error analysis, and McNemar's test.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RANDOM_SEED, CV_FOLDS, COMPARATIVE_DIR,
    get_model_metrics_dir, get_model_figures_dir,
    MODEL_DISPLAY_NAMES,
)
from modules.visualization import (
    plot_roc_curves, plot_pr_curves, plot_calibration_curves,
    plot_cv_boxplots, plot_model_comparison_heatmap, plot_error_analysis,
)


# ──────────────────────────────────────────────
#  Stratified K-Fold Cross-Validation
# ──────────────────────────────────────────────
def run_cross_validation(X_text_full, y_full, model_registry, labels, cv_folds=None,
                         max_tfidf_features=None, tfidf_ngram_range=None,
                         reviews_df=None, all_results=None):
    """
    Run stratified K-fold CV for all models.

    IMPORTANT: TF-IDF is fitted INSIDE each fold on training text only,
    then used to transform the test fold. This prevents data leakage —
    the test fold's vocabulary never influences the vectorizer.

    X_text_full: raw text Series/array (NOT pre-transformed TF-IDF).
    y_full: corresponding labels.
    Returns dict of {model_name: {metric: list_of_fold_scores}}.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from config import MAX_TFIDF_FEATURES, TFIDF_NGRAM_RANGE, PREFER_GPU

    n_folds = cv_folds or CV_FOLDS
    max_feat = max_tfidf_features or MAX_TFIDF_FEATURES
    ngram_rng = tfidf_ngram_range or TFIDF_NGRAM_RANGE

    # Detect GPU availability for models that support it (XGBoost)
    use_gpu = False
    if PREFER_GPU:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass
    gpu_status = "CUDA GPU" if use_gpu else "CPU"
    print(f"[cv] Device: {gpu_status}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    X_arr = np.array(X_text_full)
    y_arr = np.array(y_full)

    print(f"\n{'='*60}")
    print(f"STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
    print(f"(TF-IDF fitted per fold — no data leakage)")
    print(f"{'='*60}")

    all_cv_results = {}

    # Build models to run
    models_to_run = list(model_registry.keys())
    run_db_cv = False
    if all_results is not None and "distilbert" in all_results:
        run_db_cv = True
    else:
        try:
            from config import ENABLE_DISTILBERT
            if ENABLE_DISTILBERT:
                run_db_cv = True
        except ImportError:
            pass

    if run_db_cv and "distilbert" not in models_to_run:
        models_to_run.append("distilbert")

    for model_name in models_to_run:
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        metrics_dir = get_model_metrics_dir(model_name)
        cv_cache_path = metrics_dir / "cv_fold_scores.csv"
        cv_summary_path = metrics_dir / "cv_summary.csv"

        if model_name == "distilbert":
            # ── Resume logic: check for completed or partial CV results ──
            fold_metrics = {
                "accuracy": [], "precision_macro": [], "recall_macro": [],
                "f1_macro": [], "negative_recall": [], "negative_f1": [],
            }
            start_fold = 0
            if cv_cache_path.exists():
                try:
                    cached_df = pd.read_csv(cv_cache_path, index_col=0)
                    if len(cached_df) == n_folds:
                        print(f"\n[cv] {display}: CV results already exist — loading details:")
                        print(f"  [Scores File] {cv_cache_path}")
                        
                        summary = {}
                        for metric in cached_df.columns:
                            values = cached_df[metric].tolist()
                            arr = np.array(values)
                            summary[metric] = {
                                "mean": float(arr.mean()),
                                "std": float(arr.std()),
                                "values": values,
                            }
                        
                        # Generate summary CSV if missing
                        if not cv_summary_path.exists():
                            stats_rows = []
                            for metric, s in summary.items():
                                stats_rows.append({"metric": metric, "mean": s["mean"], "std": s["std"]})
                            pd.DataFrame(stats_rows).to_csv(cv_summary_path, index=False)
                            print(f"  [Summary File] {cv_summary_path} (regenerated)")
                        else:
                            print(f"  [Summary File] {cv_summary_path}")
                        
                        print("\n  Summary Metrics (mean ± std):")
                        for metric, s in summary.items():
                            print(f"    {metric}: {s['mean']:.4f} ± {s['std']:.4f}")
                        
                        print("\n  Fold-by-Fold Detailed Scores:")
                        print(cached_df.to_string())
                        
                        all_cv_results[model_name] = summary
                        continue
                    else:
                        start_fold = len(cached_df)
                        for col in fold_metrics.keys():
                            fold_metrics[col] = cached_df[col].tolist()
                        print(f"\n[cv] {display}: Found partial CV results on disk ({start_fold}/{n_folds} folds). Resuming from fold {start_fold + 1}...")
                except Exception as e:
                    print(f"  [cv] Error loading cached CV scores: {e}. Starting from scratch.")

            print(f"\n[cv] Running {n_folds}-fold CV for {display}...")
            if reviews_df is None:
                print("  [cv] Skipping DistilBERT CV — reviews_df not provided")
                continue

            # Load configuration parameters
            from config import DISTILBERT_MAX_SAMPLES, DISTILBERT_EPOCHS, DISTILBERT_BATCH_SIZE, DISTILBERT_MAX_LEN
            import torch
            from torch.utils.data import Dataset, DataLoader
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            from torch.optim import AdamW
            from transformers import get_linear_schedule_with_warmup

            # Use more tractible limits for cross-validation on resource-constrained setups
            cv_max_samples = min(15000, DISTILBERT_MAX_SAMPLES)
            cv_epochs = 1

            # Prepare subsampled dataset
            db_df = reviews_df.copy()
            if len(db_df) > cv_max_samples:
                db_df = db_df.sample(n=cv_max_samples, random_state=RANDOM_SEED).reset_index(drop=True)

            label_map = {"negative": 0, "positive": 1}
            db_df["label_enc"] = db_df["sent_label"].map(label_map)
            db_df = db_df.dropna(subset=["label_enc"])

            db_texts = db_df["text_clean"].values
            db_labels = db_df["label_enc"].values.astype(int)

            class CVDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_len):
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_len = max_len
                def __len__(self):
                    return len(self.texts)
                def __getitem__(self, idx):
                    encoding = self.tokenizer(
                        str(self.texts[idx]),
                        truncation=True, padding="max_length",
                        max_length=self.max_len, return_tensors="pt",
                    )
                    return {
                        "input_ids": encoding["input_ids"].squeeze(),
                        "attention_mask": encoding["attention_mask"].squeeze(),
                        "label": torch.tensor(self.labels[idx], dtype=torch.long),
                    }

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  [cv] DistilBERT Device: {device}")

            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            db_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

            for fold_idx, (train_idx, test_idx) in enumerate(db_skf.split(db_texts, db_labels)):
                if fold_idx < start_fold:
                    continue

                print(f"  [cv] DistilBERT Fold {fold_idx+1}/{n_folds} training...")
                X_tr_fold = db_texts[train_idx]
                X_te_fold = db_texts[test_idx]
                y_tr_fold = db_labels[train_idx]
                y_te_fold = db_labels[test_idx]

                train_ds = CVDataset(X_tr_fold, y_tr_fold, tokenizer, DISTILBERT_MAX_LEN)
                test_ds = CVDataset(X_te_fold, y_te_fold, tokenizer, DISTILBERT_MAX_LEN)
                train_loader = DataLoader(train_ds, batch_size=DISTILBERT_BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=DISTILBERT_BATCH_SIZE)

                model = DistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=2
                ).to(device)

                optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
                total_steps = len(train_loader) * cv_epochs
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
                )

                # Train
                for epoch in range(cv_epochs):
                    model.train()
                    for batch_idx, batch in enumerate(train_loader):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels_t = batch["label"].to(device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_t)
                        loss = outputs.loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                # Evaluate
                model.eval()
                all_preds, all_labels_eval = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels_t = batch["label"].to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                        all_labels_eval.extend(labels_t.cpu().numpy())

                all_preds = np.array(all_preds)
                all_labels_eval = np.array(all_labels_eval)

                inv_map = {0: "negative", 1: "positive"}
                preds_labels = [inv_map[p] for p in all_preds]
                true_labels = [inv_map[l] for l in all_labels_eval]

                acc = accuracy_score(true_labels, preds_labels)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    true_labels, preds_labels, average="macro", zero_division=0
                )
                neg_p, neg_r, neg_f, neg_s = precision_recall_fscore_support(
                    true_labels, preds_labels, labels=["negative"], average=None, zero_division=0
                )

                fold_metrics["accuracy"].append(acc)
                fold_metrics["precision_macro"].append(prec)
                fold_metrics["recall_macro"].append(rec)
                fold_metrics["f1_macro"].append(f1)
                fold_metrics["negative_recall"].append(float(neg_r[0]) if len(neg_r) > 0 else 0.0)
                fold_metrics["negative_f1"].append(float(neg_f[0]) if len(neg_f) > 0 else 0.0)

                print(f"    Fold {fold_idx+1}/{n_folds} — Acc: {acc:.4f} | F1: {f1:.4f}")

                # Save intermediate progress to cache
                try:
                    fold_df = pd.DataFrame(fold_metrics)
                    fold_df.index.name = "fold"
                    fold_df.to_csv(cv_cache_path)
                except Exception as e:
                    print(f"    [cv] Warning: Could not save intermediate CV cache: {e}")

            # Summarize metrics
            summary = {}
            for metric, values in fold_metrics.items():
                arr = np.array(values)
                summary[metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "values": values,
                }
                print(f"  {metric}: {arr.mean():.4f} ± {arr.std():.4f}")

            all_cv_results[model_name] = summary

            stats_rows = []
            for metric, s in summary.items():
                stats_rows.append({"metric": metric, "mean": s["mean"], "std": s["std"]})
            pd.DataFrame(stats_rows).to_csv(metrics_dir / "cv_summary.csv", index=False)
            continue


        # ── Resume logic: skip models with existing CV results ──
        if cv_cache_path.exists():
            try:
                cached_df = pd.read_csv(cv_cache_path, index_col=0)
                if len(cached_df) == n_folds:
                    print(f"\n[cv] {display}: CV results already exist — loading details:")
                    print(f"  [Scores File] {cv_cache_path}")
                    
                    summary = {}
                    for metric in cached_df.columns:
                        values = cached_df[metric].tolist()
                        arr = np.array(values)
                        summary[metric] = {
                            "mean": float(arr.mean()),
                            "std": float(arr.std()),
                            "values": values,
                        }
                    
                    # Generate summary CSV if missing
                    if not cv_summary_path.exists():
                        stats_rows = []
                        for metric, s in summary.items():
                            stats_rows.append({"metric": metric, "mean": s["mean"], "std": s["std"]})
                        pd.DataFrame(stats_rows).to_csv(cv_summary_path, index=False)
                        print(f"  [Summary File] {cv_summary_path} (regenerated)")
                    else:
                        print(f"  [Summary File] {cv_summary_path}")
                    
                    print("\n  Summary Metrics (mean ± std):")
                    for metric, s in summary.items():
                        print(f"    {metric}: {s['mean']:.4f} ± {s['std']:.4f}")
                    
                    print("\n  Fold-by-Fold Detailed Scores:")
                    print(cached_df.to_string())
                    
                    all_cv_results[model_name] = summary
                    continue
            except Exception as e:
                print(f"  [cv] Error loading cached CV scores for {display}: {e}. Re-running CV.")

        model_template = model_registry[model_name][0]
        print(f"\n[cv] Running {n_folds}-fold CV for {display}...")

        fold_metrics = {
            "accuracy": [], "precision_macro": [], "recall_macro": [],
            "f1_macro": [], "negative_recall": [], "negative_f1": [],
        }

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr_text = X_arr[train_idx]
            X_te_text = X_arr[test_idx]
            y_tr = y_arr[train_idx]
            y_te = y_arr[test_idx]

            # Fit TF-IDF on this fold's training data ONLY
            fold_vect = TfidfVectorizer(ngram_range=ngram_rng, max_features=max_feat, sublinear_tf=True)
            X_tr = fold_vect.fit_transform(X_tr_text)
            X_te = fold_vect.transform(X_te_text)

            # Clone model for each fold
            from sklearn.base import clone
            model = clone(model_template)

            # XGBoost needs label encoding
            if model_name == "xgboost":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_tr_enc = le.fit_transform(y_tr)
                y_te_enc = le.transform(y_te)

                if use_gpu:
                    # Use XGBoost DMatrix for GPU training with sparse data.
                    # DMatrix natively supports sparse CSR on GPU without
                    # converting to dense (which would require ~176 GiB).
                    import xgboost as xgb
                    try:
                        dtrain = xgb.DMatrix(X_tr, label=y_tr_enc)
                        dtest = xgb.DMatrix(X_te)
                        params = {
                            "max_depth": model_template.get_params()["max_depth"],
                            "learning_rate": model_template.get_params()["learning_rate"],
                            "n_estimators": model_template.get_params()["n_estimators"],
                            "objective": "binary:logistic",
                            "eval_metric": "logloss",
                            "device": "cuda",
                            "seed": RANDOM_SEED,
                        }
                        n_rounds = params.pop("n_estimators")
                        bst = xgb.train(params, dtrain, num_boost_round=n_rounds)
                        raw_preds = bst.predict(dtest)
                        preds_enc = (raw_preds > 0.5).astype(int)
                    except Exception as e:
                        print(f"  [cv] GPU DMatrix failed ({e}), falling back to CPU...")
                        model.fit(X_tr, y_tr_enc)
                        preds_enc = model.predict(X_te)
                else:
                    model.fit(X_tr, y_tr_enc)
                    preds_enc = model.predict(X_te)

                preds = le.inverse_transform(preds_enc)
            else:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)

            acc = accuracy_score(y_te, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_te, preds, average="macro", zero_division=0
            )

            # Negative class recall/f1
            neg_p, neg_r, neg_f, neg_s = precision_recall_fscore_support(
                y_te, preds, labels=["negative"], average=None, zero_division=0
            )

            fold_metrics["accuracy"].append(acc)
            fold_metrics["precision_macro"].append(prec)
            fold_metrics["recall_macro"].append(rec)
            fold_metrics["f1_macro"].append(f1)
            fold_metrics["negative_recall"].append(float(neg_r[0]) if len(neg_r) > 0 else 0.0)
            fold_metrics["negative_f1"].append(float(neg_f[0]) if len(neg_f) > 0 else 0.0)

            print(f"  Fold {fold_idx+1}/{n_folds} — Acc: {acc:.4f} | F1: {f1:.4f}")

        # Compute mean ± std
        summary = {}
        for metric, values in fold_metrics.items():
            arr = np.array(values)
            summary[metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": values,
            }
            print(f"  {metric}: {arr.mean():.4f} ± {arr.std():.4f}")

        all_cv_results[model_name] = summary

        # Save per-model CV results (enables resume on next run)
        fold_df = pd.DataFrame(fold_metrics)
        fold_df.index.name = "fold"
        fold_df.to_csv(cv_cache_path)

        stats_rows = []
        for metric, s in summary.items():
            stats_rows.append({"metric": metric, "mean": s["mean"], "std": s["std"]})
        pd.DataFrame(stats_rows).to_csv(metrics_dir / "cv_summary.csv", index=False)

    # Save combined CV comparison
    _save_cv_comparison(all_cv_results)

    return all_cv_results


def _save_cv_comparison(all_cv_results):
    """Save and plot cross-model CV comparison."""
    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, summary in all_cv_results.items():
        row = {"model": MODEL_DISPLAY_NAMES.get(model_name, model_name)}
        for metric, data in summary.items():
            row[f"{metric}_mean"] = data["mean"]
            row[f"{metric}_std"] = data["std"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(COMPARATIVE_DIR / "cv_comparison_table.csv")
    print(f"\n[cv] CV comparison saved to {COMPARATIVE_DIR / 'cv_comparison_table.csv'}")

    # Plot box plots for key metrics
    for metric in ["f1_macro", "accuracy", "negative_recall"]:
        cv_scores = {}
        for model_name, summary in all_cv_results.items():
            if metric in summary:
                cv_scores[model_name] = summary[metric]["values"]
        if cv_scores:
            plot_cv_boxplots(cv_scores, COMPARATIVE_DIR, metric_name=metric,
                             filename=f"cv_boxplot_{metric}")


# ──────────────────────────────────────────────
#  ROC-AUC Analysis
# ──────────────────────────────────────────────
def compute_roc_auc(all_results, positive_label="positive"):
    """
    Compute ROC curves and AUC for all models.
    Returns dict for overlay plotting.
    """
    print(f"\n{'='*60}")
    print("ROC-AUC ANALYSIS")
    print(f"{'='*60}")

    roc_data = {}

    for model_name, result in all_results.items():
        if result.get("prob_dict") is None:
            print(f"  [roc] Skipping {model_name} — no probability estimates")
            continue

        y_test = result["y_test"]
        y_binary = (np.array(y_test) == positive_label).astype(int)

        if positive_label in result["prob_dict"]:
            y_scores = result["prob_dict"][positive_label]
        else:
            print(f"  [roc] Skipping {model_name} — positive class not in prob_dict")
            continue

        fpr, tpr, _ = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)

        roc_data[model_name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

        # Save per-model ROC data
        metrics_dir = get_model_metrics_dir(model_name)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(metrics_dir / "roc_curve_data.csv", index=False)
        with open(metrics_dir / "roc_auc.csv", "w") as f:
            f.write(f"model,roc_auc\n{model_name},{roc_auc:.6f}\n")

        print(f"  {MODEL_DISPLAY_NAMES.get(model_name, model_name)}: ROC-AUC = {roc_auc:.4f}")

    # Plot overlay
    if roc_data:
        COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)
        plot_roc_curves(roc_data, COMPARATIVE_DIR)

    return roc_data


# ──────────────────────────────────────────────
#  PR-AUC Analysis
# ──────────────────────────────────────────────
def compute_pr_auc(all_results, positive_label="positive"):
    """
    Compute Precision-Recall curves and AUPRC for all models.
    """
    print(f"\n{'='*60}")
    print("PRECISION-RECALL AUC ANALYSIS")
    print(f"{'='*60}")

    pr_data = {}

    for model_name, result in all_results.items():
        if result.get("prob_dict") is None:
            continue

        y_test = result["y_test"]
        y_binary = (np.array(y_test) == positive_label).astype(int)

        if positive_label in result["prob_dict"]:
            y_scores = result["prob_dict"][positive_label]
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_binary, y_scores)
        auprc = average_precision_score(y_binary, y_scores)

        pr_data[model_name] = {"precision": precision, "recall": recall, "auprc": auprc}

        metrics_dir = get_model_metrics_dir(model_name)
        pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
            metrics_dir / "pr_curve_data.csv", index=False
        )
        with open(metrics_dir / "pr_auc.csv", "w") as f:
            f.write(f"model,pr_auc\n{model_name},{auprc:.6f}\n")

        print(f"  {MODEL_DISPLAY_NAMES.get(model_name, model_name)}: PR-AUC = {auprc:.4f}")

    if pr_data:
        COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)
        plot_pr_curves(pr_data, COMPARATIVE_DIR)

    return pr_data


# ──────────────────────────────────────────────
#  Calibration Analysis
# ──────────────────────────────────────────────
def compute_calibration(all_results, positive_label="positive", n_bins=10):
    """
    Compute calibration curves and Brier scores for all models.
    """
    print(f"\n{'='*60}")
    print("CALIBRATION ANALYSIS")
    print(f"{'='*60}")

    cal_data = {}

    for model_name, result in all_results.items():
        if result.get("prob_dict") is None:
            continue

        y_test = result["y_test"]
        y_binary = (np.array(y_test) == positive_label).astype(int)

        if positive_label in result["prob_dict"]:
            y_scores = result["prob_dict"][positive_label]
        else:
            continue

        prob_true, prob_pred = calibration_curve(y_binary, y_scores, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_binary, y_scores)

        cal_data[model_name] = {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "brier": brier,
        }

        metrics_dir = get_model_metrics_dir(model_name)
        pd.DataFrame({"prob_true": prob_true, "prob_pred": prob_pred}).to_csv(
            metrics_dir / "calibration_data.csv", index=False
        )
        with open(metrics_dir / "brier_score.csv", "w") as f:
            f.write(f"model,brier_score\n{model_name},{brier:.6f}\n")

        print(f"  {MODEL_DISPLAY_NAMES.get(model_name, model_name)}: Brier = {brier:.4f}")

    if cal_data:
        COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)
        plot_calibration_curves(cal_data, COMPARATIVE_DIR)

    return cal_data


# ──────────────────────────────────────────────
#  Error Analysis
# ──────────────────────────────────────────────
def run_error_analysis(all_results, reviews_df):
    """
    Analyze misclassified samples for each model.
    """
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS")
    print(f"{'='*60}")

    for model_name, result in all_results.items():
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        print(f"\n[error] Analyzing errors for {display}...")

        y_test = np.array(result["y_test"])
        preds = np.array(result["predictions"])
        X_test = result.get("X_test")

        if X_test is None:
            print(f"  Skipping — no test text data stored")
            continue

        # Build error analysis DataFrame
        error_df = pd.DataFrame({
            "text_clean": X_test.values if hasattr(X_test, "values") else X_test,
            "true_label": y_test,
            "pred_label": preds,
        })
        error_df["is_correct"] = error_df["true_label"] == error_df["pred_label"]
        error_df["text_length"] = error_df["text_clean"].astype(str).str.len()

        # Try to attach original Score
        if "Score" in reviews_df.columns:
            # Match by index from X_test if possible
            test_indices = X_test.index if hasattr(X_test, "index") else None
            if test_indices is not None:
                error_df["Score"] = reviews_df.loc[test_indices, "Score"].values

        total = len(error_df)
        errors = error_df[~error_df["is_correct"]]
        n_errors = len(errors)
        print(f"  Total test: {total} | Misclassified: {n_errors} ({n_errors/total*100:.1f}%)")

        # Error patterns
        if not errors.empty:
            pattern = errors.groupby(["true_label", "pred_label"]).size().reset_index(name="count")
            pattern = pattern.sort_values("count", ascending=False)
            print(f"  Error patterns:\n{pattern.to_string(index=False)}")

            # Length statistics
            correct_len = error_df[error_df["is_correct"]]["text_length"].mean()
            error_len = errors["text_length"].mean()
            print(f"  Avg length — Correct: {correct_len:.0f} chars | Errors: {error_len:.0f} chars")

        # Save
        figures_dir = get_model_figures_dir(model_name)
        metrics_dir = get_model_metrics_dir(model_name)

        errors.to_csv(metrics_dir / "misclassified_samples.csv", index=False)
        plot_error_analysis(error_df, figures_dir, model_name)

        # Save summary stats
        summary = {
            "total_test": total,
            "total_errors": n_errors,
            "error_rate": n_errors / total,
            "avg_correct_length": float(error_df[error_df["is_correct"]]["text_length"].mean()),
            "avg_error_length": float(errors["text_length"].mean()) if not errors.empty else 0,
        }
        with open(metrics_dir / "error_analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


# ──────────────────────────────────────────────
#  McNemar's Test (model comparison)
# ──────────────────────────────────────────────
def mcnemar_test(all_results):
    """
    Perform McNemar's test between all pairs of models.
    Returns a DataFrame of p-values.
    """
    from scipy.stats import chi2

    print(f"\n{'='*60}")
    print("McNEMAR'S TEST — PAIRWISE MODEL COMPARISON")
    print(f"{'='*60}")

    model_names = list(all_results.keys())
    n = len(model_names)
    pvalue_matrix = pd.DataFrame(np.ones((n, n)), index=model_names, columns=model_names)

    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = model_names[i], model_names[j]
            y_test1 = np.array(all_results[m1]["y_test"])
            y_test2 = np.array(all_results[m2]["y_test"])
            pred1 = np.array(all_results[m1]["predictions"])
            pred2 = np.array(all_results[m2]["predictions"])

            # Verify shapes and check if the test sets are identical
            if len(pred1) != len(pred2) or not np.array_equal(y_test1, y_test2):
                display1 = MODEL_DISPLAY_NAMES.get(m1, m1)
                display2 = MODEL_DISPLAY_NAMES.get(m2, m2)
                print(f"  [warning] Skipping McNemar's test for {display1} vs {display2} (different test sets/sizes).")
                pvalue_matrix.loc[m1, m2] = np.nan
                pvalue_matrix.loc[m2, m1] = np.nan
                continue

            correct1 = pred1 == y_test1
            correct2 = pred2 == y_test1

            # Contingency: b = m1 correct & m2 wrong, c = m1 wrong & m2 correct
            b = np.sum(correct1 & ~correct2)
            c = np.sum(~correct1 & correct2)

            if b + c == 0:
                p_value = 1.0
            else:
                # McNemar's chi-squared with continuity correction
                chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
                p_value = 1 - chi2.cdf(chi2_stat, df=1)

            pvalue_matrix.loc[m1, m2] = p_value
            pvalue_matrix.loc[m2, m1] = p_value

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            display1 = MODEL_DISPLAY_NAMES.get(m1, m1)
            display2 = MODEL_DISPLAY_NAMES.get(m2, m2)
            print(f"  {display1} vs {display2}: p = {p_value:.6f} ({sig})")

    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)
    pvalue_matrix.to_csv(COMPARATIVE_DIR / "mcnemar_pvalues.csv")
    print(f"\n[mcnemar] p-value matrix saved to {COMPARATIVE_DIR / 'mcnemar_pvalues.csv'}")
    return pvalue_matrix


# ──────────────────────────────────────────────
#  Full evaluation pipeline
# ──────────────────────────────────────────────
def run_full_evaluation(all_results, reviews_df, X_text_full, y_full, model_registry, labels):
    """
    Run the complete evaluation suite:
    1. K-fold Cross-Validation (TF-IDF fitted per fold — no data leakage)
    2. ROC-AUC
    3. PR-AUC
    4. Calibration Analysis
    5. Error Analysis
    6. McNemar's Test
    7. Combined comparison heatmap
    """
    # 1. Cross-validation (receives raw text, fits TF-IDF per fold)
    cv_results = run_cross_validation(
        X_text_full, y_full, model_registry, labels,
        reviews_df=reviews_df, all_results=all_results
    )

    # 2. ROC-AUC
    roc_data = compute_roc_auc(all_results)

    # 3. PR-AUC
    pr_data = compute_pr_auc(all_results)

    # 4. Calibration
    cal_data = compute_calibration(all_results)

    # 5. Error analysis
    run_error_analysis(all_results, reviews_df)

    # 6. McNemar's test
    pvalue_matrix = mcnemar_test(all_results)

    # 7. Build comprehensive comparison
    _build_comprehensive_comparison(all_results, cv_results, roc_data, pr_data, cal_data)

    return {
        "cv_results": cv_results,
        "roc_data": roc_data,
        "pr_data": pr_data,
        "cal_data": cal_data,
        "mcnemar_pvalues": pvalue_matrix,
    }


def _build_comprehensive_comparison(all_results, cv_results, roc_data, pr_data, cal_data):
    """Build a single comprehensive comparison table across all evaluation methods."""
    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in all_results.keys():
        row = {"model": MODEL_DISPLAY_NAMES.get(model_name, model_name)}

        # Holdout metrics
        res = all_results[model_name]
        row["holdout_accuracy"] = res["accuracy"]
        row["holdout_f1_macro"] = res["f1_macro"]
        row["holdout_neg_recall"] = res["per_class"].get("negative", {}).get("recall", None)

        # CV metrics
        if model_name in cv_results:
            cv = cv_results[model_name]
            row["cv_accuracy_mean"] = cv["accuracy"]["mean"]
            row["cv_accuracy_std"] = cv["accuracy"]["std"]
            row["cv_f1_macro_mean"] = cv["f1_macro"]["mean"]
            row["cv_f1_macro_std"] = cv["f1_macro"]["std"]
            row["cv_neg_recall_mean"] = cv["negative_recall"]["mean"]
            row["cv_neg_recall_std"] = cv["negative_recall"]["std"]

        # ROC-AUC
        if model_name in roc_data:
            row["roc_auc"] = roc_data[model_name]["auc"]

        # PR-AUC
        if model_name in pr_data:
            row["pr_auc"] = pr_data[model_name]["auprc"]

        # Brier score
        if model_name in cal_data:
            row["brier_score"] = cal_data[model_name]["brier"]

        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(COMPARATIVE_DIR / "comprehensive_comparison.csv")

    # Heatmap of numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, float]]
    if numeric_cols:
        plot_model_comparison_heatmap(
            df[numeric_cols].astype(float),
            COMPARATIVE_DIR, filename="comprehensive_comparison_heatmap",
        )

    print(f"\n[evaluation] Comprehensive comparison saved")
    print(df.to_string())
