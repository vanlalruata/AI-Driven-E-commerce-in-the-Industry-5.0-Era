#!/usr/bin/env python3
"""
modules/transformer_models.py
Optional lightweight transformer model (DistilBERT) for sentiment classification.
This module is only used if ENABLE_DISTILBERT is True in config.py.
Requires: pip install transformers torch
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RANDOM_SEED, TEST_SIZE,
    DISTILBERT_MAX_SAMPLES, DISTILBERT_EPOCHS,
    DISTILBERT_BATCH_SIZE, DISTILBERT_MAX_LEN,
    get_model_metrics_dir, get_model_figures_dir, get_model_artifacts_dir,
    MODEL_DISPLAY_NAMES, PREFER_GPU,
)


def check_transformer_deps():
    """Check if transformers and torch are available."""
    try:
        import transformers
        import torch
        return True
    except ImportError:
        print("[transformer] transformers and/or torch not installed.")
        print("[transformer] Install with: pip install transformers torch")
        return False


def load_distilbert(reviews_df):
    """
    Load a previously trained DistilBERT model from disk.
    Returns the result dict if a saved model is found, or None otherwise.
    """
    artifacts_dir = get_model_artifacts_dir("distilbert")
    model_path = artifacts_dir / "distilbert_model"
    tokenizer_path = artifacts_dir / "distilbert_tokenizer"
    config_path = artifacts_dir / "training_config.json"

    # Check if all required files exist
    if not (model_path.exists() and tokenizer_path.exists() and config_path.exists()):
        return None

    if not check_transformer_deps():
        return None

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix,
    )

    print(f"\n{'='*60}")
    print("LOADING PREVIOUSLY TRAINED DISTILBERT")
    print(f"{'='*60}")

    # Load saved config
    with open(config_path) as f:
        cfg = json.load(f)

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    try:
        model = DistilBertForSequenceClassification.from_pretrained(str(model_path)).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained(str(tokenizer_path))
    except Exception as e:
        print(f"  [transformer] Failed to load saved model: {e}")
        return None

    print(f"  ✓ Model loaded from {artifacts_dir}")

    # Reconstruct the same test split for evaluation metrics
    df = reviews_df.copy()
    label_map = {"negative": 0, "positive": 1}
    df["label_enc"] = df["sent_label"].map(label_map)
    df = df.dropna(subset=["label_enc"])

    _, X_test, _, y_test = train_test_split(
        df["text_clean"].values, df["label_enc"].values.astype(int),
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["label_enc"].values,
    )

    # Evaluate on test set
    class _EvalDataset(Dataset):
        def __init__(self, texts, labels, tok, max_len):
            self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tok(str(self.texts[idx]), truncation=True, padding="max_length",
                           max_length=self.max_len, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "label": torch.tensor(self.labels[idx], dtype=torch.long)}

    test_ds = _EvalDataset(X_test, y_test, tokenizer, DISTILBERT_MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=DISTILBERT_BATCH_SIZE)

    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels_t.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels_arr = np.array(all_labels)

    inv_map = {0: "negative", 1: "positive"}
    pred_labels = [inv_map[p] for p in all_preds]
    true_labels = [inv_map[l] for l in all_labels_arr]
    labels_list = ["negative", "positive"]

    acc = accuracy_score(true_labels, pred_labels)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0,
    )
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_list)

    per_class = {}
    for cls in labels_list:
        p, r, f, s = precision_recall_fscore_support(
            true_labels, pred_labels, labels=[cls], average=None, zero_division=0,
        )
        per_class[cls] = {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": int(s[0])}

    print(f"  Accuracy: {acc:.4f} | Precision(macro): {prec_m:.4f} | Recall(macro): {rec_m:.4f} | F1(macro): {f1_m:.4f}")

    result = {
        "model_name": "distilbert",
        "model": None,
        "predictions": np.array(pred_labels),
        "probabilities": all_probs,
        "prob_dict": {"negative": all_probs[:, 0], "positive": all_probs[:, 1]},
        "accuracy": acc,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_test": np.array(true_labels),
        "X_test": pd.Series(X_test),
    }

    print(f"  ✓ DistilBERT loaded successfully (skipped retraining)")
    return result


def train_distilbert(reviews_df, save_results=True):
    """
    Fine-tune DistilBERT on the sentiment dataset.
    Uses a subsample for tractability on modest hardware.
    """
    if not check_transformer_deps():
        return None

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        DistilBertTokenizer, DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix,
    )
    from modules.visualization import plot_confusion_matrix

    print(f"\n{'='*60}")
    print("DISTILBERT TRAINING")
    print(f"{'='*60}")

    # Label encoding (no subsampling for exact match with classical models)
    df = reviews_df.copy()
    label_map = {"negative": 0, "positive": 1}
    df["label_enc"] = df["sent_label"].map(label_map)
    df = df.dropna(subset=["label_enc"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"].values, df["label_enc"].values.astype(int),
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["label_enc"].values,
    )

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    class SentimentDataset(Dataset):
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

    train_ds = SentimentDataset(X_train, y_train, tokenizer, DISTILBERT_MAX_LEN)
    test_ds = SentimentDataset(X_test, y_test, tokenizer, DISTILBERT_MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=DISTILBERT_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=DISTILBERT_BATCH_SIZE)

    # Model — use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * DISTILBERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )

    # Training loop
    for epoch in range(DISTILBERT_EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{DISTILBERT_EPOCHS} — Batch {batch_idx+1}/{len(train_loader)} — Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{DISTILBERT_EPOCHS} — Avg Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Map back to string labels
    inv_map = {0: "negative", 1: "positive"}
    pred_labels = [inv_map[p] for p in all_preds]
    true_labels = [inv_map[l] for l in all_labels]
    labels_list = ["negative", "positive"]

    acc = accuracy_score(true_labels, pred_labels)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0,
    )
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_list)

    print(f"\n  DistilBERT Results:")
    print(f"  Accuracy: {acc:.4f} | Precision(macro): {prec_m:.4f} | Recall(macro): {rec_m:.4f} | F1(macro): {f1_m:.4f}")

    per_class = {}
    for cls in labels_list:
        p, r, f, s = precision_recall_fscore_support(
            true_labels, pred_labels, labels=[cls], average=None, zero_division=0,
        )
        per_class[cls] = {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": int(s[0])}
        print(f"  {cls}: P={p[0]:.4f} R={r[0]:.4f} F1={f[0]:.4f}")

    result = {
        "model_name": "distilbert",
        "model": None,  # too large to return in-memory; saved to disk
        "predictions": np.array(pred_labels),
        "probabilities": all_probs,
        "prob_dict": {
            "negative": all_probs[:, 0],
            "positive": all_probs[:, 1],
        },
        "accuracy": acc,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_test": np.array(true_labels),
        "X_test": pd.Series(X_test),
    }

    if save_results:
        metrics_dir = get_model_metrics_dir("distilbert")
        figures_dir = get_model_figures_dir("distilbert")
        artifacts_dir = get_model_artifacts_dir("distilbert")

        pd.DataFrame(report).transpose().to_csv(metrics_dir / "classification_report.csv")
        pd.DataFrame(per_class).T.to_csv(metrics_dir / "per_class_metrics.csv")
        plot_confusion_matrix(cm, labels_list, figures_dir, "distilbert")

        # Save model
        model.save_pretrained(str(artifacts_dir / "distilbert_model"))
        tokenizer.save_pretrained(str(artifacts_dir / "distilbert_tokenizer"))

        config_data = {
            "model_name": "distilbert",
            "accuracy": acc, "precision_macro": prec_m,
            "recall_macro": rec_m, "f1_macro": f1_m,
            "per_class": per_class,
            "max_samples": len(df),
            "epochs": DISTILBERT_EPOCHS,
            "batch_size": DISTILBERT_BATCH_SIZE,
            "max_len": DISTILBERT_MAX_LEN,
        }
        with open(artifacts_dir / "training_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"  [transformer] DistilBERT results saved to {artifacts_dir}")

    return result
