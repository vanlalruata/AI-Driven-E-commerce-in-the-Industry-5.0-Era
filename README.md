# AI-Driven E-commerce in the Industry 5.0 Era: A Parallel-Empirical Framework, Decision Support, and Methodological Caution

# Journal

Strategic Business Research

# Publisher

Elsevier

# Corresponding author

Vanlalruata Hnamte

# First author

Runu Patgiri

# Date of submission

04th June 2026

# 1st revision Review Complete Date

16th June 2026

# 2nd revision Review Complete Date

05th July 2026

# 3rd revision Review Complete Date

26th July 2026

# 4th revision Review Complete Date

27th July 2026

# Accepted Date

29th July 2026

# DOI

<https://doi.org/>

# ScienceDirect Link

<https://www.sciencedirect.com/science/article/pii/>

# Abstract

India's e-commerce ecosystem is evolving rapidly under Artificial Intelligence (AI), digital transformation, and the human-centric pillars of Industry 5.0, contributing to the Viksit Bharat@2047 agenda. Grounded in the Resource-Based View (RBV), Technology-Organization-Environment (TOE), and Technology Acceptance Model (TAM), this study presents a parallel-empirical framework and methodological caution for analysing consumer sentiment and transactional sales performance. Evaluating seven sentiment classifiers on a large product review corpus shows that transformer-based deep learning achieves superior predictive accuracy, while lightweight classical models provide competitive, sustainable, and explainable alternatives suitable for resource-constrained enterprises. Crucially, a data quality linkage audit demonstrates that public review corpora and transactional sales records are disjoint, highlighting a key methodological warning against assuming seamless secondary data integration in digital retail research. Consequently, we demonstrate independent parallel analytics streams—sentiment model benchmarking and operational sales distribution analytics—which reveal significant revenue inequality (Gini) while Herfindahl–Hirschman indices indicate a competitive product market, alongside negligible cross-marketplace price dispersion. Finally, we discuss how these empirical insights translate into a conceptual enterprise decision-support framework and data governance guidelines under the Digital Personal Data Protection (DPDP) Act of 2023.

# How to cite

Runu Patgiri, Vanlalruata Hnamte, Gurram Ramakrishna,
AI-Driven E-commerce in the Industry 5.0 Era: A Parallel-Empirical Framework, Decision Support, and Methodological Caution,
Strategic Business Research,
Volume 2,
Issue 1,
2026,
100248,
ISSN 3051-0643,
<https://doi.org/>.
(<https://www.sciencedirect.com/science/article/pii/>)

# Note

If you find this code and paper useful, kindly consider to cite from your valuable work.

## Overview

This project provides an end-to-end, reproducible pipeline for:

- **Multi-model comparative sentiment classification** on Amazon product reviews (Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost, TF-IDF+LR, optional DistilBERT).
- **Robust model evaluation** with k-fold cross-validation, ROC-AUC, PR-AUC, calibration analysis, error analysis, and McNemar's statistical test.
- **Class imbalance handling** via SMOTE, class weighting, and random oversampling/undersampling.
- **Sales analytics** with statistical significance testing (Gini coefficient, HHI, Lorenz curves, ANOVA, Kruskal-Wallis, t-tests).
- **Data linkage auditing** — ASIN matching verification between review and sales datasets.
- **Integrated review–sales analysis** with SKU/ASIN sentiment summaries.

The pipeline is **menu-driven** via `main.py`, allowing individual components to be run independently or as a full pipeline.

## Key Features

- **Comparative model benchmarking**: 6 classical ML models + optional transformer, all trained on the same TF-IDF representation for fair comparison.
- **Stratified 10-fold cross-validation** with mean ± std metrics per model.
- **Publication-quality figures** (PDF + PNG + EPS) for all analysis types.
- **Per-model result directories**: Every model's metrics, figures, and artifacts are stored in a dedicated named directory.
- **Statistical rigor**: ANOVA/Kruskal-Wallis for MRP, t-tests for profitability, Gini/HHI for revenue concentration.
- **Data quality audit**: ASIN overlap analysis, SKU coverage, temporal alignment checks.
- **Class imbalance experiments**: Systematic comparison of 5 strategies with focus on minority-class recall.

## Requirements

- Python 3.8+
- Core packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `scipy`
- Additional: `xgboost`, `imbalanced-learn`
- Optional (for DistilBERT): `transformers`, `torch`

Install via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib scipy xgboost imbalanced-learn
# Optional for DistilBERT:
pip install transformers torch
```

## Project Structure

```
industry5.0_ecommerce_sentimental_prediction/
├── main.py                          # Menu-driven entry point
├── config.py                        # Centralized configuration
├── modules/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading & preprocessing
│   ├── sentiment_models.py          # All ML model training (LR, SVM, NB, RF, XGBoost)
│   ├── transformer_models.py        # DistilBERT (optional)
│   ├── model_evaluation.py          # CV, ROC-AUC, PR-AUC, calibration, error analysis
│   ├── class_imbalance.py           # SMOTE, class weighting, resampling strategies
│   ├── statistical_tests.py         # Gini, HHI, ANOVA, Kruskal-Wallis, t-tests
│   ├── sales_analytics.py           # Sales analysis with descriptive statistics
│   ├── data_audit.py                # ASIN matching & data consistency checks
│   └── visualization.py             # All plotting functions (publication quality)
├── results/                         # Organized output root (auto-created)
│   ├── models/
│   │   ├── logistic_regression/     # Per-model: metrics/ figures/ artifacts/
│   │   ├── svm/
│   │   ├── naive_bayes/
│   │   ├── random_forest/
│   │   ├── xgboost/
│   │   └── distilbert/              # Optional
│   ├── comparative/                 # Cross-model comparison tables & plots
│   ├── statistical_tests/           # All stat test outputs
│   ├── data_audit/                  # ASIN matching & data quality reports
│   └── sales_analytics/             # Sales outputs
├── amazon_ecom_analysis_integrated_full.py  # Legacy script (preserved)
├── tfidf_lr_sentiment.py                    # Legacy script (preserved)
├── amazon_ecom_analysis.py                  # Legacy script (preserved)
├── dataset_inspector.py                     # Legacy script (preserved)
```

## Quick Start

### Interactive Menu Mode

```bash
python main.py --reviews "H:/Datasets/ecommerce/ecommerce_review/Reviews.csv" --sales_folder "H:/Datasets/ecommerce/ecommerce_sale"
```

This opens an interactive menu:

```
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
```

### Non-Interactive Mode (Full Pipeline)

```bash
python main.py --reviews "path/to/Reviews.csv" --sales_folder "path/to/sales/" --run 12
```

### Run a Specific Step

```bash
python main.py --run 3   # Train all models only
python main.py --run 6   # Run evaluation only (auto-loads data + trains if needed)
python main.py --run 9   # Run statistical tests only
```

## Command-line Arguments

| Argument | Type | Default | Description |
| ---------- | ------ | --------- | ------------- |
| `--reviews` | str | `H:/Datasets/...` | Path to Amazon `Reviews.csv` |
| `--sales_folder` | str | `H:/Datasets/...` | Folder containing sales CSV files |
| `--run` | int | None | Run specific menu option non-interactively |

Additional configuration (CV folds, TF-IDF features, model hyperparameters, etc.) can be modified in `config.py`.

## Output Structure

Each model's results are organized in a consistent directory hierarchy:

```
results/models/<model_name>/
├── metrics/
│   ├── classification_report.csv
│   ├── per_class_metrics.csv
│   ├── cv_fold_scores.csv
│   ├── cv_summary.csv
│   ├── roc_auc.csv
│   ├── pr_auc.csv
│   ├── brier_score.csv
│   ├── calibration_data.csv
│   ├── misclassified_samples.csv
│   └── error_analysis_summary.json
├── figures/
│   ├── confusion_matrix_<model>.pdf/png/eps
│   ├── error_analysis_<model>.pdf/png/eps
│   ├── roc_curve.pdf/png/eps
│   ├── pr_curve.pdf/png/eps
│   └── calibration_curve.pdf/png/eps
└── artifacts/
    ├── model.joblib
    ├── vectorizer.joblib
    └── training_config.json
```

Comparative and cross-cutting results are in:

```
results/
├── comparative/
│   ├── model_comparison_table.csv
│   ├── cv_comparison_table.csv
│   ├── comprehensive_comparison.csv
│   ├── imbalance_experiment_table.csv
│   ├── mcnemar_pvalues.csv
│   ├── roc_curves_overlay.pdf/png/eps
│   ├── pr_curves_overlay.pdf/png/eps
│   ├── calibration_curves.pdf/png/eps
│   └── cv_boxplot_*.pdf/png/eps
├── statistical_tests/
│   ├── revenue_concentration.csv
│   ├── lorenz_curve_*.pdf/png/eps
│   ├── mrp_dispersion_test.csv
│   ├── mrp_pairwise_comparisons.csv
│   └── channel_profitability_test.csv
├── data_audit/
│   ├── data_linkage_audit.txt
│   ├── data_audit.json
│   ├── asin_overlap.pdf/png/eps
│   ├── matched_asins.csv
│   ├── reviews_only_asins.csv
│   └── sales_only_asins.csv
└── sales_analytics/
    ├── merged_sales.csv
    ├── top_skus.csv / top_skus_stats.csv
    ├── sales_by_region.csv / sales_region_stats.csv
    ├── channel_profitability_descriptive.csv
    └── mrp_dispersion_stats.csv
```

## Models Compared

| Model | Type | Class Weight | Key Feature |
| ------- | ------ | ------------- | ------------- |
| TF-IDF+LR | Linear | — | Proposed model, optimized (unbalanced) |
| Logistic Regression | Linear | Balanced | Interpretable baseline |
| SVM (LinearSVC) | Linear | Balanced | Calibrated via CalibratedClassifierCV |
| Multinomial Naive Bayes | Probabilistic | — | Fast, generative model |
| Random Forest | Ensemble | Balanced | Non-linear, feature importance |
| XGBoost | Boosting | scale_pos_weight | State-of-art gradient boosting |
| DistilBERT | Transformer | — | Lightweight deep learning (optional) |

## Evaluation Methods

- **Stratified 10-Fold Cross-Validation** with per-fold metrics
- **ROC-AUC** with overlay curves for all models
- **PR-AUC** (critical for imbalanced classes)
- **Calibration Curves** (reliability diagrams) with Brier scores
- **Error Analysis** — patterns in misclassified samples
- **McNemar's Test** — pairwise statistical comparison between models

## Class Imbalance Strategies

| Strategy | Method | Expected Effect |
| ---------- | -------- | ---------------- |
| Baseline | No handling | Current approach |
| Class Weighting | `class_weight='balanced'` | Penalize majority class errors more |
| SMOTE | Synthetic oversampling | Generate synthetic minority samples |
| Random Oversampling | Duplicate minority | Simple replication |
| Random Undersampling | Reduce majority | May lose information |

## Statistical Tests

| Test | Applied To | Purpose |
| ------ | ----------- | --------- |
| Gini Coefficient | SKU/Region revenue | Measure concentration inequality |
| Lorenz Curve | Revenue distribution | Visualize concentration |
| HHI | Revenue distribution | Market concentration index |
| ANOVA / Kruskal-Wallis | MRP across marketplaces | Test pricing differences |
| Welch's t-test / Mann-Whitney | Channel profitability | Compare Shiprocket vs INCREFF |
| Shapiro-Wilk | Pre-test | Normality assumption check |
| Levene's Test | Pre-test | Variance homogeneity check |

## Configuration

Edit `config.py` to modify:

- File paths and output directories
- Random seed, CV folds, test size
- TF-IDF hyperparameters
- DistilBERT settings (enable/disable, max samples, epochs)
- Figure DPI and formats
- Model color palette for plots

## Reusing Trained Models

```python
import joblib
from pathlib import Path

model_dir = Path("results/models/logistic_regression/artifacts")
vect = joblib.load(model_dir / "vectorizer.joblib")
clf = joblib.load(model_dir / "model.joblib")

texts = ["Great product, highly recommend!", "Terrible quality, waste of money."]
X = vect.transform(texts)
preds = clf.predict(X)
print(list(zip(texts, preds)))
```

## Legacy Scripts

The original scripts are preserved and still functional:

- `amazon_ecom_analysis_integrated_full.py` — original integrated pipeline
- `tfidf_lr_sentiment.py` — standalone TF-IDF + LR sentiment
- `amazon_ecom_analysis.py` — original e-commerce analysis
- `dataset_inspector.py` — dataset inspection utility

## License

MIT License

## Acknowledgments

- Built with `scikit-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `xgboost`, `imbalanced-learn`
- Amazon Reviews dataset format inspired by common Kaggle distributions

## Changelog

- **v2.0**: Complete modular rewrite with multi-model comparison, cross-validation, statistical tests, class imbalance handling, data audit, and menu-driven interface.
- **v1.0**: Initial integrated pipeline (sentiment + sales + reporting).
