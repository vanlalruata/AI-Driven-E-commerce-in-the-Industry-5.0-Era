# generate_v5.py
import re
from pathlib import Path

def main():
    artifact_dir = Path("C:/Users/Maruata/.gemini/antigravity-ide/brain/70577be6-d245-4c7c-b436-e35c0362e391/scratch")
    v4_path = artifact_dir / "sync_and_highlight_v4.py"
    
    with open(v4_path, "r", encoding="utf-8") as f:
        code = f.read()
        
    # We want to insert our new replacements right before:
    # "    print(f\"Loaded {len(replacements)} replacements.\")"
    
    new_replacements = """
    # --- New Replacements for Shortened Captions & Expanded Paragraphs ---
    
    # 1. Figure captions
    replacements.append((
        r"\\caption{Proposed end-to-end framework architecture.}",
        r"\\caption{\\update{Proposed end-to-end framework architecture.}}"
    ))
    replacements.append((
        r"\\caption{Confusion matrix for the proposed TF-IDF+LR sentiment model on the 15.00\\% holdout test set ($N=78873$).}",
        r"\\caption{\\update{Confusion matrix for the proposed TF-IDF+LR sentiment model on the 15.00\\% holdout test set ($N=78873$).}}"
    ))
    replacements.append((
        r"\\caption{ROC and Precision-Recall curves overlay for the seven sentiment classifiers on the holdout set.}",
        r"\\caption{\\update{ROC and Precision-Recall curves overlay for the seven sentiment classifiers on the holdout set.}}"
    ))
    replacements.append((
        r"\\caption{Probability calibration curves (reliability diagram) for the seven classifiers.}",
        r"\\caption{\\update{Probability calibration curves (reliability diagram) for the seven classifiers.}}"
    ))
    replacements.append((
        r"\\caption{Heatmap of McNemar's pairwise statistical significance p-values on the holdout test set.}",
        r"\\caption{\\update{Heatmap of McNemar's pairwise statistical significance p-values on the holdout test set.}}"
    ))
    replacements.append((
        r"\\caption{Error analysis diagnostics for the proposed TF-IDF+LR classifier.}",
        r"\\caption{\\update{Error analysis diagnostics for the proposed TF-IDF+LR classifier.}}"
    ))
    replacements.append((
        r"\\caption{Ten-fold cross-validation performance boxplots.}",
        r"\\caption{\\update{Ten-fold cross-validation performance boxplots.}}"
    ))
    replacements.append((
        r"\\caption{Imbalance recall comparison plot, mapping negative-class recall against overall classification accuracy across five balancing strategies.}",
        r"\\caption{\\update{Imbalance recall comparison plot, mapping negative-class recall against overall classification accuracy across five balancing strategies.}}"
    ))
    replacements.append((
        r"\\caption{E-commerce transactional distributions, showing revenue concentration across top Indian regions (States) and individual products (SKUs).}",
        r"\\caption{\\update{E-commerce transactional distributions, showing revenue concentration across top Indian regions (States) and individual products (SKUs).}}"
    ))
    replacements.append((
        r"\\caption{Lorenz curves for SKU and regional revenue.}",
        r"\\caption{\\update{Lorenz curves for SKU and regional revenue.}}"
    ))
    replacements.append((
        r"\\caption{Boxplots of Maximum Retail Price (MRP) dispersion across ten digital marketplaces in India ($N=2586$).}",
        r"\\caption{\\update{Boxplots of Maximum Retail Price (MRP) dispersion across ten digital marketplaces in India ($N=2586$).}}"
    ))
    replacements.append((
        r"\\caption{Venn diagram of the catalogue overlap audit of Unique ASINs.}",
        r"\\caption{\\update{Venn diagram of the catalogue overlap audit of Unique ASINs.}}"
    ))
    
    # 2. Expanded body paragraphs
    paragraph_665_target = (
        "To further evaluate the model performance and probability calibrations, the confusion matrix for the proposed TF-IDF+LR model is shown in Figure \\\\ref{fig:tfidflr_confusion_matrix}. The confusion matrix highlights the classifier's performance in detail, showing that the baseline model is highly accurate in the dominant positive class but exhibits misclassifications in the minority negative class. This imbalance is critical because misclassifying customer complaints as positive reviews prevents early warning detection, necessitating the balancing techniques analysed in Section \\\\ref{sec:class_imbalance}. The receiver operating characteristic (ROC) and precision-recall (PR) curves for all seven classifiers are plotted in Figure \\\\ref{fig:model_roc_pr}, illustrating the superior discriminative capability of DistilBERT and SVM. The ROC and PR curves compare the trade-offs between sensitivity and specificity; given the class imbalance, the PR-AUC curves are more representative of model utility than the ROC-AUC, confirming that DistilBERT and SVM are highly robust, whereas XGBoost and Naive Bayes suffer significant precision drops at higher recall thresholds. Figure \\\\ref{fig:calibration_curves} shows the probability calibration curves (reliability diagrams) for the classifiers, demonstrating the high calibration of SVM and DistilBERT relative to other classifiers. The calibration curves measure the reliability of the predicted probabilities, validating their use in risk-based routing, where uncertain reviews (confidence near 0.5) are escalated to human moderators. The stability of the classifiers was validated using the ten-fold cross-validation boxplots presented in Figure \\\\ref{fig:cv_boxplots}, supporting Hypothesis H1a. The cross-validation boxplots show that classical models are highly stable across folds, whereas DistilBERT exhibits a wider variance due to training constraints. The tight distributions for SVM and TF-IDF+LR confirm their stability across folds, whereas DistilBERT displays wider variance (due to subsampling limits and single-epoch training in CV), highlighting the sensitivity of transformers to training scale and convergence limits. Furthermore, the pairwise statistical significance p-value heatmap resulting from McNemar's test is shown in Figure \\\\ref{fig:mcnemar_heatmap}, confirming that all classifier boundaries differ significantly, except for the default logistic regression and TF-IDF+LR. The McNemar's test heatmap provides statistical verification that the model differences are robust and not due to random variation. Cells indicate p-values from chi-squared testing of prediction contingency tables. The non-significant difference between default Logistic Regression and our proposed TF-IDF+LR ($p = 0.6047 > 0.05$) indicates equivalent decision boundaries, while all other comparisons are highly significant ($p < 0.001$), confirming distinct error patterns across other model architectures. Finally, a diagnostic error analysis for the proposed TF-IDF+LR classifier is illustrated in Figure \\\\ref{fig:error_analysis_tfidf_lr}, detailing the prediction confidence and string lengths of misclassifications. The error analysis diagnostics show that document length is not a primary driver of error, while the concentration of errors near the 0.5 probability threshold justifies a human-in-the-loop buffer region. The overlap in length distributions indicates that document length is not a primary driver of misclassification, while the concentration of errors near the 0.5 probability threshold highlights that prediction uncertainty is the main source of error, justifying a human-in-the-loop buffer region."
    )
    paragraph_665_replacement = (
        "To further evaluate the model performance and probability calibrations, the confusion matrix for the proposed TF-IDF+LR model is shown in Figure \\\\ref{fig:tfidflr_confusion_matrix}. The confusion matrix highlights the classifier's performance in detail, showing that the baseline model is highly accurate in the dominant positive class but exhibits misclassifications in the minority negative class. This imbalance is critical because misclassifying customer complaints as positive reviews prevents early warning detection, necessitating the balancing techniques analysed in Section \\\\ref{sec:class_imbalance}. The receiver operating characteristic (ROC) and precision-recall (PR) curves for all seven classifiers are plotted in Figure \\\\ref{fig:model_roc_pr}, illustrating the superior discriminative capability of DistilBERT and SVM. The ROC and PR curves compare the trade-offs between sensitivity and specificity; given the class imbalance, the PR-AUC curves are more representative of model utility than the ROC-AUC, confirming that DistilBERT and SVM are highly robust, whereas XGBoost and Naive Bayes suffer significant precision drops at higher recall thresholds. Figure \\\\ref{fig:calibration_curves} shows the probability calibration curves (reliability diagrams) for the classifiers, demonstrating the high calibration of SVM and DistilBERT relative to other classifiers. The calibration curves measure the reliability of the predicted probabilities, validating their use in risk-based routing, where uncertain reviews (confidence near 0.5) are escalated to human moderators. The stability of the classifiers was validated using the ten-fold cross-validation boxplots presented in Figure \\\\ref{fig:cv_boxplots}, supporting Hypothesis H1a. The cross-validation boxplots show that classical models are highly stable across folds, whereas DistilBERT exhibits a wider variance due to training constraints. \\\\update{The tight distributions for SVM and TF-IDF+LR confirm their stability across folds, whereas DistilBERT displays wider variance (due to subsampling limits and single-epoch training in CV), highlighting the sensitivity of transformers to training scale and convergence limits.} Furthermore, the pairwise statistical significance p-value heatmap resulting from McNemar's test is shown in Figure \\\\ref{fig:mcnemar_heatmap}, confirming that all classifier boundaries differ significantly, except for the default logistic regression and TF-IDF+LR. The McNemar's test heatmap provides statistical verification that the model differences are robust and not due to random variation. \\\\update{Cells indicate p-values from chi-squared testing of prediction contingency tables. The non-significant difference between default Logistic Regression and our proposed TF-IDF+LR ($p = 0.6047 > 0.05$) indicates equivalent decision boundaries, while all other comparisons are highly significant ($p < 0.001$), confirming distinct error patterns across other model architectures.} Finally, a diagnostic error analysis for the proposed TF-IDF+LR classifier is illustrated in Figure \\\\ref{fig:error_analysis_tfidf_lr}, detailing the prediction confidence and string lengths of misclassifications. The error analysis diagnostics show that document length is not a primary driver of error, while the concentration of errors near the 0.5 probability threshold justifies a human-in-the-loop buffer region. \\\\update{The overlap in length distributions indicates that document length is not a primary driver of misclassification, while the concentration of errors near the 0.5 probability threshold highlights that prediction uncertainty is the main source of error, justifying a human-in-the-loop buffer region.}"
    )
    replacements.append((paragraph_665_target, paragraph_665_replacement))

    paragraph_43_target = (
        "Class-balancing experiments demonstrate that while the baseline model achieves strong accuracy ($96.19\\\\%$), it suffers from a lower negative-class recall ($84.80\\\\%$). Implementing synthetic oversampling (SMOTE) or class weighting significantly improves the negative-class recall to $91.68\\\\%$ and $93.61\\\\%$ respectively, with only a minor decline in overall accuracy. For Industry 5.0 systems, SMOTE represents the optimal balance, yielding a negative-class F1-score of $87.14\\\\%$ and ensuring that customer dissatisfaction signals are not overlooked. The trade-offs between overall accuracy and negative-class recall across different balancing strategies are visually summarised in Figure \\\\ref{fig:imbalance_comparison}, supporting Hypothesis H1b. The imbalance recall comparison (Figure \\\\ref{fig:imbalance_comparison}) illustrates the operational trade-offs of class balancing. It shows that as negative recall improves (moving from baseline to class weighting), the overall accuracy declines slightly. The trade-off curve illustrates that while baseline and SMOTE preserve high accuracy, class weighting maximises negative recall ($93.61\\\\%$) at the cost of a minor accuracy drop, guiding operators in selecting the optimal deployment strategy based on risk aversion. This visualisation is significant for e-commerce operators, as it maps the Pareto frontier of model performance, helping them select the optimal strategy depending on whether they prioritise overall accuracy (baseline/SMOTE) or risk mitigation (class weighting)."
    )
    paragraph_43_replacement = (
        "Class-balancing experiments demonstrate that while the baseline model achieves strong accuracy ($96.19\\\\%$), it suffers from a lower negative-class recall ($84.80\\\\%$). Implementing synthetic oversampling (SMOTE) or class weighting significantly improves the negative-class recall to $91.68\\\\%$ and $93.61\\\\%$ respectively, with only a minor decline in overall accuracy. For Industry 5.0 systems, SMOTE represents the optimal balance, yielding a negative-class F1-score of $87.14\\\\%$ and ensuring that customer dissatisfaction signals are not overlooked. The trade-offs between overall accuracy and negative-class recall across different balancing strategies are visually summarised in Figure \\\\ref{fig:imbalance_comparison}, supporting Hypothesis H1b. The imbalance recall comparison (Figure \\\\ref{fig:imbalance_comparison}) illustrates the operational trade-offs of class balancing. It shows that as negative recall improves (moving from baseline to class weighting), the overall accuracy declines slightly. \\\\update{The trade-off curve illustrates that while baseline and SMOTE preserve high accuracy, class weighting maximises negative recall ($93.61\\\\%$) at the cost of a minor accuracy drop, guiding operators in selecting the optimal deployment strategy based on risk aversion.} This visualisation is significant for e-commerce operators, as it maps the Pareto frontier of model performance, helping them select the optimal strategy depending on whether they prioritise overall accuracy (baseline/SMOTE) or risk mitigation (class weighting)."
    )
    replacements.append((paragraph_43_target, paragraph_43_replacement))

    paragraph_44_target = (
        "These findings align with prior research in retail analytics and operations management, which indicates that regional sales concentrations can lead to supply chain vulnerabilities. Figure \\\\ref{fig:sales_distributions} illustrates the top ten Indian regions (states) and Stock Keeping Units (SKUs) by revenue. The geographical and product sales distributions reveal the skewness of e-commerce revenue, showing that a small number of states and SKUs generate most of the platform income. The steep decline in both distributions visually demonstrates the high Pareto skewness of transactional revenue, highlighting the dominance of key hubs and individual high-demand items in the distribution. This concentration is significant because it indicates that fulfilment networks and regional warehousing should be optimised specifically for these high-volume hubs to minimise transit times and costs. To formally visualise this concentration inequality, Figure \\\\ref{fig:lorenz_curves} presents the Lorenz curves at the SKU and regional levels, showing high skewness, which supports Hypothesis H2a. The Gini coefficients of 0.6843 (SKU) and 0.8169 (State) confirm a high concentration, especially at the geographical level. This inequality has important supply chain implications: it shows that minor disruptions in key states (such as Maharashtra or Karnataka) or supply issues for top SKUs can cause severe revenue drops, highlighting the need for dual sourcing and regional inventory redundancy. For Industry 5.0, these indicators should guide stocking and logistics planning, allowing platforms to build resilience against regional supply chain shocks."
    )
    paragraph_44_replacement = (
        "These findings align with prior research in retail analytics and operations management, which indicates that regional sales concentrations can lead to supply chain vulnerabilities. Figure \\\\ref{fig:sales_distributions} illustrates the top ten Indian regions (states) and Stock Keeping Units (SKUs) by revenue. The geographical and product sales distributions reveal the skewness of e-commerce revenue, showing that a small number of states and SKUs generate most of the platform income. \\\\update{The steep decline in both distributions visually demonstrates the high Pareto skewness of transactional revenue, highlighting the dominance of key hubs and individual high-demand items in the distribution.} This concentration is significant because it indicates that fulfilment networks and regional warehousing should be optimised specifically for these high-volume hubs to minimise transit times and costs. To formally visualise this concentration inequality, Figure \\\\ref{fig:lorenz_curves} presents the Lorenz curves at the SKU and regional levels, showing high skewness, which supports Hypothesis H2a. The Gini coefficients of 0.6843 (SKU) and 0.8169 (State) confirm a high concentration, especially at the geographical level. This inequality has important supply chain implications: it shows that minor disruptions in key states (such as Maharashtra or Karnataka) or supply issues for top SKUs can cause severe revenue drops, highlighting the need for dual sourcing and regional inventory redundancy. For Industry 5.0, these indicators should guide stocking and logistics planning, allowing platforms to build resilience against regional supply chain shocks."
    )
    replacements.append((paragraph_44_target, paragraph_44_replacement))

    paragraph_47_target = (
        "As detailed in Section \\\\ref{sec:method}, the Data Linkage Audit revealed a $0.00\\\\%$ ASIN match rate between the Amazon reviews corpus and the transactional sales files, as well as a temporal gap of ten years (1999–2012 reviews vs. 2022 sales). Therefore, we cannot establish causal relationships or conduct temporal lag modelling (e.g., Granger causality) to prove that sentiment spikes precede revenue declines for specific SKUs. Support for Hypothesis H3a is provided by the complete absence of overlapped ASIN catalogue mappings. This absolute divergence between the customer feedback and the transactional marketplace catalogue is visually illustrated in Figure \\\\ref{fig:asin_overlap}. The Venn diagram (Figure \\\\ref{fig:asin_overlap}) shows that there are no overlapping product identifiers between the review and transaction databases. The absolute separation indicates that any causal linking of review sentiment directly to transaction-level revenue drop lags is statistically unsupported, necessitating parallel decision-support modules. This visual separation is significant because it prevents researchers from making unsupported causal claims (e.g., that negative reviews cause sales declines for specific products). By presenting this limitation clearly, we demonstrate a major challenge in secondary e-commerce research and highlight the importance of data provenance and catalogue alignment under the DPDP Act of 2023."
    )
    paragraph_47_replacement = (
        "As detailed in Section \\\\ref{sec:method}, the Data Linkage Audit revealed a $0.00\\\\%$ ASIN match rate between the Amazon reviews corpus and the transactional sales files, as well as a temporal gap of ten years (1999–2012 reviews vs. 2022 sales). Therefore, we cannot establish causal relationships or conduct temporal lag modelling (e.g., Granger causality) to prove that sentiment spikes precede revenue declines for specific SKUs. Support for Hypothesis H3a is provided by the complete absence of overlapped ASIN catalogue mappings. This absolute divergence between the customer feedback and the transactional marketplace catalogue is visually illustrated in Figure \\\\ref{fig:asin_overlap}. The Venn diagram (Figure \\\\ref{fig:asin_overlap}) shows that there are no overlapping product identifiers between the review and transaction databases. \\\\update{The absolute separation indicates that any causal linking of review sentiment directly to transaction-level revenue drop lags is statistically unsupported, necessitating parallel decision-support modules.} This visual separation is significant because it prevents researchers from making unsupported causal claims (e.g., that negative reviews cause sales declines for specific products). By presenting this limitation clearly, we demonstrate a major challenge in secondary e-commerce research and highlight the importance of data provenance and catalogue alignment under the DPDP Act of 2023."
    )
    replacements.append((paragraph_47_target, paragraph_47_replacement))
"""
    
    # Insert before print(f"Loaded {len(replacements)} replacements.")
    target_str = '    print(f"Loaded {len(replacements)} replacements.")'
    code_mod = code.replace(target_str, new_replacements + "\n" + target_str)
    
    # Save as sync_and_highlight_v5.py in the workspace scratch folder
    v5_path = Path("scratch/sync_and_highlight_v5.py")
    with open(v5_path, "w", encoding="utf-8") as f:
        f.write(code_mod)
        
    print("sync_and_highlight_v5.py generated successfully.")

if __name__ == "__main__":
    main()
