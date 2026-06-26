import shutil
import re
from pathlib import Path

def main():
    project_root = Path("g:/PycharmProjects/PythonProject/industry5.0_ecommerce_sentimental_prediction")
    main_path = project_root / "manuscript" / "main.tex"
    rev_path = project_root / "manuscript" / "main_revision.tex"
    
    print("Syncing main_revision.tex with main.tex...")
    shutil.copyfile(main_path, rev_path)
    
    with open(rev_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    replacements = []
    
    def wrap_update(text):
        return f"\\update{{{text}}}"
        
    # 1. Title
    replacements.append((
        r"\title{AI-Driven E-commerce in the Industry 5.0 Era: A Parallel-Empirical Framework, Decision Support, and Methodological Caution}",
        r"\title{\update{AI-Driven E-commerce in the Industry 5.0 Era: A Parallel-Empirical Framework, Decision Support, and Methodological Caution}}"
    ))
    
    # 2. Abstract
    abstract_content = (
        "India's e-commerce ecosystem is rapidly evolving under the combined influence of "
        "Artificial Intelligence (AI), digital transformation, and the human-centric pillars of Industry 5.0, "
        "contributing to the Viksit Bharat@2047 agenda. However, existing research often lacks theoretical anchoring "
        "and rigorous validation. Grounded in the Resource-Based View (RBV), technology-organisation-environment (TOE), "
        "and Technology Acceptance Model (TAM), this study presents a parallel-empirical framework and methodological "
        "caution for analysing consumer sentiment and transactional sales performance. We benchmarked seven sentiment "
        "classifiers on a corpus of 525814 reviews. The deep DistilBERT model achieved the highest holdout accuracy "
        "of 97.69\\% and negative-class recall of 92.18\\%, whereas the lightweight SVM and TF-IDF+LR models "
        "(Brier scores of 0.0270 and 0.0292) provided sustainable, explainable alternatives. Crucially, a rigorous "
        "data quality and linkage audit revealed that the public review corpus and Indian transactional sales datasets "
        "were disjoint (ASIN overlap: 0.00\\%), presenting a fundamental methodological warning against the assumption "
        "of seamless secondary e-commerce data integration. Rather than asserting a unified causal link, this study "
        "demonstrates parallel analytics modules, namely sentiment-model benchmarking and transactional sales analytics. "
        "Sales analytics on the transactional dataset (178405 rows) revealed a significant concentration: a Gini "
        "coefficient of 0.6843 for Stock Keeping Units (SKUs) and 0.8169 for regional revenue. Furthermore, Kruskal-Wallis "
        "H-testing shows a statistically significant but negligible Maximum Retail Price (MRP) dispersion across ten "
        "digital marketplaces ($H = 17.18$, $p = 0.046$, effect size $\\epsilon^2 = 0.0003$). Finally, we discuss how "
        "these parallel empirical insights can be operationalised under the Digital Personal Data Protection (DPDP) "
        "Act of 2023, outlining an operational decision-support matrix and recommended governance guidelines for emerging markets."
    )
    replacements.append((
        abstract_content,
        f"\\update{{{abstract_content}}}"
    ))
    
    # 3. Intro Paragraph 2
    intro_p2 = (
        "This study addresses these gaps by presenting a parallel-empirical framework and methodological caution, "
        "structured around the Resource-Based View (RBV), the Technology-Organization-Environment (TOE) framework, "
        "and the Technology Acceptance Model (TAM). We evaluate seven sentiment models—ranging from classical machine "
        "learning to deep transformer architectures–on a large-scale product review corpus and perform operational sales, "
        "price dispersion, and fulfilment data audits on a detailed transactional dataset."
    )
    replacements.append((intro_p2, f"\\update{{{intro_p2}}}"))
    
    # 4. Intro Paragraph 3
    intro_p3 = (
        "Recognising this, we analysed these datasets as parallel empirical components, representing methodological "
        "caution for researchers and managers against assuming seamless secondary e-commerce data integration. Rather "
        "than asserting a unified causal link, this study demonstrates parallel analytics modules: sentiment-model "
        "benchmarking and transactional-sales analytics. This framing prevents unsupported causal linkages while "
        "demonstrating the practical application of both components in a parallel Industry 5.0 decision support blueprint."
    )
    replacements.append((intro_p3, f"\\update{{{intro_p3}}}"))
    
    # 5. Hypotheses Intro
    hyp_intro = "We formulate three core Research Questions (RQs) and four testable Hypotheses (Hs) that are directly evaluated via statistical and mathematical modeling:"
    replacements.append((hyp_intro, f"\\update{{{hyp_intro}}}"))
    
    # 6. Hypotheses items
    replacements.append((
        r"\item \textit{H1a}: Classical models (SVM, TF-IDF+LR) achieve holdout classification accuracy comparable to deep transformers (DistilBERT) with statistically equivalent performance and probability calibrations.",
        r"\item \textit{H1a}: \update{Classical models (SVM, TF-IDF+LR) achieve holdout classification accuracy comparable to deep transformers (DistilBERT) with statistically equivalent performance and probability calibrations.}"
    ))
    replacements.append((
        r"\item \textit{H1b}: Incorporating class-balancing techniques (e.g., SMOTE, class weighting) on classical models significantly improves minority negative-class recall at the expense of overall accuracy.",
        r"\item \textit{H1b}: \update{Incorporating class-balancing techniques (e.g., SMOTE, class weighting) on classical models significantly improves minority negative-class recall at the expense of overall accuracy.}"
    ))
    replacements.append((
        r"\item \textit{H2a}: E-commerce transactional sales exhibit high concentration at both the SKU and regional levels, which can be formally measured using Gini coefficients and the Herfindahl-Hirschman Index (HHI).",
        r"\item \textit{H2a}: \update{E-commerce transactional sales exhibit high concentration at both the SKU and regional levels, which can be formally measured using Gini coefficients and the Herfindahl-Hirschman Index (HHI).}"
    ))
    replacements.append((
        r"\item \textit{H2b}: Maximum Retail Prices (MRPs) across Indian digital marketplaces exhibit statistically significant but practically negligible dispersion, indicating a synchronized master catalogue.",
        r"\item \textit{H2b}: \update{Maximum Retail Prices (MRPs) across Indian digital marketplaces exhibit statistically significant but practically negligible dispersion, indicating a synchronized master catalogue.}"
    ))
    replacements.append((
        r"\item \textbf{RQ3 (Data Linkage Audit)}: Do public Amazon review corpora and Indian transactional e-commerce sales datasets show sufficient alignment to support integrated empirical predictions?",
        r"\item \textbf{RQ3 (Data Linkage Audit)}: \update{Do public Amazon review corpora and Indian transactional e-commerce sales datasets show sufficient alignment to support integrated empirical predictions?}"
    ))
    replacements.append((
        r"\item \textit{H3}: A formal data linkage audit exposes metadata, catalog, and temporal disjointness in secondary datasets, demonstrating the empirical limitations of integrating disparate secondary retail streams.",
        r"\item \textit{H3}: \update{A formal data linkage audit exposes metadata, catalog, and temporal disjointness in secondary datasets, demonstrating the empirical limitations of integrating disparate secondary retail streams.}"
    ))
    
    # 7. Implications paragraph
    impl_para = "The broader implications of these results for Industry 5.0 (sustainability, human-centricity, resilience) and DPDP Act compliance are framed as interpretive translations and managerial guidelines rather than direct empirical findings."
    replacements.append((impl_para, f"\\update{{{impl_para}}}"))
    
    # 8. Contribution 4
    contrib4 = "Fourth, we integrate these empirical findings into a regulatory-aware, human-in-the-loop decision-support blueprint aligned with Industry 5.0 principles and Viksit Bharat@2047 \\citep{nitiaayog2026dpi}."
    replacements.append((contrib4, f"\\update{{{contrib4}}}"))
    
    # 9. Table 2 Intro
    table2_intro = "To bridge these theoretical paradigms with the practical requirements of e-commerce operations, we explicitly operationalised the three core pillars of Industry 5.0 through specific empirical indicators and decision thresholds evaluated in this study, as summarised in Table \\ref{tab:industry5_operationalisation}."
    replacements.append((table2_intro, f"\\update{{{table2_intro}}}"))
    
    # 10. Table 2 content cells
    replacements.append((
        r"\caption{Industry 5.0 empirical translation and operationalisation matrix.}",
        r"\caption{\update{Industry 5.0 empirical translation and operationalisation matrix.}}"
    ))
    replacements.append((
        r"\textbf{Pillar} & \textbf{Conceptual Definition} & \textbf{Empirical Indicator / Measure} & \textbf{Decision Threshold \& Implementation}",
        r"\textbf{\update{Pillar}} & \textbf{\update{Conceptual Definition}} & \textbf{\update{Empirical Indicator / Measure}} & \textbf{\update{Decision Threshold \& Implementation}}"
    ))
    replacements.append((
        r"Human-Centricity & Preserving human agency, avoiding automated bias, and enabling human-in-the-loop oversight. & Platt-calibrated probability confidence levels $P(pos)$ and error diagnostic distributions. & Dual-boundary routing: Route reviews with $0.4 \leq P(pos) \leq 0.6$ (uncertain region) to human moderators.",
        r"\update{Human-Centricity} & \update{Preserving human agency, avoiding automated bias, and enabling human-in-the-loop oversight.} & \update{Platt-calibrated probability confidence levels $P(pos)$ and error diagnostic distributions.} & \update{Dual-boundary routing: Route reviews with $0.4 \leq P(pos) \leq 0.6$ (uncertain region) to human moderators.}"
    ))
    replacements.append((
        r"Sustainability (Green AI) & Minimising environmental and carbon footprints of digital infrastructure. & Algorithmic time complexity (linear $O(N \cdot L)$ vs. quadratic $O(N \cdot L^2 \cdot D \cdot H)$); training runtime and hardware requirements. & Prefer linear, CPU-trainable classifiers (TF-IDF+LR) for high-frequency operational batching.",
        r"\update{Sustainability (Green AI)} & \update{Minimising environmental and carbon footprints of digital infrastructure.} & \update{Algorithmic time complexity (linear $O(N \cdot L)$ vs. quadratic $O(N \cdot L^2 \cdot D \cdot H)$); training runtime and hardware requirements.} & \update{Prefer linear, CPU-trainable classifiers (TF-IDF+LR) for high-frequency operational batching.}"
    ))
    replacements.append((
        r"Resilience & Protecting supply chains and platforms against localized demand and supply shocks. & Product-level (SKU) and geographical revenue concentrations (Gini coefficients, HHI). & Dynamic logistics buffer allocation: high-Gini SKUs ($G=0.6843$) and states ($G=0.8169$) trigger supply chain alerts upon negative sentiment flags. \\ \hline",
        r"\update{Resilience} & \update{Protecting supply chains and platforms against localized demand and supply shocks.} & \update{Product-level (SKU) and geographical revenue concentrations (Gini coefficients, HHI).} & \update{Dynamic logistics buffer allocation: high-Gini SKUs ($G=0.6843$) and states ($G=0.8169$) trigger supply chain alerts upon negative sentiment flags.} \\ \hline"
    ))
    
    # 11. Architecture Diagram paragraph
    arch_para = (
        "Proposed Architecture Diagram: Figure \\ref{fig:proposed_architecture} visually represents the end-to-end "
        "flow of the sentiment analysis pipeline and its integration within the parallel e-commerce analytics blueprint. "
        "It starts with two disjoint streams: consumer reviews (textual information including ratings) and transactional "
        "sales datasets (containing price and product mappings). The preprocessing stage normalises these inputs separately. "
        "To address the class imbalance and enhance classification realism, the sentiment pipeline introduces two key "
        "methodological enhancements: (1) \\textbf{Sublinear Term Frequency Scaling} within the TF-IDF vectorisation step "
        "(replacing raw term frequency $tf$ with $1 + \\log(tf)$ to mitigate document length bias), and (2) \\textbf{Platt-Scaling Calibration} "
        "(using the logistic sigmoid function for Logistic Regression and a 3-fold \\texttt{CalibratedClassifierCV} for SVM) "
        "to map the raw decision boundaries to reliable posterior probabilities $P(pos)$. These calibrated probabilities are "
        "subsequently evaluated using a confusion matrix and integrated into a dynamic thresholding module. Finally, "
        "a human-in-the-loop decision boundary is implemented, routing reviews near the uncertainty threshold to human "
        "moderators, which directly operationalises Industry 5.0 human-centricity and compliance with the DPDP Act's data "
        "quality requirements. This diagram serves as a structural blueprint, illustrating how disjoint e-commerce datasets "
        "can be audited and operationalised in parallel to support managerial decision-making while ensuring data privacy "
        "(DPDP Act compliance) and algorithmic explainability."
    )
    replacements.append((arch_para, f"\\update{{{arch_para}}}"))
    
    # 12. Section 4.5 Data-Quality Audit
    replacements.append((
        r"\subsection{Data-Quality Audit of Multi-Channel Fulfilment Integration}",
        r"\subsection{\update{Data-Quality Audit of Multi-Channel Fulfilment Integration}}"
    ))
    sec45_body = (
        "Rather than an empirical comparison of alternative fulfilment channels, the profitability analysis serves as "
        "a data-quality audit finding that highlights the structural siloing of e-commerce databases. While the INCREFF "
        "channel records yield unit profitability metrics (mean: \\rupee~15.07 $\\pm$ \\rupee~30.22 per unit, $N = 10$, "
        "median: \\rupee~5.50), the transactional logs for the Shiprocket channel contain zero valid rows ($N = 0$) due to "
        "missing database records and API integration failures. Consequently, standard comparative statistical tests "
        "(e.g., Welch's t-test) cannot be performed. This missing data is represented in Figure \\ref{fig:channel_profitability} "
        "as a blank comparator, demonstrating how the absence of standardised application programming interfaces (APIs) "
        "and unified data logging restricts multi-channel operational visibility. In an Industry 5.0 framework, this "
        "finding highlights the need for robust, standardised data governance (such as through ONDC-compatible schemas) "
        "to enable transparent, automated fulfilment auditing."
    )
    replacements.append((sec45_body, f"\\update{{{sec45_body}}}"))
    replacements.append((
        r"\caption{Fulfilment unit profitability distribution (INCREFF channel only, $N=10$, mean: \rupee~15.07 $\pm$ \rupee~30.22).}",
        r"\caption{\update{Fulfilment unit profitability distribution (INCREFF channel only, $N=10$, mean: \rupee~15.07 $\pm$ \rupee~30.22).}}"
    ))
    
    # 13. Section 4.6 Title, body text, table, and concluding paragraph
    replacements.append((
        r"\subsection{Operational Decision Support Matrix and Escalation Logic}",
        r"\subsection{\update{Operational Decision Support Matrix and Escalation Logic}}"
    ))
    
    sec46_body = (
        "To bridge the parallel analytics streams (sentiment classification and sales operational metrics) under a unified "
        "management control loop, we propose a concrete decision support matrix and routing logic, which has been summarised "
        "in Table \\ref{tab:decision_matrix}. This matrix maps calibrated sentiment classification probabilities, $P(pos)$, "
        "against SKU-level and geographic Gini concentration thresholds to guide inventory allocation and customer relationship management."
    )
    replacements.append((sec46_body, f"\\update{{{sec46_body}}}"))
    
    replacements.append((
        "We define three sentiment routing zones based on calibrated prediction confidence:",
        "\\update{We define three sentiment routing zones based on calibrated prediction confidence:}"
    ))
    replacements.append((
        r"\item \textbf{Automated CRM Routing ($P(pos) \ge 0.6$ or $P(pos) \le 0.4$ with low SKU/regional Gini coefficients)}: Clear sentiment signals on non-critical, low-concentration items are processed automatically by downstream email or refund systems without human intervention.",
        r"\item \textbf{\update{Automated CRM Routing ($P(pos) \ge 0.6$ or $P(pos) \le 0.4$ with low SKU/regional Gini coefficients)}}: \update{Clear sentiment signals on non-critical, low-concentration items are processed automatically by downstream email or refund systems without human intervention.}"
    ))
    replacements.append((
        r"\item \textbf{Human-in-the-Loop Buffer ($0.4 < P(pos) < 0.6$)}: Predictively ambiguous reviews are routed to senior moderators for manual verification. This operationalises the human-centric pillar of Industry 5.0, preventing algorithmic misclassification from driving incorrect customer responses.",
        r"\item \textbf{\update{Human-in-the-Loop Buffer ($0.4 < P(pos) < 0.6$)}}: \update{Predictively ambiguous reviews are routed to senior moderators for manual verification. This operationalises the human-centric pillar of Industry 5.0, preventing algorithmic misclassification from driving incorrect customer responses.}"
    ))
    replacements.append((
        r"\item \textbf{Escalation Routing ($P(pos) \le 0.4$ on high-concentration SKUs or regions)}: Negative sentiment signals for high-risk products (SKUs exceeding the Gini threshold of 0.6843 or regions exceeding the Gini threshold of 0.8169) bypass automated ticketing and are escalated directly to supply chain and inventory managers for immediate dual-sourcing or quality audits.",
        r"\item \textbf{\update{Escalation Routing ($P(pos) \le 0.4$ on high-concentration SKUs or regions)}}: \update{Negative sentiment signals for high-risk products (SKUs exceeding the Gini threshold of 0.6843 or regions exceeding the Gini threshold of 0.8169) bypass automated ticketing and are escalated directly to supply chain and inventory managers for immediate dual-sourcing or quality audits.}"
    ))
    
    replacements.append((
        r"\caption{Operational Decision Support Matrix and Escalation Logic}",
        r"\caption{\update{Operational Decision Support Matrix and Escalation Logic}}"
    ))
    replacements.append((
        r"\textbf{Sentiment Range} & \textbf{SKU/Region Concentration} & \textbf{Operational Action} & \textbf{System Route}",
        r"\textbf{\update{Sentiment Range}} & \textbf{\update{SKU/Region Concentration}} & \textbf{\update{Operational Action}} & \textbf{\update{System Route}}"
    ))
    replacements.append((
        r"$P(pos) \ge 0.6$ & Any Concentration & Automated promotion log & Standard Batch Dashboard \\",
        r"\update{$P(pos) \ge 0.6$} & \update{Any Concentration} & \update{Automated promotion log} & \update{Standard Batch Dashboard} \\"
    ))
    replacements.append((
        r"$0.4 < P(pos) < 0.6$ & Any Concentration & Manual sentiment verification & Human-in-the-Loop Buffer \\",
        r"\update{$0.4 < P(pos) < 0.6$} & \update{Any Concentration} & \update{Manual sentiment verification} & \update{Human-in-the-Loop Buffer} \\"
    ))
    replacements.append((
        r"$P(pos) \le 0.4$ & Low (SKU Gini $\le 0.6843$) & Automated refund/response tick. & Standard CRM ticketing \\",
        r"\update{$P(pos) \le 0.4$} & \update{Low (SKU Gini $\le 0.6843$)} & \update{Automated refund/response tick.} & \update{Standard CRM ticketing} \\"
    ))
    replacements.append((
        r"$P(pos) \le 0.4$ & High (SKU Gini $> 0.6843$) & Immediate supply audit & Supply Chain Manager Alert \\",
        r"\update{$P(pos) \le 0.4$} & \update{High (SKU Gini $> 0.6843$)} & \update{Immediate supply audit} & \update{Supply Chain Manager Alert} \\"
    ))
    replacements.append((
        r"$P(pos) \le 0.4$ & High (Region Gini $> 0.8169$) & Regional inventory reallocation & Logistics Coordinator Alert \\",
        r"\update{$P(pos) \le 0.4$} & \update{High (Region Gini $> 0.8169$)} & \update{Regional inventory reallocation} & \update{Logistics Coordinator Alert} \\"
    ))
    
    sec46_concl = (
        "This decision matrix operationalises resilience and human centricity. By routing high-risk negative feedback on "
        "dominant SKUs or concentrated regions directly to human operators, the framework helps platforms mitigate supply chain "
        "vulnerabilities and prevent cascading regional service failures."
    )
    replacements.append((sec46_concl, f"\\update{{{sec46_concl}}}"))
    
    # 14. Section 5 Ethical, Regulatory, and Governance Discussion
    replacements.append((
        r"\section{Ethical, Regulatory, and Governance Discussion}",
        r"\section{\update{Ethical, Regulatory, and Governance Discussion}}"
    ))
    
    sec5_intro = (
        "The integration of AI into e-commerce operations introduces complex regulatory and ethical responsibilities. "
        "In this section, we distinguish between our direct empirical findings (such as disjoint catalogues and marketplace "
        "price consistency) and the normative managerial guidelines required to achieve regulatory compliance and ethical alignment "
        "under the Industry 5.0 paradigm."
    )
    replacements.append((sec5_intro, f"\\update{{{sec5_intro}}}"))
    
    replacements.append((
        r"\subsection{Empirical Constraints and Regulatory Risks}",
        r"\subsection{\update{Empirical Constraints and Regulatory Risks}}"
    ))
    replacements.append((
        "Our data linkage audit and sales analyses expose critical structural risks that have direct regulatory implications under India's Digital Personal Data Protection (DPDP) Act of 2023:",
        "\\update{Our data linkage audit and sales analyses expose critical structural risks that have direct regulatory implications under India's Digital Personal Data Protection (DPDP) Act of 2023:}"
    ))
    
    replacements.append((
        r"\item \textbf{Data Disjointness and Consent Breach}: The $0.00\%$ ASIN overlap and temporal mismatch between reviews and transactional sales data show that databases in emerging markets are highly siloed. From a regulatory perspective, attempting to merge such disjoint databases without a unified catalogue identifier creates compliance risks. If data fiduciaries attempt to link customer reviews to personal transactional profiles without explicit, renewed consent, they violate the DPDP Act's strict purpose limitation and data minimisation mandates.",
        r"\item \textbf{\update{Data Disjointness and Consent Breach}}: \update{The $0.00\%$ ASIN overlap and temporal mismatch between reviews and transactional sales data show that databases in emerging markets are highly siloed. From a regulatory perspective, attempting to merge such disjoint databases without a unified catalogue identifier creates compliance risks. If data fiduciaries attempt to link customer reviews to personal transactional profiles without explicit, renewed consent, they violate the DPDP Act's strict purpose limitation and data minimisation mandates.}"
    ))
    replacements.append((
        r"\item \textbf{Marketplace Price Synchronization}: The negligible price dispersion across the ten digital marketplaces ($\epsilon^2 = 0.0003$) suggests that catalog pricing is highly synchronized. While this prevents retail price gouging, it also indicates a high centralisation of pricing databases, which could facilitate algorithmic collusion or anticompetitive coordination if platforms share pricing engines without antitrust firewalls.",
        r"\item \textbf{\update{Marketplace Price Synchronization}}: \update{The negligible price dispersion across the ten digital marketplaces ($\epsilon^2 = 0.0003$) suggests that catalog pricing is highly synchronized. While this prevents retail price gouging, it also indicates a high centralisation of pricing databases, which could facilitate algorithmic collusion or anticompetitive coordination if platforms share pricing engines without antitrust firewalls.}"
    ))
    
    replacements.append((
        r"\subsection{Managerial Guidelines for DPDP Compliance}",
        r"\subsection{\update{Managerial Guidelines for DPDP Compliance}}"
    ))
    replacements.append((
        "To mitigate these regulatory risks, platform operators should implement three specific engineering and governance controls:",
        "\\update{To mitigate these regulatory risks, platform operators should implement three specific engineering and governance controls:}"
    ))
    replacements.append((
        r"\item \textbf{Granular Consent Capture}: Systems must implement dynamic consent architectures where users explicitly opt-in for cross-service data integration. E-commerce platforms cannot rely on blanket terms of service to merge feedback and transactional databases.",
        r"\item \textbf{\update{Granular Consent Capture}}: \update{Systems must implement dynamic consent architectures where users explicitly opt-in for cross-service data integration. E-commerce platforms cannot rely on blanket terms of service to merge feedback and transactional databases.}"
    ))
    replacements.append((
        r"\item \textbf{Anonymization Pipelines}: Automated text-scrubbing filters must strip all personally identifiable information (PII)—including names, locations, and transaction IDs—from textual reviews before they are ingested by sentiment classifiers.",
        r"\item \textbf{\update{Anonymization Pipelines}}: \update{Automated text-scrubbing filters must strip all personally identifiable information (PII)—including names, locations, and transaction IDs—from textual reviews before they are ingested by sentiment classifiers.}"
    ))
    replacements.append((
        r"\item \textbf{Immutable Provenance Logging}: Fiduciaries should deploy cryptographic, audit-ready data provenance logs to verify that customer feedback is never processed outside its original context.",
        r"\item \textbf{\update{Immutable Provenance Logging}}: \update{Fiduciaries should deploy cryptographic, audit-ready data provenance logs to verify that customer feedback is never processed outside its original context.}"
    ))
    
    replacements.append((
        r"\subsection{Barriers to Deployment and ONDC's Standardizing Role}",
        r"\subsection{\update{Barriers to Deployment and ONDC's Standardizing Role}}"
    ))
    replacements.append((
        "Transitioning from a parallel-empirical framework to an operational, live decision-support system involves significant practical hurdles, particularly for small and medium enterprises (SMEs):",
        "\\update{Transitioning from a parallel-empirical framework to an operational, live decision-support system involves significant practical hurdles, particularly for small and medium enterprises (SMEs):}"
    ))
    replacements.append((
        r"\item \textbf{API Integration and Maintenance Overhead}: Connecting disparate databases (e.g., INCREFF inventory logs, third-party logistics APIs, and customer review frontends) requires substantial software engineering resources. For SMEs, the cost of building custom API adapters and maintaining real-time data sync is a primary barrier.",
        r"\item \textbf{\update{API Integration and Maintenance Overhead}}: \update{Connecting disparate databases (e.g., INCREFF inventory logs, third-party logistics APIs, and customer review frontends) requires substantial software engineering resources. For SMEs, the cost of building custom API adapters and maintaining real-time data sync is a primary barrier.}"
    ))
    replacements.append((
        r"\item \textbf{Computational and Human-in-the-Loop Costs}: Operating deep learning models (like DistilBERT) requires expensive GPU instances. While classical models (SVM) reduce computational costs, the human-in-the-loop buffer ($0.4 < P(pos) < 0.6$) requires continuous human labour, introducing ongoing operational expenses.",
        r"\item \textbf{\update{Computational and Human-in-the-Loop Costs}}: \update{Operating deep learning models (like DistilBERT) requires expensive GPU instances. While classical models (SVM) reduce computational costs, the human-in-the-loop buffer ($0.4 < P(pos) < 0.6$) requires continuous human labour, introducing ongoing operational expenses.}"
    ))
    replacements.append((
        r"\item \textbf{ONDC as a Democratizing Protocol}: The Open Network for Digital Commerce (ONDC) represents a critical pathway to overcoming these barriers. ONDC's open-spec network protocols provide standardised schemas for product catalogues, transactions, and reviews. By adopting ONDC's open APIs, platforms can eliminate the need for proprietary database integration, allow SMEs to access calibrated sentiment and inventory analytics, and enforce compliance with data-sharing guidelines without expensive custom infrastructure.",
        r"\item \textbf{\update{ONDC as a Democratizing Protocol}}: \update{The Open Network for Digital Commerce (ONDC) represents a critical pathway to overcoming these barriers. ONDC's open-spec network protocols provide standardised schemas for product catalogues, transactions, and reviews. By adopting ONDC's open APIs, platforms can eliminate the need for proprietary database integration, allow SMEs to access calibrated sentiment and inventory analytics, and enforce compliance with data-sharing guidelines without expensive custom infrastructure.}"
    ))
    

    # --- New Replacements for Shortened Captions & Expanded Paragraphs ---
    
    # 1. Figure captions
    replacements.append((
        r"\caption{Proposed end-to-end framework architecture.}",
        r"\caption{\update{Proposed end-to-end framework architecture.}}"
    ))
    replacements.append((
        r"\caption{Confusion matrix for the proposed TF-IDF+LR sentiment model on the 15.00\% holdout test set ($N=78873$).}",
        r"\caption{\update{Confusion matrix for the proposed TF-IDF+LR sentiment model on the 15.00\% holdout test set ($N=78873$).}}"
    ))
    replacements.append((
        r"\caption{ROC and Precision-Recall curves overlay for the seven sentiment classifiers on the holdout set.}",
        r"\caption{\update{ROC and Precision-Recall curves overlay for the seven sentiment classifiers on the holdout set.}}"
    ))
    replacements.append((
        r"\caption{Probability calibration curves (reliability diagram) for the seven classifiers.}",
        r"\caption{\update{Probability calibration curves (reliability diagram) for the seven classifiers.}}"
    ))
    replacements.append((
        r"\caption{Heatmap of McNemar's pairwise statistical significance p-values on the holdout test set.}",
        r"\caption{\update{Heatmap of McNemar's pairwise statistical significance p-values on the holdout test set.}}"
    ))
    replacements.append((
        r"\caption{Error analysis diagnostics for the proposed TF-IDF+LR classifier.}",
        r"\caption{\update{Error analysis diagnostics for the proposed TF-IDF+LR classifier.}}"
    ))
    replacements.append((
        r"\caption{Ten-fold cross-validation performance boxplots.}",
        r"\caption{\update{Ten-fold cross-validation performance boxplots.}}"
    ))
    replacements.append((
        r"\caption{Imbalance recall comparison plot, mapping negative-class recall against overall classification accuracy across five balancing strategies.}",
        r"\caption{\update{Imbalance recall comparison plot, mapping negative-class recall against overall classification accuracy across five balancing strategies.}}"
    ))
    replacements.append((
        r"\caption{E-commerce transactional distributions, showing revenue concentration across top Indian regions (States) and individual products (SKUs).}",
        r"\caption{\update{E-commerce transactional distributions, showing revenue concentration across top Indian regions (States) and individual products (SKUs).}}"
    ))
    replacements.append((
        r"\caption{Lorenz curves for SKU and regional revenue.}",
        r"\caption{\update{Lorenz curves for SKU and regional revenue.}}"
    ))
    replacements.append((
        r"\caption{Boxplots of Maximum Retail Price (MRP) dispersion across ten digital marketplaces in India ($N=2586$).}",
        r"\caption{\update{Boxplots of Maximum Retail Price (MRP) dispersion across ten digital marketplaces in India ($N=2586$).}}"
    ))
    replacements.append((
        r"\caption{Venn diagram of the catalogue overlap audit of Unique ASINs.}",
        r"\caption{\update{Venn diagram of the catalogue overlap audit of Unique ASINs.}}"
    ))
    
    # 2. Expanded body paragraphs
    paragraph_665_target = (
        "To further evaluate the model performance and probability calibrations, the confusion matrix for the proposed TF-IDF+LR model is shown in Figure \\ref{fig:tfidflr_confusion_matrix}. The confusion matrix highlights the classifier's performance in detail, showing that the baseline model is highly accurate in the dominant positive class but exhibits misclassifications in the minority negative class. This imbalance is critical because misclassifying customer complaints as positive reviews prevents early warning detection, necessitating the balancing techniques analysed in Section \\ref{sec:class_imbalance}. The receiver operating characteristic (ROC) and precision-recall (PR) curves for all seven classifiers are plotted in Figure \\ref{fig:model_roc_pr}, illustrating the superior discriminative capability of DistilBERT and SVM. The ROC and PR curves compare the trade-offs between sensitivity and specificity; given the class imbalance, the PR-AUC curves are more representative of model utility than the ROC-AUC, confirming that DistilBERT and SVM are highly robust, whereas XGBoost and Naive Bayes suffer significant precision drops at higher recall thresholds. Figure \\ref{fig:calibration_curves} shows the probability calibration curves (reliability diagrams) for the classifiers, demonstrating the high calibration of SVM and DistilBERT relative to other classifiers. The calibration curves measure the reliability of the predicted probabilities, validating their use in risk-based routing, where uncertain reviews (confidence near 0.5) are escalated to human moderators. The stability of the classifiers was validated using the ten-fold cross-validation boxplots presented in Figure \\ref{fig:cv_boxplots}, supporting Hypothesis H1a. The cross-validation boxplots show that classical models are highly stable across folds, whereas DistilBERT exhibits a wider variance due to training constraints. The tight distributions for SVM and TF-IDF+LR confirm their stability across folds, whereas DistilBERT displays wider variance (due to subsampling limits and single-epoch training in CV), highlighting the sensitivity of transformers to training scale and convergence limits. Furthermore, the pairwise statistical significance p-value heatmap resulting from McNemar's test is shown in Figure \\ref{fig:mcnemar_heatmap}, confirming that all classifier boundaries differ significantly, except for the default logistic regression and TF-IDF+LR. The McNemar's test heatmap provides statistical verification that the model differences are robust and not due to random variation. Cells indicate p-values from chi-squared testing of prediction contingency tables. The non-significant difference between default Logistic Regression and our proposed TF-IDF+LR ($p = 0.6047 > 0.05$) indicates equivalent decision boundaries, while all other comparisons are highly significant ($p < 0.001$), confirming distinct error patterns across other model architectures. Finally, a diagnostic error analysis for the proposed TF-IDF+LR classifier is illustrated in Figure \\ref{fig:error_analysis_tfidf_lr}, detailing the prediction confidence and string lengths of misclassifications. The error analysis diagnostics show that document length is not a primary driver of error, while the concentration of errors near the 0.5 probability threshold justifies a human-in-the-loop buffer region. The overlap in length distributions indicates that document length is not a primary driver of misclassification, while the concentration of errors near the 0.5 probability threshold highlights that prediction uncertainty is the main source of error, justifying a human-in-the-loop buffer region."
    )
    paragraph_665_replacement = (
        "To further evaluate the model performance and probability calibrations, the confusion matrix for the proposed TF-IDF+LR model is shown in Figure \\ref{fig:tfidflr_confusion_matrix}. The confusion matrix highlights the classifier's performance in detail, showing that the baseline model is highly accurate in the dominant positive class but exhibits misclassifications in the minority negative class. This imbalance is critical because misclassifying customer complaints as positive reviews prevents early warning detection, necessitating the balancing techniques analysed in Section \\ref{sec:class_imbalance}. The receiver operating characteristic (ROC) and precision-recall (PR) curves for all seven classifiers are plotted in Figure \\ref{fig:model_roc_pr}, illustrating the superior discriminative capability of DistilBERT and SVM. The ROC and PR curves compare the trade-offs between sensitivity and specificity; given the class imbalance, the PR-AUC curves are more representative of model utility than the ROC-AUC, confirming that DistilBERT and SVM are highly robust, whereas XGBoost and Naive Bayes suffer significant precision drops at higher recall thresholds. Figure \\ref{fig:calibration_curves} shows the probability calibration curves (reliability diagrams) for the classifiers, demonstrating the high calibration of SVM and DistilBERT relative to other classifiers. The calibration curves measure the reliability of the predicted probabilities, validating their use in risk-based routing, where uncertain reviews (confidence near 0.5) are escalated to human moderators. The stability of the classifiers was validated using the ten-fold cross-validation boxplots presented in Figure \\ref{fig:cv_boxplots}, supporting Hypothesis H1a. The cross-validation boxplots show that classical models are highly stable across folds, whereas DistilBERT exhibits a wider variance due to training constraints. \\update{The tight distributions for SVM and TF-IDF+LR confirm their stability across folds, whereas DistilBERT displays wider variance (due to subsampling limits and single-epoch training in CV), highlighting the sensitivity of transformers to training scale and convergence limits.} Furthermore, the pairwise statistical significance p-value heatmap resulting from McNemar's test is shown in Figure \\ref{fig:mcnemar_heatmap}, confirming that all classifier boundaries differ significantly, except for the default logistic regression and TF-IDF+LR. The McNemar's test heatmap provides statistical verification that the model differences are robust and not due to random variation. \\update{Cells indicate p-values from chi-squared testing of prediction contingency tables. The non-significant difference between default Logistic Regression and our proposed TF-IDF+LR ($p = 0.6047 > 0.05$) indicates equivalent decision boundaries, while all other comparisons are highly significant ($p < 0.001$), confirming distinct error patterns across other model architectures.} Finally, a diagnostic error analysis for the proposed TF-IDF+LR classifier is illustrated in Figure \\ref{fig:error_analysis_tfidf_lr}, detailing the prediction confidence and string lengths of misclassifications. The error analysis diagnostics show that document length is not a primary driver of error, while the concentration of errors near the 0.5 probability threshold justifies a human-in-the-loop buffer region. \\update{The overlap in length distributions indicates that document length is not a primary driver of misclassification, while the concentration of errors near the 0.5 probability threshold highlights that prediction uncertainty is the main source of error, justifying a human-in-the-loop buffer region.}"
    )
    replacements.append((paragraph_665_target, paragraph_665_replacement))

    paragraph_43_target = (
        "Class-balancing experiments demonstrate that while the baseline model achieves strong accuracy ($96.19\\%$), it suffers from a lower negative-class recall ($84.80\\%$). Implementing synthetic oversampling (SMOTE) or class weighting significantly improves the negative-class recall to $91.68\\%$ and $93.61\\%$ respectively, with only a minor decline in overall accuracy. For Industry 5.0 systems, SMOTE represents the optimal balance, yielding a negative-class F1-score of $87.14\\%$ and ensuring that customer dissatisfaction signals are not overlooked. The trade-offs between overall accuracy and negative-class recall across different balancing strategies are visually summarised in Figure \\ref{fig:imbalance_comparison}, supporting Hypothesis H1b. The imbalance recall comparison (Figure \\ref{fig:imbalance_comparison}) illustrates the operational trade-offs of class balancing. It shows that as negative recall improves (moving from baseline to class weighting), the overall accuracy declines slightly. The trade-off curve illustrates that while baseline and SMOTE preserve high accuracy, class weighting maximises negative recall ($93.61\\%$) at the cost of a minor accuracy drop, guiding operators in selecting the optimal deployment strategy based on risk aversion. This visualisation is significant for e-commerce operators, as it maps the Pareto frontier of model performance, helping them select the optimal strategy depending on whether they prioritise overall accuracy (baseline/SMOTE) or risk mitigation (class weighting)."
    )
    paragraph_43_replacement = (
        "Class-balancing experiments demonstrate that while the baseline model achieves strong accuracy ($96.19\\%$), it suffers from a lower negative-class recall ($84.80\\%$). Implementing synthetic oversampling (SMOTE) or class weighting significantly improves the negative-class recall to $91.68\\%$ and $93.61\\%$ respectively, with only a minor decline in overall accuracy. For Industry 5.0 systems, SMOTE represents the optimal balance, yielding a negative-class F1-score of $87.14\\%$ and ensuring that customer dissatisfaction signals are not overlooked. The trade-offs between overall accuracy and negative-class recall across different balancing strategies are visually summarised in Figure \\ref{fig:imbalance_comparison}, supporting Hypothesis H1b. The imbalance recall comparison (Figure \\ref{fig:imbalance_comparison}) illustrates the operational trade-offs of class balancing. It shows that as negative recall improves (moving from baseline to class weighting), the overall accuracy declines slightly. \\update{The trade-off curve illustrates that while baseline and SMOTE preserve high accuracy, class weighting maximises negative recall ($93.61\\%$) at the cost of a minor accuracy drop, guiding operators in selecting the optimal deployment strategy based on risk aversion.} This visualisation is significant for e-commerce operators, as it maps the Pareto frontier of model performance, helping them select the optimal strategy depending on whether they prioritise overall accuracy (baseline/SMOTE) or risk mitigation (class weighting)."
    )
    replacements.append((paragraph_43_target, paragraph_43_replacement))

    paragraph_44_target = (
        "These findings align with prior research in retail analytics and operations management, which indicates that regional sales concentrations can lead to supply chain vulnerabilities. Figure \\ref{fig:sales_distributions} illustrates the top ten Indian regions (states) and Stock Keeping Units (SKUs) by revenue. The geographical and product sales distributions reveal the skewness of e-commerce revenue, showing that a small number of states and SKUs generate most of the platform income. The steep decline in both distributions visually demonstrates the high Pareto skewness of transactional revenue, highlighting the dominance of key hubs and individual high-demand items in the distribution. This concentration is significant because it indicates that fulfilment networks and regional warehousing should be optimised specifically for these high-volume hubs to minimise transit times and costs. To formally visualise this concentration inequality, Figure \\ref{fig:lorenz_curves} presents the Lorenz curves at the SKU and regional levels, showing high skewness, which supports Hypothesis H2a. The Gini coefficients of 0.6843 (SKU) and 0.8169 (State) confirm a high concentration, especially at the geographical level. This inequality has important supply chain implications: it shows that minor disruptions in key states (such as Maharashtra or Karnataka) or supply issues for top SKUs can cause severe revenue drops, highlighting the need for dual sourcing and regional inventory redundancy. For Industry 5.0, these indicators should guide stocking and logistics planning, allowing platforms to build resilience against regional supply chain shocks."
    )
    paragraph_44_replacement = (
        "These findings align with prior research in retail analytics and operations management, which indicates that regional sales concentrations can lead to supply chain vulnerabilities. Figure \\ref{fig:sales_distributions} illustrates the top ten Indian regions (states) and Stock Keeping Units (SKUs) by revenue. The geographical and product sales distributions reveal the skewness of e-commerce revenue, showing that a small number of states and SKUs generate most of the platform income. \\update{The steep decline in both distributions visually demonstrates the high Pareto skewness of transactional revenue, highlighting the dominance of key hubs and individual high-demand items in the distribution.} This concentration is significant because it indicates that fulfilment networks and regional warehousing should be optimised specifically for these high-volume hubs to minimise transit times and costs. To formally visualise this concentration inequality, Figure \\ref{fig:lorenz_curves} presents the Lorenz curves at the SKU and regional levels, showing high skewness, which supports Hypothesis H2a. The Gini coefficients of 0.6843 (SKU) and 0.8169 (State) confirm a high concentration, especially at the geographical level. This inequality has important supply chain implications: it shows that minor disruptions in key states (such as Maharashtra or Karnataka) or supply issues for top SKUs can cause severe revenue drops, highlighting the need for dual sourcing and regional inventory redundancy. For Industry 5.0, these indicators should guide stocking and logistics planning, allowing platforms to build resilience against regional supply chain shocks."
    )
    replacements.append((paragraph_44_target, paragraph_44_replacement))

    paragraph_47_target = (
        "As detailed in Section \\ref{sec:method}, the Data Linkage Audit revealed a $0.00\\%$ ASIN match rate between the Amazon reviews corpus and the transactional sales files, as well as a temporal gap of ten years (1999–2012 reviews vs. 2022 sales). Therefore, we cannot establish causal relationships or conduct temporal lag modelling (e.g., Granger causality) to prove that sentiment spikes precede revenue declines for specific SKUs. Support for Hypothesis H3a is provided by the complete absence of overlapped ASIN catalogue mappings. This absolute divergence between the customer feedback and the transactional marketplace catalogue is visually illustrated in Figure \\ref{fig:asin_overlap}. The Venn diagram (Figure \\ref{fig:asin_overlap}) shows that there are no overlapping product identifiers between the review and transaction databases. The absolute separation indicates that any causal linking of review sentiment directly to transaction-level revenue drop lags is statistically unsupported, necessitating parallel decision-support modules. This visual separation is significant because it prevents researchers from making unsupported causal claims (e.g., that negative reviews cause sales declines for specific products). By presenting this limitation clearly, we demonstrate a major challenge in secondary e-commerce research and highlight the importance of data provenance and catalogue alignment under the DPDP Act of 2023."
    )
    paragraph_47_replacement = (
        "As detailed in Section \\ref{sec:method}, the Data Linkage Audit revealed a $0.00\\%$ ASIN match rate between the Amazon reviews corpus and the transactional sales files, as well as a temporal gap of ten years (1999–2012 reviews vs. 2022 sales). Therefore, we cannot establish causal relationships or conduct temporal lag modelling (e.g., Granger causality) to prove that sentiment spikes precede revenue declines for specific SKUs. Support for Hypothesis H3a is provided by the complete absence of overlapped ASIN catalogue mappings. This absolute divergence between the customer feedback and the transactional marketplace catalogue is visually illustrated in Figure \\ref{fig:asin_overlap}. The Venn diagram (Figure \\ref{fig:asin_overlap}) shows that there are no overlapping product identifiers between the review and transaction databases. \\update{The absolute separation indicates that any causal linking of review sentiment directly to transaction-level revenue drop lags is statistically unsupported, necessitating parallel decision-support modules.} This visual separation is significant because it prevents researchers from making unsupported causal claims (e.g., that negative reviews cause sales declines for specific products). By presenting this limitation clearly, we demonstrate a major challenge in secondary e-commerce research and highlight the importance of data provenance and catalogue alignment under the DPDP Act of 2023."
    )
    replacements.append((paragraph_47_target, paragraph_47_replacement))

    print(f"Loaded {len(replacements)} replacements.")
    success_count = 0
    for idx, (old, new) in enumerate(replacements, 1):
        if old in content:
            content = content.replace(old, new)
            print(f"Block {idx} SUCCESS")
            success_count += 1
        else:
            print(f"Block {idx} FAILED")
            first_chars = old[:60] if isinstance(old, str) else str(old)[:60]
            print(f"  Target starts with: '{first_chars}'")
            
    with open(rev_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Highlighting complete: {success_count}/{len(replacements)} succeeded.")

if __name__ == "__main__":
    main()
