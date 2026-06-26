import os
import re
import shutil

main_path = r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\main.tex"
rev_path = r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\main_revision.tex"

# Copy main.tex to main_revision.tex to sync them completely
shutil.copyfile(main_path, rev_path)

with open(rev_path, "r", encoding="utf-8") as f:
    content = f.read()

# List of replacements to apply highlights
replacements = []

# 1. Title
replacements.append((
    r"\title{AI-Driven E-commerce in the Industry 5.0 Era: A Parallel-Empirical Framework, Decision Support, and Methodological Caution}",
    r"\title{\update{AI-Driven E-commerce in the Industry 5.0 Era: A Parallel-Empirical Framework, Decision Support, and Methodological Caution}}"
))

# 2. Abstract
abstract_pattern = r"\\begin\{abstract\}(.*?)\\end\{abstract\}"
match = re.search(abstract_pattern, content, re.DOTALL)
if match:
    abstract_text = match.group(1).strip()
    replacements.append((
        f"\\begin{{abstract}}\n{abstract_text}\n\\end{{abstract}}",
        f"\\begin{{abstract}}\n\\update{{{abstract_text}}}\n\\end{{abstract}}"
    ))

# 3. Introduction Paragraph 2
intro_p2 = (
    "This study addresses these gaps by presenting a parallel-empirical framework and methodological caution, "
    "structured around the Resource-Based View (RBV), the Technology-Organisation-Environment (TOE) framework, "
    "and the Technology Acceptance Model (TAM). We evaluate seven sentiment models—ranging from classical machine "
    "learning to deep transformer architectures–on a large-scale product review corpus and perform operational sales, "
    "price dispersion, and fulfilment data audits on a detailed transactional dataset."
)
replacements.append((intro_p2, f"\\update{{{intro_p2}}}"))

# 4. Introduction Paragraph 3
intro_p3 = (
    "Recognising this, we analysed these datasets as parallel empirical components, representing a methodological "
    "caution for researchers and managers against assuming seamless secondary e-commerce data integration. Rather "
    "than asserting a unified causal link, the paper demonstrates parallel analytics modules: sentiment-model "
    "benchmarking and transactional-sales analytics. This framing prevents unsupported causal linkages while "
    "demonstrating the practical application of both components in a parallel Industry 5.0 decision-support blueprint."
)
replacements.append((intro_p3, f"\\update{{{intro_p3}}}"))

# 5. Hypotheses Intro sentence
hyp_intro = "We formulate three core Research Questions (RQs) and four testable Hypotheses (Hs) that are directly evaluated via statistical and mathematical modelling:"
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
    r"\item \textit{H2b}: Maximum Retail Prices (MRPs) across Indian digital marketplaces exhibit statistically significant but practically negligible dispersion, indicating a synchronised master catalogue.",
    r"\item \textit{H2b}: \update{Maximum Retail Prices (MRPs) across Indian digital marketplaces exhibit statistically significant but practically negligible dispersion, indicating a synchronised master catalogue.}"
))
replacements.append((
    r"\item \textbf{RQ3 (Data Linkage Audit)}: Do public Amazon review corpora and Indian transactional e-commerce sales datasets show sufficient alignment to support integrated empirical predictions?",
    r"\item \textbf{RQ3 (Data Linkage Audit)}: \update{Do public Amazon review corpora and Indian transactional e-commerce sales datasets show sufficient alignment to support integrated empirical predictions?}"
))
replacements.append((
    r"\item \textit{H3}: A formal data linkage audit exposes metadata, catalogue, and temporal disjointness in secondary datasets, demonstrating the empirical limitations of integrating disparate secondary retail streams.",
    r"\item \textit{H3}: \update{A formal data linkage audit exposes metadata, catalogue, and temporal disjointness in secondary datasets, demonstrating the empirical limitations of integrating disparate secondary retail streams.}"
))

# 7. Implications paragraph in Section 1
impl_para = "The broader implications of these results for Industry 5.0 (sustainability, human-centricity, resilience) and DPDP Act compliance are framed as interpretive translations and managerial guidelines rather than direct empirical findings."
replacements.append((impl_para, f"\\update{{{impl_para}}}"))

# 8. Intro Contribution 4
contrib4 = "Fourth, we integrate these empirical findings into a regulatory-aware, human-in-the-loop decision-support blueprint aligned with Industry 5.0 principles and Viksit Bharat@2047 \cite{nitiaayog2026dpi}."
replacements.append((contrib4, f"\\update{{{contrib4}}}"))

# 9. Table 2 Operationalisation Matrix introduction paragraph and table captions/cells
table2_intro = "To bridge these theoretical paradigms with the practical requirements of e-commerce operations, we explicitly operationalise the three core pillars of Industry 5.0 through specific empirical indicators and decision thresholds evaluated in this study, as summarised in Table \\ref{tab:industry5_operationalisation}."
replacements.append((table2_intro, f"\\update{{{table2_intro}}}"))

table2_content = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\small\n"
    "\\renewcommand{\\arraystretch}{1.2}\n"
    "\\caption{Industry 5.0 empirical translation and operationalisation matrix.}\n"
    "\\label{tab:industry5_operationalisation}\n"
    "\\begin{tabular}{p{2.5cm}p{4cm}p{4cm}p{4.5cm}}\n"
    "\\hline\n"
    "\\textbf{Pillar} & \\textbf{Conceptual Definition} & \\textbf{Empirical Indicator / Measure} & \\textbf{Decision Threshold \\& Implementation} \\\\ \\hline\n"
    "Human-Centricity & Preserving human agency, avoiding automated bias, and enabling human-in-the-loop oversight. & Platt-calibrated probability confidence levels $P(pos)$ and error diagnostic distributions. & Dual-boundary routing: Route reviews with $0.4 \\leq P(pos) \\leq 0.6$ (uncertain region) to human moderators. \\\\\n"
    "Sustainability (Green AI) & Minimising environmental and carbon footprints of digital infrastructure. & Algorithmic time complexity (linear $O(N \\cdot L)$ vs. quadratic $O(N \\cdot L^2 \\cdot D \\cdot H)$); training runtime and hardware requirements. & Prefer linear, CPU-trainable classifiers (TF-IDF+LR) for high-frequency operational batching. \\\\\n"
    "Resilience & Protecting supply chains and platforms against localised demand and supply shocks. & Product-level (SKU) and geographical revenue concentrations (Gini coefficients, HHI). & Dynamic logistics buffer allocation: high-Gini SKUs ($G=0.6843$) and states ($G=0.8169$) trigger supply chain alerts upon negative sentiment flags. \\\\ \\hline\n"
    "\\end{tabular}\n"
    "\\end{table}"
)

table2_updated = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\small\n"
    "\\renewcommand{\\arraystretch}{1.2}\n"
    "\\caption{\\update{Industry 5.0 empirical translation and operationalisation matrix.}}\n"
    "\\label{tab:industry5_operationalisation}\n"
    "\\begin{tabular}{p{2.5cm}p{4cm}p{4cm}p{4.5cm}}\n"
    "\\hline\n"
    "\\textbf{\\update{Pillar}} & \\textbf{\\update{Conceptual Definition}} & \\textbf{\\update{Empirical Indicator / Measure}} & \\textbf{\\update{Decision Threshold \\& Implementation}} \\\\ \\hline\n"
    "\\update{Human-Centricity} & \\update{Preserving human agency, avoiding automated bias, and enabling human-in-the-loop oversight.} & \\update{Platt-calibrated probability confidence levels $P(pos)$ and error diagnostic distributions.} & \\update{Dual-boundary routing: Route reviews with $0.4 \\leq P(pos) \\leq 0.6$ (uncertain region) to human moderators.} \\\\\n"
    "\\update{Sustainability (Green AI)} & \\update{Minimising environmental and carbon footprints of digital infrastructure.} & \\update{Algorithmic time complexity (linear $O(N \\cdot L)$ vs. quadratic $O(N \\cdot L^2 \\cdot D \\cdot H)$); training runtime and hardware requirements.} & \\update{Prefer linear, CPU-trainable classifiers (TF-IDF+LR) for high-frequency operational batching.} \\\\\n"
    "\\update{Resilience} & \\update{Protecting supply chains and platforms against localised demand and supply shocks.} & \\update{Product-level (SKU) and geographical revenue concentrations (Gini coefficients, HHI).} & \\update{Dynamic logistics buffer allocation: high-Gini SKUs ($G=0.6843$) and states ($G=0.8169$) trigger supply chain alerts upon negative sentiment flags.} \\\\ \\hline\n"
    "\\end{tabular}\n"
    "\\end{table}"
)
replacements.append((table2_content, table2_updated))

# 10. Architecture diagram paragraph
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

# 11. Section 4.5 Title, body text and figure caption
sec45_content = (
    "\\subsection{Data-Quality Audit of Multi-Channel Fulfilment Integration}\n"
    "Rather than an empirical comparison of alternative fulfilment channels, the profitability analysis serves as a data-quality audit finding that highlights the structural siloing of e-commerce databases. While the INCREFF channel records yield unit profitability metrics (mean: \\rupee~15.07 $\\pm$ \\rupee~30.22 per unit, $N = 10$, median: \\rupee~5.50), the transactional logs for the Shiprocket channel contain zero valid rows ($N = 0$) due to missing database records and API integration failures. Consequently, standard comparative statistical tests (e.g., Welch's t-test) cannot be performed. This missing data is represented in Figure \\ref{fig:channel_profitability} as a blank comparator, demonstrating how the absence of standardised application programming interfaces (APIs) and unified data logging restricts multi-channel operational visibility. In an Industry 5.0 framework, this finding highlights the need for robust, standardised data governance (such as through ONDC-compatible schemas) to enable transparent, automated fulfilment auditing.\n"
    "\n"
    "\\begin{figure}[htbp]\n"
    "    \\centering\n"
    "    \\includegraphics[width=0.75\\linewidth]{figures/channel_profitability.eps}\n"
    "    \\caption{Fulfilment unit profitability distribution (INCREFF channel only, $N=10$, mean: \\rupee~15.07 $\\pm$ \\rupee~30.22). Shiprocket records are absent ($N=0$) due to database constraints, serving as an empirical audit marker of system disjointness rather than a comparative performance index.}\n"
    "    \\label{fig:channel_profitability}\n"
    "\\end{figure}"
)

sec45_updated = (
    "\\subsection{\\update{Data-Quality Audit of Multi-Channel Fulfilment Integration}}\n"
    "\\update{Rather than an empirical comparison of alternative fulfilment channels, the profitability analysis serves as a data-quality audit finding that highlights the structural siloing of e-commerce databases. While the INCREFF channel records yield unit profitability metrics (mean: \\rupee~15.07 $\\pm$ \\rupee~30.22 per unit, $N = 10$, median: \\rupee~5.50), the transactional logs for the Shiprocket channel contain zero valid rows ($N = 0$) due to missing database records and API integration failures. Consequently, standard comparative statistical tests (e.g., Welch's t-test) cannot be performed. This missing data is represented in Figure \\ref{fig:channel_profitability} as a blank comparator, demonstrating how the absence of standardised application programming interfaces (APIs) and unified data logging restricts multi-channel operational visibility. In an Industry 5.0 framework, this finding highlights the need for robust, standardised data governance (such as through ONDC-compatible schemas) to enable transparent, automated fulfilment auditing.}\n"
    "\n"
    "\\begin{figure}[htbp]\n"
    "    \\centering\n"
    "    \\includegraphics[width=0.75\\linewidth]{figures/channel_profitability.eps}\n"
    "    \\caption{\\update{Fulfilment unit profitability distribution (INCREFF channel only, $N=10$, mean: \\rupee~15.07 $\\pm$ \\rupee~30.22). Shiprocket records are absent ($N=0$) due to database constraints, serving as an empirical audit marker of system disjointness rather than a comparative performance index.}}\n"
    "    \\label{fig:channel_profitability}\n"
    "\\end{figure}"
)
replacements.append((sec45_content, sec45_updated))

# 12. Section 4.6 Title, body text, table, and concluding paragraph
sec46_content = (
    "\\subsection{Operational Decision Support Matrix and Escalation Logic}\n"
    "To bridge the parallel analytics streams (sentiment classification and sales operational metrics) under a unified management control loop, we propose a concrete decision support matrix and routing logic (summarised in Table \\ref{tab:decision_matrix}). This matrix maps calibrated sentiment classification probabilities, $P(pos)$, against SKU-level and geographic Gini concentration thresholds to guide inventory allocation and customer relationship management. \n"
    "\n"
    "We define three sentiment routing zones based on calibrated prediction confidence:\n"
    "\\begin{itemize}\n"
    "    \\item \\textbf{Automated CRM Routing ($P(pos) \\ge 0.6$ or $P(pos) \\le 0.4$ with low SKU/regional Gini coefficients)}: Clear sentiment signals on non-critical, low-concentration items are processed automatically by downstream email or refund systems without human intervention.\n"
    "    \\item \\textbf{Human-in-the-Loop Buffer ($0.4 < P(pos) < 0.6$)}: Predictively ambiguous reviews are routed to senior moderators for manual verification. This operationalises the human-centric pillar of Industry 5.0, preventing algorithmic misclassification from driving incorrect customer responses.\n"
    "    \\item \\textbf{Escalation Routing ($P(pos) \\le 0.4$ on high-concentration SKUs or regions)}: Negative sentiment signals for high-risk products (SKUs exceeding the Gini threshold of 0.6843 or regions exceeding the Gini threshold of 0.8169) bypass automated ticketing and are escalated directly to supply chain and inventory managers for immediate dual-sourcing or quality audits.\n"
    "\\end{itemize}\n"
    "\n"
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\renewcommand{\\arraystretch}{1.15}\n"
    "\\caption{Operational Decision Support Matrix and Escalation Logic}\n"
    "\\label{tab:decision_matrix}\n"
    "\\begin{tabular}{llll}\n"
    "\\hline\n"
    "\\textbf{Sentiment Range} & \\textbf{SKU/Region Concentration} & \\textbf{Operational Action} & \\textbf{System Route} \\\\ \\hline\n"
    "$P(pos) \\ge 0.6$ & Any Concentration & Automated promotion log & Standard Batch Dashboard \\\\\n"
    "$0.4 < P(pos) < 0.6$ & Any Concentration & Manual sentiment verification & Human-in-the-Loop Buffer \\\\\n"
    "$P(pos) \\le 0.4$ & Low (SKU Gini $\\le 0.6843$) & Automated refund/response tick. & Standard CRM ticketing \\\\\n"
    "$P(pos) \\le 0.4$ & High (SKU Gini $> 0.6843$) & Immediate supply audit & Supply Chain Manager Alert \\\\\n"
    "$P(pos) \\le 0.4$ & High (Region Gini $> 0.8169$) & Regional inventory reallocation & Logistics Coordinator Alert \\\\ \\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
    "\n"
    "This decision matrix operationalises resilience and human centricity. By routing high-risk negative feedback on dominant SKUs or concentrated regions directly to human operators, the framework helps platforms mitigate supply chain vulnerabilities and prevent cascading regional service failures."
)

sec46_updated = (
    "\\subsection{\\update{Operational Decision Support Matrix and Escalation Logic}}\n"
    "\\update{To bridge the parallel analytics streams (sentiment classification and sales operational metrics) under a unified management control loop, we propose a concrete decision support matrix and routing logic (summarised in Table \\ref{tab:decision_matrix}). This matrix maps calibrated sentiment classification probabilities, $P(pos)$, against SKU-level and geographic Gini concentration thresholds to guide inventory allocation and customer relationship management. }\n"
    "\n"
    "\\update{We define three sentiment routing zones based on calibrated prediction confidence:}\n"
    "\\begin{itemize}\n"
    "    \\item \\textbf{\\update{Automated CRM Routing ($P(pos) \\ge 0.6$ or $P(pos) \\le 0.4$ with low SKU/regional Gini coefficients)}}: \\update{Clear sentiment signals on non-critical, low-concentration items are processed automatically by downstream email or refund systems without human intervention.}\n"
    "    \\item \\textbf{\\update{Human-in-the-Loop Buffer ($0.4 < P(pos) < 0.6$)}}: \\update{Predictively ambiguous reviews are routed to senior moderators for manual verification. This operationalises the human-centric pillar of Industry 5.0, preventing algorithmic misclassification from driving incorrect customer responses.}\n"
    "    \\item \\textbf{\\update{Escalation Routing ($P(pos) \\le 0.4$ on high-concentration SKUs or regions)}}: \\update{Negative sentiment signals for high-risk products (SKUs exceeding the Gini threshold of 0.6843 or regions exceeding the Gini threshold of 0.8169) bypass automated ticketing and are escalated directly to supply chain and inventory managers for immediate dual-sourcing or quality audits.}\n"
    "\\end{itemize}\n"
    "\n"
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\renewcommand{\\arraystretch}{1.15}\n"
    "\\caption{\\update{Operational Decision Support Matrix and Escalation Logic}}\n"
    "\\label{tab:decision_matrix}\n"
    "\\begin{tabular}{llll}\n"
    "\\hline\n"
    "\\textbf{\\update{Sentiment Range}} & \\textbf{\\update{SKU/Region Concentration}} & \\textbf{\\update{Operational Action}} & \\textbf{\\update{System Route}} \\\\ \\hline\n"
    "\\update{$P(pos) \\ge 0.6$} & \\update{Any Concentration} & \\update{Automated promotion log} & \\update{Standard Batch Dashboard} \\\\\n"
    "\\update{$0.4 < P(pos) < 0.6$} & \\update{Any Concentration} & \\update{Manual sentiment verification} & \\update{Human-in-the-Loop Buffer} \\\\\n"
    "\\update{$P(pos) \\le 0.4$} & \\update{Low (SKU Gini $\\le 0.6843$)} & \\update{Automated refund/response tick.} & \\update{Standard CRM ticketing} \\\\\n"
    "\\update{$P(pos) \\le 0.4$} & \\update{High (SKU Gini $> 0.6843$)} & \\update{Immediate supply audit} & \\update{Supply Chain Manager Alert} \\\\\n"
    "\\update{$P(pos) \\le 0.4$} & \\update{High (Region Gini $> 0.8169$)} & \\update{Regional inventory reallocation} & \\update{Logistics Coordinator Alert} \\\\ \\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
    "\n"
    "\\update{This decision matrix operationalises resilience and human centricity. By routing high-risk negative feedback on dominant SKUs or concentrated regions directly to human operators, the framework helps platforms mitigate supply chain vulnerabilities and prevent cascading regional service failures.}"
)
replacements.append((sec46_content, sec46_updated))

# 13. Section 5 Ethical, Regulatory and Governance
sec5_content = (
    "The integration of AI into e-commerce operations introduces complex regulatory and ethical responsibilities. In this section, we distinguish between our direct empirical findings (such as disjoint catalogues and marketplace price consistency) and the normative managerial guidelines required to achieve regulatory compliance and ethical alignment under the Industry 5.0 paradigm.\n"
    "\n"
    "\\subsection{Empirical Constraints and Regulatory Risks}\n"
    "Our data linkage audit and sales analyses expose critical structural risks that have direct regulatory implications under India's Digital Personal Data Protection (DPDP) Act of 2023:\n"
    "\\begin{itemize}\n"
    "    \\item \\textbf{Data Disjointness and Consent Breach}: The $0.00\%$ ASIN overlap and temporal mismatch between reviews and transactional sales data show that databases in emerging markets are highly siloed. From a regulatory perspective, attempting to merge such disjoint databases without a unified catalogue identifier creates compliance risks. If data fiduciaries attempt to link customer reviews to personal transactional profiles without explicit, renewed consent, they violate the DPDP Act's strict purpose limitation and data minimisation mandates.\n"
    "    \\item \\textbf{Marketplace Price Synchronization}: The negligible price dispersion across the ten digital marketplaces ($\\epsilon^2 = 0.0003$) suggests that catalog pricing is highly synchronized. While this prevents retail price gouging, it also indicates a high centralisation of pricing databases, which could facilitate algorithmic collusion or anticompetitive coordination if platforms share pricing engines without antitrust firewalls.\n"
    "\\end{itemize}\n"
    "\n"
    "\\subsection{Managerial Guidelines for DPDP Compliance}\n"
    "To mitigate these regulatory risks, platform operators should implement three specific engineering and governance controls:\n"
    "\\begin{itemize}\n"
    "    \\item \\textbf{Granular Consent Capture}: Systems must implement dynamic consent architectures where users explicitly opt-in for cross-service data integration. E-commerce platforms cannot rely on blanket terms of service to merge feedback and transactional databases.\n"
    "    \\item \\textbf{Anonymization Pipelines}: Automated text-scrubbing filters must strip all personally identifiable information (PII)—including names, locations, and transaction IDs—from textual reviews before they are ingested by sentiment classifiers.\n"
    "    \\item \\textbf{Immutable Provenance Logging}: Fiduciaries should deploy cryptographic, audit-ready data provenance logs to verify that customer feedback is never processed outside its original context.\n"
    "\\end{itemize}\n"
    "\n"
    "\\subsection{Barriers to Deployment and ONDC's Standardizing Role}\n"
    "Transitioning from a parallel-empirical framework to an operational, live decision-support system involves significant practical hurdles, particularly for small and medium enterprises (SMEs):\n"
    "\\begin{itemize}\n"
    "    \\item \\textbf{API Integration and Maintenance Overhead}: Connecting disparate databases (e.g., INCREFF inventory logs, third-party logistics APIs, and customer review frontends) requires substantial software engineering resources. For SMEs, the cost of building custom API adapters and maintaining real-time data sync is a primary barrier.\n"
    "    \\item \\textbf{Computational and Human-in-the-Loop Costs}: Operating deep learning models (like DistilBERT) requires expensive GPU instances. While classical models (SVM) reduce computational costs, the human-in-the-loop buffer ($0.4 < P(pos) < 0.6$) requires continuous human labour, introducing ongoing operational expenses.\n"
    "    \\item \\textbf{ONDC as a Democratizing Protocol}: The Open Network for Digital Commerce (ONDC) represents a critical pathway to overcoming these barriers. ONDC's open-spec network protocols provide standardised schemas for product catalogues, transactions, and reviews. By adopting ONDC's open APIs, platforms can eliminate the need for proprietary database integration, allow SMEs to access calibrated sentiment and inventory analytics, and enforce compliance with data-sharing guidelines without expensive custom infrastructure.\n"
    "\\end{itemize}"
)

sec5_updated = (
    "\\update{The integration of AI into e-commerce operations introduces complex regulatory and ethical responsibilities. In this section, we distinguish between our direct empirical findings (such as disjoint catalogues and marketplace price consistency) and the normative managerial guidelines required to achieve regulatory compliance and ethical alignment under the Industry 5.0 paradigm.}\n"
    "\n"
    "\\subsection{\\update{Empirical Constraints and Regulatory Risks}}\n"
    "\\update{Our data linkage audit and sales analyses expose critical structural risks that have direct regulatory implications under India's Digital Personal Data Protection (DPDP) Act of 2023:}\n"
    "\\begin{itemize}\n"
    "    \\item \\textbf{\\update{Data Disjointness and Consent Breach}}: \\update{The $0.00\%$ ASIN overlap and temporal mismatch between reviews and transactional sales data show that databases in emerging markets are highly siloed. From a regulatory perspective, attempting to merge such disjoint databases without a unified catalogue identifier creates compliance risks. If data fiduciaries attempt to link customer reviews to personal transactional profiles without explicit, renewed consent, they violate the DPDP Act's strict purpose limitation and data minimisation mandates.}\n"
    "    \\item \\textbf{\\update{Marketplace Price Synchronization}}: \\update{The negligible price dispersion across the ten digital marketplaces ($\\epsilon^2 = 0.0003$) suggests that catalog pricing is highly synchronized. While this prevents retail price gouging, it also indicates a high centralisation of pricing databases, which could facilitate algorithmic collusion or anticompetitive coordination if platforms share pricing engines without antitrust firewalls.}\n"
    "\\end{itemize}\n"
    "\n"
    "\\subsection{\\update{Managerial Guidelines for DPDP Compliance}}\n"
    "\\update{To mitigate these regulatory risks, platform operators should implement three specific engineering and governance controls:}\n"
    "\\begin{itemize}\n"
    "    \\item \\textbf{\\update{Granular Consent Capture}}: \\update{Systems must implement dynamic consent architectures where users explicitly opt-in for cross-service data integration. E-commerce platforms cannot rely on blanket terms of service to merge feedback and transactional databases.}\n"
    "    \\item \\textbf{\\update{Anonymization Pipelines}}: \\update{Automated text-scrubbing filters must strip all personally identifiable information (PII)—including names, locations, and transaction IDs—from textual reviews before they are ingested by sentiment classifiers.}\n"
    "    \\item \\textbf{\\update{Immutable Provenance Logging}}: \\update{Fiduciaries should deploy cryptographic, audit-ready data provenance logs to verify that customer feedback is never processed outside its original context.}\n"
    "\\end{itemize}\n"
    "\n"
    "\\subsection{\\update{Barriers to Deployment and ONDC's Standardizing Role}}\n"
    "\\update{Transitioning from a parallel-empirical framework to an operational, live decision-support system involves significant practical hurdles, particularly for small and medium enterprises (SMEs):}\n"
    "\\begin{itemize}\n"
    "    \\item \\textbf{\\update{API Integration and Maintenance Overhead}}: \\update{Connecting disparate databases (e.g., INCREFF inventory logs, third-party logistics APIs, and customer review frontends) requires substantial software engineering resources. For SMEs, the cost of building custom API adapters and maintaining real-time data sync is a primary barrier.}\n"
    "    \\item \\textbf{\\update{Computational and Human-in-the-Loop Costs}}: \\update{Operating deep learning models (like DistilBERT) requires expensive GPU instances. While classical models (SVM) reduce computational costs, the human-in-the-loop buffer ($0.4 < P(pos) < 0.6$) requires continuous human labour, introducing ongoing operational expenses.}\n"
    "    \\item \\textbf{\\update{ONDC as a Democratizing Protocol}}: \\update{The Open Network for Digital Commerce (ONDC) represents a critical pathway to overcoming these barriers. ONDC's open-spec network protocols provide standardised schemas for product catalogues, transactions, and reviews. By adopting ONDC's open APIs, platforms can eliminate the need for proprietary database integration, allow SMEs to access calibrated sentiment and inventory analytics, and enforce compliance with data-sharing guidelines without expensive custom infrastructure.}\n"
    "\\end{itemize}"
)
replacements.append((sec5_content, sec5_updated))

# 14. Section 6 New validation and generalizability limitation (highlighted)
# We already wrote it to main.tex. In main_revision.tex, we should highlight only the new text:
sec6_content = (
    "Furthermore, the managerial relevance of our proposed decision support matrix remains speculative as it has not "
    "been validated through real-world business applications such as retailer case studies, manager interviews, "
    "user experiments, or live organisational implementation. Additionally, the external validity and generalisability "
    "of our empirical findings are constrained by our reliance on a single product review corpus and a single "
    "transactional database. Future research should address these limitations by validating the framework in "
    "real-world retail environments and conducting replication studies across other major Indian e-commerce platforms "
    "(e.g., Flipkart, Myntra, Meesho, or live ONDC registries) to confirm generalisability."
)
replacements.append((sec6_content, f"\\update{{{sec6_content}}}"))

# 15. Supplementary Material declarations
supp_decl = (
    "\\section*{Supplementary Material}\n"
    "Supplementary material associated with this article can be found in the online version of this document. It provides extended definitions, detailed algorithmic time complexity derivations, cross-validation metrics, and additional statistical tables."
)
supp_decl_updated = (
    "\\section*{\\update{Supplementary Material}}\n"
    "\\update{Supplementary material associated with this article can be found in the online version of this document. It provides extended definitions, detailed algorithmic time complexity derivations, cross-validation metrics, and additional statistical tables.}"
)
replacements.append((supp_decl, supp_decl_updated))

success_count = 0
for idx, (target, repl) in enumerate(replacements):
    if target in content:
        content = content.replace(target, repl)
        print(f"Replacement {idx+1} SUCCESS")
        success_count += 1
    else:
        print(f"Replacement {idx+1} FAILED")
        # Let's print a small diagnostics
        first_few = target[:60]
        if first_few in content:
            print(f"  Found first 60 chars of target: '{first_few}'")
        else:
            print(f"  First 60 chars not found: '{first_few}'")

if success_count == len(replacements):
    with open(rev_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("All highlights applied successfully to main_revision.tex!")
else:
    print(f"Highlights incomplete: {success_count}/{len(replacements)} succeeded.")
