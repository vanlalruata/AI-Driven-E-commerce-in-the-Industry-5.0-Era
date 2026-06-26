import shutil
import re

main_path = r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\main.tex"
rev_path = r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\manuscript\main_revision.tex"

# Copy main.tex to main_revision.tex to sync them completely
shutil.copyfile(main_path, rev_path)

with open(main_path, "r", encoding="utf-8") as f:
    main_content = f.read()

with open(rev_path, "r", encoding="utf-8") as f:
    rev_content = f.read()

# Define blocks to highlight
# For each block: (start_sub, end_sub, format_function)
# format_function takes the exact match string and returns the replacement with \update

def wrap_update(text):
    return f"\\update{{{text}}}"

blocks = [
    # 1. Title
    (
        r"\title{AI-Driven E-commerce",
        r"Methodological Caution}",
        lambda text: text.replace(
            "AI-Driven E-commerce in the Industry 5.0 Era: A Parallel-Empirical Framework, Decision Support, and Methodological Caution",
            "\\update{AI-Driven E-commerce in the Industry 5.0 Era: A Parallel-Empirical Framework, Decision Support, and Methodological Caution}"
        )
    ),
    # 2. Abstract
    (
        r"\begin{abstract}",
        r"\end{abstract}",
        lambda text: text.replace(
            "\\begin{abstract}\n",
            "\\begin{abstract}\n\\update{"
        ).replace(
            "\n\\end{abstract}",
            "}\n\\end{abstract}"
        )
    ),
    # 3. Intro Paragraph 2
    (
        "This study addresses these gaps by presenting a parallel-empirical framework",
        "detailed transactional dataset.",
        wrap_update
    ),
    # 4. Intro Paragraph 3
    (
        "Recognising this, we analysed these datasets as parallel",
        "decision-support blueprint.",
        wrap_update
    ),
    # 5. Hypotheses Intro
    (
        "We formulate three core Research Questions (RQs) and four testable Hypotheses",
        "mathematical modelling:",
        wrap_update
    ),
    # 6. H1a
    (
        r"\item \textit{H1a}: Classical models",
        "probability calibrations.",
        lambda text: text.replace(
            r"\item \textit{H1a}: ",
            r"\item \textit{H1a}: \update{"
        ) + "}"
    ),
    # 7. H1b
    (
        r"\item \textit{H1b}: Incorporating class-balancing",
        "overall accuracy.",
        lambda text: text.replace(
            r"\item \textit{H1b}: ",
            r"\item \textit{H1b}: \update{"
        ) + "}"
    ),
    # 8. H2a
    (
        r"\item \textit{H2a}: E-commerce transactional sales",
        "Index (HHI).",
        lambda text: text.replace(
            r"\item \textit{H2a}: ",
            r"\item \textit{H2a}: \update{"
        ) + "}"
    ),
    # 9. H2b
    (
        r"\item \textit{H2b}: Maximum Retail Prices",
        "master catalogue.",
        lambda text: text.replace(
            r"\item \textit{H2b}: ",
            r"\item \textit{H2b}: \update{"
        ) + "}"
    ),
    # 10. RQ3
    (
        r"\item \textbf{RQ3 (Data Linkage Audit)}:",
        "empirical predictions?",
        lambda text: text.replace(
            r"\item \textbf{RQ3 (Data Linkage Audit)}: ",
            r"\item \textbf{RQ3 (Data Linkage Audit)}: \update{"
        ) + "}"
    ),
    # 11. H3
    (
        r"\item \textit{H3}: A formal data linkage",
        "secondary retail streams.",
        lambda text: text.replace(
            r"\item \textit{H3}: ",
            r"\item \textit{H3}: \update{"
        ) + "}"
    ),
    # 12. Implications
    (
        "The broader implications of these results for Industry 5.0",
        "empirical findings.",
        wrap_update
    ),
    # 13. Contribution 4
    (
        "Fourth, we integrate these empirical findings into a regulatory-aware",
        r"\cite{nitiaayog2026dpi}.",
        wrap_update
    ),
    # 14. Table 2 Intro
    (
        "To bridge these theoretical paradigms with the practical requirements of e-commerce",
        r"Table \ref{tab:industry5_operationalisation}.",
        wrap_update
    ),
    # 15. Table 2 content
    (
        r"\begin{table}[htbp]" + "\n" + r"\centering" + "\n" + r"\small" + "\n" + r"\renewcommand{\arraystretch}{1.2}" + "\n" + r"\caption{Industry 5.0 empirical translation",
        r"\label{tab:industry5_operationalisation}" + "\n" + r"\begin{tabular}{p{2.5cm}p{4cm}p{4cm}p{4.5cm}}" + "\n" + r"\hline" + "\n" + r"\textbf{Pillar} & \textbf{Conceptual Definition} & \textbf{Empirical Indicator / Measure} & \textbf{Decision Threshold \& Implementation} \\ \hline" + "\n" + r"Human-Centricity & Preserving human agency, avoiding automated bias, and enabling human-in-the-loop oversight. & Platt-calibrated probability confidence levels $P(pos)$ and error diagnostic distributions. & Dual-boundary routing: Route reviews with $0.4 \leq P(pos) \leq 0.6$ (uncertain region) to human moderators. \\" + "\n" + r"Sustainability (Green AI) & Minimising environmental and carbon footprints of digital infrastructure. & Algorithmic time complexity (linear $O(N \cdot L)$ vs. quadratic $O(N \cdot L^2 \cdot D \cdot H)$); training runtime and hardware requirements. & Prefer linear, CPU-trainable classifiers (TF-IDF+LR) for high-frequency operational batching. \\" + "\n" + r"Resilience & Protecting supply chains and platforms against localised demand and supply shocks. & Product-level (SKU) and geographical revenue concentrations (Gini coefficients, HHI). & Dynamic logistics buffer allocation: high-Gini SKUs ($G=0.6843$) and states ($G=0.8169$) trigger supply chain alerts upon negative sentiment flags. \\ \hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}",
        lambda text: (
            text.replace(
                r"\caption{Industry 5.0 empirical translation and operationalisation matrix.}",
                r"\caption{\update{Industry 5.0 empirical translation and operationalisation matrix.}}"
            ).replace(
                r"\textbf{Pillar} & \textbf{Conceptual Definition} & \textbf{Empirical Indicator / Measure} & \textbf{Decision Threshold \& Implementation}",
                r"\textbf{\update{Pillar}} & \textbf{\update{Conceptual Definition}} & \textbf{\update{Empirical Indicator / Measure}} & \textbf{\update{Decision Threshold \& Implementation}}"
            ).replace(
                r"Human-Centricity & Preserving human agency, avoiding automated bias, and enabling human-in-the-loop oversight. & Platt-calibrated probability confidence levels $P(pos)$ and error diagnostic distributions. & Dual-boundary routing: Route reviews with $0.4 \leq P(pos) \leq 0.6$ (uncertain region) to human moderators. \\",
                r"\update{Human-Centricity} & \update{Preserving human agency, avoiding automated bias, and enabling human-in-the-loop oversight.} & \update{Platt-calibrated probability confidence levels $P(pos)$ and error diagnostic distributions.} & \update{Dual-boundary routing: Route reviews with $0.4 \leq P(pos) \leq 0.6$ (uncertain region) to human moderators.} \\"
            ).replace(
                r"Sustainability (Green AI) & Minimising environmental and carbon footprints of digital infrastructure. & Algorithmic time complexity (linear $O(N \cdot L)$ vs. quadratic $O(N \cdot L^2 \cdot D \cdot H)$); training runtime and hardware requirements. & Prefer linear, CPU-trainable classifiers (TF-IDF+LR) for high-frequency operational batching. \\",
                r"\update{Sustainability (Green AI)} & \update{Minimising environmental and carbon footprints of digital infrastructure.} & \update{Algorithmic time complexity (linear $O(N \cdot L)$ vs. quadratic $O(N \cdot L^2 \cdot D \cdot H)$); training runtime and hardware requirements.} & \update{Prefer linear, CPU-trainable classifiers (TF-IDF+LR) for high-frequency operational batching.} \\"
            ).replace(
                r"Resilience & Protecting supply chains and platforms against localised demand and supply shocks. & Product-level (SKU) and geographical revenue concentrations (Gini coefficients, HHI). & Dynamic logistics buffer allocation: high-Gini SKUs ($G=0.6843$) and states ($G=0.8169$) trigger supply chain alerts upon negative sentiment flags. \\",
                r"\update{Resilience} & \update{Protecting supply chains and platforms against localised demand and supply shocks.} & \update{Product-level (SKU) and geographical revenue concentrations (Gini coefficients, HHI).} & \update{Dynamic logistics buffer allocation: high-Gini SKUs ($G=0.6843$) and states ($G=0.8169$) trigger supply chain alerts upon negative sentiment flags.} \\"
            )
        )
    ),
    # 16. Architecture Diagram Paragraph
    (
        "Proposed Architecture Diagram: Figure \\ref{fig:proposed_architecture} visually represents the end-to-end",
        "algorithmic explainability.",
        wrap_update
    ),
    # 17. Section 4.5
    (
        "\\subsection{Data-Quality Audit of Multi-Channel Fulfilment Integration}",
        "serving as an empirical audit marker of system disjointness rather than a comparative performance index.}\n    \\label{fig:channel_profitability}\n\\end{figure}",
        lambda text: (
            text.replace(
                "\\subsection{Data-Quality Audit of Multi-Channel Fulfilment Integration}",
                "\\subsection{\\update{Data-Quality Audit of Multi-Channel Fulfilment Integration}}"
            ).replace(
                "Rather than an empirical comparison of alternative fulfilment channels, the profitability analysis serves as a data-quality audit finding that highlights the structural siloing of e-commerce databases.",
                "\\update{Rather than an empirical comparison of alternative fulfilment channels, the profitability analysis serves as a data-quality audit finding that highlights the structural siloing of e-commerce databases."
            ).replace(
                "In an Industry 5.0 framework, this finding highlights the need for robust, standardised data governance (such as through ONDC-compatible schemas) to enable transparent, automated fulfilment auditing.\n\n\\begin{figure}[htbp]",
                "In an Industry 5.0 framework, this finding highlights the need for robust, standardised data governance (such as through ONDC-compatible schemas) to enable transparent, automated fulfilment auditing.}\n\n\\begin{figure}[htbp]"
            ).replace(
                "\\caption{Fulfilment unit profitability distribution",
                "\\caption{\\update{Fulfilment unit profitability distribution"
            ).replace(
                "rather than a comparative performance index.}",
                "rather than a comparative performance index.}}"
            )
        )
    ),
    # 18. Section 4.6
    (
        "\\subsection{Operational Decision Support Matrix and Escalation Logic}",
        "prevent cascading regional service failures.",
        lambda text: (
            text.replace(
                "\\subsection{Operational Decision Support Matrix and Escalation Logic}",
                "\\subsection{\\update{Operational Decision Support Matrix and Escalation Logic}}"
            ).replace(
                "To bridge the parallel analytics streams (sentiment classification and sales operational metrics) under a unified management control loop, we propose a concrete decision support matrix and routing logic (summarised in Table \\ref{tab:decision_matrix}). This matrix maps calibrated sentiment classification probabilities, $P(pos)$, against SKU-level and geographic Gini concentration thresholds to guide inventory allocation and customer relationship management. \n\nWe define three sentiment routing zones based on calibrated prediction confidence:\n\\begin{itemize}",
                "\\update{To bridge the parallel analytics streams (sentiment classification and sales operational metrics) under a unified management control loop, we propose a concrete decision support matrix and routing logic (summarised in Table \\ref{tab:decision_matrix}). This matrix maps calibrated sentiment classification probabilities, $P(pos)$, against SKU-level and geographic Gini concentration thresholds to guide inventory allocation and customer relationship management. }\n\n\\update{We define three sentiment routing zones based on calibrated prediction confidence:}\n\\begin{itemize}"
            ).replace(
                "\\item \\textbf{Automated CRM Routing ($P(pos) \\ge 0.6$ or $P(pos) \\le 0.4$ with low SKU/regional Gini coefficients)}: Clear sentiment signals on non-critical, low-concentration items are processed automatically by downstream email or refund systems without human intervention.",
                "\\item \\textbf{\\update{Automated CRM Routing ($P(pos) \\ge 0.6$ or $P(pos) \\le 0.4$ with low SKU/regional Gini coefficients)}}: \\update{Clear sentiment signals on non-critical, low-concentration items are processed automatically by downstream email or refund systems without human intervention.}"
            ).replace(
                "\\item \\textbf{Human-in-the-Loop Buffer ($0.4 < P(pos) < 0.6$)}: Predictively ambiguous reviews are routed to senior moderators for manual verification. This operationalises the human-centric pillar of Industry 5.0, preventing algorithmic misclassification from driving incorrect customer responses.",
                "\\item \\textbf{\\update{Human-in-the-Loop Buffer ($0.4 < P(pos) < 0.6$)}}: \\update{Predictively ambiguous reviews are routed to senior moderators for manual verification. This operationalises the human-centric pillar of Industry 5.0, preventing algorithmic misclassification from driving incorrect customer responses.}"
            ).replace(
                "\\item \\textbf{Escalation Routing ($P(pos) \\le 0.4$ on high-concentration SKUs or regions)}: Negative sentiment signals for high-risk products (SKUs exceeding the Gini threshold of 0.6843 or regions exceeding the Gini threshold of 0.8169) bypass automated ticketing and are escalated directly to supply chain and inventory managers for immediate dual-sourcing or quality audits.",
                "\\item \\textbf{\\update{Escalation Routing ($P(pos) \\le 0.4$ on high-concentration SKUs or regions)}}: \\update{Negative sentiment signals for high-risk products (SKUs exceeding the Gini threshold of 0.6843 or regions exceeding the Gini threshold of 0.8169) bypass automated ticketing and are escalated directly to supply chain and inventory managers for immediate dual-sourcing or quality audits.}"
            ).replace(
                "\\caption{Operational Decision Support Matrix and Escalation Logic}",
                "\\caption{\\update{Operational Decision Support Matrix and Escalation Logic}}"
            ).replace(
                "\\textbf{Sentiment Range} & \\textbf{SKU/Region Concentration} & \\textbf{Operational Action} & \\textbf{System Route}",
                "\\textbf{\\update{Sentiment Range}} & \\textbf{\\update{SKU/Region Concentration}} & \\textbf{\\update{Operational Action}} & \\textbf{\\update{System Route}}"
            ).replace(
                "$P(pos) \\ge 0.6$ & Any Concentration & Automated promotion log & Standard Batch Dashboard \\\\",
                "\\update{$P(pos) \\ge 0.6$} & \\update{Any Concentration} & \\update{Automated promotion log} & \\update{Standard Batch Dashboard} \\\\"
            ).replace(
                "$0.4 < P(pos) < 0.6$ & Any Concentration & Manual sentiment verification & Human-in-the-Loop Buffer \\\\",
                "\\update{$0.4 < P(pos) < 0.6$} & \\update{Any Concentration} & \\update{Manual sentiment verification} & \\update{Human-in-the-Loop Buffer} \\\\"
            ).replace(
                "$P(pos) \\le 0.4$ & Low (SKU Gini $\\le 0.6843$) & Automated refund/response tick. & Standard CRM ticketing \\\\",
                "\\update{$P(pos) \\le 0.4$} & \\update{Low (SKU Gini $\\le 0.6843$)} & \\update{Automated refund/response tick.} & \\update{Standard CRM ticketing} \\\\"
            ).replace(
                "$P(pos) \\le 0.4$ & High (SKU Gini $> 0.6843$) & Immediate supply audit & Supply Chain Manager Alert \\\\",
                "\\update{$P(pos) \\le 0.4$} & \\update{High (SKU Gini $> 0.6843$)} & \\update{Immediate supply audit} & \\update{Supply Chain Manager Alert} \\\\"
            ).replace(
                "$P(pos) \\le 0.4$ & High (Region Gini $> 0.8169$) & Regional inventory reallocation & Logistics Coordinator Alert \\\\",
                "\\update{$P(pos) \\le 0.4$} & \\update{High (Region Gini $> 0.8169$)} & \\update{Regional inventory reallocation} & \\update{Logistics Coordinator Alert} \\\\"
            ).replace(
                "This decision matrix operationalises resilience and human centricity. By routing high-risk negative feedback on dominant SKUs or concentrated regions directly to human operators, the framework helps platforms mitigate supply chain vulnerabilities and prevent cascading regional service failures.",
                "\\update{This decision matrix operationalises resilience and human centricity. By routing high-risk negative feedback on dominant SKUs or concentrated regions directly to human operators, the framework helps platforms mitigate supply chain vulnerabilities and prevent cascading regional service failures.}"
            )
        )
    ),
    # 19. Section 5
    (
        "\\section{Ethical, Regulatory, and Governance Discussion}",
        "expensive custom infrastructure.\n    \\end{itemize}",
        lambda text: (
            text.replace(
                "\\section{Ethical, Regulatory, and Governance Discussion}",
                "\\section{\\update{Ethical, Regulatory, and Governance Discussion}}"
            ).replace(
                "The integration of AI into e-commerce operations introduces complex regulatory and ethical responsibilities. In this section, we distinguish between our direct empirical findings (such as disjoint catalogues and marketplace price consistency) and the normative managerial guidelines required to achieve regulatory compliance and ethical alignment under the Industry 5.0 paradigm.",
                "\\update{The integration of AI into e-commerce operations introduces complex regulatory and ethical responsibilities. In this section, we distinguish between our direct empirical findings (such as disjoint catalogues and marketplace price consistency) and the normative managerial guidelines required to achieve regulatory compliance and ethical alignment under the Industry 5.0 paradigm.}"
            ).replace(
                "\\subsection{Empirical Constraints and Regulatory Risks}",
                "\\subsection{\\update{Empirical Constraints and Regulatory Risks}}"
            ).replace(
                "Our data linkage audit and sales analyses expose critical structural risks that have direct regulatory implications under India's Digital Personal Data Protection (DPDP) Act of 2023:\n\\begin{itemize}",
                "\\update{Our data linkage audit and sales analyses expose critical structural risks that have direct regulatory implications under India's Digital Personal Data Protection (DPDP) Act of 2023:}\n\\begin{itemize}"
            ).replace(
                "\\item \\textbf{Data Disjointness and Consent Breach}: The $0.00\\%$ ASIN overlap and temporal mismatch between reviews and transactional sales data show that databases in emerging markets are highly siloed. From a regulatory perspective, attempting to merge such disjoint databases without a unified catalogue identifier creates compliance risks. If data fiduciaries attempt to link customer reviews to personal transactional profiles without explicit, renewed consent, they violate the DPDP Act's strict purpose limitation and data minimisation mandates.",
                "\\item \\textbf{\\update{Data Disjointness and Consent Breach}}: \\update{The $0.00\\%$ ASIN overlap and temporal mismatch between reviews and transactional sales data show that databases in emerging markets are highly siloed. From a regulatory perspective, attempting to merge such disjoint databases without a unified catalogue identifier creates compliance risks. If data fiduciaries attempt to link customer reviews to personal transactional profiles without explicit, renewed consent, they violate the DPDP Act's strict purpose limitation and data minimisation mandates.}"
            ).replace(
                "\\item \\textbf{Marketplace Price Synchronization}: The negligible price dispersion across the ten digital marketplaces ($\\epsilon^2 = 0.0003$) suggests that catalog pricing is highly synchronized. While this prevents retail price gouging, it also indicates a high centralisation of pricing databases, which could facilitate algorithmic collusion or anticompetitive coordination if platforms share pricing engines without antitrust firewalls.",
                "\\item \\textbf{\\update{Marketplace Price Synchronization}}: \\update{The negligible price dispersion across the ten digital marketplaces ($\\epsilon^2 = 0.0003$) suggests that catalog pricing is highly synchronized. While this prevents retail price gouging, it also indicates a high centralisation of pricing databases, which could facilitate algorithmic collusion or anticompetitive coordination if platforms share pricing engines without antitrust firewalls.}"
            ).replace(
                "\\subsection{Managerial Guidelines for DPDP Compliance}",
                "\\subsection{\\update{Managerial Guidelines for DPDP Compliance}}"
            ).replace(
                "To mitigate these regulatory risks, platform operators should implement three specific engineering and governance controls:\n\\begin{itemize}",
                "\\update{To mitigate these regulatory risks, platform operators should implement three specific engineering and governance controls:}\n\\begin{itemize}"
            ).replace(
                "\\item \\textbf{Granular Consent Capture}: Systems must implement dynamic consent architectures where users explicitly opt-in for cross-service data integration. E-commerce platforms cannot rely on blanket terms of service to merge feedback and transactional databases.",
                "\\item \\textbf{\\update{Granular Consent Capture}}: \\update{Systems must implement dynamic consent architectures where users explicitly opt-in for cross-service data integration. E-commerce platforms cannot rely on blanket terms of service to merge feedback and transactional databases.}"
            ).replace(
                "\\item \\textbf{Anonymization Pipelines}: Automated text-scrubbing filters must strip all personally identifiable information (PII)—including names, locations, and transaction IDs—from textual reviews before they are ingested by sentiment classifiers.",
                "\\item \\textbf{\\update{Anonymization Pipelines}}: \\update{Automated text-scrubbing filters must strip all personally identifiable information (PII)—including names, locations, and transaction IDs—from textual reviews before they are ingested by sentiment classifiers.}"
            ).replace(
                "\\item \\textbf{Immutable Provenance Logging}: Fiduciaries should deploy cryptographic, audit-ready data provenance logs to verify that customer feedback is never processed outside its original context.",
                "\\item \\textbf{\\update{Immutable Provenance Logging}}: \\update{Fiduciaries should deploy cryptographic, audit-ready data provenance logs to verify that customer feedback is never processed outside its original context.}"
            ).replace(
                "\\subsection{Barriers to Deployment and ONDC's Standardizing Role}",
                "\\subsection{\\update{Barriers to Deployment and ONDC's Standardizing Role}}"
            ).replace(
                "Transitioning from a parallel-empirical framework to an operational, live decision-support system involves significant practical hurdles, particularly for small and medium enterprises (SMEs):\n\\begin{itemize}",
                "\\update{Transitioning from a parallel-empirical framework to an operational, live decision-support system involves significant practical hurdles, particularly for small and medium enterprises (SMEs):}\n\\begin{itemize}"
            ).replace(
                "\\item \\textbf{API Integration and Maintenance Overhead}: Connecting disparate databases (e.g., INCREFF inventory logs, third-party logistics APIs, and customer review frontends) requires substantial software engineering resources. For SMEs, the cost of building custom API adapters and maintaining real-time data sync is a primary barrier.",
                "\\item \\textbf{\\update{API Integration and Maintenance Overhead}}: \\update{Connecting disparate databases (e.g., INCREFF inventory logs, third-party logistics APIs, and customer review frontends) requires substantial software engineering resources. For SMEs, the cost of building custom API adapters and maintaining real-time data sync is a primary barrier.}"
            ).replace(
                "\\item \\textbf{Computational and Human-in-the-Loop Costs}: Operating deep learning models (like DistilBERT) requires expensive GPU instances. While classical models (SVM) reduce computational costs, the human-in-the-loop buffer ($0.4 < P(pos) < 0.6$) requires continuous human labour, introducing ongoing operational expenses.",
                "\\item \\textbf{\\update{Computational and Human-in-the-Loop Costs}}: \\update{Operating deep learning models (like DistilBERT) requires expensive GPU instances. While classical models (SVM) reduce computational costs, the human-in-the-loop buffer ($0.4 < P(pos) < 0.6$) requires continuous human labour, introducing ongoing operational expenses.}"
            ).replace(
                "\\item \\textbf{ONDC as a Democratizing Protocol}: The Open Network for Digital Commerce (ONDC) represents a critical pathway to overcoming these barriers. ONDC's open-spec network protocols provide standardised schemas for product catalogues, transactions, and reviews. By adopting ONDC's open APIs, platforms can eliminate the need for proprietary database integration, allow SMEs to access calibrated sentiment and inventory analytics, and enforce compliance with data-sharing guidelines without expensive custom infrastructure.",
                "\\item \\textbf{\\update{ONDC as a Democratizing Protocol}}: \\update{The Open Network for Digital Commerce (ONDC) represents a critical pathway to overcoming these barriers. ONDC's open-spec network protocols provide standardised schemas for product catalogues, transactions, and reviews. By adopting ONDC's open APIs, platforms can eliminate the need for proprietary database integration, allow SMEs to access calibrated sentiment and inventory analytics, and enforce compliance with data-sharing guidelines without expensive custom infrastructure.}"
            )
        )
    ),
    # 20. Section 6 New validation and generalizability limitation (highlighted)
    (
        "Furthermore, the managerial relevance of our proposed decision support matrix remains speculative",
        "registries) to confirm generalisability.",
        wrap_update
    ),
    # 21. Supplementary Material declarations
    (
        "\\section*{Supplementary Material}\nSupplementary material associated with this article can be found in the online",
        "additional statistical tables.",
        lambda text: (
            text.replace(
                "\\section*{Supplementary Material}",
                "\\section*{\\update{Supplementary Material}}"
            ).replace(
                "Supplementary material associated with this article can be found in the online version of this document. It provides extended definitions, detailed algorithmic time complexity derivations, cross-validation metrics, and additional statistical tables.",
                "\\update{Supplementary material associated with this article can be found in the online version of this document. It provides extended definitions, detailed algorithmic time complexity derivations, cross-validation metrics, and additional statistical tables.}"
            )
        )
    )
]

success = 0
for idx, (start_sub, end_sub, format_func) in enumerate(blocks):
    # Find start and end position in main_content
    start_pos = main_content.find(start_sub)
    if start_pos == -1:
        print(f"Block {idx+1} start not found: '{start_sub[:50]}'")
        continue
    
    # Search for end_sub starting from start_pos
    end_pos_offset = main_content[start_pos:].find(end_sub)
    if end_pos_offset == -1:
        print(f"Block {idx+1} end not found: '{end_sub[:50]}'")
        continue
    
    exact_match = main_content[start_pos : start_pos + end_pos_offset + len(end_sub)]
    
    # Locate exact_match in rev_content
    rev_start = rev_content.find(exact_match)
    if rev_start == -1:
        print(f"Block {idx+1} exact match not found in main_revision.tex! First few chars of match: '{exact_match[:60]}'")
        # Try finding with flexible whitespace
        continue
    
    replacement = format_func(exact_match)
    rev_content = rev_content[:rev_start] + replacement + rev_content[rev_start + len(exact_match):]
    print(f"Block {idx+1} SUCCESS")
    success += 1

if success == len(blocks):
    with open(rev_path, "w", encoding="utf-8") as f:
        f.write(rev_content)
    print("All highlights applied successfully to main_revision.tex!")
else:
    print(f"Highlights incomplete: {success}/{len(blocks)} succeeded.")
