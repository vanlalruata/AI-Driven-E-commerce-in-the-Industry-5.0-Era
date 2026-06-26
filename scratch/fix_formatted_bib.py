import re

with open(r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\scratch\formatted_bib.txt", "r", encoding="utf-8") as f:
    bib = f.read()

# Fix Chen & Esperan\cc
bib = bib.replace(
    r"\bibitem[Chen & Esperan\cc(2022)]{chen2022impact}",
    r"\bibitem[Chen \& Esperan\c{c}a(2022)]{chen2022impact}"
)
bib = bib.replace(
    r"Chen, D. & Esperan\cc (2022). The Impact of Artificial Intelligence on Firm Performance: An Application of the Resource-Based View to e-Commerce Firms. \textit{Frontiers in Psychology}, \textit{13}, 884830. \url{https://doi.org/10.3389/fpsyg.2022.884830}",
    r"Chen, D., \& Esperan\c{c}a, J. P. (2022). The impact of artificial intelligence on firm performance: An application of the resource-based view to e-commerce firms. \textit{Frontiers in Psychology}, \textit{13}, 884830. \url{https://doi.org/10.3389/fpsyg.2022.884830}"
)

# Fix Daza & N'e
bib = bib.replace(
    r"\bibitem[Daza & N'e(2024)]{DAZA2024100267}",
    r"\bibitem[Daza et~al.(2024)]{DAZA2024100267}"
)
bib = bib.replace(
    r"Daza, A. & N'e (2024). Sentiment Analysis on E-Commerce Product Reviews Using Machine Learning and Deep Learning Algorithms: A Bibliometric Analysis, Systematic Literature Review, Challenges and Future Works. \textit{International Journal of Information Management Data Insights}, \textit{4}(2), 100267. \url{https://doi.org/10.1016/j.jjimei.2024.100267}",
    r"Daza, A., Gonz\'{a}lez Rueda, N. D., Aguilar S\'{a}nchez, M. S., Robles Esp\'{i}ritu, W. F., \& Chauca Qui\~{n}ones, M. E. (2024). Sentiment analysis on e-commerce product reviews using machine learning and deep learning algorithms: A bibliometric analysis, systematic literature review, challenges and future works. \textit{International Journal of Information Management Data Insights}, \textit{4}(2), 100267. \url{https://doi.org/10.1016/j.jjimei.2024.100267}"
)

# Fix Chenavaz & Dimitrov (R'e)
bib = bib.replace(
    r"\bibitem[R'e(2025)]{Chenavaz31122025}",
    r"\bibitem[Chenavaz \& Dimitrov(2025)]{Chenavaz31122025}"
)
bib = bib.replace(
    r"R'e (2025). Artificial intelligence and dynamic pricing: a systematic literature review. \textit{Journal of Applied Economics}, \textit{28}(1), 2466140. \url{https://doi.org/10.1080/15140326.2025.2466140}",
    r"Chenavaz, R. Y., \& Dimitrov, S. (2025). Artificial intelligence and dynamic pricing: A systematic literature review. \textit{Journal of Applied Economics}, \textit{28}(1), 2466140. \url{https://doi.org/10.1080/15140326.2025.2466140}"
)

# Fix Tóth et al. (T'o)
bib = bib.replace(
    r"\bibitem[T'o(2023)]{TOTH2023102260}",
    r"\bibitem[T\'{o}th et~al.(2023)]{TOTH2023102260}"
)
bib = bib.replace(
    r"T'o, A. (2023). The human-centric Industry 5.0 collaboration architecture. \textit{MethodsX}, \textit{11}, 102260. \url{https://doi.org/10.1016/j.mex.2023.102260}",
    r"T\'{o}th, A., Nagy, L., Kennedy, R., Bohu\v{s}, B., Abonyi, J., \& Ruppert, T. (2023). The human-centric Industry 5.0 collaboration architecture. \textit{MethodsX}, \textit{11}, 102260. \url{https://doi.org/10.1016/j.mex.2023.102260}"
)

# Clean up other minor things
bib = bib.replace(
    r"Bawack, R.E., Wamba, S.F., \& Carillo, K.D.A. (2022). Artificial intelligence in E-Commerce: a bibliometric study and literature review. \textit{Electronic markets}, \textit{32}(1), 297–338. \url{https://doi.org/10.1007/s12525-022-00537-z}",
    r"Bawack, R. E., Wamba, S. F., Carillo, K. D. A., \& Akter, S. (2022). Artificial intelligence in e-commerce: A bibliometric study and literature review. \textit{Electronic Markets}, \textit{32}(1), 297–338. \url{https://doi.org/10.1007/s12525-022-00537-z}"
)

with open(r"g:\PycharmProjects\PythonProject\industry5.0_ecommerce_sentimental_prediction\scratch\formatted_bib_fixed.txt", "w", encoding="utf-8") as f:
    f.write(bib)

print("Fixed bibliography file written!")
