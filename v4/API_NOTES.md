# Sahodrift v4 API notes

| API | Verification | Confidence |
| --- | --- | ---: |
| `numpy.quantile(..., method="higher")` | Already exercised in the v3 Python 3.11 environment | 10 |
| `scipy.stats.beta.ppf` | Already exercised for exact Clopper Pearson intervals in v3 | 10 |
| `sklearn` detector APIs | Versions and signatures inherited from the frozen v3 environment | 10 |
| Conformal risk control Equation 4 | Read from arXiv HTML version 4 dated 13 June 2025 | 9 |
| `streamlit.file_uploader`, `download_button`, `dataframe`, `line_chart`, `selectbox`, `slider` | Signatures inspected from installed Streamlit 1.64.0 before implementation; local HTTP smoke test returned 200 | 10 |
| `Semi_ATE.STDF.records_from_file` | Source inspected in installed 0.1.28; generated FAR, PIR, PTR, PRR binary stream round tripped through the parser | 9 |
| `reportlab.pdfgen.canvas.Canvas` | Audit PDF generated and verified by `%PDF` header in `v4/test_demo.py` | 9 |
| `sklearn.linear_model.LogisticRegression` | Installed scikit learn 1.9.1 API exercised in domain classifier weighting | 10 |

`libomp` is not installed, so the local XGBoost wheel remains unusable. The requested foundation model packages and weights are not installed. Streamlit, ReportLab and Semi ATE STDF were dry run, inspected and installed in the project virtual environment.
