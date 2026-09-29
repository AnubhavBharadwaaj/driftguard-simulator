from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).resolve().parent / "out" / "literature.json"


def main() -> None:
    records = [
        {
            "paper": "Ni, Rui, Zhuo, Li, Wen and Nie, A Novel Approach to Reducing Testing Costs and Minimizing Defect Escapes Using Dynamic Neighborhood Range and Shapley Values",
            "found": False,
            "source_url": "https://doi.org/10.1145/3716826",
            "task": "Select semiconductor test patterns dynamically while using Shapley values to explain which measurements matter.",
            "data": "The DOI record and an ACM special issue summary were accessible, but no legally open full manuscript or author copy was located; data type and size could not therefore be verified.",
            "method": "Dynamic neighborhood range analysis followed by Shapley based test selection, according to the publisher's public summary.",
            "headline_numbers": "Unavailable from a legally open full text. The bibliographic record identifies a 20 page ACM TODAES article, but no result number is repeated here without its table context.",
            "limitations": "Only bibliographic metadata and a publisher summary were legally accessible, so numerical claims, experimental conditions and failure cases remain unverified.",
            "difference_from_driftguard": "The paper optimizes which production tests to retain, whereas DriftGuard predicts burn in anomalies and end of test values from early trajectories. DriftGuard also enforces lot split calibration, an explicit false alarm budget, slot and tester deconfounding, and robustness checks under a shifted simulator."
        },
        {
            "paper": "Ahmed, Baraldi, Zio and Lewitschnig, burn in decision framework; plus the 2024 Springer chapter An Artificial Intelligence Based Framework for Burn in Reduction",
            "found": True,
            "source_url": "https://re.public.polimi.it/retrieve/4fa2352a-8f8d-487d-b99f-e66a59101432/P674.pdf",
            "additional_source_url": "https://www.sciencedirect.com/science/article/pii/S036083522500261X",
            "task": "Predict production batch quality before burn in so the manufacturer can decide how many devices and which burn in tests are required.",
            "data": "The open ESREL precursor uses a synthetic semiconductor batch study. The 2025 open access journal extension reports both a synthetic case study and confidential real semiconductor production data; the public article description does not disclose the real sample count.",
            "method": "Piecewise Aggregate Approximation and PCA compress process time series, Clopper Pearson bounds estimate batch quality, and a Bayesian optimized probabilistic SVR predicts early life failures. The related 2024 framework also separates machine signal reconstruction, wafer map anomaly detection and electrical test regression.",
            "headline_numbers": "The accessible public description supports successful validation on synthetic and real production data but does not expose a safe standalone error number. The open ESREL paper is an earlier batch defective count study, not a direct per part 168 hour forecast benchmark.",
            "limitations": "The real factory data are confidential, the public metadata omit sample size, and batch quality prediction is not the same as finding rare defective parts at a fixed 0.5 percent false alarm rate.",
            "difference_from_driftguard": "Ahmed et al. predict batch level quality and burn in policy from production process data. DriftGuard makes per part, lot contextual decisions, forecasts three electrical quantities, and reports calibration, defect type recall, false rejects and distribution shift separately."
        },
        {
            "paper": "Zhai, Li and Chen, Modeling product degradation with heterogeneity: a general random effects Wiener process approach",
            "found": True,
            "source_url": "https://www.tandfonline.com/doi/pdf/10.1080/24725854.2024.2434125",
            "task": "Model heterogeneous degradation when both unit drift and diffusion intensity vary across a population.",
            "data": "Monte Carlo experiments plus two real degradation sets: infrared LED devices and hard disk magnetic head wear. The open article text does not frame the cases as a common train test sample count benchmark.",
            "method": "A random effects Wiener process uses a normal distribution for drift and an independent inverse Gaussian distribution for diffusion; variational Bayesian EM estimates parameters and parametric bootstrap builds confidence intervals.",
            "headline_numbers": "The paper validates parameter recovery in Monte Carlo experiments and compares fit on both real cases; no single cross case MAE is claimed in the abstract. Article Tables and figures report case specific parameter and fit comparisons rather than a universal prediction score.",
            "limitations": "The stochastic path model assumes a Wiener form and independence between the two random effects; it is not an anomaly classifier and does not directly control false alarms.",
            "difference_from_driftguard": "Zhai et al. provide a principled population degradation model with simultaneous drift and volatility heterogeneity. DriftGuard instead combines lot context, conformal anomaly decisions and multi output forecasting, but could use their richer random effects prior in future simulator versions."
        },
        {
            "paper": "Statistical outlier screening for latent defects, IEEE document 6531962",
            "found": False,
            "source_url": "https://ieeexplore.ieee.org/document/6531962",
            "task": "Screen latent semiconductor defects from multivariate wafer probe parameters.",
            "data": "The public abstract states that three latent returns were analyzed, but no legally open full manuscript was located and the complete device count was not verifiable.",
            "method": "Multivariate statistical outlier models on parametric wafer probe data, as described by the public IEEE abstract.",
            "headline_numbers": "Unavailable from a legally open full text; no metric has been copied from secondary summaries.",
            "limitations": "The open abstract is insufficient to audit thresholds, leakage, false rejects, data splits or experimental sample size.",
            "difference_from_driftguard": "The 2013 work screens static wafer probe outliers associated with latent returns. DriftGuard evaluates time dependent burn in trajectories, lot effects, forecast intervals, shifted physics and explicit per lot false alarm behavior."
        },
        {
            "paper": "Shen, Wen and Akoglu, FoMo 0D",
            "found": True,
            "source_url": "https://arxiv.org/abs/2409.05672",
            "task": "Zero shot tabular outlier detection without selecting or tuning a new model for each dataset.",
            "data": "Synthetic Gaussian mixture prior for pretraining; evaluation on 57 real ADBench datasets against 26 baselines, with five seed tables in the appendix.",
            "method": "A prior data fitted transformer receives the unlabeled dataset as context. Router attention reduces sample attention from quadratic to linear scaling; default experiments use 500 routers and a 5000 row context.",
            "headline_numbers": "Table 1: no significant AUROC difference from the second best baseline across all 57 datasets, p equals 0.106 versus kNN. Table 3 and Appendix Figure 6: 7.7 ms per sample at 5000 context on GPU and at least 7 times faster overall; Appendix Table 5 studies 1000, 2000 and 5000 training contexts with inference up to 10000.",
            "limitations": "The prior is a simple GMM, pretraining is synthetic only, dimensionality and context generalization are strongest near training conditions, and the study is static rather than streaming. The authors explicitly propose richer priors, real plus synthetic pretraining, larger context, contaminated context and simulator mismatch stress tests.",
            "difference_from_driftguard": "FoMo 0D is a general zero shot point anomaly model, while DriftPFN narrows the prior to lot structured burn in physics and adds trajectory quantiles, defect type and responsible parameter heads. DriftGuard also tests per lot calibration, wrong drift law, slot effects and adaptive measurements, none of which are FoMo 0D's target."
        },
        {
            "paper": "Kluttermann et al., FoMo X: Modular Explainability Signals for Outlier Detection Foundation Models",
            "found": True,
            "source_url": "https://arxiv.org/abs/2603.17570",
            "task": "Add useful diagnostics to a frozen outlier detection PFN without expensive post hoc explanations.",
            "data": "Synthetic simulator tasks train the diagnostic heads; transfer is evaluated on ADBench real datasets. The paper reports the cardio example at FoMo 0D AUROC 0.88.",
            "method": "Small MLP heads read frozen PFN embeddings. A four tier severity head learns simulator labels, while an uncertainty head distills ten pass Monte Carlo dropout variability into one pass.",
            "headline_numbers": "Figure 4 and Section 5: severity reaches 59 percent balanced accuracy and uncertainty reaches about 99 percent Spearman correlation on synthetic validation. Figure 6: nearly all ADBench uncertainty correlations exceed 90 percent; Figure 7: both heads add under 2 percent inference overhead, under 1 percent each; training took 12 and 18 hours.",
            "limitations": "Severity has no direct real world ground truth in ADBench and is interpreted as intrinsic reliability, while a separate AUROC prediction head collapses from about 0.95 synthetic correlation to about minus 0.01 on ADBench. Diagnostics therefore transfer unevenly when the simulator does not match reality.",
            "difference_from_driftguard": "FoMo X explains a generic frozen PFN with distilled severity and uncertainty. DriftPFN predicts defect class, responsible electrical parameter and future quantiles directly from a semiconductor lot, but its current synthetic only training inherits the same simulator transfer risk highlighted by FoMo X."
        },
        {
            "paper": "Bashari, Sesia and Romano, Robust Conformal Outlier Detection under Contaminated Reference Data",
            "found": True,
            "source_url": "https://proceedings.mlr.press/v267/bashari25a.html",
            "task": "Control inlier false positives when the calibration reference set is contaminated by unlabeled outliers, while recovering lost detection power.",
            "data": "Synthetic experiments, three real tabular benchmarks, and six visual outlier tasks. Main visual results average 100 random splits; tabular experiments use 5000 training and 2500 calibration samples by default.",
            "method": "Compute ordinary conformal scores on the contaminated reference. Use the detector to rank suspicious calibration points. Spend a limited label budget only on this subset. Remove confirmed outliers but retain labeled inliers. Recompute conformal p values; validity is protected under the paper's label trim condition.",
            "headline_numbers": "Table 1 at alpha 0.01 and 5 percent contamination: Standard relative power 1.000 at type I error 0.004; Label Trim 1.786 at 0.008; invalid Naive Trim 4.160 at 0.036. Figure 4 reports that 40 to 50 annotations make Label Trim nearly indistinguishable from the infeasible oracle on shuttle.",
            "limitations": "Conservativeness requires the non adversarial score ordering in Assumption 2.3, formally E of the inlier score CDF at the contaminated calibration quantile below 1 minus alpha. Cleaning improves power but requires labels and can lose its guarantee if labels or score ordering are adversarial.",
            "difference_from_driftguard": "DriftGuard currently assumes known healthy calibration parts and uses finite sample conformal p values to allocate a false alarm budget. This paper identifies the exact failure mode when that healthy set is contaminated and supplies a defensible future extension: active review and removal of only confirmed calibration outliers."
        },
        {
            "paper": "Li, Tian, Wang, Chen and Ren, Adaptive Active Learning for Online Reliability Prediction of Satellite Electronics",
            "found": True,
            "source_url": "https://arxiv.org/abs/2603.09058",
            "task": "Choose which satellite electronic units and which times to sample when telemetry is limited.",
            "data": "Numerical studies averaged over 1000 replications plus a Tiangong space station case with 12 devices.",
            "method": "A Wiener degradation model combines a generalized Arrhenius link, device random effects and spatial correlation. Stage one selects spatially representative units; stage two selects times by balancing unit information, model uncertainty and the current degradation rate.",
            "headline_numbers": "Table 1, constrained windows: at year 12 active sampling error is 23.323 percent versus 25.386 percent for uniform sampling in one scenario. Table 2: at year 10.75 error is 1.508 percent versus 1.690 percent; Table 3 reports final stage 29.132 percent versus 30.108 percent when the schedule is unknown.",
            "limitations": "Benefits shrink late in life and can reverse slightly, for example Table 2 at year 12 gives 30.006 percent active versus 29.821 percent uniform. The six page preprint focuses on population reliability, not rare defect recall at a fixed false alarm budget.",
            "difference_from_driftguard": "The paper optimizes telemetry acquisition for a spatially correlated satellite population. DriftGuard's adaptive readouts operate per component in an oven and optimize defect decisions, but the paper's joint spatial and temporal information criterion is a stronger future replacement for the current gray zone and value of information heuristics."
        }
    ]
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
