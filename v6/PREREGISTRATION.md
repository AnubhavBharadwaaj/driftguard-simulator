# Sahodrift v6 preregistration

Committed before executing any v6 test-lot evaluation.

## Frozen inputs and exclusions

* Branch v6 starts from v5 commit `2fa8fddbf0987023e7d52a481127c11b2488e29b`.
* Outputs under `out/v2` through `out/v5` are immutable inputs.
* Simulator seeds 0 through 4 are the comparison set. Fresh confirmation uses seeds 5 through 9, which have not been used for model or threshold choice.
* No test label may select a merger, threshold, evidence source, hyperparameter or checkpoint policy.

## Fixed Sahodrift choices

* The merger is preregistered as `twice_mean`. It is not reselected in v6 because the v5 C2a comparison has inadequate 0.5 percent resolution in two seeds and zero recall for all rules in another.
* The nominal REJECT threshold is alpha 0.005 at 24 hours.
* The fresh-seed 168 hour confirmation uses the preregistered conservative threshold alpha 0.0025.
* Each method uses the identical healthy calibration rows for a seed and a deterministic upper-tail conformal p value `(1 + count(reference score >= test score))/(n + 1)`.
* A row is marked `infeasible_at_alpha` when `1/(n+1) > alpha`; its diagnostic prediction is retained separately and is not presented as an attainable operating point.

## Simulator rival stack

R1 is within-lot median absolute deviation over the existing early features. R2 is a 200-tree IsolationForest fitted independently per lot with contamination 0.05 and random state 42. R3 is the labelled sklearn fallback, a GradientBoostingClassifier with 500 estimators, depth 5, learning rate 0.03 and seed-specific random state. R4 combines PCA reconstruction, tenth-neighbour distance, novelty LOF, robust covariance distance and lot MAD by the maximum within-frame percentile rank. R5 is the ridge forecast-slope score. No method may receive a method-specific calibration subset.

All five rivals and Sahodrift are evaluated for seeds 0 through 4 on in-prior and shifted tests at 24 and 168 hours, at alpha 0.005 and 0.0025. Primary reporting includes per-seed FAR, pooled recall, pooled Clopper-Pearson intervals and the number of seeds above 0.75 percent FAR.

## Fourth-source candidate gate

The sole candidate is a within-lot Minimum Covariance Determinant Mahalanobis distance built over the six level-and-slope channels at the decision checkpoint. It is converted to a lot-cross conformal p value using the same healthy calibration rows and added as a fourth evidence source before `twice_mean` merging.

The candidate passes only if, on simulator seeds 0 through 4 at 24 hours in-prior and alpha 0.005:

1. pooled recall is strictly greater than the frozen v5 Sahodrift result, and
2. at most one seed exceeds 0.75 percent FAR.

Only if both conditions pass will the candidate be evaluated once on the frozen Aabhas split. Regardless of outcome, all simulator results are retained.

## Aabhas diagnosis

The frozen Aabhas split and labels are used only after the candidate gate. The diagnosis reports defect-class counts for Sahodrift-only, R4-only, both and neither at 24 hours. At 168 hours, the preregistered comparison explicitly retains plain lot MAD even if it beats Sahodrift.

## BurnTrace resolvable budgets

The frozen BurnTrace split is evaluated at alpha 0.01 and 0.02 for Sahodrift, R1, R2, R4 and R5 at 24 and 168 hours. Every method uses the same 144 healthy calibration parts. Exact Clopper-Pearson intervals are reported. Existing alpha 0.005 rows remain infeasible and are not reinterpreted.

## Fresh-seed confirmation

Seeds 5 through 9 are evaluated once. Acceptance is checked separately at 24 hours with alpha 0.005 and 168 hours with alpha 0.0025. A checkpoint passes when mean achieved FAR is at most 0.5 percent and no more than one seed exceeds 0.75 percent. Failure is reported without tuning or replacement seeds.
