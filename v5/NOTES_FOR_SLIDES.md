# Sahodrift v5 slide evidence

Every number below names its JSON path. Earlier v2, v3 and v4 outputs remain frozen.

## Point: fair selection changes the scientific comparison

Evidence: after each candidate merger receives the same leave-one-lot-out recalibration on C2a, twice the mean is selected with 11.33 percent mean recall and 0.181 percent mean FAR. Fisher reaches 7.33 percent at 0.262 percent. The choice uses calibration labels only, never test labels.

JSON paths: `protocol.selected_merger`, `protocol.selection_summary`, `protocol.selection_on_c2a_recalibrated`

## Point: v5 meets its predeclared 24 hour in-prior acceptance rule

Evidence: at nominal alpha 0.005, mean FAR is 0.173 percent, pooled FAR is 0.199 percent with 95 percent Clopper-Pearson interval 0.091 to 0.377 percent, and pooled recall is 20.43 percent. No seed exceeds 0.75 percent FAR. Healthy calibration counts range from 580 to 1,076, so the nominal decision is resolution-feasible in every seed.

JSON path: `operating_aggregates[detector=v5_cross_conformal,test_set=inprior,decision_hour=24,target_alpha=0.005]`

## Point: the conservative factor-two sensitivity sacrifices recall for a tighter tail

Evidence: at alpha 0.0025, 24 hour in-prior mean FAR is 0.079 percent and pooled recall is 16.13 percent. No seed exceeds 0.75 percent.

JSON path: `operating_aggregates[detector=v5_cross_conformal_conservative,test_set=inprior,decision_hour=24,target_alpha=0.0025]`

## Point: later checkpoints need the conservative rule

Evidence: nominal 168 hour in-prior FAR rises to 0.614 percent and three seeds exceed 0.75 percent. The conservative 168 hour run reduces mean FAR to 0.222 percent with pooled recall 39.78 percent. This is reported as a limitation, not hidden by averaging.

JSON paths: `operating_aggregates[detector=v5_cross_conformal,test_set=inprior,decision_hour=168]`, `operating_aggregates[detector=v5_cross_conformal_conservative,test_set=inprior,decision_hour=168]`

## Point: the review tier becomes affordable when information arrives

Evidence: for the 80 percent CRC target, mean in-prior review load falls from 483 good parts per 1,000 at 24 hours to 299 at 96 hours and 18 at 168 hours. At 24 hours, separately evaluating defects designed to be observable then gives 94.13 percent mean recall for D1, D2 and D4; D3 late accelerators remain in the all-defect result.

JSON path: `review_crc.rows`

## Point: the public inspector is reproducible product evidence

Evidence: the supplied v5 app passed 18 tests, built, and was deployed at https://sahodrift-app.vercel.app. It includes bisection-based counterfactual rescoring, 168 hour forecasts with intervals, similar parts, exact Shapley values, binary STDF handling in both byte orders, and a hash-chained override log.

JSON path: `demo.public_url`

## Point: CORA DOI correction

Evidence: DataCite identifies `10.34810/data3204` as the Perez-Farre et al. electrical and thermal Si IGBT accelerated power-cycling dataset. `10.34810/data3206` is an unrelated colistin-delivery dataset. The source Readme contains the typo.

JSON path: `realdata_v5.cora_identifier_erratum`

## Claims we must not make

* Do not claim exact finite-sample validity for this lot-cross outlier procedure. The jackknife+ factor-two theorem and averaged-p-value argument motivate the alpha-over-two sensitivity run but are not an exact theorem for our construction.
* Do not say the nominal rule controls 0.5 percent FAR at every hour or under every shift. The per-seed tails and 168 hour overshoot are visible in the JSON.
* Do not call synthetic recall factory performance. CORA, Aabhas, BurnTrace and Bosch answer different external-validity questions.
* Do not compare rival headline points without naming their split, calibration count, decision hour and achieved FAR.
* Do not cite `10.34810/data3206` for CORA.
* Do not claim a screen recording was produced. No supported recorder was available in this run.
