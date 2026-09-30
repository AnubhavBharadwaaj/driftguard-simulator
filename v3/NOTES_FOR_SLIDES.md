# Sahodrift v3 notes for six slides

## Slide 1: The operating problem

Sahodrift treats early screening as a risk controlled decision, not a promise to identify every future failure. Across five held out simulation seeds, the 24 hour REJECT tier found 28.35 percent of defects while sending 0.318 percent of healthy parts to rejection, from 93 total defects at `headline.reject_tier_24h_inprior` in `out/v3/driftguard_results_v3.json`.

The shifted test exposes the remaining risk instead of hiding it. Under shifted conditions, recall fell to 12.48 percent and false alarms rose to 0.575 percent at `headline.shifted_reject_tier` in `out/v3/driftguard_results_v3.json`.

## Slide 2: Technical approach

Calibration lots are split into C1 for component evidence and C2 for final calibration and method choice, so held out test labels never select the detector. The selected method is `twice_mean` at `combiners.selected_on_calibration` and the split is recorded at `calibration.c1_c2_split` in `out/v3/driftguard_results_v3.json`.

The detector combines an early parameter signal with a conditional residual signal, then issues decisions at an explicit read hour. Results for 24, 96 and 168 hours appear under `detectors` in `out/v3/driftguard_results_v3.json`.

## Slide 3: Feasibility and viability

Common unsupervised baselines can appear stronger when their false alarm cost is left uncontrolled. The default IsolationForest family reached 44.78 percent recall but 4.39 percent false alarms across five seeds at the `R2_isolation_forest`, `default`, `inprior`, 24 hour row in `methods` of `out/v3/baselines_rival_stack.json`.

The current 24 hour model is feasible as a conservative rejection aid, but it is not ready to replace the full screening process. It misses about 716.52 of every 1000 defects at `headline.reject_tier_24h_inprior.escapes_per_1000_defects_mean` in `out/v3/driftguard_results_v3.json`.

## Slide 4: Impact and benefits

At the conservative tier, about 3.18 healthy parts per 1000 would be sent to rejection. This value is at `headline.reject_tier_24h_inprior.review_per_1000_good_mean` in `out/v3/driftguard_results_v3.json`.

A nominal 95 percent recall review setting is not economically usable in the present experiment because it sends all healthy parts to review. The measured 1000 reviews per 1000 healthy parts are at `headline.review_tier_recall95_cost.review_per_1000_good_mean` in `out/v3/driftguard_results_v3.json`.

## Slide 5: Public data sanity check

The best early to late gEOL forecast used a condition matched ratio and produced 0.00317 V mean absolute error across eight devices. The method and result are at `headline.best_forecaster` and `headline.real_data_best` in `out/v3/driftguard_results_v3.json`.

This result supports the forecasting workflow but does not prove burn in performance. The dataset limitation is stated at `realdata_unused[0].caveat` in `out/v3/driftguard_results_v3.json`.

## Slide 6: Research and references

Five dependence aware evidence rules and one learned rule were compared, with one rule selected using C2 calibration only. Their calibration and held out summaries appear under `combiners.validation` and `combiners.held_out_test` in `out/v3/driftguard_results_v3.json`.

Every reported decision identifies its read hour, target, achieved false alarm rate, recall, sample count and confidence interval. These fields appear for every row under `detectors` in `out/v3/driftguard_results_v3.json`.

## Claims we must NOT make

1. Do not claim validation on production burn in data.
2. Do not claim practical 95 percent recall at 24 hours.
3. Do not claim that CORA or Bosch was evaluated because both local archives are incomplete.
4. Do not call the supervised baseline XGBoost because it is an explicit sklearn fallback.
5. Do not claim that the eight device gEOL check represents semiconductor production diversity.
6. Do not claim that a foundation model was benchmarked.

## Three separating numbers

1. Sahodrift REJECT tier: 28.35 percent recall at 0.318 percent false alarms. Evidence is at `headline.reject_tier_24h_inprior` in `out/v3/driftguard_results_v3.json`.
2. Default IsolationForest family: 44.78 percent recall at 4.39 percent false alarms. Evidence is at the `R2_isolation_forest`, `default`, `inprior`, 24 hour row in `methods` of `out/v3/baselines_rival_stack.json`.
3. Public gEOL forecast: 0.00317 V mean absolute error across eight devices. Evidence is at `headline.real_data_best` in `out/v3/driftguard_results_v3.json`.
