# Sahodrift v6 slide evidence

Every number below has a JSON path. The experiment choices were frozen in commit `0db8ea58cdc4d2941a2ae1dfede7583d8866b9ea` before the v6 test runs.

## Point: Sahodrift remains the strongest 24 hour simulator result under the strict tail rule

Evidence: at alpha 0.005, Sahodrift reaches 20.43 percent pooled recall with 0.173 percent mean FAR and no seed above 0.75 percent. R2 reaches 21.51 percent recall but spends 0.623 percent mean FAR and has one seed above 0.75 percent. R3 reaches 13.98 percent at 0.541 percent mean FAR. R1, R4 and R5 reach 3.23, 5.38 and 3.23 percent recall respectively.

JSON path: `simulator_rivals.operating_aggregates[test_set=inprior,decision_hour=24,target_alpha=0.005]`

## Point: the fresh-seed confirmation passes

Evidence: on previously unused seeds 5 through 9, the preregistered 24 hour rule has 0.391 percent mean FAR, 0.348 percent pooled FAR and 25.77 percent pooled recall. One seed reaches 1.427 percent FAR, which is within the preregistered allowance of at most one seed above 0.75 percent. The 168 hour half-budget rule has 0.042 percent mean FAR, 0.046 percent pooled FAR, 46.39 percent pooled recall and no seed above 0.75 percent. Both checkpoints pass.

JSON path: `fresh_seeds.operating_aggregates`

## Point: the proposed MCD evidence source was rejected

Evidence: adding the within-lot six-channel MCD source reduces 24 hour pooled recall from 20.43 to 11.83 percent. Its pooled FAR is 0.133 percent and no seed exceeds 0.75 percent, but it fails the preregistered recall-improvement condition. It was therefore not run on Aabhas.

JSON paths: `candidate_MCD.simulator_aggregate`, `candidate_MCD.gate_passed`, `candidate_MCD.aabhas_run`

## Point: R4's Aabhas advantage is concentrated in drift and step defects

Evidence: at 24 hours R4 uniquely catches 59 steep drifters and 19 step defects that Sahodrift misses. Sahodrift uniquely catches 28 correlation breaks and 30 extreme-level defects that R4 misses. Neither catches any of the eight centre-hider defects. Overall, R4 reaches 33.26 percent recall at 0.373 percent FAR; Sahodrift reaches 28.57 percent at 0.501 percent.

JSON paths: `aabhas.per_defect_class_24h`, `aabhas.results.24`

## Point: plain lot MAD is better on Aabhas at 168 hours

Evidence: R1 lot MAD reaches 45.43 percent recall at 0.140 percent FAR, compared with Sahodrift's 40.75 percent at 0.420 percent and R4's 44.73 percent at 0.301 percent. By 168 hours the Aabhas anomalies have accumulated large within-lot level separations, which directly suits a robust univariate maximum-deviation rule. Sahodrift's evidence merging is more conservative when only one channel becomes dominant.

JSON path: `aabhas.results.168`

## Point: BurnTrace becomes interpretable at resolvable budgets

Evidence: with only 144 healthy calibration parts, alpha 0.005 remains infeasible. At 24 hours and alpha 0.01, Sahodrift catches 3 of 13 defects, 23.08 percent, with zero false alarms; R1 catches 4 of 13, 30.77 percent, at 0.465 percent FAR. At 168 hours and alpha 0.01, Sahodrift catches 4 of 13 at 0.465 percent FAR, while R1 catches 7 of 13 at the same FAR. The exact 95 percent recall interval for R1 is 25.13 to 80.78 percent, showing the severe uncertainty from thirteen defects.

JSON path: `burntrace.rows`

## Point: the public inspector was verified end to end

Evidence: the supplied v6 application passed all 18 tests and its production build. The deployed guided demo loads 1,632 parts, the official example opens as `LOT-A01-P045` with a REJECT reason card, the performance table and curve render, and the binary demonstration writes and reads 64 STDF V4 files containing 19,584 PTR records.

JSON path: `demo`

## Claims we must not make

* Do not claim the MCD addition improves Sahodrift. It failed the preregistered simulator gate.
* Do not claim Sahodrift wins every dataset or checkpoint. R4 wins Aabhas at 24 hours and plain lot MAD wins it at 168 hours.
* Do not describe the single 1.427 percent fresh-seed FAR as hidden. It is explicitly part of the accepted tail rule.
* Do not compare BurnTrace methods at 0.5 percent; the calibration resolution makes that operating point infeasible.
* Do not present the thirteen BurnTrace defects as a precise ranking. Use the exact confidence intervals.
* Do not call the sklearn gradient boosting fallback XGBoost.
* Do not claim equal achieved FAR merely because methods share alpha. Report both alpha and achieved FAR.
