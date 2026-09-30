# Sahodrift v4 slide notes

## Slide 1: problem and operating contract

Point: Sahodrift separates automatic rejection from human review because the two decisions have different error costs. Evidence: the five seed operating rows, including decision hour, recall, achieved FAR, counts and intervals, are at `operating_rows` in `out/v4/sahodrift_results_v4.json`.

Point: the conservative 24 hour result is not yet at the promised budget. Evidence: the in prior REJECT row at target 0.5 percent has 13.88 percent recall and 0.665 percent achieved FAR over 93 defects and 4,522 healthy parts at `operating_rows`.

## Slide 2: technical approach

Point: source calibration, merger selection and final calibration now use three disjoint groups of lots. Evidence: C1, C2a and C2b are documented at `protocol.split`, and twice the mean remains the calibration selected merger at `protocol.selected_merger`.

Point: the protocol cost is visible rather than hidden. Evidence: the five row reproduction to v4 ablation is at `ablation_protocol`; v4 reports 13.88 percent recall at 0.665 percent FAR instead of repeating the higher v2 headline under a stricter protocol.

## Slide 3: feasibility and risk

Point: the new REVIEW score avoids the old every part goes to review failure, but it is still expensive. Evidence: measured healthy review loads are 527.9, 803.3 and 896.6 per 1000 for the 80, 90 and 95 percent targets at `operating_rows`.

Point: shifted operation needs a conservative fallback. Evidence: unweighted shifted FAR is 0.922 percent, while the domain alarm fallback reduces it to 0.470 percent at a recall cost from 8.40 to 7.90 percent at `shift_robust.aggregate`.

## Slide 4: impact and benefit

Point: the prototype makes the yield cost explicit next to every sensitivity claim. Evidence: false rejects or reviews per 1000 healthy parts are included on every synthetic operating row in `operating_rows` and every seed row in `shift_robust.seed_rows`.

Point: the inspector produces a traceable decision rather than an unexplained score. Evidence: its JSON and PDF audit contain the input hash, model version, calibration hash, threshold and every part decision; the round trip test is at `demo.local_test`.

## Slide 5: public evidence and rivals

Point: on Aabhas at the same 24 hour decision, Sahodrift reaches 28.81 percent recall at 0.649 percent FAR, close to the repository's 29.7 percent at 0.54 percent. Evidence: both values, source commit and split are at `rival_benchmark_v4.benchmarks[0]`.

Point: BurnTrace exposes a clear failure rather than a success story. Evidence: Sahodrift detects none of 13 held out anomalies at both 24 and 168 hours; the wide 95 percent recall interval and split difference are at `rival_benchmark_v4.benchmarks[1]`.

## Slide 6: research and validation roadmap

Point: the REVIEW thresholds implement the finite sample conformal risk correction from Conformal Risk Control. Evidence: the exact criterion and paper location are recorded in `v4/literature_v4.json`, while thresholds and calibration defect counts are at `review_crc`.

Point: sequential D3 detection is research in progress, not a validated claim. Evidence: checkpoint results and the conditional validity warning are at `sequential_d3`, and the work package is marked partial at `meta.wp_status.B4`.

## Claims we must NOT make

1. Do not claim the 24 hour v4 REJECT tier controls realised FAR below 0.5 percent on the in prior test. Its mean is 0.665 percent.
2. Do not claim a practical low cost 95 percent recall REVIEW tier. It sends about 897 healthy parts per 1000 to review.
3. Do not claim production burn in validation from Bosch, CORA, gEOL, Aabhas or BurnTrace.
4. Do not claim exact XGBoost. The row remains a labelled scikit learn fallback because libomp is missing.
5. Do not claim foundation model results or DriftPFN v4 results. No model row was executed.
6. Do not claim Sahodrift wins on BurnTrace. It detected zero of 13 held out anomalies in the selected split.
7. Do not call the sequential checkpoint experiment anytime valid. Its conditional validity has not been proved.
8. Do not claim public deployment. Only the local Streamlit app and HTTP smoke test are complete.
9. Do not claim all Phase B datasets, SPRC, standards or literature were completed. Their status and reasons are explicit under `meta.wp_status`.

## Five numbers that most separate the project at matched conditions

1. In prior v4 REJECT: 13.88 percent recall at 0.665 percent FAR, 24 hours, 93 defects and 4,522 healthy parts. Path: `operating_rows`.
2. Shift fallback: 7.90 percent recall at 0.470 percent FAR, 24 hours, 87 defects and 4,158 healthy parts. Path: `shift_robust.aggregate`.
3. Aabhas 24 hours: 28.81 percent recall at 0.649 percent FAR, 427 defects and 23,573 healthy parts. Path: `rival_benchmark_v4.benchmarks[0].sahodrift[0]`.
4. Aabhas 168 hours: 40.75 percent recall at 0.445 percent FAR on the same held out parts. Path: `rival_benchmark_v4.benchmarks[0].sahodrift[1]`.
5. BurnTrace R1 at 24 hours: 30.77 percent recall at 0.465 percent FAR while Sahodrift is 0 percent at the same 0.465 percent FAR. Path: `rival_benchmark_v4.benchmarks[1]`.
