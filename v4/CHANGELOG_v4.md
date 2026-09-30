# Sahodrift v4 changelog

1. Created local branch `v4` from the completed v3 head.
2. Preserved every file under `out/v3/` and the pinned v2 result.
3. Replaced two-way calibration with lot-disjoint C1, C2a and C2b groups.
4. Added real calibration-size subsampling with five draws per requested size.
5. Added a continuous Fisher review score and conformal risk control thresholds.
6. Added the five-row v2 to v4 protocol ablation.
7. Added per-defect-type recall and sample counts to v4 operating rows.
8. Added independent schema adapters for the pinned Aabhas and BurnTrace public benchmarks.
9. Ran Sahodrift at 24 and 168 hours on Aabhas and BurnTrace and ran all five reference baselines on BurnTrace.
10. Added a local Streamlit inspector with CSV and STDF v4 input, reason cards, operating tier controls, and JSON and PDF audit exports.
11. Added a tested append only override log and an STDF binary round trip.
12. Added domain weighted conformal experiments and a conservative domain alarm fallback for the shifted set.
13. Added an explicitly exploratory sequential D3 e process across 24, 48, 96 and 168 hours.
14. Corrected the CORA DOI from the dataset's own Readme and assembled a single v4 evidence index with work package status and limitations.
