# Sahodrift v3 changelog

## WP0 setup

1. Created the local `v3` branch from pinned commit `cd25ef045f4c58f1168ccf805195efc1c6420de3`.
2. Created `v3/` and `out/v3/` without changing the published v2 JSON.
3. Captured the available Python 3.11 package environment in `v3/env_v2_freeze.txt`.
4. Recreated an isolated Python 3.11 environment from `requirements.txt` after a successful dependency dry run.
5. Ran seed 0 through the original v2 pipeline and saved the raw result to `out/v3/regression_seed0_raw.json`.
6. Unpacked the supplied top 23 rival source archive outside the repository for audit only. Rival code and data are not copied into this repository.

## Simulation and evidence package

1. Added a lot disjoint C1 and C2 calibration protocol with randomized tie handling.
2. Compared Fisher, Cauchy, harmonic mean, Tippett, twice mean and learned logistic evidence combinations without using test labels for selection.
3. Added explicit 24, 96 and 168 hour operating points with REJECT and REVIEW tiers.
4. Reimplemented five audited rival approach families without copying rival code.
5. Added benign shift traps and matched false alarm comparisons.
6. Evaluated early to late forecasting on eight public Aalborg gEOL power cycling devices.
7. Recorded CORA and Bosch as unavailable because their local archives are incomplete.
8. Added five Sahodrift v3 inspector cards, including a strong D4 example and honest failure cases.
9. Added leave one device out CORA IGBT forecasting at 15, 25 and 50 percent of observed life using the corrected extracted MAT files.
10. Added a chronological Bosch production line benchmark with training only feature selection and calibration only operating thresholds.
