# Sahodrift v3 changelog

## WP0 setup

1. Created the local `v3` branch from pinned commit `cd25ef045f4c58f1168ccf805195efc1c6420de3`.
2. Created `v3/` and `out/v3/` without changing the published v2 JSON.
3. Captured the available Python 3.11 package environment in `v3/env_v2_freeze.txt`.
4. Recreated an isolated Python 3.11 environment from `requirements.txt` after a successful dependency dry run.
5. Ran seed 0 through the original v2 pipeline and saved the raw result to `out/v3/regression_seed0_raw.json`.
6. Unpacked the supplied top 23 rival source archive outside the repository for audit only. Rival code and data are not copied into this repository.
