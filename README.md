# DriftGuard simulator and benchmark

DriftGuard is a research prototype for SIH26170, AI driven anomaly detection in component burn in and screening. The repository contains the physics motivated synthetic data generator, fixed baselines, conformal detection methods, hierarchical forecasting, adaptive readout experiments, deconfounding logic and the experimental DriftPFN lot context model.

The repository also includes the final machine readable experiment handback at [`out/driftguard_results_v2.json`](out/driftguard_results_v2.json).

## Important scope statement

These results come from synthetic experiments and a limited public data pipeline check. They do not represent performance on ISRO production data, qualification grade hardware or an operational semiconductor manufacturing line. The shifted test set is included to measure sensitivity to simulator mismatch.

## Main files

* `simulate_v2.py` and `config_v2.yaml`: lot generator, checkpoints, oven slot effects, broad prior and shifted evaluation distribution
* `module_a_v2.py`: conformal component scores and conditional residual detection
* `hier.py`: empirical Bayes power law forecaster
* `conformal.py`: fixed alpha and Benjamini Hochberg evaluation utilities
* `adaptive.py`: gray zone and value of information readout policies
* `deconfound.py`: tester offset and oven slot correction
* `driftpfn/`: lot in context transformer experiment
* `run_v2.py`: main v2 experiment runner
* `finalize_v2.py`: assembles the final JSON handback

The v1 files are retained because the v2 postmortem compares against them.

## Environment

Python 3.11 was used for the recorded run.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

The complete experiment is computationally heavier than a normal unit test. It can train neural models, run multiple seeds and optionally attempt public dataset checks.

```bash
python run_v2.py
python finalize_v2.py
```

For the original benchmark:

```bash
python run_all.py
```

TabPFN is not listed as a required dependency because local model access may require separate licence acceptance and credentials. The final results JSON records that experiment as skipped. External datasets, downloaded papers, virtual environments and trained checkpoints are intentionally not committed.

## Reproducibility and integrity

The experiment design keeps training, calibration and test lots separate. Thresholds and conformal quantities are set on calibration lots. The results JSON records seeds, package versions, work package status, limitations and every place where a new method lost to a baseline.

## Results

See [`out/driftguard_results_v2.json`](out/driftguard_results_v2.json) for the complete structured results. Treat headline metrics together with the `integrity`, `wp_status`, `postmortem` and shifted set fields.

## Project status

Research prototype. Not for production screening, safety qualification or automated reject decisions.
