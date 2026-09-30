# Sahodrift simulator, benchmark and inspector

Sahodrift, formerly DriftGuard, is a research prototype for SIH26170, AI driven anomaly detection in component burn in and screening. It combines lot relative evidence, drift evidence and conditional residual evidence, then calibrates reject or review decisions on lots that are disjoint from model and rule selection.

Machine readable handbacks are preserved at `out/driftguard_results_v2.json`, `out/v3/driftguard_results_v3.json` and `out/v4/sahodrift_results_v4.json`. The v4 handback is the current evidence index.

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
* `v3/run_v3.py`: two way conformal calibration and rival stack
* `v4/run_phase_a.py`: three way calibration, protocol ablation, CRC review tier and trap experiments
* `v4/rival_benchmark.py`: independent Aabhas and BurnTrace adapters
* `v4/run_shift_sequential.py`: shifted set and exploratory sequential experiments
* `app/`: local Streamlit inspector, reason cards, audit export and STDF v4 ingestion

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

For v4:

```bash
.venv/bin/python v4/run_phase_a.py
.venv/bin/python v4/rival_benchmark.py
.venv/bin/python v4/run_shift_sequential.py
.venv/bin/python v4/test_demo.py
.venv/bin/python v4/finalize_v4.py
.venv/bin/streamlit run app/main.py
```

TabPFN is not listed as a required dependency because local model access may require separate licence acceptance and credentials. The final results JSON records that experiment as skipped. External datasets, downloaded papers, virtual environments and trained checkpoints are intentionally not committed.

## Reproducibility and integrity

The experiment design keeps training, calibration and test lots separate. Thresholds and conformal quantities are set on calibration lots. The results JSON records seeds, package versions, work package status, limitations and every place where a new method lost to a baseline.

## Evidence labels and dataset licences

Synthetic rows are labelled simulation evidence. Aabhas and BurnTrace are cross benchmark evidence tied to their repository commit and split note. CORA and gEOL are CC BY public laboratory data. Bosch data is governed by Kaggle competition rules and is never redistributed here. Rival repositories and datasets remain uncommitted under `data_ext/`.

## Results and limitations

Treat every recall together with its achieved FAR, decision hour, sample count and interval. V4 closes the calibration selection leak and degenerate review threshold, but its in prior 0.5 percent REJECT row spends 0.665 percent FAR and the review tiers remain expensive. Public cross benchmarks are mixed: the method is competitive on Aabhas at 24 hours but detected no BurnTrace anomalies in the small held out split. The sequential result is explicitly exploratory.

## Project status

Research prototype. Not for production screening, safety qualification or automated reject decisions.
