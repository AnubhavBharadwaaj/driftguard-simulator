# Verified API notes

## Existing v2 environment

Python 3.11 is required. The default Homebrew Python 3.14 installation on this machine currently fails while importing `pyexpat`, so it is not used.

The following project entry points were inspected directly before use.

| API | Verified signature or behavior | Confidence |
| --- | --- | --- |
| `run_v2.seed_run(cfg, seed)` | Returns one deterministic seed result without writing files | 10/10 |
| `simulate_v2.generate_evaluation(cfg, seed)` | Returns train, calibration, in prior test, and shifted test frames | 10/10 |
| `conformal.pvalues_from_calibration(cal_healthy_scores, scores)` | Computes finite sample upper tail conformal p values | 10/10 |
| `conformal.evaluate(y, flags, lots, defect_types)` | Reports achieved FAR, recall, review load, FDP, lot FAR, and recall by defect class | 10/10 |

New dependencies will be added only after a dry run dependency report and direct source signature inspection.
