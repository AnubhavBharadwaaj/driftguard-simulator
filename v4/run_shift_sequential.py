from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "v4"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "v3")); sys.path.insert(0, str(ROOT / "v4"))

from conformal import evaluate
from v3.run_v3 import mergers
from run_phase_a import all_protocol_data, cp_interval, final_pvalues


PARAMS = ["leakage_uA", "iddq_uA", "delay_ns"]


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def metric(frame: pd.DataFrame, flags: np.ndarray, detector: str, seed: int, hour: int = 24) -> dict:
    y = frame.is_defect.to_numpy(int); result = evaluate(y, flags, frame.lot_id, frame.defect_type)
    tp = int(np.sum(flags & (y == 1))); fp = int(np.sum(flags & (y == 0)))
    return {
        "seed": seed, "detector": detector, "decision_hour": hour, "target_far_pct": 0.5,
        "recall_pct": result["recall_pct"], "achieved_far_pct": result["achieved_far_pct"],
        "n_pos": int(np.sum(y == 1)), "n_good": int(np.sum(y == 0)),
        "recall_cp95": cp_interval(tp, int(np.sum(y == 1))), "far_cp95": cp_interval(fp, int(np.sum(y == 0))),
        "recall_by_type": result["recall_by_type"],
    }


def features(frame: pd.DataFrame) -> np.ndarray:
    columns = [f"{parameter}_{hour}h" for parameter in PARAMS for hour in [0, 24]]
    return frame[columns].to_numpy(float)


def weighted_pvalues(ref_scores: np.ndarray, test_scores: np.ndarray, ref_weights: np.ndarray, test_weights: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty(len(test_scores))
    order = np.argsort(ref_scores)
    sorted_scores = ref_scores[order]; sorted_weights = ref_weights[order]
    suffix = np.cumsum(sorted_weights[::-1])[::-1]
    total = float(ref_weights.sum())
    for index, (score, weight) in enumerate(zip(test_scores, test_weights)):
        position = int(np.searchsorted(sorted_scores, score, side="left"))
        greater = float(suffix[position]) if position < len(suffix) else 0.0
        output[index] = (greater + float(weight) * rng.random()) / (total + float(weight))
    return output


def shift_rows(data: dict, method: str) -> tuple[list[dict], dict]:
    seed = data["seed"]; target = data["shifted"]["frame"]; c2b = data["c2b"]["frame"]
    base = final_pvalues(data, method, "shifted")
    rows = [metric(target, base <= 0.005, "unweighted_conformal", seed)]
    good = c2b.is_defect.to_numpy() == 0
    x_ref = features(c2b)[good]; x_target = features(target)
    x_all = np.vstack([x_ref, x_target]); means = np.nanmean(x_all, axis=0); scales = np.nanstd(x_all, axis=0) + 1e-8
    x_all = (np.nan_to_num(x_all, nan=means) - means) / scales
    domain = np.r_[np.zeros(len(x_ref)), np.ones(len(x_target))]
    classifier = LogisticRegression(max_iter=1000, random_state=seed).fit(x_all, domain)
    probability = classifier.predict_proba(x_all)[:, 1]
    auc = float(roc_auc_score(domain, probability))
    odds = np.clip(probability / np.clip(1 - probability, 1e-6, None), 0.1, 10.0)
    ref_weights = odds[:len(x_ref)]; test_weights = odds[len(x_ref):]
    ref_score = -np.log(np.clip(data["c2b"]["merged"][method][good], 1e-300, 1.0))
    test_score = -np.log(np.clip(data["shifted"]["merged"][method], 1e-300, 1.0))
    weighted = weighted_pvalues(ref_score, test_score, ref_weights, test_weights, seed + 417)
    rows.append(metric(target, weighted <= 0.005, "domain_weighted_conformal", seed))
    fallback_alpha = 0.003 if auc >= 0.70 else 0.005
    fallback = weighted <= fallback_alpha
    rows.append(metric(target, fallback, "domain_alarm_conservative_fallback", seed))
    return rows, {"seed": seed, "domain_classifier_auc": auc, "fallback_triggered": auc >= 0.70, "fallback_alpha": fallback_alpha, "weight_clip": [0.1, 10.0]}


def checkpoint_score(reference: pd.DataFrame, target: pd.DataFrame, hour: int) -> tuple[np.ndarray, np.ndarray]:
    reference_scores = []
    target_scores = []
    for parameter in PARAMS:
        column = f"{parameter}_{hour}h"
        median = float(reference[column].median()); mad = max(float((reference[column] - median).abs().median()), 1e-9)
        reference_scores.append(np.abs(reference[column].to_numpy(float) - median) / (1.4826 * mad))
        target_scores.append(np.abs(target[column].to_numpy(float) - median) / (1.4826 * mad))
    return np.max(reference_scores, axis=0), np.max(target_scores, axis=0)


def sequential_rows(data: dict) -> dict:
    seed = data["seed"]; reference = data["c1"]; test = data["inprior"]["frame"]
    good_ref = reference.is_defect.to_numpy() == 0
    rng = np.random.default_rng(seed + 881)
    eprocess = np.ones(len(test)); first = np.full(len(test), np.nan)
    snapshots = []
    for hour in [24, 48, 96, 168]:
        ref_score, score = checkpoint_score(reference, test, hour)
        p = np.empty(len(score))
        for index, value in enumerate(score):
            p[index] = (np.sum(ref_score[good_ref] >= value) + rng.random()) / (int(good_ref.sum()) + 1)
        eincrement = 0.5 / np.sqrt(np.clip(p, 1e-12, 1.0))
        eprocess *= eincrement
        flags = eprocess >= 200.0
        first[np.isnan(first) & flags] = hour
        snapshots.append(metric(test, flags, "exploratory_eprocess", seed, hour))
    y = test.is_defect.to_numpy(int); d3 = test.defect_type.eq("D3_late_accelerator").to_numpy()
    d3_by_hour = {str(hour): 100 * float(np.mean(first[d3] <= hour)) for hour in [24, 48, 96, 168]}
    detected = np.isfinite(first) & (y == 1)
    oven_hours_saved = float(np.sum(168 - first[detected])) if detected.any() else 0.0
    return {
        "seed": seed, "method": "product of p-to-e calibrators, exploratory because conditional checkpoint validity is not proved",
        "whole_horizon_threshold": 200.0, "snapshots": snapshots, "d3_recall_by_hour_pct": d3_by_hour,
        "oven_hours_saved_detected_defects": oven_hours_saved,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows); output = []
    for detector, local in frame.groupby("detector"):
        output.append({
            "detector": detector, "decision_hour": 24, "target_far_pct": 0.5,
            "recall_pct": float(local.recall_pct.mean()), "achieved_far_pct": float(local.achieved_far_pct.mean()),
            "n_pos": int(local.n_pos.sum()), "n_good": int(local.n_good.sum()),
            "aggregation": "mean of five seed percentages; per-seed CIs retained in seed_rows",
        })
    return output


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_v2.yaml").read_text())
    phase = json.loads((OUT / "phase_a_results.json").read_text()); method = phase["protocol"]["selected_merger"]
    shift = []; diagnostics = []; sequential = []
    for seed in range(5):
        data = all_protocol_data(seed, cfg)
        local, diagnostic = shift_rows(data, method); shift.extend(local); diagnostics.append(diagnostic)
        sequential.append(sequential_rows(data))
    shift_payload = {
        "status": "partial", "seed_rows": shift, "aggregate": aggregate(shift), "diagnostics": diagnostics,
        "not_run": {
            "hierarchical_lot_calibration": "requires a formally specified new-lot hierarchy; a nearest-lot shortcut would leak target structure",
            "within_lot_calibration": "not run because labelled healthy target-lot calibration parts were not available",
            "adaptive_alpha": "not claimed without the formal online update and delay analysis",
            "exchangeability_martingale": "represented only by the domain-alarm fallback; no theorem-level martingale implementation is claimed",
        },
    }
    sequential_payload = {"status": "exploratory_partial", "seed_results": sequential, "validity_warning": "The p-to-e product is not a production claim until conditional validity across repeated component measurements is established."}
    (OUT / "shift_robust_v4.json").write_text(json.dumps(clean(shift_payload), indent=2, allow_nan=False) + "\n")
    (OUT / "sequential_d3_v4.json").write_text(json.dumps(clean(sequential_payload), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"shift": shift_payload["aggregate"], "sequential_status": sequential_payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
