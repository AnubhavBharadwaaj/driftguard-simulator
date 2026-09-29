from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import catboost
import mapie
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from baselines import dpat_score, delta_score, evaluate_flags, fit_isolation, isolation_score, pr_auc, static_limit_score, threshold_for_far
from fuse import choose_k, fused_flags, inspector_card, measure_latency
from module_a import DriftGuardA
from module_b import ModuleB, calibrate_intervals, interval_metrics
from simulate import PARAMS, impute_by_family, simulate_dataset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def mean_sd(values):
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    if not len(a): return {"mean": None, "sd": None}
    return {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0}


def sig4(x):
    if x is None: return None
    if isinstance(x, bool): return x
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x): return None
        if x == 0: return 0.0
        return float(f"{x:.4g}")
    if isinstance(x, dict): return {str(k): sig4(v) for k, v in x.items()}
    if isinstance(x, list): return [sig4(v) for v in x]
    return x


def mae_block(test, predictions, defects_only=False):
    mask = test["is_defect"].to_numpy() == 1 if defects_only else np.ones(len(test), dtype=bool)
    out = {}
    for p in PARAMS:
        y = test[f"{p}_168h"].to_numpy(dtype=float)
        out[p] = float(np.mean(np.abs(y[mask] - predictions[p][mask]))) if mask.any() else np.nan
    return out


def run_seed(cfg, seed):
    raw, meta = simulate_dataset(cfg, seed)
    if seed == 0:
        raw.to_csv(OUT / "parts_seed0.csv", index=False)
    train0 = raw[raw.split == "train"].copy().reset_index(drop=True)
    cal0 = raw[raw.split == "calibration"].copy().reset_index(drop=True)
    test0 = raw[raw.split == "test"].copy().reset_index(drop=True)
    cols = [f"{p}_{t}h" for p in PARAMS for t in cfg["simulation"]["checkpoints_h"]]
    train, cal, test = impute_by_family(train0, [train0, cal0, test0], cols)
    limits = {p: float(cfg["simulation"]["parameters"][p]["limit"]) for p in PARAMS}
    ycal = cal["is_defect"].to_numpy(); ytest = test["is_defect"].to_numpy(); dtype = test["defect_type"].to_numpy()

    iso = fit_isolation(train, cfg, seed)
    dg_a = DriftGuardA(cfg).fit(train)
    a_scores_cal = {
        "static_limit": static_limit_score(cal, limits), "delta_limit": delta_score(cal),
        "dpat_6sigma": dpat_score(cal), "isolation_forest_global": isolation_score(iso, cal),
        "driftguard_A": dg_a.score(cal),
    }
    dg_test_score, a_detail = dg_a.score(test, return_detail=True)
    a_scores_test = {
        "static_limit": static_limit_score(test, limits), "delta_limit": delta_score(test),
        "dpat_6sigma": dpat_score(test), "isolation_forest_global": isolation_score(iso, test),
        "driftguard_A": dg_test_score,
    }
    fars = [float(x) for x in cfg["module_a"]["operating_far_pct"]]
    module_a = {}; thresholds = {}
    for name, score in a_scores_test.items():
        ops = []; thresholds[name] = {}
        for far in fars:
            threshold = 1.0 if name == "static_limit" else threshold_for_far(a_scores_cal[name], ycal, far)
            thresholds[name][far] = threshold
            ev = evaluate_flags(ytest, score >= threshold, dtype); ev["target_far_pct"] = far
            ops.append(ev)
        module_a[name] = {"pr_auc": pr_auc(ytest, score), "operating_points": ops}

    mb = ModuleB(cfg, seed).fit(train)
    cal_pred = mb.predict(cal); test_pred = mb.predict(test)
    module_b = {}
    for name, pdict in test_pred.items():
        module_b[name] = {"mae": mae_block(test, pdict), "mae_defects_only": mae_block(test, pdict, True)}
    conf = calibrate_intervals(cal, cal_pred, float(cfg["module_b"]["conformal_nominal_pct"]))
    intervals = interval_metrics(test, test_pred, conf["q"])

    k = choose_k(cal, cal_pred, conf["q"], limits, cfg)
    fused = {}; fused_components = None; fused_flags_05 = None
    for far in fars:
        flags, components = fused_flags(test, dg_test_score, thresholds["driftguard_A"][far], test_pred, conf["q"], limits, k, cfg)
        ev = evaluate_flags(ytest, flags, dtype); ev["target_far_pct"] = far; fused[far] = ev
        if abs(far - 0.5) < 1e-9: fused_components = components; fused_flags_05 = flags
    card = inspector_card(test, train, fused_flags_05, fused_components, a_detail, test_pred, conf["q"], limits, cfg)
    latency = measure_latency(test, dg_a, mb, conf["q"], limits, k, thresholds["driftguard_A"][0.5], cfg)
    return {
        "n_parts": len(raw), "defect_rate_pct": 100.0 * raw.is_defect.mean(), "module_a": module_a,
        "module_b": module_b, "intervals": intervals, "q": conf["q"], "k": k, "fused": fused,
        "oven_hours": fused[0.5]["true_positives_per_1000"] * float(cfg["fusion"]["oven_hours_after_24h"]),
        "latency": latency, "card": card,
    }


def aggregate(cfg, runs, elapsed):
    defect_types = ["D1_high_in_limit", "D2_fast_drifter", "D3_late_accelerator", "D4_multivariate"]
    fars = [float(x) for x in cfg["module_a"]["operating_far_pct"]]
    methods_a = []
    for name in ["static_limit", "delta_limit", "dpat_6sigma", "isolation_forest_global", "driftguard_A"]:
        ops = []
        for oi, far in enumerate(fars):
            evs = [r["module_a"][name]["operating_points"][oi] for r in runs]
            ops.append({"target_far_pct": far, "achieved_far_pct": mean_sd([e["achieved_far_pct"] for e in evs]), "recall_pct": mean_sd([e["recall_pct"] for e in evs]), "false_negatives_per_100_defects": mean_sd([e["false_negatives_per_100_defects"] for e in evs]), "false_rejects_per_1000": mean_sd([e["false_rejects_per_1000"] for e in evs])})
        by = {d: float(np.nanmean([r["module_a"][name]["operating_points"][1]["recall_by_type"][d] for r in runs])) for d in defect_types}
        methods_a.append({"name": name, "pr_auc": mean_sd([r["module_a"][name]["pr_auc"] for r in runs]), "operating_points": ops, "recall_pct_by_defect_type_at_0p5_far": by})
    methods_b = []
    for name in ["persistence", "linear_extrapolation", "power_law_fixed_n", "lot_median_ratio", "svr", "catboost", "driftguard_B"]:
        mae = {p: mean_sd([r["module_b"][name]["mae"][p] for r in runs]) for p in PARAMS}
        maed = {p: mean_sd([r["module_b"][name]["mae_defects_only"][p] for r in runs]) for p in PARAMS}
        methods_b.append({"name": name, "mae": mae, "mae_defects_only": maed})
    reductions = {}
    for p in PARAMS:
        reductions[p] = float(np.mean([100.0 * (r["module_b"]["linear_extrapolation"]["mae"][p] - r["module_b"]["driftguard_B"]["mae"][p]) / r["module_b"]["linear_extrapolation"]["mae"][p] for r in runs]))
    fused_ops = []
    for far in fars:
        evs = [r["fused"][far] for r in runs]
        fused_ops.append({"target_far_pct": far, "achieved_far_pct": mean_sd([e["achieved_far_pct"] for e in evs]), "recall_pct": mean_sd([e["recall_pct"] for e in evs]), "false_negatives_per_100_defects": mean_sd([e["false_negatives_per_100_defects"] for e in evs]), "false_rejects_per_1000": mean_sd([e["false_rejects_per_1000"] for e in evs])})
    missed = {d: float(np.nanmean([100.0 - r["fused"][0.5]["recall_by_type"][d] for r in runs])) for d in defect_types}
    defect_shares = {}
    # The configured injection distribution is the honest population target; realized shares vary because defects are rare.
    for d, prob in cfg["simulation"]["defect_type_probs"].items():
        desc = {"D1_high_in_limit": "High relative level remains inside the datasheet limit", "D2_fast_drifter": "Normal initial value with accelerated drift", "D3_late_accelerator": "Healthy at 24 h with acceleration after 96 h", "D4_multivariate": "Marginally normal parameters with an abnormal combination"}[d]
        defect_shares[d] = {"share_pct": 100.0 * float(prob), "description": desc}
    m_a = {x["name"]: x for x in methods_a}; m_b = {x["name"]: x for x in methods_b}
    result = {
        "meta": {"date": "2026-09-18", "seeds": cfg["simulation"]["seeds"], "n_lots": int(cfg["simulation"]["n_lots"]), "n_parts_mean": float(np.mean([r["n_parts"] for r in runs])), "defect_rate_pct": float(np.mean([r["defect_rate_pct"] for r in runs])), "python": platform.python_version(), "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "catboost": catboost.__version__, "mapie": mapie.__version__, "pyyaml": yaml.__version__}, "runtime_s": elapsed, "conformal_impl": "manual"},
        "simulator": {"parameters": PARAMS, "limits": {p: float(cfg["simulation"]["parameters"][p]["limit"]) for p in PARAMS}, "checkpoints_h": cfg["simulation"]["checkpoints_h"], "drift_law": "x(t) = x0 * (1 + a * (t/168)^n), with an added post 96 h acceleration term for D3", "defect_types": defect_shares, "nuisance": "Multiplicative measurement noise of 1 to 2 percent, three tester offsets up to plus or minus 3 percent, 1 percent missing readouts, and recorded board slot", "split": "24 train lots, 8 calibration lots, and 8 test lots per seed; all splits are by lot"},
        "module_a": {"decision_time_h": 24, "methods": methods_a},
        "module_b": {"methods": methods_b, "best_vs_linear_mae_reduction_pct": reductions, "interval": {"nominal_pct": 90, "pooled_coverage_pct": mean_sd([r["intervals"]["pooled_coverage_pct"] for r in runs]), "per_lot_coverage_min_pct": float(np.mean([r["intervals"]["per_lot_min_pct"] for r in runs])), "per_lot_coverage_median_pct": float(np.mean([r["intervals"]["per_lot_median_pct"] for r in runs])), "mean_width": {p: float(np.mean([r["intervals"]["mean_width"][p] for r in runs])) for p in PARAMS}, "calibration": "Manual split conformal on eight calibration lots, with leave one lot out residual quantiles recorded during calibration"}},
        "fused": {"rule": "Flag if a static limit fails, Module A crosses its calibrated threshold, predicted slope crosses the stricter safety or lot bound, or the 90 percent upper interval crosses the datasheet limit", "safety_slope": {"formula": "S_safe = (L - x0 - g) / 168", "guard_band": "g = 10 percent of L", "k": float(np.mean([r["k"] for r in runs]))}, "operating_points": fused_ops, "missed_by_defect_type_at_0p5_far": missed, "review_queue_per_1000_parts": mean_sd([r["fused"][0.5]["review_queue_per_1000"] for r in runs])},
        "operations": {"oven_hours_freed_per_1000_parts": mean_sd([r["oven_hours"] for r in runs]), "latency_ms_per_lot_of_100": mean_sd([r["latency"] for r in runs])},
        "inspector_card_example": next((r["card"] for r in runs if r["card"]), {}),
        "headline": {"static_limit_recall_pct": m_a["static_limit"]["operating_points"][1]["recall_pct"]["mean"], "dpat_recall_pct": m_a["dpat_6sigma"]["operating_points"][1]["recall_pct"]["mean"], "driftguard_full_recall_pct": fused_ops[1]["recall_pct"]["mean"], "at_far_pct": 0.5, "driftguard_false_rejects_per_1000": fused_ops[1]["false_rejects_per_1000"]["mean"], "mae_linear_leakage_uA": m_b["linear_extrapolation"]["mae"]["leakage_uA"]["mean"], "mae_driftguard_leakage_uA": m_b["driftguard_B"]["mae"]["leakage_uA"]["mean"]},
        "integrity": {"test_lots_touched_once": True, "thresholds_set_on": "calibration lots", "hyperparameter_search": "No result driven search. Model settings and simulator constants were fixed in config.yaml before the five test evaluations. Only operating thresholds, power law n, and slope bound k use training or calibration lots as specified.", "where_driftguard_loses": [], "notes": ["Synthetic evidence only. Final claims require real ISRO burn in data.", "Static limit uses its physical threshold of 1.0 at every nominal operating point, so its achieved false alarm rate need not match the target.", "Manual conformal was used to preserve grouped calibration control even though MAPIE was installed."]}
    }
    # Honest loss audit.
    at05 = {m["name"]: m["operating_points"][1]["recall_pct"]["mean"] for m in methods_a}
    for name, value in at05.items():
        if name != "driftguard_A" and value > at05["driftguard_A"]:
            result["integrity"]["where_driftguard_loses"].append(f"{name} had higher Module A recall at the nominal 0.5 percent FAR")
    for p in PARAMS:
        best = min(methods_b, key=lambda m: m["mae"][p]["mean"])
        if best["name"] != "driftguard_B": result["integrity"]["where_driftguard_loses"].append(f"{best['name']} had lower {p} MAE than driftguard_B")
    return sig4(result)


def main():
    start = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    runs = []
    for seed in cfg["simulation"]["seeds"]:
        print(f"seed {seed}", flush=True); runs.append(run_seed(cfg, int(seed)))
    result = aggregate(cfg, runs, time.perf_counter() - start)
    (OUT / "driftguard_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["headline"], indent=2))


if __name__ == "__main__":
    main()
