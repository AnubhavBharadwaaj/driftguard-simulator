from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "v5"
sys.path[:0] = [str(ROOT), str(ROOT / "v3"), str(ROOT / "v4")]

from conformal import evaluate, fisher_p
from module_a_v2 import component_scores, conditional_residual_scores
from simulate_v2 import generate_evaluation, impute
from v3.run_v3 import mergers
from v4.run_phase_a import crc_threshold, split_three

DEFECT_TYPES = ["D1_high_in_limit", "D2_fast_drifter", "D3_late_accelerator", "D4_multivariate"]
DETECTABLE_24 = {"D1_high_in_limit", "D2_fast_drifter", "D4_multivariate"}


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def cp(successes: int, total: int) -> list[float | None]:
    if not total:
        return [None, None]
    lo = 0.0 if successes == 0 else beta.ppf(0.025, successes, total - successes + 1)
    hi = 1.0 if successes == total else beta.ppf(0.975, successes + 1, total - successes)
    return [100 * float(lo), 100 * float(hi)]


def at_hour(frame: pd.DataFrame, hour: int) -> pd.DataFrame:
    out = frame.copy()
    if hour != 24:
        for p in ("leakage_uA", "iddq_uA", "delay_ns"):
            out[f"{p}_24h"] = out[f"{p}_{hour}h"]
    return out


def raw_scores(frame: pd.DataFrame, cfg: dict, hour: int) -> tuple[np.ndarray, np.ndarray]:
    local = at_hour(frame, hour)
    components, _, _ = component_scores(local, cfg)
    conditional = conditional_residual_scores(local)
    return components, conditional


def upper_p(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, float))
    val = np.asarray(values, float)
    ge = len(ref) - np.searchsorted(ref, val, side="left")
    return (1.0 + ge) / (len(ref) + 1.0)


def source_p_cross(cal: pd.DataFrame, targets: dict[str, pd.DataFrame], cfg: dict, hour: int):
    """Lot-held-out source p values for calibration; all healthy calibration for tests."""
    cal_comp, cal_cond = raw_scores(cal, cfg, hour)
    healthy = cal.is_defect.to_numpy() == 0
    lots = cal.lot_id.to_numpy()
    cal_pc = np.zeros_like(cal_comp)
    cal_pcond = np.zeros(len(cal))
    for lot in np.unique(lots):
        idx = lots == lot
        ref = healthy & ~idx
        for j in range(cal_comp.shape[1]):
            cal_pc[idx, j] = upper_p(cal_comp[ref, j], cal_comp[idx, j])
        cal_pcond[idx] = upper_p(cal_cond[ref], cal_cond[idx])
    cal_sources = np.column_stack([fisher_p(cal_pc[:, [0, 1, 2, 6]]), cal_pcond])
    output = {"calibration": cal_sources}
    for name, frame in targets.items():
        comp, cond = raw_scores(frame, cfg, hour)
        pc = np.column_stack([upper_p(cal_comp[healthy, j], comp[:, j]) for j in range(comp.shape[1])])
        output[name] = np.column_stack([fisher_p(pc[:, [0, 1, 2, 6]]), upper_p(cal_cond[healthy], cond)])
    return output


def final_cross_p(cal: pd.DataFrame, cal_merged: np.ndarray, test_merged: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    score_cal = -np.log(np.clip(cal_merged, 1e-300, 1.0))
    score_test = -np.log(np.clip(test_merged, 1e-300, 1.0))
    healthy = cal.is_defect.to_numpy() == 0
    lots = cal.lot_id.to_numpy()
    pcal = np.ones(len(cal))
    for lot in np.unique(lots):
        idx = lots == lot
        pcal[idx] = upper_p(score_cal[healthy & ~idx], score_cal[idx])
    ptest = upper_p(score_cal[healthy], score_test)
    n = int(healthy.sum())
    return pcal, ptest, {"n_healthy_calibration": n, "min_attainable_p": 1.0 / (n + 1.0)}


def select_rows(seed: int, c1: pd.DataFrame, c2a: pd.DataFrame, cfg: dict) -> list[dict]:
    # Source evidence uses C1; every merger is then recalibrated identically inside C2a.
    sources = source_p_cross(c1, {"c2a": c2a}, cfg, 24)["c2a"]
    merged = mergers(sources)
    rows = []
    for method, value in merged.items():
        pcal, _, res = final_cross_p(c2a, value, value)
        flags = pcal <= 0.005
        ev = evaluate(c2a.is_defect, flags, c2a.lot_id, c2a.defect_type)
        rows.append({"seed": seed, "method": method, "recall_pct": ev["recall_pct"],
                     "achieved_far_pct": ev["achieved_far_pct"], **res,
                     "infeasible_at_alpha": res["min_attainable_p"] > 0.005})
    return rows


def metric(seed: int, method: str, test_set: str, hour: int, frame: pd.DataFrame,
           pvalue: np.ndarray, alpha: float, res: dict) -> dict:
    flags = pvalue <= alpha
    ev = evaluate(frame.is_defect, flags, frame.lot_id, frame.defect_type)
    y = frame.is_defect.to_numpy() == 1
    fp = int(np.sum(flags & ~y)); tp = int(np.sum(flags & y))
    return {"seed": seed, "detector": method, "test_set": test_set, "tier": "REJECT",
            "target_alpha": alpha, "target_far_pct": 100 * alpha, "decision_hour": hour,
            "achieved_far_pct": ev["achieved_far_pct"], "recall_pct": ev["recall_pct"],
            "false_rejects": fp, "true_rejects": tp, "n_good": int((~y).sum()), "n_pos": int(y.sum()),
            "far_cp95": cp(fp, int((~y).sum())), "recall_cp95": cp(tp, int(y.sum())),
            "recall_by_type": ev["recall_by_type"], **res,
            "infeasible_at_alpha": res["min_attainable_p"] > alpha}


def review_rows(seed: int, test_set: str, hour: int, cal: pd.DataFrame, test: pd.DataFrame,
                cal_sources: np.ndarray, test_sources: np.ndarray) -> list[dict]:
    cal_score = -2.0 * np.log(np.clip(cal_sources, 1e-12, 1.0)).sum(axis=1)
    test_score = -2.0 * np.log(np.clip(test_sources, 1e-12, 1.0)).sum(axis=1)
    scopes = [("all_defects", np.ones(len(cal), bool))]
    if hour == 24:
        scopes.append(("detectable_at_24h_D1_D2_D4", cal.defect_type.isin(DETECTABLE_24).to_numpy()))
    rows = []
    for scope, mask in scopes:
        defects = (cal.is_defect.to_numpy() == 1) & mask
        for target in (0.8, 0.95):
            rule = crc_threshold(cal_score[defects], target)
            flags = test_score >= rule["threshold"]
            y = test.is_defect.to_numpy() == 1
            eval_mask = y if scope == "all_defects" else test.defect_type.isin(DETECTABLE_24).to_numpy()
            recall = 100 * float(np.mean(flags[eval_mask])) if eval_mask.any() else None
            good = ~y
            rows.append({"seed": seed, "test_set": test_set, "decision_hour": hour, "scope": scope,
                         "target_recall_pct": 100 * target, "recall_pct": recall,
                         "review_per_1000_good": 1000 * float(np.mean(flags[good])),
                         "n_evaluation_defects": int(eval_mask.sum()), **rule})
    return rows


def aggregate_reject(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    out = []
    keys = ["detector", "test_set", "decision_hour", "target_alpha"]
    for group, local in df.groupby(keys):
        fp = int(local.false_rejects.sum()); ng = int(local.n_good.sum())
        tp = int(local.true_rejects.sum()); np_ = int(local.n_pos.sum())
        out.append(dict(zip(keys, group)) | {
            "mean_achieved_far_pct": float(local.achieved_far_pct.mean()),
            "sd_achieved_far_pct": float(local.achieved_far_pct.std(ddof=1)),
            "per_seed_far_pct": [{"seed": int(r.seed), "far_pct": float(r.achieved_far_pct)} for r in local.itertuples()],
            "seeds_above_nominal_alpha": int(np.sum(local.achieved_far_pct > 100 * local.target_alpha)),
            "seeds_above_0_75_pct": int(np.sum(local.achieved_far_pct > 0.75)),
            "pooled_far_pct": 100 * fp / ng, "pooled_far_cp95": cp(fp, ng),
            "mean_recall_pct": float(local.recall_pct.mean()), "pooled_recall_pct": 100 * tp / np_,
            "pooled_recall_cp95": cp(tp, np_), "n_healthy_calibration_by_seed": local.n_healthy_calibration.tolist(),
            "min_attainable_p_by_seed": local.min_attainable_p.tolist(),
            "any_infeasible_at_alpha": bool(local.infeasible_at_alpha.any())})
    return out


def main():
    started = time.time()
    cfg = yaml.safe_load((ROOT / "config_v2.yaml").read_text())
    seeds = cfg["simulation"]["seeds"]
    selections = []; cached = {}
    for seed in seeds:
        sets = generate_evaluation(cfg, seed)
        train, cal, tin, tsh = impute(sets["train"], [sets["train"], sets["calibration"], sets["test_inprior"], sets["test_shifted"]], cfg)
        c1, c2a, c2b = split_three(cal, seed)
        selections.extend(select_rows(seed, c1, c2a, cfg))
        cached[seed] = (cal, tin, tsh)
    sdf = pd.DataFrame(selections)
    summary = sdf.groupby("method").agg(mean_recall_pct=("recall_pct", "mean"), mean_far_pct=("achieved_far_pct", "mean")).reset_index()
    eligible = summary[summary.mean_far_pct <= 0.5]
    chosen = str((eligible if len(eligible) else summary).sort_values(["mean_recall_pct", "mean_far_pct"], ascending=[False, True]).iloc[0].method)

    rejects = []; reviews = []
    for seed, (cal, tin, tsh) in cached.items():
        for hour in (24, 96, 168):
            src = source_p_cross(cal, {"inprior": tin, "shifted": tsh}, cfg, hour)
            merged_cal = mergers(src["calibration"])[chosen]
            for name, test in (("inprior", tin), ("shifted", tsh)):
                merged_test = mergers(src[name])[chosen]
                _, ptest, res = final_cross_p(cal, merged_cal, merged_test)
                if hour in (24, 168):
                    rejects.append(metric(seed, "v5_cross_conformal", name, hour, test, ptest, 0.005, res))
                    rejects.append(metric(seed, "v5_cross_conformal_conservative", name, hour, test, ptest, 0.0025, res))
                reviews.extend(review_rows(seed, name, hour, cal, test, src["calibration"], src[name]))

    aggregates = aggregate_reject(rejects)
    main_row = next(x for x in aggregates if x["detector"] == "v5_cross_conformal" and x["test_set"] == "inprior" and x["decision_hour"] == 24)
    acceptance = main_row["mean_achieved_far_pct"] <= 0.5 and main_row["seeds_above_0_75_pct"] <= 1
    payload = {
        "schema_version": "5.0", "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "protocol": {"selected_merger": chosen, "selection_on_c2a_recalibrated": selections,
                     "selection_summary": summary.to_dict("records"),
                     "cross_conformal": {"description": "Each calibration lot is ranked against healthy parts in every other calibration lot; tests are ranked against all healthy calibration parts.",
                       "nominal_alpha": 0.005, "conservative_alpha": 0.0025,
                       "theorem_used": "Barber et al. (2021), Theorem 1: jackknife+ coverage is at least 1 minus 2 alpha; Appendix A.2 states average p-values are valid up to a factor of two. This motivates, but does not prove exact validity for, the conservative alpha/2 sensitivity run.",
                       "outlier_caveat": "Bates et al. (2023) establishes marginal super-uniformity for split-conformal outlier p-values and warns that conditional validity can fail. Our lot-cross procedure is therefore reported empirically, not claimed exact."}},
        "operating_rows": rejects, "operating_aggregates": aggregates,
        "review_crc": {"criterion": "Angelopoulos et al. Eq. 4: (n * empirical risk + 1)/(n + 1) <= alpha", "rows": reviews},
        "demo": {"public_url": "https://sahodrift-app.vercel.app", "supersedes_streamlit": True},
        "realdata_v5": {"cora_identifier_erratum": {"correct_doi": "10.34810/data3204", "incorrect_readme_doi": "10.34810/data3206", "verification": "DataCite title for data3204 is the Perez-Farre et al. Si IGBT accelerated power-cycling dataset; data3206 is an unrelated colistin delivery dataset."}},
        "integrity": {"acceptance_passed": bool(acceptance), "limitations": [] if acceptance else ["The predeclared v5 in-prior FAR acceptance criterion failed. No test-lot tuning was performed."],
                      "earlier_outputs_modified": False, "runtime_seconds": time.time() - started,
                      "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sahodrift_results_v5.json").write_text(json.dumps(serial(payload), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"chosen": chosen, "acceptance": acceptance, "main": main_row, "runtime_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
