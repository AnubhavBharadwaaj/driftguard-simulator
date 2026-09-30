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
OUT = ROOT / "out" / "v4"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v3"))

from baselines import dpat_score, static_limit_score
from conformal import evaluate, fisher_p, pvalues_from_calibration
from module_a_v2 import component_scores, conditional_residual_scores
from simulate_v2 import PARAMS, generate_evaluation, impute
from v3.run_v3 import component_pvalues, mergers, smooth_conformal_p, make_traps, rival_baselines


DEFECT_TYPES = ["D1_high_in_limit", "D2_fast_drifter", "D3_late_accelerator", "D4_multivariate"]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")


def cp_interval(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    low = 0.0 if successes == 0 else beta.ppf(0.025, successes, total - successes + 1)
    high = 1.0 if successes == total else beta.ppf(0.975, successes + 1, total - successes)
    return [100 * float(low), 100 * float(high)]


def split_three(cal: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lots = np.array(sorted(cal.lot_id.unique()))
    rng = np.random.default_rng(seed + 4170)
    rng.shuffle(lots)
    groups = np.array_split(lots, 3)
    frames = [cal[cal.lot_id.isin(set(group))].reset_index(drop=True) for group in groups]
    return frames[0], frames[1], frames[2]


def source_pvalues(c1: pd.DataFrame, target: pd.DataFrame, cfg: dict, rng) -> tuple[np.ndarray, np.ndarray]:
    c1_components, _, _ = component_scores(c1, cfg)
    target_components, _, _ = component_scores(target, cfg)
    component_p = component_pvalues(c1, c1_components, target_components, rng)
    c1_conditional = conditional_residual_scores(c1)
    target_conditional = conditional_residual_scores(target)
    good = c1.is_defect.to_numpy() == 0
    conditional_p = smooth_conformal_p(c1_conditional[good], target_conditional, rng)
    sources = np.column_stack([fisher_p(component_p[:, [0, 1, 2, 6]]), conditional_p])
    review_score = -2.0 * np.sum(np.log(np.clip(sources, 1e-12, 1.0)), axis=1)
    return sources, review_score


def all_protocol_data(seed: int, cfg: dict) -> dict:
    sets = generate_evaluation(cfg, seed)
    train, cal, tin, tsh = impute(
        sets["train"],
        [sets["train"], sets["calibration"], sets["test_inprior"], sets["test_shifted"]],
        cfg,
    )
    c1, c2a, c2b = split_three(cal, seed)
    rng = np.random.default_rng(seed + 9400)
    data = {"seed": seed, "train": train, "cal": cal, "c1": c1, "c2a": c2a, "c2b": c2b, "tests": {"inprior": tin, "shifted": tsh}}
    for name, frame in [("c2a", c2a), ("c2b", c2b), ("inprior", tin), ("shifted", tsh)]:
        sources, review = source_pvalues(c1, frame, cfg, rng)
        data[name] = {"frame": frame, "sources": sources, "merged": mergers(sources), "review_score": review}
    return data


def selection_rows(data: dict) -> list[dict]:
    frame = data["c2a"]["frame"]
    output = []
    for method, pvalue in data["c2a"]["merged"].items():
        result = evaluate(frame.is_defect, pvalue <= 0.005, frame.lot_id, frame.defect_type)
        output.append({"seed": data["seed"], "method": method, "recall_pct": result["recall_pct"], "achieved_far_pct": result["achieved_far_pct"]})
    return output


def choose_global(selection: list[dict]) -> str:
    frame = pd.DataFrame(selection)
    summary = frame.groupby("method").agg(recall_pct=("recall_pct", "mean"), achieved_far_pct=("achieved_far_pct", "mean")).reset_index()
    eligible = summary[summary.achieved_far_pct <= 1.0]
    candidates = eligible if len(eligible) else summary
    return str(candidates.sort_values(["recall_pct", "achieved_far_pct"], ascending=[False, True]).iloc[0].method)


def final_pvalues(data: dict, method: str, target: str, seed_offset: int = 0, reference_indices: np.ndarray | None = None) -> np.ndarray:
    c2b = data["c2b"]["frame"]
    good = c2b.is_defect.to_numpy() == 0
    ref = -np.log(np.clip(data["c2b"]["merged"][method], 1e-300, 1.0))[good]
    if reference_indices is not None:
        ref = ref[reference_indices]
    score = -np.log(np.clip(data[target]["merged"][method], 1e-300, 1.0))
    rng = np.random.default_rng(data["seed"] * 10000 + 731 + seed_offset)
    return smooth_conformal_p(ref, score, rng)


def crc_threshold(defect_scores: np.ndarray, target_recall: float) -> dict:
    scores = np.sort(np.asarray(defect_scores, float))
    n = len(scores)
    alpha = 1.0 - target_recall
    candidates = np.r_[np.nextafter(scores[0], -np.inf), np.unique(scores)] if n else np.array([-np.inf])
    valid = []
    for threshold in candidates:
        empirical_fnr = float(np.mean(scores < threshold)) if n else 1.0
        corrected = (n * empirical_fnr + 1.0) / (n + 1.0)
        if corrected <= alpha + 1e-12:
            valid.append((float(threshold), empirical_fnr, corrected))
    if not valid:
        return {"threshold": float("-inf"), "empirical_fnr": 0.0, "corrected_risk": 1.0 / (n + 1.0), "n_calibration_defects": n, "feasible": False}
    threshold, empirical, corrected = max(valid, key=lambda row: row[0])
    return {"threshold": threshold, "empirical_fnr": empirical, "corrected_risk": corrected, "n_calibration_defects": n, "feasible": True}


def metric_row(seed: int, detector: str, test_set: str, tier: str, target: str, hour: int, frame: pd.DataFrame, flags: np.ndarray) -> dict:
    y = frame.is_defect.to_numpy(int)
    result = evaluate(y, flags, frame.lot_id, frame.defect_type)
    defect = y == 1
    good = y == 0
    tp = int(np.sum(flags & defect))
    fp = int(np.sum(flags & good))
    n_by_type = {name: int(np.sum(frame.defect_type.to_numpy() == name)) for name in DEFECT_TYPES}
    return {
        "seed": seed,
        "detector": detector,
        "test_set": test_set,
        "tier": tier,
        "target": target,
        "decision_hour": hour,
        "achieved_far_pct": result["achieved_far_pct"],
        "recall_pct": result["recall_pct"],
        "false_rejects_per_1000_good": 1000 * fp / max(1, int(good.sum())),
        "review_per_1000_good": 1000 * fp / max(1, int(good.sum())),
        "n_pos": int(defect.sum()),
        "n_good": int(good.sum()),
        "recall_cp95": cp_interval(tp, int(defect.sum())),
        "far_cp95": cp_interval(fp, int(good.sum())),
        "recall_by_type": result["recall_by_type"],
        "n_by_type": n_by_type,
    }


def sample_good_lots(c2b: pd.DataFrame, target_n: int, draw: int, seed: int) -> np.ndarray:
    good_frame = c2b[c2b.is_defect == 0].reset_index(drop=True)
    if target_n >= len(good_frame):
        return np.arange(len(good_frame))
    rng = np.random.default_rng(seed * 1000 + target_n * 7 + draw)
    lots = np.array(sorted(good_frame.lot_id.unique()))
    rng.shuffle(lots)
    selected = []
    total = 0
    for lot in lots:
        idx = np.flatnonzero(good_frame.lot_id.to_numpy() == lot)
        selected.extend(idx.tolist())
        total += len(idx)
        if total >= target_n:
            break
    return np.asarray(selected, int)


def defect_heavy_lot(test: pd.DataFrame, seed: int, fraction: float = 0.15) -> pd.DataFrame:
    result = test.copy().reset_index(drop=True)
    lot_sizes = result.groupby("lot_id").size().sort_values(ascending=False)
    lot = lot_sizes.index[0]
    indices = np.flatnonzero(result.lot_id.to_numpy() == lot)
    rng = np.random.default_rng(seed + 1550)
    chosen = rng.choice(indices, size=max(1, int(round(fraction * len(indices)))), replace=False)
    for parameter in PARAMS:
        result.loc[chosen, f"{parameter}_24h"] *= 1.75
    result.loc[chosen, "is_defect"] = 1
    result.loc[chosen, "defect_type"] = "D2_fast_drifter"
    return result


def deterministic_sources(cal: pd.DataFrame, target: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    cal_components, _, _ = component_scores(cal, cfg)
    target_components, _, _ = component_scores(target, cfg)
    good = cal.is_defect.to_numpy() == 0
    pcomp = np.column_stack([pvalues_from_calibration(cal_components[good, j], target_components[:, j]) for j in range(cal_components.shape[1])])
    cal_cond = conditional_residual_scores(cal)
    target_cond = conditional_residual_scores(target)
    pcond = pvalues_from_calibration(cal_cond[good], target_cond)
    return np.column_stack([fisher_p(pcomp[:, [0, 1, 2, 6]]), pcond]), cal_components


def ablation_for_seed(data: dict, method: str, cfg: dict) -> list[dict]:
    seed = data["seed"]
    test = data["inprior"]["frame"]
    rows = []
    # v2 reproduction: the whole calibration set supplies source and final references.
    cal = data["cal"]
    cal_sources, _ = deterministic_sources(cal, cal, cfg)
    test_sources, _ = deterministic_sources(cal, test, cfg)
    cal_good = cal.is_defect.to_numpy() == 0
    cal_score = -np.log(np.clip(fisher_p(cal_sources), 1e-300, 1.0))
    test_score = -np.log(np.clip(fisher_p(test_sources), 1e-300, 1.0))
    p_v2 = pvalues_from_calibration(cal_score[cal_good], test_score)
    rows.append(metric_row(seed, "v2_reproduction", "inprior", "REJECT", "FAR 0.5%", 24, test, p_v2 <= 0.005))

    # Disjoint C1/C2 with deterministic p values and Fisher.
    c1 = data["c1"]
    c2 = pd.concat([data["c2a"]["frame"], data["c2b"]["frame"]], ignore_index=True)
    c2_sources, _ = deterministic_sources(c1, c2, cfg)
    test_sources, _ = deterministic_sources(c1, test, cfg)
    good2 = c2.is_defect.to_numpy() == 0
    c2_score = -np.log(np.clip(fisher_p(c2_sources), 1e-300, 1.0))
    test_score = -np.log(np.clip(fisher_p(test_sources), 1e-300, 1.0))
    p_disjoint = pvalues_from_calibration(c2_score[good2], test_score)
    rows.append(metric_row(seed, "plus_disjoint_c1_c2", "inprior", "REJECT", "FAR 0.5%", 24, test, p_disjoint <= 0.005))

    rng = np.random.default_rng(seed + 6200)
    p_random = smooth_conformal_p(c2_score[good2], test_score, rng)
    rows.append(metric_row(seed, "plus_randomised_ties", "inprior", "REJECT", "FAR 0.5%", 24, test, p_random <= 0.005))

    # v3 approximation with twice mean and two-way C1/C2.
    c2_twice = mergers(c2_sources)["twice_mean"]
    test_twice = mergers(test_sources)["twice_mean"]
    p_v3 = smooth_conformal_p(-np.log(np.clip(c2_twice[good2], 1e-300, 1.0)), -np.log(np.clip(test_twice, 1e-300, 1.0)), rng)
    rows.append(metric_row(seed, "v3_twice_mean", "inprior", "REJECT", "FAR 0.5%", 24, test, p_v3 <= 0.005))

    p_v4 = final_pvalues(data, method, "inprior")
    rows.append(metric_row(seed, "v4_three_way", "inprior", "REJECT", "FAR 0.5%", 24, test, p_v4 <= 0.005))
    return rows


def aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    metrics = ["achieved_far_pct", "recall_pct", "false_rejects_per_1000_good", "review_per_1000_good"]
    for group, local in frame.groupby(keys, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        row = dict(zip(keys, group))
        for metric in metrics:
            values = pd.to_numeric(local[metric], errors="coerce").dropna()
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row["n_pos"] = int(local.n_pos.sum())
        row["n_good"] = int(local.n_good.sum())
        tp = int(round(sum(local.recall_pct * local.n_pos / 100)))
        fp = int(round(sum(local.achieved_far_pct * local.n_good / 100)))
        row["recall_cp95"] = cp_interval(tp, row["n_pos"])
        row["far_cp95"] = cp_interval(fp, row["n_good"])
        by_type = {}
        n_by_type = {}
        for defect_type in DEFECT_TYPES:
            n = int(sum(item.get(defect_type, 0) for item in local.n_by_type))
            hit = 0.0
            for counts, recalls in zip(local.n_by_type, local.recall_by_type):
                local_n = int(counts.get(defect_type, 0))
                local_recall = recalls.get(defect_type)
                if local_n and local_recall is not None and np.isfinite(local_recall):
                    hit += local_n * local_recall / 100
            n_by_type[defect_type] = n
            by_type[defect_type] = 100 * hit / n if n else None
        row["n_by_type"] = n_by_type
        row["recall_by_type"] = by_type
        output.append(row)
    return output


def main() -> None:
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config_v2.yaml").read_text())
    seeds = [int(seed) for seed in cfg["simulation"]["seeds"]]
    data_by_seed = []
    selection = []
    for seed in seeds:
        print(f"v4 protocol seed {seed}", flush=True)
        data = all_protocol_data(seed, cfg)
        data_by_seed.append(data)
        selection.extend(selection_rows(data))
    chosen = choose_global(selection)
    print(f"selected merger: {chosen}", flush=True)

    pooled_review_defects = np.concatenate([
        data["c2b"]["review_score"][data["c2b"]["frame"].is_defect.to_numpy() == 1]
        for data in data_by_seed
    ])
    review_thresholds = {target: crc_threshold(pooled_review_defects, target / 100.0) for target in [80, 90, 95]}

    operating = []
    calibration_size = []
    ablation = []
    baseline_rows = []
    trap_rows = []
    limits = {parameter: float(cfg["simulation"]["parameters"][parameter]["limit"]) for parameter in PARAMS}
    for data in data_by_seed:
        seed = data["seed"]
        for set_name in ["inprior", "shifted"]:
            frame = data[set_name]["frame"]
            pvalue = final_pvalues(data, chosen, set_name)
            for far in [0.1, 0.5, 1.0]:
                operating.append(metric_row(seed, f"sahodrift_{chosen}", set_name, "REJECT", f"FAR {far}%", 24, frame, pvalue <= far / 100.0))
            score = data[set_name]["review_score"]
            for target, detail in review_thresholds.items():
                flags = score >= detail["threshold"]
                row = metric_row(seed, "sahodrift_fisher_review_score", set_name, "REVIEW", f"recall {target}%", 24, frame, flags)
                row["risk_control"] = detail
                operating.append(row)

            # Static limits and dynamic PAT are calibrated to the same FAR using C2b good parts.
            c2b = data["c2b"]["frame"]
            for baseline, fn in [("static_limit", lambda f: static_limit_score(f, limits)), ("dynamic_pat_6sigma", dpat_score)]:
                cal_score = fn(c2b)
                test_score = fn(frame)
                good = c2b.is_defect.to_numpy() == 0
                threshold = np.quantile(cal_score[good], 0.995, method="higher")
                baseline_rows.append(metric_row(seed, baseline, set_name, "REJECT", "FAR 0.5%", 24, frame, test_score >= threshold))

        ablation.extend(ablation_for_seed(data, chosen, cfg))

        # Calibration size experiment on the in-prior test set.
        c2b = data["c2b"]["frame"]
        good_frame = c2b[c2b.is_defect == 0].reset_index(drop=True)
        raw_ref = -np.log(np.clip(data["c2b"]["merged"][chosen], 1e-300, 1.0))[c2b.is_defect.to_numpy() == 0]
        test_score = -np.log(np.clip(data["inprior"]["merged"][chosen], 1e-300, 1.0))
        for target_n in [100, 200, 400, 800, len(good_frame)]:
            draws = 1 if target_n >= len(good_frame) else 5
            for draw in range(draws):
                idx = sample_good_lots(c2b, target_n, draw, seed)
                pvalue = smooth_conformal_p(raw_ref[idx], test_score, np.random.default_rng(seed * 10000 + target_n + draw))
                row = metric_row(seed, "calibration_resolution", "inprior", "REJECT", "FAR 0.5%", 24, data["inprior"]["frame"], pvalue <= 0.005)
                row.update({"requested_n": "all" if target_n == len(good_frame) else int(target_n), "actual_n": int(len(idx)), "draw": draw, "theoretical_grid_pct": 100.0 / (len(idx) + 1)})
                calibration_size.append(row)

        # Sahodrift and independently reimplemented rival trap rows.
        traps = make_traps(data["inprior"]["frame"])
        traps["defect_heavy_lot_15pct"] = defect_heavy_lot(data["inprior"]["frame"], seed)
        for trap_name, trap in traps.items():
            rng = np.random.default_rng(seed + 8800)
            sources, _ = source_pvalues(data["c1"], trap, cfg, rng)
            merged = mergers(sources)[chosen]
            good_ref = data["c2b"]["frame"].is_defect.to_numpy() == 0
            ref_score = -np.log(np.clip(data["c2b"]["merged"][chosen], 1e-300, 1.0))[good_ref]
            pvalue = smooth_conformal_p(ref_score, -np.log(np.clip(merged, 1e-300, 1.0)), rng)
            flags = pvalue <= 0.005
            ev = evaluate(trap.is_defect, flags, trap.lot_id, trap.defect_type)
            trap_rows.append({"seed": seed, "detector": f"sahodrift_{chosen}", "variant": "recalibrated_0p5", "trap": trap_name, "false_alarm_pct": ev["achieved_far_pct"], "recall_pct": ev["recall_pct"], "n_good": int(np.sum(trap.is_defect.to_numpy() == 0)), "n_pos": int(np.sum(trap.is_defect.to_numpy() == 1))})

        rival_tests = {"inprior": data["inprior"]["frame"], "shifted": data["shifted"]["frame"], **traps}
        rivals = rival_baselines(data["train"], data["cal"], rival_tests, seed)
        for row in rivals:
            if row["test_set"] in traps:
                trap_rows.append({"seed": seed, "detector": row["id"], "variant": row["variant"], "trap": row["test_set"], "false_alarm_pct": row["achieved_far_pct"], "recall_pct": row["recall_pct"], "n_good": int(np.sum(traps[row["test_set"]].is_defect.to_numpy() == 0)), "n_pos": int(np.sum(traps[row["test_set"]].is_defect.to_numpy() == 1))})
        # Preserve exact v3 rival metrics and label the R3 fallback explicitly.
        write_json(OUT / f"rival_seed_{seed}.json", rivals)

        write_json(OUT / "checkpoint_phase_a.json", {"chosen_merger": chosen, "selection": selection, "operating": operating, "calibration_size": calibration_size, "ablation": ablation, "baselines": baseline_rows, "traps": trap_rows})

    operating_summary = aggregate(operating, ["detector", "test_set", "tier", "target", "decision_hour"])
    ablation_summary = aggregate(ablation, ["detector", "test_set", "tier", "target", "decision_hour"])
    baseline_summary = aggregate(baseline_rows, ["detector", "test_set", "tier", "target", "decision_hour"])
    trap_summary = (
        pd.DataFrame(trap_rows)
        .groupby(["detector", "variant", "trap"], dropna=False)
        .agg(false_alarm_pct_mean=("false_alarm_pct", "mean"), false_alarm_pct_sd=("false_alarm_pct", "std"), recall_pct_mean=("recall_pct", "mean"), n_good=("n_good", "sum"), n_pos=("n_pos", "sum"))
        .reset_index()
        .fillna({"false_alarm_pct_sd": 0.0})
        .to_dict("records")
    )
    selection_summary = pd.DataFrame(selection).groupby("method").agg(recall_pct_mean=("recall_pct", "mean"), achieved_far_pct_mean=("achieved_far_pct", "mean")).reset_index().to_dict("records")

    results = {
        "meta": {
            "date": "2026-09-30",
            "branch": "v4",
            "seeds": seeds,
            "runtime_s": time.perf_counter() - start,
            "hardware": f"{platform.machine()} {platform.platform()}",
            "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
            "wp_status": {
                "A1": {"status": "done", "reason": "Three-way lot calibration, real calibration-size draws and continuous CRC review score implemented"},
                "A2": {"status": "done", "reason": "Five-row protocol ablation completed across five seeds"},
                "A3": {"status": "done", "reason": "Per-type counts and recalls included for all v4 operating rows"},
                "A4": {"status": "partial", "reason": "Static and dynamic PAT rows plus Sahodrift traps complete; exact XGBoost remains gated by missing libomp and rival trap expansion remains"},
            },
        },
        "protocol": {"split": "C1 source calibration, C2a merger selection, C2b final recalibration", "selected_merger": chosen, "selection_on_c2a": selection_summary},
        "calibration_size": calibration_size,
        "review_crc": {"source": "Angelopoulos et al. Conformal Risk Control, arXiv:2208.02814, Equation 4", "thresholds": review_thresholds},
        "detectors": operating_summary + baseline_summary,
        "ablation_protocol": ablation_summary,
        "recall_by_type": [{key: row[key] for key in ["detector", "test_set", "tier", "target", "decision_hour", "recall_by_type", "n_by_type"]} for row in operating_summary + baseline_summary],
        "traps_v4": trap_summary,
        "integrity": {
            "selection_uses": "C2a lots only",
            "final_thresholds_use": "C2b lots only",
            "test_labels_used_for_selection": False,
            "limitations": ["Exact XGBoost is gated by missing libomp", "CRC guarantees require exchangeability", "95 percent review risk control can be infeasible when the pooled calibration defect count is too small"],
        },
    }
    write_json(OUT / "phase_a_results.json", results)
    write_json(OUT / "ablation_protocol.json", ablation_summary)
    write_json(OUT / "traps_v4.json", {"summary": trap_summary, "per_seed": trap_rows})
    print(json.dumps(jsonable({"selected_merger": chosen, "review_thresholds": review_thresholds, "ablation": ablation_summary}), indent=2))


if __name__ == "__main__":
    main()
