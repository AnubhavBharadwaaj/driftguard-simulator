from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from baselines import evaluate_flags, robust_center_scale, static_limit_score
from module_b import feature_frame
from simulate import PARAMS


def lot_slope_stats(df: pd.DataFrame) -> dict:
    out = {}
    for lot, idx0 in df.groupby("lot_id").groups.items():
        idx = np.asarray(list(idx0), dtype=int); out[str(lot)] = {}
        for p in PARAMS:
            slope = (df.loc[idx, f"{p}_24h"].to_numpy(dtype=float) - df.loc[idx, f"{p}_0h"].to_numpy(dtype=float)) / 24.0
            med, sig = robust_center_scale(slope)
            out[str(lot)][p] = {"median": med, "mad_sigma": sig}
    return out


def safety_components(df: pd.DataFrame, pred: dict, q: dict, limits: dict, k: float, guard_fraction: float) -> dict:
    stats = lot_slope_stats(df)
    slope_flag = np.zeros(len(df), dtype=bool); interval_flag = np.zeros(len(df), dtype=bool)
    parameter_flags = {p: {"slope": np.zeros(len(df), dtype=bool), "interval": np.zeros(len(df), dtype=bool)} for p in PARAMS}
    details = {}
    for lot, idx0 in df.groupby("lot_id").groups.items():
        idx = np.asarray(list(idx0), dtype=int); details[str(lot)] = {}
        for p in PARAMS:
            x0 = df.loc[idx, f"{p}_0h"].to_numpy(dtype=float)
            forecast = pred["driftguard_B"][p][idx]
            predicted_slope = (forecast - x0) / 168.0
            safe = (limits[p] - x0 - guard_fraction * limits[p]) / 168.0
            lot_bound = stats[str(lot)][p]["median"] + k * stats[str(lot)][p]["mad_sigma"]
            bound = np.minimum(safe, lot_bound)
            sf = predicted_slope > bound
            inf = forecast + q[p] > limits[p]
            slope_flag[idx] |= sf; interval_flag[idx] |= inf
            parameter_flags[p]["slope"][idx] = sf; parameter_flags[p]["interval"][idx] = inf
            details[str(lot)][p] = {"lot_bound": lot_bound}
    return {"slope_flag": slope_flag, "interval_flag": interval_flag, "parameter_flags": parameter_flags, "lot_stats": stats, "details": details}


def choose_k(cal: pd.DataFrame, cal_pred: dict, q: dict, limits: dict, cfg: dict) -> float:
    target = float(cfg["fusion"]["k_target_far_pct"])
    healthy = cal["is_defect"].to_numpy() == 0
    chosen = float(cfg["fusion"]["lot_slope_k_grid"][-1])
    for k in [float(x) for x in cfg["fusion"]["lot_slope_k_grid"]]:
        c = safety_components(cal, cal_pred, q, limits, k, float(cfg["fusion"]["guard_band_fraction"]))
        far = 100.0 * float(np.mean(c["slope_flag"][healthy]))
        if far <= target:
            chosen = k; break
    return chosen


def fused_flags(df: pd.DataFrame, a_score: np.ndarray, a_threshold: float, pred: dict, q: dict, limits: dict, k: float, cfg: dict):
    static = static_limit_score(df, limits) >= 1.0
    comp = safety_components(df, pred, q, limits, k, float(cfg["fusion"]["guard_band_fraction"]))
    module_a = a_score >= a_threshold
    flags = static | module_a | comp["slope_flag"] | comp["interval_flag"]
    return flags, {"static": static, "module_a": module_a, **comp}


def inspector_card(test: pd.DataFrame, train: pd.DataFrame, flags: np.ndarray, components: dict, a_detail: dict, pred: dict, q: dict, limits: dict, cfg: dict) -> dict:
    candidates = np.flatnonzero((test["defect_type"].to_numpy() == "D2_fast_drifter") & flags)
    if len(candidates) == 0:
        candidates = np.flatnonzero((test["is_defect"].to_numpy() == 1) & flags)
    if len(candidates) == 0:
        return {}
    i = int(candidates[0]); row = test.iloc[i]
    ratios = {p: pred["driftguard_B"][p][i] / limits[p] for p in PARAMS}
    p = max(ratios, key=ratios.get); pi = PARAMS.index(p)
    lot = str(row["lot_id"]); stat = a_detail["lot_stats"][lot][p]
    x0 = float(row[f"{p}_0h"]); x24 = float(row[f"{p}_24h"]); forecast = float(pred["driftguard_B"][p][i])
    upper = forecast + q[p]; lower = forecast - q[p]
    predicted_slope = (forecast - x0) / 168.0
    safety_slope = (limits[p] - x0 - float(cfg["fusion"]["guard_band_fraction"]) * limits[p]) / 168.0
    reasons = []
    if components["static"][i]: reasons.append("static limit exceeded")
    if components["module_a"][i]: reasons.append("lot relative early anomaly")
    if components["parameter_flags"][p]["slope"][i]: reasons.append("predicted slope exceeds safety bound")
    if components["parameter_flags"][p]["interval"][i]: reasons.append("upper interval exceeds datasheet limit")
    Xtr = feature_frame(train).to_numpy(); Xte = feature_frame(test.iloc[[i]].reset_index(drop=True)).to_numpy()
    scaler = StandardScaler().fit(Xtr); dist = np.sum((scaler.transform(Xtr) - scaler.transform(Xte)[0]) ** 2, axis=1)
    nn = np.argsort(dist)[:3]
    return {
        "part_id": str(row["part_id"]), "lot_id": lot, "parameter": p,
        "x0": x0, "x24": x24, "lot_median_24h": float(stat["median24"]),
        "ratio_to_lot_median": x24 / max(float(stat["median24"]), 1e-9),
        "robust_z": float(a_detail["zlevel"][i, pi]), "datasheet_max": float(limits[p]),
        "static_pass": bool(not components["static"][i]), "forecast_168": forecast,
        "interval_168": [float(lower), float(upper)], "actual_168": float(row[f"{p}_168h"]),
        "predicted_slope": predicted_slope, "safety_slope": safety_slope,
        "reason_codes": reasons, "nearest_neighbours": {"k": 3, "defective": int(train.iloc[nn]["is_defect"].sum())},
        "true_label": str(row["defect_type"]),
    }


def measure_latency(test: pd.DataFrame, model_a, model_b, q: dict, limits: dict, k: float, threshold: float, cfg: dict) -> float:
    sample = test.sample(n=100, replace=len(test) < 100, random_state=991).copy().reset_index(drop=True)
    sample["lot_id"] = "LATENCY_LOT"
    repeats = int(cfg["fusion"]["latency_repeats"]); times = []
    for _ in range(repeats):
        start = time.perf_counter()
        a = model_a.score(sample)
        p = model_b.predict(sample)
        fused_flags(sample, a, threshold, p, q, limits, k, cfg)
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))
