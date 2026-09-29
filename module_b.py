from __future__ import annotations

import math
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from baselines import robust_center_scale
from simulate import PARAMS


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for p in PARAMS:
        x0 = df[f"{p}_0h"].to_numpy(dtype=float)
        x24 = df[f"{p}_24h"].to_numpy(dtype=float)
        out[f"{p}_x0"] = x0
        out[f"{p}_x24"] = x24
        out[f"{p}_delta"] = x24 - x0
        out[f"{p}_ratio"] = x24 / np.maximum(np.abs(x0), 1e-9)
        z0 = np.zeros(len(df)); z24 = np.zeros(len(df)); lot_slope = np.zeros(len(df))
        for _, idx0 in df.groupby("lot_id").groups.items():
            idx = np.asarray(list(idx0), dtype=int)
            m0, s0 = robust_center_scale(x0[idx])
            m24, s24 = robust_center_scale(x24[idx])
            z0[idx] = (x0[idx] - m0) / s0
            z24[idx] = (x24[idx] - m24) / s24
            lot_slope[idx] = np.median((x24[idx] - x0[idx]) / 24.0)
        out[f"{p}_z0"] = z0
        out[f"{p}_z24"] = z24
        out[f"{p}_lot_median_slope"] = lot_slope
    out["family_1"] = (df["device_family"].to_numpy() == 1).astype(float)
    out["tester_1"] = (df["tester_id"].to_numpy() == 1).astype(float)
    out["tester_2"] = (df["tester_id"].to_numpy() == 2).astype(float)
    return out.astype(float)


def fit_power_n(train: pd.DataFrame, p: str, grid: list[float]) -> float:
    x0 = train[f"{p}_0h"].to_numpy(dtype=float)
    x24 = train[f"{p}_24h"].to_numpy(dtype=float)
    y = train[f"{p}_168h"].to_numpy(dtype=float)
    best = (np.inf, 0.35)
    for n in grid:
        a = (x24 / np.maximum(x0, 1e-9) - 1.0) / ((24.0 / 168.0) ** n)
        pred = x0 * (1.0 + a)
        mae = float(np.mean(np.abs(y - pred)))
        if mae < best[0]: best = (mae, float(n))
    return best[1]


class ModuleB:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg; self.seed = seed
        self.models = {}; self.power_n = {}; self.family_ratio = {}

    def fit(self, train: pd.DataFrame) -> "ModuleB":
        X = feature_frame(train).to_numpy()
        b = self.cfg["module_b"]
        for pi, p in enumerate(PARAMS):
            y = train[f"{p}_168h"].to_numpy(dtype=float)
            svr = make_pipeline(StandardScaler(), SVR(C=float(b["svr_c"]), epsilon=float(b["svr_epsilon"]), gamma=b["svr_gamma"]))
            svr.fit(X, y)
            cat = CatBoostRegressor(
                iterations=int(b["catboost_iterations"]), depth=int(b["catboost_depth"]),
                learning_rate=float(b["catboost_learning_rate"]), l2_leaf_reg=float(b["catboost_l2_leaf_reg"]),
                loss_function="MAE", verbose=False, random_seed=self.seed + pi,
                thread_count=int(self.cfg["runtime"]["cpu_threads"]), allow_writing_files=False,
            )
            cat.fit(X, y)
            self.models[p] = {"svr": svr, "catboost": cat}
            self.power_n[p] = fit_power_n(train, p, [float(x) for x in b["power_n_grid"]])
            ratios = train[f"{p}_168h"] / np.maximum(train[f"{p}_24h"], 1e-9)
            self.family_ratio[p] = train.assign(_ratio=ratios).groupby("device_family")["_ratio"].median().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        X = feature_frame(df).to_numpy()
        result = {name: {} for name in ["persistence", "linear_extrapolation", "power_law_fixed_n", "lot_median_ratio", "svr", "catboost", "driftguard_B"]}
        for p in PARAMS:
            x0 = df[f"{p}_0h"].to_numpy(dtype=float); x24 = df[f"{p}_24h"].to_numpy(dtype=float)
            persistence = x24
            linear = x24 + 6.0 * (x24 - x0)
            n = self.power_n[p]
            a = (x24 / np.maximum(x0, 1e-9) - 1.0) / ((24.0 / 168.0) ** n)
            power = x0 * (1.0 + a)
            ratio = df["device_family"].map(self.family_ratio[p]).to_numpy(dtype=float)
            lot_ratio = x24 * ratio
            svr = self.models[p]["svr"].predict(X)
            cat = self.models[p]["catboost"].predict(X)
            result["persistence"][p] = persistence
            result["linear_extrapolation"][p] = linear
            result["power_law_fixed_n"][p] = power
            result["lot_median_ratio"][p] = lot_ratio
            result["svr"][p] = svr
            result["catboost"][p] = cat
            result["driftguard_B"][p] = 0.5 * (svr + cat)
        return result


def conformal_quantile(residuals: np.ndarray, nominal_pct: float) -> float:
    r = np.sort(np.asarray(residuals, dtype=float))
    rank = int(math.ceil((len(r) + 1) * nominal_pct / 100.0))
    rank = min(max(rank, 1), len(r))
    return float(r[rank - 1])


def calibrate_intervals(cal: pd.DataFrame, pred: dict, nominal_pct: float) -> dict:
    q = {}; loo = {}
    for p in PARAMS:
        y = cal[f"{p}_168h"].to_numpy(dtype=float)
        residual = np.abs(y - pred["driftguard_B"][p])
        q[p] = conformal_quantile(residual, nominal_pct)
        loo[p] = {}
        for lot in cal["lot_id"].unique():
            mask = cal["lot_id"].to_numpy() != lot
            loo[p][str(lot)] = conformal_quantile(residual[mask], nominal_pct)
    return {"q": q, "leave_one_lot_out_q": loo}


def interval_metrics(test: pd.DataFrame, pred: dict, q: dict) -> dict:
    cover_all = []; widths = {}; lot_cover = []
    for p in PARAMS:
        center = pred["driftguard_B"][p]; y = test[f"{p}_168h"].to_numpy(dtype=float)
        lo = center - q[p]; hi = center + q[p]
        cover = (y >= lo) & (y <= hi)
        cover_all.extend(cover.tolist()); widths[p] = 2.0 * q[p]
    for lot, idx0 in test.groupby("lot_id").groups.items():
        idx = np.asarray(list(idx0), dtype=int); vals = []
        for p in PARAMS:
            center = pred["driftguard_B"][p][idx]; y = test.loc[idx, f"{p}_168h"].to_numpy(dtype=float)
            vals.extend(((y >= center - q[p]) & (y <= center + q[p])).tolist())
        lot_cover.append(100.0 * float(np.mean(vals)))
    return {"pooled_coverage_pct": 100.0 * float(np.mean(cover_all)), "per_lot_min_pct": float(np.min(lot_cover)), "per_lot_median_pct": float(np.median(lot_cover)), "mean_width": widths}
