from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from simulate import PARAMS


def robust_center_scale(values: np.ndarray, floor: float = 1e-6) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    return med, max(1.4826 * mad, floor)


def static_limit_score(df: pd.DataFrame, limits: dict) -> np.ndarray:
    pieces = []
    for p in PARAMS:
        pieces.append(df[f"{p}_0h"].to_numpy() / limits[p])
        pieces.append(df[f"{p}_24h"].to_numpy() / limits[p])
    return np.nanmax(np.column_stack(pieces), axis=1)


def delta_score(df: pd.DataFrame) -> np.ndarray:
    vals = []
    for p in PARAMS:
        x0 = df[f"{p}_0h"].to_numpy()
        x24 = df[f"{p}_24h"].to_numpy()
        vals.append(np.abs(x24 - x0) / np.maximum(np.abs(x0), 1e-9))
    return np.nanmax(np.column_stack(vals), axis=1)


def dpat_score(df: pd.DataFrame, floor: float = 1e-6) -> np.ndarray:
    scores = np.zeros(len(df))
    for _, idx in df.groupby("lot_id").groups.items():
        idx = np.asarray(list(idx))
        local = []
        for p in PARAMS:
            x = df.loc[idx, f"{p}_24h"].to_numpy(dtype=float)
            med, sig = robust_center_scale(x, floor)
            local.append(np.abs(x - med) / sig)
        scores[idx] = np.nanmax(np.column_stack(local), axis=1)
    return scores


def isolation_features(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for p in PARAMS:
        x0 = df[f"{p}_0h"].to_numpy()
        x24 = df[f"{p}_24h"].to_numpy()
        cols.extend([x0, x24, (x24 - x0) / np.maximum(np.abs(x0), 1e-9)])
    return np.column_stack(cols)


def fit_isolation(train: pd.DataFrame, cfg: dict, seed: int) -> IsolationForest:
    return IsolationForest(
        n_estimators=int(cfg["module_a"]["isolation_forest_estimators"]),
        max_samples=min(int(cfg["module_a"]["isolation_forest_max_samples"]), len(train)),
        contamination="auto", random_state=seed, n_jobs=int(cfg["runtime"]["cpu_threads"]),
    ).fit(isolation_features(train))


def isolation_score(model: IsolationForest, df: pd.DataFrame) -> np.ndarray:
    return -model.score_samples(isolation_features(df))


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y.astype(int), score))


def threshold_for_far(scores: np.ndarray, labels: np.ndarray, far_pct: float) -> float:
    healthy = np.asarray(scores)[np.asarray(labels) == 0]
    q = 1.0 - far_pct / 100.0
    try:
        return float(np.quantile(healthy, q, method="higher"))
    except TypeError:
        return float(np.quantile(healthy, q, interpolation="higher"))


def evaluate_flags(y: np.ndarray, flags: np.ndarray, defect_types: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int); flags = np.asarray(flags, dtype=bool)
    healthy = y == 0; defective = y == 1
    far = 100.0 * float(np.mean(flags[healthy])) if healthy.any() else np.nan
    recall = 100.0 * float(np.mean(flags[defective])) if defective.any() else np.nan
    by_type = {}
    for d in ["D1_high_in_limit", "D2_fast_drifter", "D3_late_accelerator", "D4_multivariate"]:
        m = np.asarray(defect_types) == d
        by_type[d] = 100.0 * float(np.mean(flags[m])) if m.any() else np.nan
    return {
        "achieved_far_pct": far,
        "recall_pct": recall,
        "false_negatives_per_100_defects": 100.0 - recall if np.isfinite(recall) else np.nan,
        "false_rejects_per_1000": 1000.0 * float(np.mean(flags[healthy])) if healthy.any() else np.nan,
        "recall_by_type": by_type,
        "review_queue_per_1000": 1000.0 * float(np.mean(flags)),
        "true_positives_per_1000": 1000.0 * float(np.sum(flags & defective) / len(y)),
    }
