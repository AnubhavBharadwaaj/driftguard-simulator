from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

from baselines import robust_center_scale
from simulate import PARAMS


def _family_stats(train: pd.DataFrame, floor: float) -> dict:
    stats = {}
    for fam, g in train.groupby("device_family"):
        stats[int(fam)] = {}
        for p in PARAMS:
            x0 = g[f"{p}_0h"].to_numpy()
            x24 = g[f"{p}_24h"].to_numpy()
            slope = (x24 - x0) / 24.0
            stats[int(fam)][p] = {
                "x0": robust_center_scale(x0, floor),
                "x24": robust_center_scale(x24, floor),
                "slope": robust_center_scale(slope, floor),
            }
    return stats


def _lot_normalized(df: pd.DataFrame, family_stats: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    floor = float(cfg["module_a"]["robust_sigma_floor"])
    target_n = float(cfg["module_a"]["shrinkage_target_n"])
    zlevel = np.zeros((len(df), 3)); zslope = np.zeros((len(df), 3))
    detail = {}
    for lot, idx0 in df.groupby("lot_id").groups.items():
        idx = np.asarray(list(idx0), dtype=int)
        fam = int(df.loc[idx[0], "device_family"])
        weight = min(1.0, len(idx) / target_n)
        detail[lot] = {}
        for pi, p in enumerate(PARAMS):
            x0 = df.loc[idx, f"{p}_0h"].to_numpy(dtype=float)
            x24 = df.loc[idx, f"{p}_24h"].to_numpy(dtype=float)
            slope = (x24 - x0) / 24.0
            lot_med24, lot_sig24 = robust_center_scale(x24, floor)
            lot_meds, lot_sigs = robust_center_scale(slope, floor)
            fam_med24, fam_sig24 = family_stats[fam][p]["x24"]
            fam_meds, fam_sigs = family_stats[fam][p]["slope"]
            med24 = weight * lot_med24 + (1.0 - weight) * fam_med24
            sig24 = max(floor, weight * lot_sig24 + (1.0 - weight) * fam_sig24)
            meds = weight * lot_meds + (1.0 - weight) * fam_meds
            sigs = max(floor, weight * lot_sigs + (1.0 - weight) * fam_sigs)
            zlevel[idx, pi] = (x24 - med24) / sig24
            zslope[idx, pi] = (slope - meds) / sigs
            detail[lot][p] = {"median24": med24, "sigma24": sig24, "median_slope": meds, "sigma_slope": sigs}
    return zlevel, zslope, detail


class DriftGuardA:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.family_stats = None
        self.mcd = None

    def fit(self, train: pd.DataFrame) -> "DriftGuardA":
        floor = float(self.cfg["module_a"]["robust_sigma_floor"])
        self.family_stats = _family_stats(train, floor)
        zlevel, _, _ = _lot_normalized(train, self.family_stats, self.cfg)
        clean = np.all(np.isfinite(zlevel), axis=1)
        try:
            self.mcd = MinCovDet(
                support_fraction=float(self.cfg["module_a"]["mincov_support_fraction"]),
                random_state=0,
            ).fit(zlevel[clean])
        except Exception:
            self.mcd = None
        return self

    def score(self, df: pd.DataFrame, return_detail: bool = False):
        zlevel, zslope, detail = _lot_normalized(df, self.family_stats, self.cfg)
        components = [np.max(np.abs(zlevel), axis=1), np.max(np.abs(zslope), axis=1)]
        if self.mcd is not None:
            md = np.sqrt(np.maximum(0.0, self.mcd.mahalanobis(zlevel)))
        else:
            md = np.sqrt(np.sum(zlevel ** 2, axis=1))
        components.append(md)
        score = np.max(np.column_stack(components), axis=1)
        if return_detail:
            return score, {"zlevel": zlevel, "zslope": zslope, "mahalanobis": md, "lot_stats": detail}
        return score
