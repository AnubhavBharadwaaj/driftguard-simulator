from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "v5"
sys.path[:0] = [str(ROOT), str(ROOT / "v3"), str(ROOT / "v4"), str(ROOT / "v5")]

from conformal import evaluate
from simulate_v2 import PARAMS
from v3.run_v3 import early_features, lot_mad_score, mergers
from v4.rival_benchmark import aabhas_frames, burntrace_frames, at_hour
from run_v5 import cp, final_cross_p, source_p_cross


def threshold(cal_score, healthy, alpha):
    return float(np.quantile(np.asarray(cal_score)[healthy], 1 - alpha, method="higher"))


def row(dataset, method, hour, cal, test, cscore, tscore, alpha, fit_note):
    healthy = cal.is_defect.to_numpy() == 0
    ncal = int(healthy.sum()); min_p = 1 / (ncal + 1); infeasible = min_p > alpha
    cut = threshold(cscore, healthy, alpha)
    flags = np.asarray(tscore) >= cut
    y = test.is_defect.to_numpy() == 1
    fp = int(np.sum(flags & ~y)); tp = int(np.sum(flags & y))
    return {"dataset": dataset, "detector": method, "decision_hour": hour,
            "target_far_pct": 100 * alpha, "achieved_far_pct": None if infeasible else 100 * fp / max(1, (~y).sum()),
            "recall_pct": None if infeasible else 100 * tp / max(1, y.sum()), "diagnostic_recall_pct": 100 * tp / max(1, y.sum()),
            "n_pos": int(y.sum()), "n_good": int((~y).sum()),
            "n_healthy_calibration": ncal, "min_attainable_p": min_p, "infeasible_at_alpha": infeasible,
            "far_cp95": cp(fp, int((~y).sum())), "recall_cp95": cp(tp, int(y.sum())),
            "calibration_rule": "same healthy calibration parts; empirical upper 1-alpha quantile", "fit_note": fit_note}


def cheap_baselines(dataset, train, cal, test, hour):
    xtr, xcal, xtest = early_features(train), early_features(cal), early_features(test)
    good_train = train.is_defect.to_numpy() == 0
    good_cal = cal.is_defect.to_numpy() == 0
    methods = {}
    methods["R1_lot_mad"] = (lot_mad_score(cal, xcal), lot_mad_score(test, xtest), "no supervised fit")

    def iforest(frame, x):
        out = np.zeros(len(frame))
        for _, idx0 in frame.groupby("lot_id").groups.items():
            idx = np.asarray(list(idx0), int)
            model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1).fit(x[idx])
            out[idx] = -model.decision_function(x[idx])
        return out
    methods["R2_isolation_forest"] = (iforest(cal, xcal), iforest(test, xtest), "fit independently inside each calibration or test lot, matching public stack")

    scaler = StandardScaler().fit(xtr[good_train]); ztr = scaler.transform(xtr)
    pca = PCA(n_components=0.95, random_state=26170).fit(ztr[good_train])
    nn = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(ztr[good_train])
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(ztr[good_train])
    # Deterministic cap is disclosed; robust covariance on 70k rows is needlessly cubic.
    rng = np.random.default_rng(26170 + hour)
    mcd_idx = np.flatnonzero(good_train)
    if len(mcd_idx) > 12000:
        mcd_idx = rng.choice(mcd_idx, 12000, replace=False)
    mcd = MinCovDet(random_state=26170).fit(ztr[mcd_idx])
    def ensemble(frame, x):
        z = scaler.transform(x)
        parts = [np.sum((z - pca.inverse_transform(pca.transform(z))) ** 2, axis=1),
                 nn.kneighbors(z)[0][:, -1], -lof.decision_function(z), mcd.mahalanobis(z), lot_mad_score(frame, x)]
        return np.max(np.column_stack([pd.Series(a).rank(pct=True).to_numpy() for a in parts]), axis=1)
    methods["R4_unsupervised_ensemble"] = (ensemble(cal, xcal), ensemble(test, xtest), "full training split except disclosed 12,000 healthy row MCD numerical cap")

    ridge = {p: Ridge(alpha=1).fit(xtr, train[f"{p}_168h"]) for p in PARAMS}
    def slope(frame, x):
        pred = np.column_stack([ridge[p].predict(x) for p in PARAMS])
        current = frame[[f"{p}_24h" for p in PARAMS]].to_numpy(float)
        return lot_mad_score(frame, (pred - current) / 144.0)
    methods["R5_ridge_forecast_slope"] = (slope(cal, xcal), slope(test, xtest), "full training split")
    return [row(dataset, name, hour, cal, test, cs, ts, 0.005, note) for name, (cs, ts, note) in methods.items()]


def evaluate_dataset(dataset, loader, commit, published, cfg):
    train, cal, test, meta = loader()
    for column in [f"{p}_{h}h" for p in PARAMS for h in (0, 24, 168)]:
        fill = float(np.nanmedian(train[column]))
        for frame in (train, cal, test): frame[column] = frame[column].fillna(fill)
    rows = []
    for hour in (24, 168):
        tr, ca, te = at_hour(train, hour), at_hour(cal, hour), at_hour(test, hour)
        src = source_p_cross(ca, {"test": te}, cfg, 24)
        cm = mergers(src["calibration"])["twice_mean"]; tm = mergers(src["test"])["twice_mean"]
        _, p, res = final_cross_p(ca, cm, tm)
        for alpha in ((0.005, 0.0054) if dataset == "Aabhas" and hour == 24 else (0.005,)):
            flags = p <= alpha; y = te.is_defect.to_numpy() == 1
            fp = int(np.sum(flags & ~y)); tp = int(np.sum(flags & y))
            infeasible = res["min_attainable_p"] > alpha
            rows.append({"dataset": dataset, "detector": "Sahodrift_v5", "decision_hour": hour,
                         "target_far_pct": 100 * alpha, "achieved_far_pct": None if infeasible else 100 * fp / max(1, (~y).sum()),
                         "recall_pct": None if infeasible else 100 * tp / max(1, y.sum()),
                         "diagnostic_recall_pct": 100 * tp / max(1, y.sum()), "n_pos": int(y.sum()), "n_good": int((~y).sum()),
                         "far_cp95": cp(fp, int((~y).sum())), "recall_cp95": cp(tp, int(y.sum())), **res,
                         "infeasible_at_alpha": infeasible,
                         "calibration_rule": "v5 lot-cross conformal using every healthy calibration part"})
        rows.extend(cheap_baselines(dataset, tr, ca, te, hour))
    return {"dataset": dataset, "commit": commit, "split": meta, "rows": rows, "published_points": published}


def main():
    start = time.time(); cfg = yaml.safe_load((ROOT / "config_v2.yaml").read_text())
    results = [
        evaluate_dataset("Aabhas", aabhas_frames, "3d3b1e152f65abd2ad2f90fdc6f755b83f2de73c",
                         [{"hour": 24, "recall_pct": 29.7, "far_pct": 0.54, "split": "their published split"}], cfg),
        evaluate_dataset("BurnTrace", burntrace_frames, "a8796e197f242d87768272db8b055dda21120c69",
                         [{"hour": 24, "recall_pct": 21.1, "far_pct": 2.55, "split": "their published split"},
                          {"hour": 168, "recall_pct": 72.2, "far_pct": 1.04, "split": "their published split"}], cfg),
    ]
    payload = {"benchmarks": results, "R3_status": "not run: the full 500-tree supervised fallback exceeded the v5 30-minute cap in earlier profiling; no approximation substituted", "runtime_seconds": time.time() - start}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rival_benchmark_v5.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"runtime_seconds": time.time() - start, "datasets": {x["dataset"]: len(x["rows"]) for x in results}}, indent=2))


if __name__ == "__main__": main()
