from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score


def pvalues_from_calibration(cal_healthy_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(cal_healthy_scores, dtype=float))
    s = np.asarray(scores, dtype=float)
    ge = len(ref) - np.searchsorted(ref, s, side="left")
    return (1.0 + ge) / (len(ref) + 1.0)


def bonferroni_minp(p_components: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, p_components.shape[1] * np.min(p_components, axis=1))


def fisher_p(p_components: np.ndarray) -> np.ndarray:
    p = np.clip(p_components, 1e-12, 1.0); stat = -2.0 * np.sum(np.log(p), axis=1)
    return chi2.sf(stat, 2 * p.shape[1])


def bh_flags(pvals: np.ndarray, lots: np.ndarray, q: float) -> np.ndarray:
    flags = np.zeros(len(pvals), dtype=bool)
    for lot in np.unique(lots):
        idx = np.flatnonzero(lots == lot); p = pvals[idx]; order = np.argsort(p); ranked = p[order]
        eligible = np.flatnonzero(ranked <= q * (np.arange(1, len(ranked) + 1) / len(ranked)))
        if len(eligible): flags[idx[order[:eligible[-1] + 1]]] = True
    return flags


def evaluate(y: np.ndarray, flags: np.ndarray, lots: np.ndarray, defect_types: np.ndarray) -> dict:
    y=np.asarray(y,int); flags=np.asarray(flags,bool); healthy=y==0; defect=y==1
    far=100*np.mean(flags[healthy]) if healthy.any() else np.nan; recall=100*np.mean(flags[defect]) if defect.any() else np.nan
    per=[]
    for lot in np.unique(lots):
        m=(lots==lot)&healthy
        if m.any(): per.append(100*np.mean(flags[m]))
    by={}
    for d in ["D1_high_in_limit","D2_fast_drifter","D3_late_accelerator","D4_multivariate"]:
        m=defect_types==d; by[d]=100*np.mean(flags[m]) if m.any() else np.nan
    discoveries=int(flags.sum()); false=int(np.sum(flags&healthy))
    return {"achieved_far_pct":float(far),"recall_pct":float(recall),"false_rejects_per_1000":float(1000*np.mean(flags[healthy])) if healthy.any() else np.nan,
            "review_queue_per_1000":float(1000*np.mean(flags)),"fdp_pct":float(100*false/discoveries) if discoveries else 0.0,
            "per_lot_far_pct":{"min":float(np.min(per)) if per else np.nan,"median":float(np.median(per)) if per else np.nan,"max":float(np.max(per)) if per else np.nan},
            "recall_by_type":by}


def detector_report(name: str, test_set: str, y: np.ndarray, anomaly_score: np.ndarray, pvals: np.ndarray, lots: np.ndarray, defect_types: np.ndarray, cfg: dict) -> dict:
    ops=[]
    for far in cfg["detectors"]["operating_far_pct"]:
        ev=evaluate(y,pvals<=float(far)/100.0,lots,defect_types)
        ops.append({"target_far_pct":float(far),"achieved_far_pct":ev["achieved_far_pct"],"recall_pct":ev["recall_pct"],"false_rejects_per_1000":ev["false_rejects_per_1000"]})
    bh=[]
    for q in cfg["detectors"]["bh_q"]:
        ev=evaluate(y,bh_flags(pvals,lots,float(q)),lots,defect_types)
        bh.append({"q":float(q),"fdp_pct":ev["fdp_pct"],"recall_pct":ev["recall_pct"]})
    ev05=evaluate(y,pvals<=0.005,lots,defect_types)
    return {"name":name,"test_set":test_set,"pr_auc":float(average_precision_score(y,anomaly_score)),"operating_points":ops,
            "recall_pct_by_defect_type_at_0p5_far":ev05["recall_by_type"],"bh":bh,"per_lot_far_pct":ev05["per_lot_far_pct"],
            "_pvals":pvals,"_score":anomaly_score}
