from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.linear_model import HuberRegressor

from baselines import robust_center_scale
from conformal import bonferroni_minp, fisher_p, pvalues_from_calibration
from simulate_v2 import PARAMS


def component_scores(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, list[str], dict]:
    comps=np.zeros((len(df),7)); detail={}
    for lot, idx0 in df.groupby("lot_id").groups.items():
        idx=np.asarray(list(idx0),int); detail[str(lot)]={}
        zlev=[]; zsl=[]
        for p in PARAMS:
            x0=df.loc[idx,f"{p}_0h"].to_numpy(float); x24=df.loc[idx,f"{p}_24h"].to_numpy(float)
            m,s=robust_center_scale(x24); sm,ss=robust_center_scale((x24-x0)/24.0)
            zlev.append(np.abs((x24-m)/s)); zsl.append(np.abs(((x24-x0)/24.0-sm)/ss)); detail[str(lot)][p]={"median24":m,"sigma24":s,"slope_median":sm,"slope_sigma":ss}
        zl=np.column_stack(zlev); zs=np.column_stack(zsl)
        try: md=np.sqrt(np.maximum(0,MinCovDet(support_fraction=float(cfg["detectors"]["mincov_support_fraction"]),random_state=0).fit(zl).mahalanobis(zl)))
        except Exception: md=np.sqrt(np.sum(zl**2,axis=1))
        comps[idx,:3]=zl; comps[idx,3:6]=zs; comps[idx,6]=md
    return comps,["level_leakage","level_iddq","level_delay","slope_leakage","slope_iddq","slope_delay","mahalanobis"],detail


def calibrate_component_pvalues(cal: pd.DataFrame, cal_components: np.ndarray, test_components: np.ndarray) -> np.ndarray:
    healthy=cal["is_defect"].to_numpy()==0
    return np.column_stack([pvalues_from_calibration(cal_components[healthy,j],test_components[:,j]) for j in range(test_components.shape[1])])


def combined_pvalues(cal: pd.DataFrame, cal_components: np.ndarray, test_components: np.ndarray) -> dict:
    """Combine components and conformalise the aggregate statistic.

    The component p-values are dependent, so Fisher's chi-square reference is
    not a calibrated final p-value.  A second healthy-calibration conformal
    layer restores control of the overall false-alarm budget.
    """
    healthy=cal["is_defect"].to_numpy()==0
    pcal=calibrate_component_pvalues(cal,cal_components,cal_components)
    ptest=calibrate_component_pvalues(cal,cal_components,test_components)
    raw_cal_min=bonferroni_minp(pcal); raw_test_min=bonferroni_minp(ptest)
    raw_cal_fisher=fisher_p(pcal[:,[0,1,2,6]]); raw_test_fisher=fisher_p(ptest[:,[0,1,2,6]])
    minp=pvalues_from_calibration(-np.log(np.clip(raw_cal_min[healthy],1e-300,1.0)),-np.log(np.clip(raw_test_min,1e-300,1.0)))
    fisher=pvalues_from_calibration(-np.log(np.clip(raw_cal_fisher[healthy],1e-300,1.0)),-np.log(np.clip(raw_test_fisher,1e-300,1.0)))
    return {"minp":minp,"fisher":fisher,"components":ptest,"raw_minp":raw_test_min,"raw_fisher":raw_test_fisher}


def conditional_residual_scores(df: pd.DataFrame) -> np.ndarray:
    scores=np.zeros((len(df),3))
    logs=np.log(np.maximum(df[[f"{p}_24h" for p in PARAMS]].to_numpy(float),1e-9))
    for _,idx0 in df.groupby("lot_id").groups.items():
        idx=np.asarray(list(idx0),int); local=logs[idx]
        for j in range(3):
            other=[k for k in range(3) if k!=j]
            try:
                model=HuberRegressor(epsilon=1.35,alpha=0.0,max_iter=200).fit(local[:,other],local[:,j]); resid=local[:,j]-model.predict(local[:,other])
            except Exception:
                resid=local[:,j]-np.median(local[:,j])
            m,s=robust_center_scale(resid); scores[idx,j]=np.abs((resid-m)/s)
    return np.max(scores,axis=1)


def deconfounded_copy(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    for lot,idx0 in out.groupby("lot_id").groups.items():
        idx=np.asarray(list(idx0),int)
        for p in PARAMS:
            x0=np.maximum(out.loc[idx,f"{p}_0h"].to_numpy(float),1e-9); x24=np.maximum(out.loc[idx,f"{p}_24h"].to_numpy(float),1e-9)
            lr=np.log(x24/x0); testers=out.loc[idx,"tester_id"].to_numpy(); overall=np.median(lr)
            tester_adj=np.array([np.median(lr[testers==t])-overall if np.any(testers==t) else 0 for t in testers])
            X=np.column_stack([np.ones(len(idx)),out.loc[idx,"slot_row"].to_numpy()-1.5,out.loc[idx,"slot_col"].to_numpy()-1.5])
            target=lr-tester_adj
            for _ in range(4):
                beta=np.linalg.lstsq(X,target,rcond=None)[0]; resid=target-X@beta; _,sc=robust_center_scale(resid); w=np.minimum(1,1.5*sc/np.maximum(np.abs(resid),1e-9)); beta=np.linalg.lstsq(X*w[:,None],target*w,rcond=None)[0]
            corrected=lr-tester_adj-(X@beta-beta[0]); out.loc[idx,f"{p}_24h"]=x0*np.exp(corrected)
    return out
