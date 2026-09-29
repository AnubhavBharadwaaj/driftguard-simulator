from __future__ import annotations

import numpy as np

from conformal import pvalues_from_calibration
from simulate_v2 import PARAMS


def readout_score(df,time_h:float)->np.ndarray:
    scores=np.zeros(len(df))
    for lot,idx0 in df.groupby("lot_id").groups.items():
        idx=np.asarray(list(idx0),int); local=[]
        for p in PARAMS:
            r=np.log(np.maximum(df.loc[idx,f"{p}_{int(time_h)}h"],1e-9)/np.maximum(df.loc[idx,f"{p}_0h"],1e-9))
            med=np.median(r); mad=max(1e-8,1.4826*np.median(np.abs(r-med))); local.append(np.abs((r-med)/mad))
        scores[idx]=np.max(np.column_stack(local),axis=1)
    return scores


def adaptive_curve(cal,test,p24_cal,p24_test,eb,limits,cfg,policy:str):
    healthy=cal["is_defect"].to_numpy()==0; alpha=float(cfg["adaptive"]["target_far_pct"])/100
    cal_scores={t:readout_score(cal,t) for t in [48,72,96]}; test_scores={t:readout_score(test,t) for t in [48,72,96]}
    cal_p={t:pvalues_from_calibration(cal_scores[t][healthy],cal_scores[t]) for t in cal_scores}
    test_p={t:pvalues_from_calibration(cal_scores[t][healthy],test_scores[t]) for t in test_scores}
    y=test["is_defect"].to_numpy(); dtype=test["defect_type"].to_numpy(); curve=[]; detection10=None
    pred=eb.predict(test); risk=np.max(np.column_stack([pred["mean"][p]/limits[p] for p in PARAMS]),axis=1)
    for budget in cfg["adaptive"]["budgets_pct"]:
        n=int(round(len(test)*float(budget)/100)); base=p24_test.copy(); eligible=np.flatnonzero(base>alpha)
        if policy=="gray_zone": rank=eligible[np.argsort(base[eligible])]
        else: rank=eligible[np.argsort(np.abs(risk[eligible]-1.0))]
        selected48=rank[:n]; combined=base.copy(); combined[selected48]=np.minimum(combined[selected48],test_p[48][selected48])
        still=selected48[combined[selected48]>alpha]; n72=min(len(still),n//2); selected72=still[np.argsort(combined[still])[:n72]] if n72 else np.array([],int)
        combined[selected72]=np.minimum(combined[selected72],test_p[72][selected72])
        flags=combined<=alpha; defect=y==1
        recall=100*np.mean(flags[defect]) if defect.any() else np.nan
        def rec_type(d):
            m=dtype==d; return 100*np.mean(flags[m]) if m.any() else np.nan
        detect_time=np.full(len(test),168,int); detect_time[p24_test<=alpha]=24
        m=(detect_time==168)&np.isin(np.arange(len(test)),selected48)&(test_p[48]<=alpha); detect_time[m]=48
        m=(detect_time==168)&np.isin(np.arange(len(test)),selected72)&(test_p[72]<=alpha); detect_time[m]=72
        m=(detect_time==168)&(test_p[96]<=alpha); detect_time[m]=96
        detect_time[(detect_time==168)&(~flags)]=-1
        hours=np.sum(np.maximum(0,168-detect_time[(defect)&(detect_time>0)]))/len(test)*1000
        curve.append({"extra_readout_pct":float(budget),"recall_pct":float(recall),"recall_D2_pct":float(rec_type("D2_fast_drifter")),"recall_D3_pct":float(rec_type("D3_late_accelerator")),"oven_hours_freed_per_1000":float(hours)})
        if float(budget)==10:
            shares={str(t):100*np.mean(detect_time[defect]==t) for t in [24,48,72,96,168]}; shares["never"]=100*np.mean(detect_time[defect]==-1); detection10=shares
    return curve,detection10
