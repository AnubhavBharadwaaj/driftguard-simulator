from __future__ import annotations

import numpy as np
import pandas as pd

from conformal import pvalues_from_calibration
from simulate_v2 import PARAMS


def transform_spikes(base:pd.DataFrame,dtype:str,rng:np.random.Generator,cfg:dict,strong=False)->pd.DataFrame:
    out=base.copy().reset_index(drop=True); limits={p:float(cfg["simulation"]["parameters"][p]["limit"]) for p in PARAMS}
    for i in range(len(out)):
        p=str(rng.choice(PARAMS)); med=float(base[f"{p}_24h"].median()); amp=float(rng.uniform(.65,1.0) if strong else rng.uniform(.30,1.0))
        if dtype=="D1_high_in_limit":
            val=min(.97*limits[p],med*(3.0+1.8*amp)); out.loc[i,f"{p}_0h"]=val*.99; out.loc[i,f"{p}_24h"]=val
        elif dtype=="D2_fast_drifter": out.loc[i,f"{p}_24h"]*=1.0+(.12+.35*amp)
        elif dtype=="D4_multivariate":
            for j,q in enumerate(PARAMS): out.loc[i,f"{q}_24h"]*=np.exp((1 if j%2==0 else -1)*(.16+.16*amp))
        out.loc[i,"is_defect"]=1; out.loc[i,"defect_type"]=dtype; out.loc[i,"defect_parameter"]=p; out.loc[i,"part_id"]=str(out.loc[i,"part_id"])+"_SPIKE"
    return out


def run_spikes(test,get_p,cfg,seed,strong=False):
    rng=np.random.default_rng(seed+8800+(100 if strong else 0)); rows=[]; allp={d:[] for d in ["D1_high_in_limit","D2_fast_drifter","D4_multivariate"]}
    for lot,g0 in test.groupby("lot_id"):
        g=g0.copy().reset_index(drop=True); true_p=get_p(g); true_dt=g["defect_type"].to_numpy()
        for d in allp:
            collected=[]; remaining=int(cfg["spikein"]["spikes_per_type_per_lot"])
            while remaining:
                n=min(remaining,max(1,int(len(g)*float(cfg["spikein"]["max_fraction_per_pass"]))))
                base=g.sample(n=n,replace=True,random_state=int(rng.integers(0,2**31-1)))
                sp=transform_spikes(base,d,rng,cfg,strong); aug=pd.concat([g,sp],ignore_index=True); p=get_p(aug)[len(g):]; collected.extend(p.tolist()); remaining-=n
            sp=np.asarray(collected); allp[d].extend(sp.tolist()); tm=true_dt==d
            rows.append({"lot_id":str(lot),"defect_type":d,"spike_recall_pct":100*np.mean(sp<=.005),"true_recall_pct":100*np.mean(true_p[tm]<=.005) if tm.any() else np.nan,"strong":strong})
    return pd.DataFrame(rows),allp


def summarize_pairs(pairs:pd.DataFrame):
    out={}
    for d,g in pairs.groupby("defect_type"):
        valid=g.dropna(subset=["true_recall_pct"]); corr=float(valid[["spike_recall_pct","true_recall_pct"]].corr().iloc[0,1]) if len(valid)>=3 else np.nan
        out[d]={"spike_recall_pct":float(g["spike_recall_pct"].mean()),"true_recall_pct":float(valid["true_recall_pct"].mean()) if len(valid) else np.nan,
                "mean_abs_gap_pct":float(np.mean(np.abs(valid["spike_recall_pct"]-valid["true_recall_pct"]))) if len(valid) else np.nan,"lot_correlation":corr}
    return out


def run_d3_spikes_96(cal:pd.DataFrame,test:pd.DataFrame,score_fn,cfg:dict,seed:int,strong:bool=False):
    """Inject late acceleration only at 96 h and compare spike with true D3 recall."""
    rng=np.random.default_rng(seed+9900+(100 if strong else 0)); healthy=cal["is_defect"].to_numpy()==0
    cal_score=score_fn(cal); rows=[]; pooled=[]
    for lot,g0 in test.groupby("lot_id"):
        g=g0.copy().reset_index(drop=True); base_p=pvalues_from_calibration(cal_score[healthy],score_fn(g)); collected=[]
        remaining=int(cfg["spikein"]["spikes_per_type_per_lot"])
        while remaining:
            n=min(remaining,max(1,int(len(g)*float(cfg["spikein"]["max_fraction_per_pass"]))))
            sp=g.sample(n=n,replace=True,random_state=int(rng.integers(0,2**31-1))).copy().reset_index(drop=True)
            for i in range(len(sp)):
                p=str(rng.choice(PARAMS)); amp=float(rng.uniform(.65,1.0) if strong else rng.uniform(.30,1.0))
                sp.loc[i,f"{p}_96h"]*=1.0+(.10+.30*amp)
                sp.loc[i,"is_defect"]=1; sp.loc[i,"defect_type"]="D3_late_accelerator"; sp.loc[i,"defect_parameter"]=p
                sp.loc[i,"part_id"]=str(sp.loc[i,"part_id"])+"_D3SPIKE96"
            aug=pd.concat([g,sp],ignore_index=True)
            ps=pvalues_from_calibration(cal_score[healthy],score_fn(aug)[len(g):]); collected.extend(ps.tolist()); remaining-=n
        values=np.asarray(collected); pooled.extend(values.tolist()); tm=g["defect_type"].to_numpy()=="D3_late_accelerator"
        rows.append({"lot_id":str(lot),"defect_type":"D3_late_accelerator","spike_recall_pct":100*np.mean(values<=.005),"true_recall_pct":100*np.mean(base_p[tm]<=.005) if tm.any() else np.nan,"strong":strong})
    pairs=pd.DataFrame(rows); valid=pairs.dropna(subset=["true_recall_pct"])
    corr=float(valid[["spike_recall_pct","true_recall_pct"]].corr().iloc[0,1]) if len(valid)>=3 else np.nan
    summary={"spike_recall_pct":float(pairs["spike_recall_pct"].mean()),"true_recall_pct":float(valid["true_recall_pct"].mean()) if len(valid) else np.nan,"mean_abs_gap_pct":float(np.mean(np.abs(valid["spike_recall_pct"]-valid["true_recall_pct"]))) if len(valid) else np.nan,"lot_correlation":corr}
    return pairs,np.asarray(pooled),summary
