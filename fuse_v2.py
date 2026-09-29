from __future__ import annotations

import numpy as np

from conformal import evaluate
from simulate_v2 import PARAMS


def slope_score(df,pred,limits,cfg):
    allscores=[]
    for p in PARAMS:
        x0=df[f"{p}_0h"].to_numpy(float); ps=(pred["mean"][p]-x0)/168.0; safe=(limits[p]-x0-float(cfg["fusion"]["guard_band_fraction"])*limits[p])/168.0
        allscores.append(ps-safe)
    return np.max(np.column_stack(allscores),axis=1)


def interval_score(pred,q,limits):
    vals=[]
    for p in PARAMS: vals.append(pred["mean"][p]*np.exp(q[p])/limits[p])
    return np.max(np.column_stack(vals),axis=1)


def choose_split(cal,ycal,p_a,p_slope,p_interval,alpha,cfg):
    best=None
    for split in cfg["detectors"]["fusion_splits"]:
        flags=(p_a<=alpha*split["module_a"])|(p_slope<=alpha*split["slope"])|(p_interval<=alpha*split["interval"])
        healthy=ycal==0; far=np.mean(flags[healthy]); rec=np.mean(flags[ycal==1]) if np.any(ycal==1) else 0
        candidate=(far<=alpha,rec,-far,split)
        if best is None or candidate[:3]>best[:3]: best=candidate
    return best[3]


def apply_budget(p_a,p_slope,p_interval,alpha,split):
    return (p_a<=alpha*split["module_a"])|(p_slope<=alpha*split["slope"])|(p_interval<=alpha*split["interval"])
