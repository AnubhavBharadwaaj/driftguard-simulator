from __future__ import annotations

import numpy as np

from conformal import evaluate


def compare(test,p_without,p_with):
    y=test["is_defect"].to_numpy(); dt=test["defect_type"].to_numpy(); lots=test["lot_id"].to_numpy(); healthy=y==0
    temp=test["temperature_c"].to_numpy(); lo=np.quantile(temp[healthy],.25); hi=np.quantile(temp[healthy],.75)
    def one(p):
        flags=p<=.005; ev=evaluate(y,flags,lots,dt)
        hot=100*np.mean(flags[healthy&(temp>=hi)]); cold=100*np.mean(flags[healthy&(temp<=lo)])
        d2=dt=="D2_fast_drifter"; return hot,cold,ev["recall_pct"],100*np.mean(flags[d2]) if d2.any() else np.nan
    a=one(p_without); b=one(p_with)
    return {"hot_vs_cold_far_pct":{"without":{"hot":a[0],"cold":a[1]},"with":{"hot":b[0],"cold":b[1]}},
            "recall_pct_at_0p5_far":{"without":a[2],"with":b[2]},"recall_D2_pct":{"without":a[3],"with":b[3]}}
