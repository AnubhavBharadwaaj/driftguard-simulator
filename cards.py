from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from module_b import feature_frame
from simulate_v2 import PARAMS


def build_cards(test,train,pvals,component_p,eb_pred,q,limits,seed=0):
    rng=np.random.default_rng(seed); flags=pvals<=.005; dt=test["defect_type"].to_numpy(); y=test["is_defect"].to_numpy()
    specs=[("flagged_D1",(dt=="D1_high_in_limit")&flags,True),("flagged_D2",(dt=="D2_fast_drifter")&flags,True),("flagged_D4",(dt=="D4_multivariate")&flags,True),("false_alarm",(y==0)&flags,False),("missed_defect",(y==1)&(~flags),False)]
    Xtr=feature_frame(train).to_numpy(); Xte=feature_frame(test).to_numpy(); sc=StandardScaler().fit(Xtr); A=sc.transform(Xtr); B=sc.transform(Xte)
    cards=[]
    for label,mask,strongest in specs:
        idx=np.flatnonzero(mask)
        if not len(idx): cards.append({"card_type":label,"status":"unavailable"}); continue
        i=int(idx[np.argmin(pvals[idx])] if strongest else rng.choice(idx)); row=test.iloc[i]
        ratios={p:eb_pred["mean"][p][i]/limits[p] for p in PARAMS}; p=max(ratios,key=ratios.get); x0=float(row[f"{p}_0h"]); x24=float(row[f"{p}_24h"])
        lot=test[test.lot_id==row.lot_id]; med=float(lot[f"{p}_24h"].median()); forecast=float(eb_pred["mean"][p][i]); lo=forecast*np.exp(-q[p]); hi=forecast*np.exp(q[p])
        dist=np.sum((A-B[i])**2,axis=1); nn=np.argsort(dist)[:3]
        cards.append({"card_type":label,"status":"available","part_id":str(row.part_id),"lot_id":str(row.lot_id),"parameter":p,"x0":x0,"x24":x24,
            "lot_median_24h":med,"ratio_to_lot_median":x24/max(med,1e-9),"conformal_p_value":float(pvals[i]),"component_p_values":[float(x) for x in component_p[i]],
            "forecast_168":forecast,"interval_168":[float(lo),float(hi)],"actual_168":float(row[f"{p}_168h"]),"predicted_slope":(forecast-x0)/168,
            "safety_slope":(limits[p]-x0-.1*limits[p])/168,"reason_codes":["combined conformal evidence"] if flags[i] else ["below frozen evidence threshold"],
            "predicted_defect_type":None,"predicted_type_probability":None,"counterfactual_max_24h_not_flagged":None,
            "nearest_historical_parts":[{"part_id":str(train.iloc[j].part_id),"label":str(train.iloc[j].defect_type)} for j in nn],"true_label":str(row.defect_type)})
    return cards
