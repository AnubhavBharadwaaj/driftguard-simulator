from __future__ import annotations

import math
import numpy as np
import pandas as pd

from baselines import robust_center_scale
from simulate_v2 import PARAMS


class EBPowerLaw:
    def __init__(self,cfg): self.cfg=cfg; self.n={}; self.weight={}; self.resid_sd={}
    def fit(self,train:pd.DataFrame):
        grid=[float(x) for x in self.cfg["forecaster"]["power_n_grid"]]
        for p in PARAMS:
            r=np.log(np.maximum(train[f"{p}_24h"],1e-9)/np.maximum(train[f"{p}_0h"],1e-9)).to_numpy()
            lotmed=train.assign(_r=r).groupby("lot_id")["_r"].transform("median").to_numpy()
            between=np.var(pd.Series(lotmed).groupby(train["lot_id"].to_numpy()).first().to_numpy()); within=np.var(r-lotmed)
            self.weight[p]=float(between/(between+within+1e-12))
            shr=self.weight[p]*r+(1-self.weight[p])*lotmed; y=np.log(np.maximum(train[f"{p}_168h"],1e-9)/np.maximum(train[f"{p}_0h"],1e-9)).to_numpy()
            self.n[p]=min(grid,key=lambda n:np.mean(np.abs(y-shr/((24/168)**n))))
            pred=shr/((24/168)**self.n[p]); self.resid_sd[p]=float(np.std(y-pred,ddof=1))
        return self
    def predict(self,df:pd.DataFrame,time_h:float=24.0)->dict:
        out={}; sd={}
        for p in PARAMS:
            rt=np.log(np.maximum(df[f"{p}_{int(time_h)}h"],1e-9)/np.maximum(df[f"{p}_0h"],1e-9)).to_numpy()
            lotmed=df.assign(_r=rt).groupby("lot_id")["_r"].transform("median").to_numpy()
            shr=self.weight[p]*rt+(1-self.weight[p])*lotmed
            logratio=shr/((time_h/168.0)**self.n[p]); x0=df[f"{p}_0h"].to_numpy(float)
            out[p]=x0*np.exp(logratio); sd[p]=np.full(len(df),self.resid_sd[p])
        return {"mean":out,"log_sd":sd}


def conformal_widths(cal:pd.DataFrame,pred:dict,nominal=90.0)->dict:
    q={}
    for p in PARAMS:
        r=np.abs(np.log(np.maximum(cal[f"{p}_168h"],1e-9))-np.log(np.maximum(pred["mean"][p],1e-9)))
        rank=min(len(r)-1,int(math.ceil((len(r)+1)*nominal/100))-1); q[p]=float(np.sort(r)[rank])
    return q


def interval(pred:dict,q:dict,p:str)->tuple[np.ndarray,np.ndarray]:
    c=np.maximum(pred["mean"][p],1e-9); return c*np.exp(-q[p]),c*np.exp(q[p])
