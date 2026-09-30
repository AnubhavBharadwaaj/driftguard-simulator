from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.stats import beta
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "v6"
sys.path[:0] = [str(ROOT), str(ROOT / "v3"), str(ROOT / "v4"), str(ROOT / "v5")]

from module_a_v2 import component_scores
from simulate_v2 import PARAMS, generate_evaluation, impute
from v3.run_v3 import early_features, lot_mad_score, mergers
from v4.rival_benchmark import aabhas_frames, at_hour, burntrace_frames
from v5.run_v5 import at_hour as sim_at_hour, final_cross_p, source_p_cross, upper_p

METHODS = ["Sahodrift_v5", "R1_lot_mad", "R2_isolation_forest", "R3_gradient_boosting_sklearn_fallback", "R4_unsupervised_ensemble", "R5_ridge_forecast_slope"]


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, float): return None if not math.isfinite(value) else value
    return value


def cp(k: int, n: int):
    if n == 0: return [None, None]
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n-k+1)
    hi = 1.0 if k == n else beta.ppf(0.975, k+1, n-k)
    return [100*float(lo), 100*float(hi)]


def load_seed(cfg, seed):
    sets = generate_evaluation(cfg, seed)
    _, cal, tin, tsh = impute(sets["train"], [sets["train"], sets["calibration"], sets["test_inprior"], sets["test_shifted"]], cfg)
    train = impute(sets["train"], [sets["train"]], cfg)[0]
    return train, cal, tin, tsh


def p_to_metric(seed, dataset, method, test_set, hour, alpha, frame, p, ncal):
    minp = 1/(ncal+1); infeasible = minp > alpha
    flags = np.asarray(p) <= alpha
    y = frame.is_defect.to_numpy() == 1
    fp = int(np.sum(flags & ~y)); tp = int(np.sum(flags & y)); ng = int((~y).sum()); np_ = int(y.sum())
    return {"seed": seed, "dataset": dataset, "detector": method, "test_set": test_set, "decision_hour": hour,
            "target_alpha": alpha, "target_far_pct": 100*alpha,
            "achieved_far_pct": None if infeasible else 100*fp/max(1,ng),
            "recall_pct": None if infeasible else 100*tp/max(1,np_),
            "diagnostic_far_pct": 100*fp/max(1,ng), "diagnostic_recall_pct": 100*tp/max(1,np_),
            "false_rejects": fp, "true_rejects": tp, "n_good": ng, "n_pos": np_,
            "far_cp95": cp(fp,ng), "recall_cp95": cp(tp,np_), "n_healthy_calibration": ncal,
            "min_attainable_p": minp, "infeasible_at_alpha": infeasible}


def rival_raw(train, cal, tests, seed, include_r3=True):
    xtr, xcal = early_features(train), early_features(cal)
    good_train = train.is_defect.to_numpy() == 0
    scaler = StandardScaler().fit(xtr[good_train]); ztr = scaler.transform(xtr)
    out = {}
    out["R1_lot_mad"] = (lot_mad_score(cal,xcal), {k:lot_mad_score(v,early_features(v)) for k,v in tests.items()})
    def iforest(frame):
        x=early_features(frame); s=np.zeros(len(frame))
        for _,ii in frame.groupby("lot_id").groups.items():
            idx=np.asarray(list(ii),int); model=IsolationForest(n_estimators=200,contamination=.05,random_state=42,n_jobs=-1).fit(x[idx]); s[idx]=-model.decision_function(x[idx])
        return s
    out["R2_isolation_forest"]=(iforest(cal),{k:iforest(v) for k,v in tests.items()})
    if include_r3:
        model3=GradientBoostingClassifier(n_estimators=500,max_depth=5,learning_rate=.03,random_state=seed).fit(xtr,train.is_defect)
        out["R3_gradient_boosting_sklearn_fallback"]=(model3.predict_proba(xcal)[:,1],{k:model3.predict_proba(early_features(v))[:,1] for k,v in tests.items()})
    pca=PCA(n_components=.95,random_state=seed).fit(ztr[good_train]); nn=NearestNeighbors(n_neighbors=10,n_jobs=-1).fit(ztr[good_train]); lof=LocalOutlierFactor(n_neighbors=20,novelty=True,n_jobs=-1).fit(ztr[good_train]); mcd=MinCovDet(random_state=seed).fit(ztr[good_train])
    def ensemble(frame):
        x=early_features(frame); z=scaler.transform(x)
        pieces=[np.sum((z-pca.inverse_transform(pca.transform(z)))**2,axis=1),nn.kneighbors(z)[0][:,-1],-lof.decision_function(z),mcd.mahalanobis(z),lot_mad_score(frame,x)]
        return np.max(np.column_stack([pd.Series(a).rank(pct=True).to_numpy() for a in pieces]),axis=1)
    out["R4_unsupervised_ensemble"]=(ensemble(cal),{k:ensemble(v) for k,v in tests.items()})
    ridge={p:Ridge(alpha=1).fit(xtr,train[f"{p}_168h"]) for p in PARAMS}
    def slope(frame):
        x=early_features(frame); pred=np.column_stack([ridge[p].predict(x) for p in PARAMS]); now=frame[[f"{p}_24h" for p in PARAMS]].to_numpy(float)
        return lot_mad_score(frame,(pred-now)/144.)
    out["R5_ridge_forecast_slope"]=(slope(cal),{k:slope(v) for k,v in tests.items()})
    return out


def sahodrift_p(cal, tests, cfg, hour, extra_mcd=False):
    src=source_p_cross(cal,tests,cfg,hour)
    if extra_mcd:
        raw={"calibration":within_lot_mcd(cal,cfg,hour)}|{k:within_lot_mcd(v,cfg,hour) for k,v in tests.items()}
        healthy=cal.is_defect.to_numpy()==0; lots=cal.lot_id.to_numpy(); pcal=np.ones(len(cal))
        for lot in np.unique(lots):
            idx=lots==lot; pcal[idx]=upper_p(raw["calibration"][healthy & ~idx],raw["calibration"][idx])
        src["calibration"]=np.column_stack([src["calibration"],pcal])
        for k in tests: src[k]=np.column_stack([src[k],upper_p(raw["calibration"][healthy],raw[k])])
    cm=mergers(src["calibration"])["twice_mean"]
    result={}
    for k in tests:
        _,result[k],res=final_cross_p(cal,cm,mergers(src[k])["twice_mean"])
    return result,res


def within_lot_mcd(frame,cfg,hour):
    local=sim_at_hour(frame,hour); comps,_,_=component_scores(local,cfg); out=np.zeros(len(frame))
    for _,ii in local.groupby("lot_id").groups.items():
        idx=np.asarray(list(ii),int); x=comps[idx,:6]
        try: out[idx]=np.sqrt(np.maximum(0,MinCovDet(support_fraction=.75,random_state=0).fit(x).mahalanobis(x)))
        except Exception: out[idx]=np.sqrt(np.sum(x*x,axis=1))
    return out


def aggregate(rows):
    df=pd.DataFrame([r for r in rows if not r["infeasible_at_alpha"]]); result=[]
    keys=["dataset","detector","test_set","decision_hour","target_alpha"]
    for key,local in df.groupby(keys):
        fp=int(local.false_rejects.sum()); ng=int(local.n_good.sum()); tp=int(local.true_rejects.sum()); np_=int(local.n_pos.sum())
        result.append(dict(zip(keys,key))|{"mean_achieved_far_pct":float(local.achieved_far_pct.mean()),"per_seed_far_pct":local[["seed","achieved_far_pct"]].to_dict("records"),"seeds_above_0_75_pct":int(np.sum(local.achieved_far_pct>.75)),"pooled_far_pct":100*fp/ng,"pooled_far_cp95":cp(fp,ng),"pooled_recall_pct":100*tp/np_,"pooled_recall_cp95":cp(tp,np_)})
    return result


def simulator_stack(cfg):
    rows=[]; candidate=[]
    for seed in range(5):
        train,cal,tin,tsh=load_seed(cfg,seed); ncal=int((cal.is_defect==0).sum())
        for hour in (24,168):
            tr,ca,tests=sim_at_hour(train,hour),sim_at_hour(cal,hour),{"inprior":sim_at_hour(tin,hour),"shifted":sim_at_hour(tsh,hour)}
            sp,_=sahodrift_p(cal,{"inprior":tin,"shifted":tsh},cfg,hour)
            raw=rival_raw(tr,ca,tests,seed+hour)
            for set_name,frame in (("inprior",tin),("shifted",tsh)):
                for alpha in (.005,.0025): rows.append(p_to_metric(seed,"simulator","Sahodrift_v5",set_name,hour,alpha,frame,sp[set_name],ncal))
            for method,(cs,ts) in raw.items():
                good=ca.is_defect.to_numpy()==0
                for set_name,frame in (("inprior",tin),("shifted",tsh)):
                    p=upper_p(cs[good],ts[set_name])
                    for alpha in (.005,.0025): rows.append(p_to_metric(seed,"simulator",method,set_name,hour,alpha,frame,p,ncal))
        cpv,_=sahodrift_p(cal,{"inprior":tin},cfg,24,extra_mcd=True)
        candidate.append(p_to_metric(seed,"simulator","Sahodrift_plus_within_lot_MCD","inprior",24,.005,tin,cpv["inprior"],ncal))
    ag=aggregate(rows); cag=aggregate(candidate)[0]
    base=next(x for x in ag if x["detector"]=="Sahodrift_v5" and x["test_set"]=="inprior" and x["decision_hour"]==24 and x["target_alpha"]==.005)
    passed=cag["pooled_recall_pct"]>base["pooled_recall_pct"] and cag["seeds_above_0_75_pct"]<=1
    return rows,ag,candidate,cag,passed


def fresh(cfg):
    rows=[]
    for seed in range(5,10):
        _,cal,tin,_=load_seed(cfg,seed); ncal=int((cal.is_defect==0).sum())
        for hour,alpha in ((24,.005),(168,.0025)):
            p,_=sahodrift_p(cal,{"inprior":tin},cfg,hour)
            rows.append(p_to_metric(seed,"simulator_fresh","Sahodrift_v5","inprior",hour,alpha,tin,p["inprior"],ncal))
    ag=aggregate(rows)
    for x in ag: x["acceptance_passed"]=x["mean_achieved_far_pct"]<=.5 and x["seeds_above_0_75_pct"]<=1
    return rows,ag


def external_frame_clean(loader):
    train,cal,test,meta=loader()
    for col in [f"{p}_{h}h" for p in PARAMS for h in (0,24,168)]:
        fill=float(np.nanmedian(train[col])); train[col]=train[col].fillna(fill); cal[col]=cal[col].fillna(fill); test[col]=test[col].fillna(fill)
    return train,cal,test,meta


def burntrace(cfg):
    train,cal,test,meta=external_frame_clean(burntrace_frames); rows=[]; ncal=int((cal.is_defect==0).sum())
    for hour in (24,168):
        tr,ca,te=at_hour(train,hour),at_hour(cal,hour),at_hour(test,hour)
        sp,_=sahodrift_p(ca,{"test":te},cfg,24); raw=rival_raw(tr,ca,{"test":te},26170+hour,include_r3=False); good=ca.is_defect.to_numpy()==0
        methods={"Sahodrift_v5":sp["test"]}|{m:upper_p(cs[good],ts["test"]) for m,(cs,ts) in raw.items() if m!="R3_gradient_boosting_sklearn_fallback"}
        for method,p in methods.items():
            for alpha in (.005,.01,.02): rows.append(p_to_metric(26170,"BurnTrace",method,"published_labelled_split",hour,alpha,te,p,ncal))
    return {"meta":meta,"rows":rows,"published_points":[{"hour":24,"recall_pct":21.1,"far_pct":2.55},{"hour":168,"recall_pct":72.2,"far_pct":1.04}]}


def aabhas(cfg,run_candidate):
    train,cal,test,meta=external_frame_clean(aabhas_frames); hour_results={}; class_table=None
    for hour in (24,168):
        tr,ca,te=at_hour(train,hour),at_hour(cal,hour),at_hour(test,hour); ncal=int((ca.is_defect==0).sum()); good=ca.is_defect.to_numpy()==0
        sp,_=sahodrift_p(ca,{"test":te},cfg,24); raw=rival_raw(tr,ca,{"test":te},26170+hour,include_r3=False)
        ps=sp["test"]; pr4=upper_p(raw["R4_unsupervised_ensemble"][0][good],raw["R4_unsupervised_ensemble"][1]["test"]); pr1=upper_p(raw["R1_lot_mad"][0][good],raw["R1_lot_mad"][1]["test"])
        records=[p_to_metric(26170,"Aabhas","Sahodrift_v5","frozen",hour,.005,te,ps,ncal),p_to_metric(26170,"Aabhas","R4_unsupervised_ensemble","frozen",hour,.005,te,pr4,ncal),p_to_metric(26170,"Aabhas","R1_lot_mad","frozen",hour,.005,te,pr1,ncal)]
        if hour==24:
            fs=ps<=.005; fr=pr4<=.005; defect=te.is_defect.to_numpy()==1; table=[]
            for cls in sorted(te.loc[defect,"defect_type"].unique()):
                mask=te.defect_type.to_numpy()==cls
                table.append({"defect_class":str(cls),"n":int(mask.sum()),"Sahodrift_only":int(np.sum(mask&fs&~fr)),"R4_only":int(np.sum(mask&fr&~fs)),"both":int(np.sum(mask&fr&fs)),"neither":int(np.sum(mask&~fr&~fs))})
            class_table=table
            if run_candidate:
                cpv,_=sahodrift_p(ca,{"test":te},cfg,24,extra_mcd=True)
                records.append(p_to_metric(26170,"Aabhas","Sahodrift_plus_within_lot_MCD","frozen",24,.005,te,cpv["test"],ncal))
        hour_results[str(hour)]=records
    return {"meta":meta,"per_defect_class_24h":class_table,"results":hour_results,"candidate_run":run_candidate,"published_24h":{"recall_pct":29.7,"far_pct":.54}}


def main():
    start=time.time(); cfg=yaml.safe_load((ROOT/"config_v2.yaml").read_text())
    rows,ag,crows,cag,gate=simulator_stack(cfg)
    fresh_rows,fresh_ag=fresh(cfg); burn=burntrace(cfg); aab=aabhas(cfg,gate)
    payload={"schema_version":"6.0","preregistration_commit":"0db8ea58cdc4d2941a2ae1dfede7583d8866b9ea","merger_selection":{"choice":"twice_mean","basis":"preregistered; not reselected on v6 tests"},"simulator_rivals":{"rows":rows,"operating_aggregates":ag},"candidate_MCD":{"simulator_rows":crows,"simulator_aggregate":cag,"gate_passed":gate,"aabhas_run":gate},"fresh_seeds":{"rows":fresh_rows,"operating_aggregates":fresh_ag},"aabhas":aab,"burntrace":burn,"demo":{"public_url":"https://sahodrift-app.vercel.app"},"integrity":{"runtime_seconds":time.time()-start,"environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},"limitations":[]}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"sahodrift_results_v6.json").write_text(json.dumps(serial(payload),indent=2,allow_nan=False)+"\n")
    (OUT/"simulator_rivals_v5_rules.json").write_text(json.dumps(serial({"preregistration_commit":payload["preregistration_commit"],"rows":rows,"operating_aggregates":ag,"demo":payload["demo"]}),indent=2,allow_nan=False)+"\n")
    print(json.dumps({"candidate_gate":gate,"candidate":cag,"fresh":fresh_ag,"runtime_seconds":time.time()-start},indent=2))


if __name__=="__main__": main()
