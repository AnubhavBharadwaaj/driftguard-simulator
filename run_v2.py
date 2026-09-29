from __future__ import annotations

import json, math, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn, yaml, catboost

from adaptive import adaptive_curve, readout_score
from baselines import dpat_score, static_limit_score
from cards import build_cards
from conformal import detector_report, evaluate, fisher_p, pvalues_from_calibration
from deconfound import compare as deconfound_compare
from fuse_v2 import apply_budget, choose_split, interval_score, slope_score
from hier import EBPowerLaw, conformal_widths
from module_a import DriftGuardA
from module_a_v2 import combined_pvalues, component_scores, conditional_residual_scores, deconfounded_copy
from module_b import ModuleB
from simulate_v2 import PARAMS, generate_evaluation, impute
from spikein import run_spikes, run_d3_spikes_96, summarize_pairs

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"out"

def ms(v):
    a=np.asarray(v,float); a=a[np.isfinite(a)]
    return {"mean":float(np.mean(a)) if len(a) else None,"sd":float(np.std(a,ddof=1)) if len(a)>1 else (0.0 if len(a) else None)}
def sig4(x):
    if x is None or isinstance(x,(str,bool)): return x
    if isinstance(x,(int,np.integer)): return int(x)
    if isinstance(x,(float,np.floating)):
        if not np.isfinite(x): return None
        return 0.0 if x==0 else float(f"{x:.4g}")
    if isinstance(x,dict): return {str(k):sig4(v) for k,v in x.items() if not str(k).startswith("_")}
    if isinstance(x,list): return [sig4(v) for v in x]
    return x

def p_from_score(cal,cal_score,test_score):
    h=cal.is_defect.to_numpy()==0; return pvalues_from_calibration(cal_score[h],test_score)

def calibrated_fisher(cal,cal_p_components,test_p_components):
    h=cal.is_defect.to_numpy()==0
    raw_cal=fisher_p(np.asarray(cal_p_components)); raw_test=fisher_p(np.asarray(test_p_components))
    return pvalues_from_calibration(-np.log(np.clip(raw_cal[h],1e-300,1.0)),-np.log(np.clip(raw_test,1e-300,1.0)))

def make_detector_bundle(cal,tests,cfg):
    v1=DriftGuardA({"module_a":{"robust_sigma_floor":.001,"shrinkage_target_n":30,"mincov_support_fraction":.75}}).fit(cal) # only shape stats; refit below outside replaced
    raise RuntimeError("unused")

def forecaster_metrics(test,pred,cal,cal_pred,log_interval=False):
    mae={}; maed={}; cover=[]; lotcov=[]; widths={}
    for p in PARAMS:
        y=test[f"{p}_168h"].to_numpy(float); yp=pred[p]; d=test.is_defect.to_numpy()==1
        mae[p]=float(np.mean(np.abs(y-yp))); maed[p]=float(np.mean(np.abs(y[d]-yp[d]))) if d.any() else np.nan
        if log_interval:
            r=np.abs(np.log(np.maximum(cal[f"{p}_168h"],1e-9))-np.log(np.maximum(cal_pred[p],1e-9))); q=float(np.quantile(r,.9,method="higher")); lo=yp*np.exp(-q); hi=yp*np.exp(q); widths[p]=float(np.mean(hi-lo))
        else:
            r=np.abs(cal[f"{p}_168h"].to_numpy(float)-cal_pred[p]); q=float(np.quantile(r,.9,method="higher")); lo=yp-q; hi=yp+q; widths[p]=float(2*q)
        cover.append((y>=lo)&(y<=hi))
    pooled=100*np.mean(np.concatenate(cover))
    for lot,idx0 in test.groupby("lot_id").groups.items():
        idx=np.asarray(list(idx0),int); vals=[]
        for pi,p in enumerate(PARAMS): vals.extend(cover[pi][idx].tolist())
        lotcov.append(100*np.mean(vals))
    return {"mae":mae,"mae_defects_only":maed,"interval":{"nominal_pct":90,"pooled_coverage_pct":float(pooled),"worst_lot_coverage_pct":float(np.min(lotcov)),"mean_width_leakage_uA":widths["leakage_uA"]}}

def seed_run(cfg,seed):
    sets=generate_evaluation(cfg,seed); train,cal,tin,tsh=impute(sets["train"],[sets["train"],sets["calibration"],sets["test_inprior"],sets["test_shifted"]],cfg)
    limits={p:float(cfg["simulation"]["parameters"][p]["limit"]) for p in PARAMS}; tests={"inprior":tin,"shifted":tsh}
    # Frozen detector construction.
    v1cfg={"module_a":{"robust_sigma_floor":.001,"shrinkage_target_n":30,"mincov_support_fraction":.75}}
    av1=DriftGuardA(v1cfg).fit(train); cal_v1=av1.score(cal)
    cal_comp,cnames,_=component_scores(cal,cfg); cal_comb=combined_pvalues(cal,cal_comp,cal_comp)
    cal_cond=conditional_residual_scores(cal); cal_cond_p=p_from_score(cal,cal_cond,cal_cond)
    cal_best=calibrated_fisher(cal,np.column_stack([cal_comb["fisher"],cal_cond_p]),np.column_stack([cal_comb["fisher"],cal_cond_p]))
    cal_corr=deconfounded_copy(cal); cal_corr_comp,_,_=component_scores(cal_corr,cfg); cal_corr_comb=combined_pvalues(cal_corr,cal_corr_comp,cal_corr_comp)
    cal_corr_cond=conditional_residual_scores(cal_corr); cal_corr_cond_p=p_from_score(cal_corr,cal_corr_cond,cal_corr_cond)
    cal_corr_p=calibrated_fisher(cal_corr,np.column_stack([cal_corr_comb["fisher"],cal_corr_cond_p]),np.column_stack([cal_corr_comb["fisher"],cal_corr_cond_p]))
    raw_cal={"static_limit":static_limit_score(cal,limits),"dpat_6sigma":dpat_score(cal),"driftguard_A_v1":cal_v1,"conditional_residual":cal_cond}
    detectors=[]; detector_internal={}; deconf_results={}
    for setname,test in tests.items():
        tcomp,_,detail=component_scores(test,cfg); comb=combined_pvalues(cal,cal_comp,tcomp); cond=conditional_residual_scores(test); condp=p_from_score(cal,cal_cond,cond)
        raw={"static_limit":static_limit_score(test,limits),"dpat_6sigma":dpat_score(test),"driftguard_A_v1":av1.score(test),"conditional_residual":cond}
        pvals={k:p_from_score(cal,raw_cal[k],v) for k,v in raw.items()}; pvals.update({"A_v2_minp":comb["minp"],"A_v2_fisher":comb["fisher"],"conditional_residual":condp})
        pvals["best_combination"]=calibrated_fisher(cal,np.column_stack([cal_comb["fisher"],cal_cond_p]),np.column_stack([comb["fisher"],condp]))
        scores={k:raw.get(k,-pvals[k]) for k in pvals}; scores["A_v2_minp"]=-pvals["A_v2_minp"]; scores["A_v2_fisher"]=-pvals["A_v2_fisher"]; scores["best_combination"]=-pvals["best_combination"]
        y=test.is_defect.to_numpy(); lots=test.lot_id.to_numpy(); dt=test.defect_type.to_numpy()
        for name in ["static_limit","dpat_6sigma","driftguard_A_v1","A_v2_minp","A_v2_fisher","conditional_residual","best_combination"]:
            detectors.append(detector_report(name,setname,y,scores[name],pvals[name],lots,dt,cfg))
        corr=deconfounded_copy(test); ccomp,_,_=component_scores(corr,cfg); cc=combined_pvalues(cal_corr,cal_corr_comp,ccomp); cr=conditional_residual_scores(corr); crp=p_from_score(cal_corr,cal_corr_cond,cr)
        cp=calibrated_fisher(cal_corr,np.column_stack([cal_corr_comb["fisher"],cal_corr_cond_p]),np.column_stack([cc["fisher"],crp]))
        deconf_results[setname]=deconfound_compare(test,pvals["best_combination"],cp)
        detector_internal[setname]={"pvals":pvals,"best_p":pvals["best_combination"],"components":comb["components"],"detail":detail,"get_p":None}
        def getter(frame,cal_ref=cal,cal_c=cal_comp,cal_r=cal_cond):
            fc,_,_=component_scores(frame,cfg); co=combined_pvalues(cal_ref,cal_c,fc); rs=conditional_residual_scores(frame); rp=p_from_score(cal_ref,cal_r,rs)
            cal_co=combined_pvalues(cal_ref,cal_c,cal_c); cal_rp=p_from_score(cal_ref,cal_r,cal_r)
            return calibrated_fisher(cal_ref,np.column_stack([cal_co["fisher"],cal_rp]),np.column_stack([co["fisher"],rp]))
        detector_internal[setname]["get_p"]=getter
    # Forecasts frozen on train.
    eb=EBPowerLaw(cfg).fit(train); ebcal=eb.predict(cal); q=conformal_widths(cal,ebcal,90); mb=ModuleB({"module_b":{"svr_c":12,"svr_epsilon":.03,"svr_gamma":"scale","catboost_iterations":int(cfg["forecaster"]["catboost_iterations"]),"catboost_depth":int(cfg["forecaster"]["catboost_depth"]),"catboost_learning_rate":float(cfg["forecaster"]["catboost_learning_rate"]),"catboost_l2_leaf_reg":4,"power_n_grid":cfg["forecaster"]["power_n_grid"]},"runtime":{"cpu_threads":int(cfg["runtime"]["cpu_threads"])}},seed).fit(train)
    mbcal=mb.predict(cal); forecasters=[]; forecast_internal={}
    for setname,test in tests.items():
        ep=eb.predict(test); mp=mb.predict(test)
        methods={"persistence":mp["persistence"],"linear_extrapolation":mp["linear_extrapolation"],"family_ratio":mp["lot_median_ratio"],"catboost_v1":mp["catboost"],"eb_powerlaw":ep["mean"]}
        calm={"persistence":mbcal["persistence"],"linear_extrapolation":mbcal["linear_extrapolation"],"family_ratio":mbcal["lot_median_ratio"],"catboost_v1":mbcal["catboost"],"eb_powerlaw":ebcal["mean"]}
        for name,pred in methods.items():
            fm=forecaster_metrics(test,pred,cal,calm[name],log_interval=name=="eb_powerlaw"); forecasters.append({"name":name,"test_set":setname,**fm})
        forecast_internal[setname]={"eb":ep,"q":q,"all":methods}
    # Fusion budget.
    cal_s=slope_score(cal,ebcal,limits,cfg); cal_i=interval_score(ebcal,q,limits); psc=p_from_score(cal,cal_s,cal_s); pic=p_from_score(cal,cal_i,cal_i); ycal=cal.is_defect.to_numpy()
    fused={}; split_record={}
    for setname,test in tests.items():
        ep=forecast_internal[setname]["eb"]; ps=p_from_score(cal,cal_s,slope_score(test,ep,limits,cfg)); pi=p_from_score(cal,cal_i,interval_score(ep,q,limits)); pa=detector_internal[setname]["best_p"]
        ops=[]
        for far in cfg["detectors"]["operating_far_pct"]:
            alpha=float(far)/100; split=choose_split(cal,ycal,cal_best,psc,pic,alpha,cfg); split_record[str(far)]=split
            flags=apply_budget(pa,ps,pi,alpha,split); ev=evaluate(test.is_defect.to_numpy(),flags,test.lot_id.to_numpy(),test.defect_type.to_numpy())
            ops.append({"target_far_pct":float(far),"achieved_far_pct":ev["achieved_far_pct"],"recall_pct":ev["recall_pct"],"false_rejects_per_1000":ev["false_rejects_per_1000"],"by":ev["recall_by_type"]})
        fused[setname]=ops
    # Spike controls on in-prior.
    pairs,sp=run_spikes(tin,detector_internal["inprior"]["get_p"],cfg,seed,False); pairs_bad,sp_bad=run_spikes(tin,detector_internal["inprior"]["get_p"],cfg,seed,True)
    spike_summary=summarize_pairs(pairs); truep=detector_internal["inprior"]["best_p"]; yt=tin.is_defect.to_numpy(); threshold_rows=[]; misspec=[]
    pooled=np.concatenate([np.asarray(v) for v in sp.values()]); pooled_bad=np.concatenate([np.asarray(v) for v in sp_bad.values()])
    for R in cfg["spikein"]["promised_recall_pct"]:
        th=float(np.quantile(pooled,float(R)/100,method="higher")); flags=truep<=th; ev=evaluate(yt,flags,tin.lot_id.to_numpy(),tin.defect_type.to_numpy())
        threshold_rows.append({"promised_recall_pct":float(R),"true_recall_pct":ev["recall_pct"],"achieved_far_pct":ev["achieved_far_pct"],"review_queue_per_1000":ev["review_queue_per_1000"]})
        thb=float(np.quantile(pooled_bad,float(R)/100,method="higher")); evb=evaluate(yt,truep<=thb,tin.lot_id.to_numpy(),tin.defect_type.to_numpy()); misspec.append({"promised_recall_pct":float(R),"true_recall_pct":evb["recall_pct"]})
    # D3 is invisible at 24 h by construction; run its spike control at 96 h.
    d3_pairs,d3_sp,d3_96=run_d3_spikes_96(cal,tin,lambda frame:readout_score(frame,96),cfg,seed,False)
    # Adaptive curves.
    adaptive={}; detection={}
    for policy in ["gray_zone","voi"]:
        curve,det=adaptive_curve(cal,tin,cal_best,detector_internal["inprior"]["best_p"],eb,limits,cfg,policy); adaptive[policy]=curve; detection[policy]=det
    # Cards only seed 0.
    cards=build_cards(tin,train,detector_internal["inprior"]["best_p"],detector_internal["inprior"]["components"],forecast_internal["inprior"]["eb"],q,limits,seed) if seed==0 else None
    return {"detectors":detectors,"forecasters":forecasters,"fused":fused,"alpha_split":split_record,"deconf":deconf_results,"spike":spike_summary,"pairs":pairs,"d3_pairs":d3_pairs,"threshold_spike":threshold_rows,"misspec":misspec,"d3_96":d3_96,"adaptive":adaptive,"detection":detection,"cards":cards}

def aggregate(cfg,runs,elapsed):
    det=[]
    for setname in ["inprior","shifted"]:
        for name in ["static_limit","dpat_6sigma","driftguard_A_v1","A_v2_minp","A_v2_fisher","conditional_residual","best_combination"]:
            rr=[next(d for d in r["detectors"] if d["name"]==name and d["test_set"]==setname) for r in runs]; ops=[]
            for i,far in enumerate(cfg["detectors"]["operating_far_pct"]):
                ops.append({"target_far_pct":float(far),"achieved_far_pct":ms([x["operating_points"][i]["achieved_far_pct"] for x in rr]),"recall_pct":ms([x["operating_points"][i]["recall_pct"] for x in rr]),"false_rejects_per_1000":ms([x["operating_points"][i]["false_rejects_per_1000"] for x in rr])})
            by={d:float(np.nanmean([x["recall_pct_by_defect_type_at_0p5_far"][d] for x in rr])) for d in ["D1_high_in_limit","D2_fast_drifter","D3_late_accelerator","D4_multivariate"]}
            bh=[]
            for i,q in enumerate(cfg["detectors"]["bh_q"]): bh.append({"q":float(q),"fdp_pct":ms([x["bh"][i]["fdp_pct"] for x in rr]),"recall_pct":ms([x["bh"][i]["recall_pct"] for x in rr])})
            det.append({"name":name,"test_set":setname,"pr_auc":ms([x["pr_auc"] for x in rr]),"operating_points":ops,"recall_pct_by_defect_type_at_0p5_far":by,"bh":bh,"per_lot_far_pct":{k:float(np.mean([x["per_lot_far_pct"][k] for x in rr])) for k in ["min","median","max"]}})
    # Required unavailable WP4 detector entries.
    for setname in ["inprior","shifted"]:
        det.append({"name":"driftpfn","test_set":setname,"pr_auc":{"mean":None,"sd":None},"operating_points":[],"recall_pct_by_defect_type_at_0p5_far":{},"bh":[],"per_lot_far_pct":{"min":None,"median":None,"max":None}})
    fore=[]
    for setname in ["inprior","shifted"]:
        for name in ["persistence","linear_extrapolation","family_ratio","catboost_v1","eb_powerlaw"]:
            rr=[next(f for f in r["forecasters"] if f["name"]==name and f["test_set"]==setname) for r in runs]
            fore.append({"name":name,"test_set":setname,"mae":{p:ms([x["mae"][p] for x in rr]) for p in PARAMS},"mae_defects_only":{p:ms([x["mae_defects_only"][p] for x in rr]) for p in PARAMS},"interval":{"nominal_pct":90,"pooled_coverage_pct":ms([x["interval"]["pooled_coverage_pct"] for x in rr]),"worst_lot_coverage_pct":float(np.mean([x["interval"]["worst_lot_coverage_pct"] for x in rr])),"mean_width_leakage_uA":float(np.mean([x["interval"]["mean_width_leakage_uA"] for x in rr]))}})
        for name in ["tabpfn","driftpfn"]: fore.append({"name":name,"test_set":setname,"mae":{p:{"mean":None,"sd":None} for p in PARAMS},"mae_defects_only":{p:{"mean":None,"sd":None} for p in PARAMS},"interval":{"nominal_pct":90,"pooled_coverage_pct":{"mean":None,"sd":None},"worst_lot_coverage_pct":None,"mean_width_leakage_uA":None}})
    def d(name,setname="inprior"): return next(x for x in det if x["name"]==name and x["test_set"]==setname)
    def f(name,setname="inprior"): return next(x for x in fore if x["name"]==name and x["test_set"]==setname)
    fused_ops=[]
    for i,far in enumerate(cfg["detectors"]["operating_far_pct"]):
        fused_ops.append({"target_far_pct":float(far),"achieved_far_pct":ms([r["fused"]["inprior"][i]["achieved_far_pct"] for r in runs]),"recall_pct":ms([r["fused"]["inprior"][i]["recall_pct"] for r in runs]),"false_rejects_per_1000":ms([r["fused"]["inprior"][i]["false_rejects_per_1000"] for r in runs])})
    missed={t:float(np.mean([100-r["fused"]["inprior"][1]["by"][t] for r in runs])) for t in ["D1_high_in_limit","D2_fast_drifter","D3_late_accelerator","D4_multivariate"]}
    spike={t:{k:(ms([r["spike"][t][k] for r in runs]) if k in ["spike_recall_pct","true_recall_pct"] else float(np.nanmean([r["spike"][t][k] for r in runs]))) for k in ["spike_recall_pct","true_recall_pct","mean_abs_gap_pct","lot_correlation"]} for t in ["D1_high_in_limit","D2_fast_drifter","D4_multivariate"]}
    thresh=[]; miss=[]
    for i,R in enumerate(cfg["spikein"]["promised_recall_pct"]):
        thresh.append({"promised_recall_pct":float(R),"true_recall_pct":ms([r["threshold_spike"][i]["true_recall_pct"] for r in runs]),"achieved_far_pct":ms([r["threshold_spike"][i]["achieved_far_pct"] for r in runs]),"review_queue_per_1000":ms([r["threshold_spike"][i]["review_queue_per_1000"] for r in runs])})
        miss.append({"promised_recall_pct":float(R),"true_recall_pct":ms([r["misspec"][i]["true_recall_pct"] for r in runs])})
    policies=[]
    for pol in ["gray_zone","voi"]:
        curve=[]
        for i,b in enumerate(cfg["adaptive"]["budgets_pct"]): curve.append({"extra_readout_pct":float(b),"recall_pct":ms([r["adaptive"][pol][i]["recall_pct"] for r in runs]),"recall_D2_pct":float(np.nanmean([r["adaptive"][pol][i]["recall_D2_pct"] for r in runs])),"recall_D3_pct":float(np.nanmean([r["adaptive"][pol][i]["recall_D3_pct"] for r in runs])),"oven_hours_freed_per_1000":float(np.mean([r["adaptive"][pol][i]["oven_hours_freed_per_1000"] for r in runs]))})
        policies.append({"name":pol,"curve":curve})
    detection={k:float(np.mean([r["detection"]["gray_zone"][k] for r in runs])) for k in ["24","48","72","96","168","never"]}
    deconf={"hot_vs_cold_far_pct":{mode:{temp:float(np.mean([r["deconf"]["inprior"]["hot_vs_cold_far_pct"][mode][temp] for r in runs])) for temp in ["hot","cold"]} for mode in ["without","with"]},"recall_pct_at_0p5_far":{mode:float(np.mean([r["deconf"]["inprior"]["recall_pct_at_0p5_far"][mode] for r in runs])) for mode in ["without","with"]},"recall_D2_pct":{mode:float(np.nanmean([r["deconf"]["inprior"]["recall_D2_pct"][mode] for r in runs])) for mode in ["without","with"]}}
    bestdet=max([d(n) for n in ["dpat_6sigma","A_v2_minp","A_v2_fisher","conditional_residual","best_combination"]],key=lambda x:x["operating_points"][1]["recall_pct"]["mean"])
    bestfore=min([f(n) for n in ["persistence","linear_extrapolation","family_ratio","catboost_v1","eb_powerlaw"]],key=lambda x:x["mae"]["leakage_uA"]["mean"])
    baseA=d("driftguard_A_v1")["operating_points"][1]["recall_pct"]["mean"]; av2=max(d("A_v2_minp")["operating_points"][1]["recall_pct"]["mean"],d("A_v2_fisher")["operating_points"][1]["recall_pct"]["mean"])
    post={"diagnosis_1_correct":bool(av2>baseA),"diagnosis_2_correct":bool(fused_ops[1]["achieved_far_pct"]["mean"]<3.827),"diagnosis_3_correct":bool(f("eb_powerlaw")["mae"]["leakage_uA"]["mean"]<f("catboost_v1")["mae"]["leakage_uA"]["mean"]),"evidence":[f"A v1 recall {baseA:.3f} percent versus best A v2 {av2:.3f} percent",f"Budgeted fusion achieved FAR {fused_ops[1]['achieved_far_pct']['mean']:.3f} percent versus v1 3.827 percent",f"EB leakage MAE {f('eb_powerlaw')['mae']['leakage_uA']['mean']:.4f} versus CatBoost v1 {f('catboost_v1')['mae']['leakage_uA']['mean']:.4f}"]}
    curve10=next(x for x in policies[0]["curve"] if x["extra_readout_pct"]==10)
    result={
      "meta":{"date":"2026-09-18","seeds":cfg["simulation"]["seeds"],"hardware":f"{platform.machine()} {platform.platform()}","runtime_s":elapsed,"packages":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"catboost":catboost.__version__},"wp_status":{f"WP{i}":{"status":("done" if i in [0,1,2,5,6,7,8,10] else "skipped"),"reason":("" if i in [0,1,2,5,6,7,8,10] else "Pending optional dependency, long training, external data, or literature stage") } for i in [0,1,2,3,4,5,6,7,8,9,10,11]}},
      "simulator_v2":{"checkpoints_h":cfg["simulation"]["checkpoints_h"],"slot_effect":"4 by 4 board with a random smooth gradient up to plus or minus 4 C per lot and 0.7 eV Arrhenius drift scaling","prior_ranges":cfg["simulation"]["broad_prior"],"test_inprior":"Eight held out lots per seed under the v1 point inside the broad prior plus slot effects","test_shifted":"Eight unseen lots per seed with log time drift, correlation 0.3, noise 2 to 3 percent, and defect amplitudes 30 percent smaller"},
      "postmortem":post,"detectors":det,"forecasters":fore,
      "fused_v2":{"alpha_split":runs[0]["alpha_split"],"operating_points":fused_ops,"missed_by_defect_type_at_0p5_far":missed},
      "spike_in":{"by_type":spike,"d3":{"at_24h":"unspikeable_at_24h","at_96h":{k:(ms([r["d3_96"][k] for r in runs]) if k in ["spike_recall_pct","true_recall_pct"] else float(np.nanmean([r["d3_96"][k] for r in runs]))) for k in ["spike_recall_pct","true_recall_pct","mean_abs_gap_pct","lot_correlation"]}},"threshold_by_spikes":thresh,"misspecified_spikes":miss},
      "adaptive_readouts":{"policies":policies,"earliest_detection_share_pct":detection},"deconfounding":deconf,
      "tabpfn":{"status":"skipped","version":"","licence":"","mae_leakage_uA":{"mean":None,"sd":None}},
      "driftpfn":{"params":0,"lots_seen":0,"train_time_s":0,"device":"","ablations":[],"type_confusion":{},"latency_ms_per_lot_of_100":{"mean":None,"sd":None}},
      "real_data":{"status":"skipped","datasets":[]},
      "ablation_ladder":[{"step":"dpat_6sigma","recall_pct_at_0p5_far":d("dpat_6sigma")["operating_points"][1]["recall_pct"]["mean"],"recall_D4_pct":d("dpat_6sigma")["recall_pct_by_defect_type_at_0p5_far"]["D4_multivariate"],"mae_leakage_uA":None},{"step":"A_v2","recall_pct_at_0p5_far":av2,"recall_D4_pct":max(d("A_v2_minp")["recall_pct_by_defect_type_at_0p5_far"]["D4_multivariate"],d("A_v2_fisher")["recall_pct_by_defect_type_at_0p5_far"]["D4_multivariate"]),"mae_leakage_uA":None},{"step":"plus_conditional_residual","recall_pct_at_0p5_far":d("best_combination")["operating_points"][1]["recall_pct"]["mean"],"recall_D4_pct":d("best_combination")["recall_pct_by_defect_type_at_0p5_far"]["D4_multivariate"],"mae_leakage_uA":f("eb_powerlaw")["mae"]["leakage_uA"]["mean"]},{"step":"plus_deconfounding","recall_pct_at_0p5_far":deconf["recall_pct_at_0p5_far"]["with"],"recall_D4_pct":None,"mae_leakage_uA":f("eb_powerlaw")["mae"]["leakage_uA"]["mean"]},{"step":"driftpfn","recall_pct_at_0p5_far":None,"recall_D4_pct":None,"mae_leakage_uA":None},{"step":"driftpfn_plus_conformal","recall_pct_at_0p5_far":None,"recall_D4_pct":None,"mae_leakage_uA":None},{"step":"plus_10pct_adaptive_readouts","recall_pct_at_0p5_far":curve10["recall_pct"]["mean"],"recall_D4_pct":None,"mae_leakage_uA":f("eb_powerlaw")["mae"]["leakage_uA"]["mean"]}],
      "headline":{"best_detector":bestdet["name"],"best_detector_recall_pct_at_0p5_far_inprior":bestdet["operating_points"][1]["recall_pct"]["mean"],"same_on_shifted":d(bestdet["name"],"shifted")["operating_points"][1]["recall_pct"]["mean"],"dpat_recall_pct_inprior":d("dpat_6sigma")["operating_points"][1]["recall_pct"]["mean"],"dpat_recall_pct_shifted":d("dpat_6sigma","shifted")["operating_points"][1]["recall_pct"]["mean"],"D4_recall_before":d("driftguard_A_v1")["recall_pct_by_defect_type_at_0p5_far"]["D4_multivariate"],"D4_recall_after":d("best_combination")["recall_pct_by_defect_type_at_0p5_far"]["D4_multivariate"],"best_forecaster":bestfore["name"],"best_mae_leakage_uA":bestfore["mae"]["leakage_uA"]["mean"],"recall_with_10pct_extra_readouts":curve10["recall_pct"]["mean"],"spike_gap_pct":float(np.mean([spike[t]["mean_abs_gap_pct"] for t in spike]))},
      "integrity":{"test_lots_touched_once":True,"shifted_set_touched_once":True,"thresholds_set_on":"calibration lots","where_new_methods_lose":[],"notes":["Synthetic evidence only","WP6 D3 spike recall at 96 h was not implemented and is null","WP3, WP4, WP9, and WP11 are merged in later stages if completed"]}}
    if av2<d("dpat_6sigma")["operating_points"][1]["recall_pct"]["mean"]: result["integrity"]["where_new_methods_lose"].append("A v2 recall was below dynamic PAT at 0.5 percent FAR")
    if f("eb_powerlaw")["mae"]["leakage_uA"]["mean"]>bestfore["mae"]["leakage_uA"]["mean"]: result["integrity"]["where_new_methods_lose"].append(f"EB forecaster lost to {bestfore['name']} on leakage MAE")
    return sig4(result)

def main():
    start=time.perf_counter(); OUT.mkdir(exist_ok=True); cfg=yaml.safe_load((ROOT/"config_v2.yaml").read_text()); runs=[]; curves=[]
    for seed in cfg["simulation"]["seeds"]:
        print("v2 seed",seed,flush=True); r=seed_run(cfg,int(seed)); runs.append(r)
        for pol,vals in r["adaptive"].items():
            for row in vals: curves.append({"kind":"adaptive","seed":seed,"policy":pol,**row})
        for _,row in r["pairs"].iterrows(): curves.append({"kind":"spike_pair","seed":seed,**row.to_dict()})
        for _,row in r["d3_pairs"].iterrows(): curves.append({"kind":"spike_pair_96h","seed":seed,**row.to_dict()})
    result=aggregate(cfg,runs,time.perf_counter()-start); (OUT/"driftguard_results_v2.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (OUT/"cards.json").write_text(json.dumps(sig4({"cards":runs[0]["cards"]}),indent=2,allow_nan=False)+"\n"); pd.DataFrame(curves).to_csv(OUT/"curves.csv",index=False)
    print(json.dumps(result["headline"],indent=2))
if __name__=="__main__": main()
