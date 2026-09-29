from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, yaml
from tabpfn import TabPFNRegressor, __version__ as tabpfn_version
from hier import EBPowerLaw
from module_b import feature_frame
from simulate_v2 import PARAMS, generate_evaluation, impute

ROOT=Path(__file__).resolve().parent; cfg=yaml.safe_load((ROOT/'config_v2.yaml').read_text())
def main():
    started=time.perf_counter(); runs=[]; model_ids=[]
    try:
        for seed in cfg['simulation']['seeds']:
            print('tabpfn seed',seed,flush=True); sets=generate_evaluation(cfg,int(seed)); train,cal,tin,tsh=impute(sets['train'],[sets['train'],sets['calibration'],sets['test_inprior'],sets['test_shifted']],cfg)
            eb=EBPowerLaw(cfg).fit(train); results={}
            X=feature_frame(train); maxrows=min(3000,len(train)); idx=np.random.default_rng(seed+300).choice(len(train),maxrows,replace=False); Xfit=X.iloc[idx]
            for setname,test in [('inprior',tin),('shifted',tsh)]: results[setname]={}
            for p in PARAMS:
                r=np.log(np.maximum(train[f'{p}_168h'],1e-9)/np.maximum(train[f'{p}_24h'],1e-9)).to_numpy()
                lot=train.assign(_r=r).groupby('lot_id')['_r'].transform('median').to_numpy(); target=(r-lot)[idx]
                model=TabPFNRegressor(n_estimators=1,device='cpu',random_state=int(seed),show_progress_bar=False,inference_config={'SUBSAMPLE_SAMPLES':3000})
                model.fit(Xfit,target); model_ids.append(str(getattr(model,'model_path','auto')))
                for setname,test in [('inprior',tin),('shifted',tsh)]:
                    ebp=eb.predict(test)['mean'][p]; base=np.log(np.maximum(ebp,1e-9)/np.maximum(test[f'{p}_24h'],1e-9)); resid=model.predict(feature_frame(test)); pred=test[f'{p}_24h'].to_numpy()*np.exp(base+resid); y=test[f'{p}_168h'].to_numpy(); results[setname][p]=float(np.mean(np.abs(y-pred)))
            runs.append(results)
        out={'status':'done','version':tabpfn_version,'model_version':sorted(set(model_ids)),'licence':'Apache-2.0','runtime_s':time.perf_counter()-started,'mae':{s:{p:{'mean':float(np.mean([r[s][p] for r in runs])),'sd':float(np.std([r[s][p] for r in runs],ddof=1))} for p in PARAMS} for s in ['inprior','shifted']}}
    except Exception as e:
        out={'status':'skipped','version':tabpfn_version,'model_version':[],'licence':'Apache-2.0','runtime_s':time.perf_counter()-started,'error':f'{type(e).__name__}: {e}'}
    (ROOT/'out/tabpfn_result.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
