from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score

from conformal import pvalues_from_calibration
from simulate_v2 import PARAMS, generate_evaluation, impute, sample_lot, sample_prior
from driftpfn.model import DriftPFN

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; cfg=yaml.safe_load((ROOT/'config_v2.yaml').read_text())
TYPES={'healthy':0,'D1_high_in_limit':1,'D2_fast_drifter':2,'D3_late_accelerator':3,'D4_multivariate':4}

def fill_lot(df):
    f=df.copy().reset_index(drop=True)
    for p in PARAMS:
        for t in [0,24,48,168]: f[f'{p}_{t}h']=f[f'{p}_{t}h'].fillna(f[f'{p}_{t}h'].median())
    return f

def lot_features(df,use_slot=True,use48=False):
    f=fill_lot(df); blocks=[]; globals=[]
    for t in [0,24]:
        vals=np.log(np.maximum(f[[f'{p}_{t}h' for p in PARAMS]].to_numpy(float),1e-9)); med=np.median(vals,axis=0); mad=np.maximum(1e-3,1.4826*np.median(np.abs(vals-med),axis=0)); blocks.append((vals-med)/mad); globals.extend(med.tolist()+np.log(mad).tolist())
    g=np.tile(np.array(globals,float),(len(f),1)); tester=np.eye(3)[f.tester_id.to_numpy(int)]; slot=np.column_stack([(f.slot_row.to_numpy(float)-1.5)/1.5,(f.slot_col.to_numpy(float)-1.5)/1.5]) if use_slot else np.zeros((len(f),2))
    if use48:
        vals=np.log(np.maximum(f[[f'{p}_48h' for p in PARAMS]].to_numpy(float),1e-9)); med=np.median(vals,axis=0); mad=np.maximum(1e-3,1.4826*np.median(np.abs(vals-med),axis=0)); extra=np.column_stack([(vals-med)/mad,np.ones(len(f))])
    else: extra=np.zeros((len(f),4))
    x=np.column_stack(blocks+[g,tester,slot,extra]).astype('float32')
    ydef=f.is_defect.to_numpy(np.float32); ytype=np.array([TYPES[x] for x in f.defect_type]); yparam=np.array([PARAMS.index(x) if x in PARAMS else -100 for x in f.defect_parameter]); target=np.log(np.maximum(f[[f'{p}_168h' for p in PARAMS]].to_numpy(float),1e-9)/np.maximum(f[[f'{p}_24h' for p in PARAMS]].to_numpy(float),1e-9)).astype('float32')
    return x,ydef,ytype,yparam,target

def batch_lots(rng,batch=8):
    lots=[]
    for i in range(batch): lots.append(sample_lot(rng,sample_prior(rng,cfg),cfg,f'TRAIN_{int(rng.integers(10**12))}'))
    feats=[lot_features(x) for x in lots]; m=max(len(x[0]) for x in feats); d=feats[0][0].shape[1]
    X=np.zeros((batch,m,d),np.float32); mask=np.ones((batch,m),bool); yd=np.zeros((batch,m),np.float32); yt=np.full((batch,m),-100,int); yp=np.full((batch,m),-100,int); yq=np.zeros((batch,m,3),np.float32)
    for i,(x,a,b,c,q) in enumerate(feats): n=len(x); X[i,:n]=x; mask[i,:n]=False; yd[i,:n]=a; yt[i,:n]=b; yp[i,:n]=c; yq[i,:n]=q
    return X,mask,yd,yt,yp,yq

def pinball(pred,target):
    qs=torch.tensor([.05,.5,.95],device=pred.device); e=target.unsqueeze(-1)-pred; return torch.maximum(qs*e,(qs-1)*e).mean()

def train():
    device='mps' if torch.backends.mps.is_available() else 'cpu'; torch.manual_seed(2026); rng=np.random.default_rng(2026)
    model=DriftPFN().to(device); params=sum(p.numel() for p in model.parameters()); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3); bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(35.,device=device)); ce=nn.CrossEntropyLoss(ignore_index=-100)
    start=time.perf_counter(); deadline=start+1800; lots_seen=0; steps=0; curve=[]; half_saved=False
    while time.perf_counter()<deadline and lots_seen<300000:
        X,mask,yd,yt,yp,yq=batch_lots(rng,8); X=torch.tensor(X,device=device); mask=torch.tensor(mask,device=device); valid=~mask; yd=torch.tensor(yd,device=device); yt=torch.tensor(yt,device=device); yp=torch.tensor(yp,device=device); yq=torch.tensor(yq,device=device)
        out=model(X,mask); loss=bce(out['defect'][valid],yd[valid])+0.7*pinball(out['quantiles'][valid],yq[valid])+0.3*ce(out['dtype'].reshape(-1,5),yt.reshape(-1))+0.15*ce(out['param'].reshape(-1,3),yp.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); lots_seen+=8; steps+=1
        if steps%25==0: curve.append({'lots_seen':lots_seen,'loss':float(loss.detach().cpu()),'time_s':time.perf_counter()-start}); print('lots',lots_seen,'loss',curve[-1]['loss'],flush=True)
        if not half_saved and time.perf_counter()-start>=900:
            torch.save({'model':model.state_dict(),'params':params,'lots_seen':lots_seen},OUT/'driftpfn_half.pt'); half_saved=True
    torch.save({'model':model.state_dict(),'params':params,'lots_seen':lots_seen},OUT/'driftpfn.pt'); pd.DataFrame(curve).to_csv(OUT/'driftpfn_training_curve.csv',index=False)
    return model,params,lots_seen,time.perf_counter()-start,device

@torch.no_grad()
def predict_lots(model,df,device,use_slot=True,no_context=False):
    # Evaluation frames preserve source row labels after lot splitting.  All
    # output arrays are positional, so normalize the index before grouping.
    df=df.reset_index(drop=True)
    score=np.zeros(len(df)); quant=np.zeros((len(df),3,3)); typ=np.zeros((len(df),5)); rows=[]
    model.eval()
    for _,idx0 in df.groupby('lot_id').groups.items():
        idx=np.asarray(list(idx0),int); g=df.loc[idx].reset_index(drop=True); x,*_=lot_features(g,use_slot=use_slot)
        if no_context:
            for jj,ii in enumerate(idx):
                xx=torch.tensor(x[jj:jj+1][None],device=device); o=model(xx,None); score[ii]=torch.sigmoid(o['defect'][0,0]).cpu(); quant[ii]=o['quantiles'][0,0].cpu(); typ[ii]=torch.softmax(o['dtype'][0,0],-1).cpu()
        else:
            xx=torch.tensor(x[None],device=device); o=model(xx,None); score[idx]=torch.sigmoid(o['defect'][0]).cpu().numpy(); quant[idx]=o['quantiles'][0].cpu().numpy(); typ[idx]=torch.softmax(o['dtype'][0],-1).cpu().numpy()
    return score,quant,typ

def eval_model(model,params,lots_seen,train_time,device):
    runs=[]; lat=[]; confusion=np.zeros((5,5),int)
    for seed in cfg['simulation']['seeds']:
        sets=generate_evaluation(cfg,int(seed)); train,cal,tin,tsh=impute(sets['train'],[sets['train'],sets['calibration'],sets['test_inprior'],sets['test_shifted']],cfg); cs,cq,ct=predict_lots(model,cal,device)
        run={}
        for setname,test in [('inprior',tin),('shifted',tsh)]:
            st=time.perf_counter(); s,q,t=predict_lots(model,test,device); sample=test.groupby('lot_id').head(100); ss=time.perf_counter(); predict_lots(model,sample,device); lat.append((time.perf_counter()-ss)*1000)
            p=pvalues_from_calibration(cs[cal.is_defect.to_numpy()==0],s); y=test.is_defect.to_numpy(); dt=test.defect_type.to_numpy(); ops={}
            for a in [.001,.005,.01]:
                fl=p<=a; ops[a]={'far':100*np.mean(fl[y==0]),'recall':100*np.mean(fl[y==1]) if np.any(y==1) else np.nan,'by':{d:100*np.mean(fl[dt==d]) if np.any(dt==d) else np.nan for d in ['D1_high_in_limit','D2_fast_drifter','D3_late_accelerator','D4_multivariate']}}
            maes={}; widths={}; cover=[]; lotcov=[]
            for pi,pn in enumerate(PARAMS):
                pred=test[f'{pn}_24h'].to_numpy()*np.exp(q[:,pi,1]); lo=test[f'{pn}_24h'].to_numpy()*np.exp(q[:,pi,0]); hi=test[f'{pn}_24h'].to_numpy()*np.exp(q[:,pi,2]); yy=test[f'{pn}_168h'].to_numpy(); maes[pn]=np.mean(np.abs(yy-pred)); widths[pn]=np.mean(hi-lo); cover.append((yy>=lo)&(yy<=hi))
            for lot,idx0 in test.groupby('lot_id').groups.items(): idx=np.asarray(list(idx0),int); lotcov.append(100*np.mean(np.concatenate([c[idx] for c in cover])))
            predtype=np.argmax(t,axis=1); true=np.array([TYPES[x] for x in test.defect_type]);
            for a,b in zip(true,predtype): confusion[a,b]+=1
            run[setname]={'pr_auc':average_precision_score(y,s),'ops':ops,'mae':maes,'coverage':100*np.mean(np.concatenate(cover)),'worst':min(lotcov),'width':widths}
        runs.append(run)
    def ms(vals): return {'mean':float(np.nanmean(vals)),'sd':float(np.nanstd(vals,ddof=1))}
    out={'status':'partial','reason':'Smallest compliant model trained for the 30 minute fallback budget, far below the requested 300000 lots','params':params,'lots_seen':lots_seen,'train_time_s':train_time,'device':device,'results':{},'type_confusion':confusion.tolist(),'latency_ms_per_lot_of_100':ms(lat)}
    for st in ['inprior','shifted']:
        out['results'][st]={'pr_auc':ms([r[st]['pr_auc'] for r in runs]),'ops':{str(a):{'far':ms([r[st]['ops'][a]['far'] for r in runs]),'recall':ms([r[st]['ops'][a]['recall'] for r in runs]),'by':{d:float(np.nanmean([r[st]['ops'][a]['by'][d] for r in runs])) for d in ['D1_high_in_limit','D2_fast_drifter','D3_late_accelerator','D4_multivariate']}} for a in [.001,.005,.01]},'mae':{p:ms([r[st]['mae'][p] for r in runs]) for p in PARAMS},'coverage':ms([r[st]['coverage'] for r in runs]),'worst_lot_coverage_pct':float(np.mean([r[st]['worst'] for r in runs])),'mean_width':{p:float(np.mean([r[st]['width'][p] for r in runs])) for p in PARAMS}}
    return out

def main():
    try:
        model,params,lots_seen,train_time,device=train(); out=eval_model(model,params,lots_seen,train_time,device)
    except Exception as e: out={'status':'skipped','error':f'{type(e).__name__}: {e}','params':0,'lots_seen':0,'train_time_s':0,'device':'mps' if torch.backends.mps.is_available() else 'cpu'}
    (OUT/'driftpfn_result.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
