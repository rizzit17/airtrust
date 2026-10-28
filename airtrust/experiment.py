from pathlib import Path
import copy, json, time, platform
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from scipy.stats import wilcoxon
from .data import ROOT, prepare, load_beijing
from .models import SpatialGraph, TemporalLSTM, aggregate

def metrics(y,p):
    y=np.asarray(y)*100.; p=np.maximum(np.asarray(p)*100.,0)
    return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)),n=len(y))

def tensors(split,model):
    key='own' if model=='lstm' else 'x'
    valid=np.isfinite(split[key]).reshape(len(split['y']),-1).all(1)
    d={key_:v[valid] for key_,v in split.items()}
    return d,torch.from_numpy(d[key]),torch.from_numpy(d['context']),torch.from_numpy(d['y'].astype('float32'))

def predict(model,x,c):
    model.eval()
    with torch.no_grad(): return torch.cat([model(x[i:i+4096],c[i:i+4096]) for i in range(0,len(x),4096)]).numpy()

def train_one(data,method='trust',kind='graph',rounds=20,steps=12,seed=17,save_dir=None):
    started=time.perf_counter(); torch.manual_seed(seed); torch.set_num_threads(2)
    rng=np.random.default_rng(seed)
    make=TemporalLSTM if kind=='lstm' else SpatialGraph
    model=make()
    td,x,c,y=tensors(data['splits']['train'],kind)
    vd,vx,vc,_=tensors(data['splits']['validation'],kind)
    client_ids=[np.where(td['client']==g)[0] for g in range(len(data['profiles']))]
    sizes=np.array([len(ids) for ids in client_ids])
    confidences=np.array([p['confidence'] for p in data['profiles']])
    history=[]; best=float('inf'); best_state=None; last_updates=None
    def optimize(local,ids,count):
        opt=torch.optim.Adam(local.parameters(),lr=.002)
        local.train()
        for _ in range(count):
            idx=torch.from_numpy(rng.choice(ids,size=min(128,len(ids)),replace=False))
            pred=local(x[idx],c[idx]); loss=((pred-y[idx])**2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(local.parameters(),5.)
            opt.step()
    for r in range(rounds):
        before=copy.deepcopy(model.state_dict()); states=[]
        if method=='centralized':
            optimize(model,np.arange(len(y)),steps*len(sizes))
            weights=sizes/sizes.sum()
        else:
            for ids in client_ids:
                local=make(); local.load_state_dict(before); optimize(local,ids,steps)
                states.append(copy.deepcopy(local.state_dict()))
            weights=np.ones(len(sizes)) if method=='equal' else sizes.astype(float)
            if method=='trust': weights*=confidences
            if method=='reputation':
                # Independent update-consistency baseline: inverse distance to median update.
                flat=np.stack([np.concatenate([v.numpy().ravel() for v in s.values()]) for s in states])
                weights*=1/(np.linalg.norm(flat-np.median(flat,axis=0),axis=1)+1e-4)
            model.load_state_dict(aggregate(states,weights,method))
            weights=weights/weights.sum()
            last_updates=(before,states)
        vp=predict(model,vx,vc)
        score=metrics(vd['truth'],vp)['rmse']
        history.append(dict(round=r+1,validation_rmse=score,weights=weights.tolist()))
        if score<best:
            best=score; best_state=copy.deepcopy(model.state_dict()); best_round=r+1
    model.load_state_dict(best_state)
    results=[]; predictions=[]
    for name in ['test','spatial']:
        if kind=='lstm' and name=='spatial': continue
        d,xx,cc,_=tensors(data['splits'][name],kind)
        pred=predict(model,xx,cc)
        result=dict(seed=seed,model=kind,method=method,split=name,**metrics(d['truth'],pred))
        results.append(result)
        frame=pd.DataFrame(dict(seed=seed,model=kind,method=method,split=name,station_id=data['stations'].index[d['s']],timestamp=data['times'][d['t']].astype(str),truth=d['truth']*100,prediction=np.maximum(pred*100,0)))
        predictions.append(frame)
    # Attack sees the final-round individual model update and pre-update model.
    privacy=None
    if last_updates is not None:
        before,states=last_updates; member_scores=[]; nonmember_scores=[]
        for name,destination in [('train',member_scores),('nonmember',nonmember_scores)]:
            d,xx,cc,yy=tensors(data['splits'][name],kind)
            for g,state in enumerate(states):
                ids=np.where(d['client']==g)[0]
                if not len(ids): continue
                ids=rng.choice(ids,min(len(ids),300),replace=False)
                local=make(); local.load_state_dict(before)
                a=(predict(local,xx[ids],cc[ids])-yy[ids].numpy())**2
                local.load_state_dict(state)
                b=(predict(local,xx[ids],cc[ids])-yy[ids].numpy())**2
                destination.extend((a-b).tolist())
        labels=np.r_[np.ones(len(member_scores)),np.zeros(len(nonmember_scores))]
        scores=np.r_[member_scores,nonmember_scores]
        auc=roc_auc_score(labels,scores); fpr,tpr,_=roc_curve(labels,scores)
        privacy=dict(seed=seed,model=kind,method=method,attack='individual-update loss improvement',auc=float(auc),advantage=float(max(tpr-fpr)),members=len(member_scores),nonmembers=len(nonmember_scores),note='Random reserved windows in training era; overlapping windows remain correlated. No formal privacy guarantee.')
    params=sum(p.numel() for p in model.parameters())
    overhead=dict(seed=seed,model=kind,method=method,seconds=time.perf_counter()-started,parameters=params,best_round=best_round,rounds=rounds,local_steps=steps,upload_bytes=0 if method=='centralized' else rounds*len(sizes)*params*4,download_bytes=0 if method=='centralized' else rounds*len(sizes)*params*4,raw_training_bytes=int(x.numel()*4+c.numel()*4+y.numel()*4),note='Logical float32 tensor payload, excludes protocol overhead; simulation wall-clock, not network latency')
    if save_dir:
        save_dir=Path(save_dir); save_dir.mkdir(parents=True,exist_ok=True)
        torch.save(best_state,save_dir/f'{kind}_{method}_{seed}.pt')
    return results,pd.concat(predictions),history,privacy,overhead

def baselines(data,seed):
    results=[]; predictions=[]
    # Fit spatial covariance range to calibration-period semivariances of seen stations.
    seen=[i for i,s in enumerate(data['stations'].index) if s in data['audit']['seen_stations']]
    cal=data['noisy'][:data['bounds'][0],seen]
    distances=data['distance'][np.ix_(seen,seen)]
    semivar=np.nanmean((cal[:,:,None]-cal[:,None,:])**2,axis=0)/2
    sill=float(np.nanvar(cal)); best_range=20.; error=float('inf')
    for value in [5.,10.,20.,40.,80.]:
        e=np.nanmean((sill*(1-np.exp(-distances/value))-semivar)**2)
        if e<error: error=e; best_range=value
    for name in ['test','spatial']:
        d=data['splits'][name]; out_idw=np.empty(len(d['y'])); out_krig=np.empty(len(d['y']))
        for s in np.unique(d['s']):
            mask=d['s']==s; ids=data['donors'][int(s)]; tt=d['t'][mask]
            values=data['filled'][tt-1][:,ids]
            weights=1/np.maximum(data['distance'][s,ids],.1)**2; weights/=weights.sum()
            out_idw[mask]=values@weights/100.
            cov=np.exp(-data['distance'][np.ix_(ids,ids)]/best_range)+np.eye(len(ids))*.05
            rhs=np.r_[np.exp(-data['distance'][s,ids]/best_range),1.]
            mat=np.block([[cov,np.ones((len(ids),1))],[np.ones((1,len(ids))),np.zeros((1,1))]])
            kw=np.linalg.solve(mat,rhs)[:-1]
            out_krig[mask]=np.maximum(values@kw,0)/100.
        for method,pred in [('idw',out_idw),('kriging',out_krig)]:
            results.append(dict(seed=seed,model='spatial',method=method,split=name,**metrics(d['truth'],pred)))
            predictions.append(pd.DataFrame(dict(seed=seed,model='spatial',method=method,split=name,station_id=data['stations'].index[d['s']],timestamp=data['times'][d['t']].astype(str),truth=d['truth']*100,prediction=np.maximum(pred*100,0))))
    d=data['splits']['test']; valid=np.isfinite(d['own'][:,-1])
    results.append(dict(seed=seed,model='temporal',method='persistence',split='test',**metrics(d['truth'][valid],d['own'][valid,-1])))
    return results,pd.concat(predictions),best_range

def beijing_benchmark():
    # Separate benchmark: no unsupported zero-shot Delhi-to-Beijing transfer claim.
    p=load_beijing(); n=len(p); cut=int(.70*n); valid=int(.85*n)
    filled=p.ffill(limit=3)
    truth=p.iloc[valid:].to_numpy(); last=filled.shift(1).iloc[valid:].to_numpy()
    rows=[]
    for j,s in enumerate(p):
        good=np.isfinite(truth[:,j])&np.isfinite(last[:,j])
        rows.append(dict(station=s,**metrics(truth[good,j]/100,last[good,j]/100)))
    return dict(role='Independent temporal persistence sanity benchmark only; no federated cross-city transfer evaluated',rows=len(p)*len(p.columns),stations=len(p.columns),first=str(p.index[0]),last=str(p.index[-1]),train_end=str(p.index[cut]),test_start=str(p.index[valid]),station_metrics=rows,missing_fraction=float(p.isna().mean().mean()))

def run_suite(output='results',seeds=(17,29,43),rounds=20,steps=12,robustness=True):
    out=ROOT/output; out.mkdir(parents=True,exist_ok=True)
    records=[]; predictions=[]; histories={}; privacy=[]; costs=[]; sweep=[]
    methods=['centralized','equal','fedavg','trust','median','trimmed','krum','reputation']
    for seed in seeds:
        data=prepare(seed)
        (out/f'audit_{seed}.json').write_text(json.dumps(data['audit'],indent=2),encoding='utf-8')
        rr,pp,krange=baselines(data,seed); records.extend(rr); predictions.append(pp)
        for kind,method in [('graph',m) for m in methods]+[('lstm','fedavg'),('lstm','trust'),('lstm','centralized')]:
            print(f'Seed {seed} | {kind} | {method}',flush=True)
            rr,pp,hh,attack,cost=train_one(data,method,kind,rounds,steps,seed,out/'models')
            records.extend(rr); predictions.append(pp); histories[f'{seed}_{kind}_{method}']=hh
            if attack: privacy.append(attack)
            costs.append(cost)
            pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
        if robustness:
            for fraction in [0.,.2,.6]:
                scenario=prepare(seed,corruption=fraction)
                for method in ['fedavg','trust','median','trimmed']:
                    print(f'Robustness seed {seed} | corruption {fraction} | {method}',flush=True)
                    rr,_,_,_,_=train_one(scenario,method,'graph',rounds,steps,seed)
                    sweep.extend([dict(corruption=fraction,actual_corruption=round(8*fraction)/8,**row) for row in rr])
            sweep.extend([dict(corruption=.4,actual_corruption=3/8,**r) for r in records if r['seed']==seed and r['model']=='graph' and r['method'] in ['fedavg','trust','median','trimmed']])
        pd.DataFrame(sweep).to_csv(out/'robustness.csv',index=False)
        pd.DataFrame(privacy).to_csv(out/'privacy.csv',index=False)
        pd.DataFrame(costs).to_csv(out/'communication.csv',index=False)
        (out/'history.json').write_text(json.dumps(histories),encoding='utf-8')
    table=pd.DataFrame(records)
    summary=table.groupby(['model','method','split']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),r2_mean=('r2','mean'),r2_sd=('r2','std'),runs=('seed','count')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    pred=pd.concat(predictions); pred.to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    sig=[]
    for split in ['test','spatial']:
        a=table[(table.model=='graph')&(table.method=='trust')&(table.split==split)].sort_values('seed').rmse.to_numpy()
        b=table[(table.model=='graph')&(table.method=='fedavg')&(table.split==split)].sort_values('seed').rmse.to_numpy()
        sig.append(dict(split=split,paired_seeds=len(a),wilcoxon_p=float(wilcoxon(a,b).pvalue) if len(a)>1 and np.any(a!=b) else None,note='Low statistical power with three seeds; windows are not independent replicates'))
    info=dict(seeds=list(seeds),rounds=rounds,local_steps=steps,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,kriging_range_km=krange,krum_assumed_f=1,krum_warning='Krum theoretical condition does not cover 3/8 or 5/8 corrupted scenarios; empirical comparator only',significance=sig,beijing=beijing_benchmark())
    (out/'run.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('Suite complete',flush=True)
    return out
