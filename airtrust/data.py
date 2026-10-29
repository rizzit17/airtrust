from pathlib import Path
import json, zipfile, io
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def load_delhi():
    raw=ROOT/'datasets'/'raw'
    stations=pd.DataFrame(json.loads((raw/'delhi_stations.json').read_text(encoding='utf-8'))['data'])
    stations=stations[['station_id','station_name','latitude','longitude']].dropna().set_index('station_id')
    df=pd.read_csv(raw/'delhi_pm25_hourly_2024.csv',usecols=['station_id','period_start','mean'])
    df['time']=pd.to_datetime(df.period_start)
    df['mean']=pd.to_numeric(df['mean'],errors='coerce')
    # Negative and >2000 ug/m3 readings are excluded; no clipping/imputation of labels.
    df.loc[~df['mean'].between(0,2000),'mean']=np.nan
    p=df.pivot_table(index='time',columns='station_id',values='mean',aggfunc='mean').reindex(pd.date_range('2024-01-01','2024-12-31 23:00',freq='h'))
    keep=[s for s in p if s in stations.index and p[s].notna().mean()>=.70]
    # Co-located catalog entries must not straddle seen and unseen evaluation sets.
    selected=[]; removed=[]
    for s in sorted(keep,key=lambda s:-int(p[s].notna().sum())):
        a=stations.loc[s]
        if any(np.hypot((a.latitude-stations.loc[j].latitude)*111.,(a.longitude-stations.loc[j].longitude)*97.5)<.05 for j in selected): removed.append(s)
        else: selected.append(s)
    keep=sorted(selected)
    p=p[keep]; stations=stations.loc[keep]
    return p,stations,dict(source_rows=len(df),retained_observations=int(p.notna().sum().sum()),coverage_threshold=.70,excluded_stations=int(len(pd.unique(df.station_id))-len(keep)),colocated_catalog_entries_excluded=removed,timezone='Asia/Kolkata')

def load_beijing():
    raw=ROOT/'datasets'/'raw'
    archive=raw/'beijing_station_data.zip'
    if not archive.exists(): archive=raw/'beijing_multi_site_uci.zip'
    frames=[]
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            if n.endswith('.csv') and 'PRSA_Data_' in n:
                d=pd.read_csv(io.BytesIO(z.read(n)))
                d['time']=pd.to_datetime(d[['year','month','day','hour']])
                frames.append(d[['time','station','PM2.5']])
    if not frames: raise ValueError('No station CSVs in UCI archive')
    df=pd.concat(frames)
    df.loc[~df['PM2.5'].between(0,2000),'PM2.5']=np.nan
    p=df.pivot_table(index='time',columns='station',values='PM2.5')
    return p.reindex(pd.date_range('2013-03-01','2017-02-28 23:00',freq='h'))

def client_partition(stations,k=8):
    # Geographic strips, deterministic, approximately balanced (at least 4 nodes/client).
    order=np.lexsort((stations.longitude.to_numpy(),stations.latitude.to_numpy()))
    groups=np.empty(len(stations),dtype=int)
    for g,ids in enumerate(np.array_split(order,k)): groups[ids]=g
    return groups

def simulate(p,groups,seed,corruption=.4,sensor_model='stress'):
    if sensor_model not in ('stress','kanpur_relative'): raise ValueError('Unknown sensor model')
    rng=np.random.default_rng(seed)
    ref=p.to_numpy(dtype=np.float32)
    k=int(max(groups)+1)
    bad=set(rng.permutation(k)[:round(k*corruption)].tolist())
    noisy=ref.copy(); profiles=[]
    clock=np.linspace(0,1,len(ref),dtype=np.float32)
    for g in range(k):
        corrupt=g in bad
        # Explicit scenario assumptions, not fitted or claimed hardware specifications.
        bias=35. if corrupt else 1.5
        gain=1.25 if corrupt else 1.02
        drift=30. if corrupt else 2.
        sigma=12. if corrupt else 3.
        ids=np.where(groups==g)[0]
        for s in ids:
            if sensor_model=='kanpur_relative':
                # Zheng 2018 Table 4 / abstract: 1h calibrated ratio error SD 35%/46%.
                # Mean-one lognormal is our nonnegative generative assumption, not their fitted distribution.
                relative_sd=.46 if corrupt else .35
                log_sd=np.sqrt(np.log1p(relative_sd**2))
                noisy[:,s]=ref[:,s]*rng.lognormal(-.5*log_sd**2,log_sd,len(ref))
                continue
            y=ref[:,s]*gain+bias+clock*drift+rng.normal(0,sigma,len(ref))
            dropout=rng.random(len(ref)) < (.04 if corrupt else .01)
            y[dropout]=np.nan
            if corrupt:
                for start in rng.integers(24,len(ref)-12,12): y[start:start+6]=y[start-1]
            noisy[:,s]=np.maximum(y,0)
        profiles.append(dict(client=g,corrupted=corrupt,bias_ugm3=bias,gain=gain,annual_drift_ugm3=drift,noise_sd_ugm3=sigma,dropout=.04 if corrupt else .01,stuck_blocks=12 if corrupt else 0))
        if sensor_model=='kanpur_relative':
            profiles[-1].update(bias_ugm3=0,gain=1,annual_drift_ugm3=0,noise_sd_ugm3=None,dropout=0,stuck_blocks=0,relative_error_sd=.46 if corrupt else .35,sensor_model=sensor_model)
    return noisy,profiles

def prepare(seed=17,corruption=.4,k=8,lookback=6,stride=3,fold=None,calibrate=False,sensor_model='stress'):
    p,stations,audit=load_delhi()
    n=len(p); bounds=(int(n*.10),int(n*.70),int(n*.85))
    groups=client_partition(stations,k)
    noisy,profiles=simulate(p,groups,seed,corruption,sensor_model)
    audit['sensor_model']=sensor_model
    # Select unseen nodes from geography; never use their readings as donors or labels.
    held=[]
    for g in range(k):
        ids=np.where(groups==g)[0]
        if fold is not None and fold not in range(5): raise ValueError('fold must be 0 through 4')
        if len(ids)>=4:
            if fold is None: held.append(int(ids[-1]))
            elif fold<len(ids): held.append(int(ids[fold]))
    seen=[i for i in range(len(stations)) if i not in held]
    audit['spatial_fold']=fold
    audit['input_calibration']=calibrate
    if calibrate:
        coefficients={}
        for s in seen:
            x=noisy[:bounds[0],s]; y=p.iloc[:bounds[0],s].to_numpy()
            valid=np.isfinite(x)&np.isfinite(y)
            if valid.sum()<30: raise ValueError('Insufficient co-located calibration pairs')
            slope,intercept=np.linalg.lstsq(np.column_stack((x[valid],np.ones(valid.sum()))),y[valid],rcond=None)[0]
            noisy[:,s]=np.maximum(slope*noisy[:,s]+intercept,0)
            coefficients[str(stations.index[s])]=dict(slope=float(slope),intercept=float(intercept),pairs=int(valid.sum()))
        audit['calibration_coefficients']=coefficients
    # Forward-fill inputs only, at most 3h; no backfill, no interpolated test truth.
    filled=pd.DataFrame(noisy).ffill(limit=3).to_numpy(dtype=np.float32)
    ref=p.to_numpy(dtype=np.float32)
    xy=np.column_stack(((stations.longitude-77.2)*97.5,(stations.latitude-28.65)*111.))
    distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    times=p.index
    clients=[]; donors={}; splits={key:[] for key in ['train','validation','test','spatial']}
    for s in range(len(stations)):
        candidates=[j for j in seen if groups[j]==groups[s] and j!=s]
        if len(candidates)<2: raise ValueError('Too few donor stations in client')
        ids=sorted(candidates,key=lambda j:distance[s,j])[:3]
        donors[s]=ids
        t=np.arange(max(bounds[0],lookback),n,stride)
        hist=np.stack([filled[t-lookback+i][:,ids] for i in range(lookback)],axis=1)
        # Graph messages: each donor's last six readings and relative location.
        rel=np.broadcast_to((xy[ids]-xy[s])[None,:,:]/50.,(len(t),len(ids),2))
        x=np.concatenate((hist.transpose(0,2,1)/100.,rel),axis=-1)
        # Pad to three neighbors using a repeated donor; graph pooling remains defined.
        if len(ids)<3: x=np.concatenate((x,x[:,-1:]),axis=1)
        own=np.stack([filled[t-lookback+i,s] for i in range(lookback)],axis=1)/100.
        hour=times[t].hour.to_numpy(); doy=times[t].dayofyear.to_numpy()
        context=np.column_stack((np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24),np.sin(doy*2*np.pi/366),np.cos(doy*2*np.pi/366)))
        valid=np.isfinite(x).all(axis=(1,2)) & np.isfinite(ref[t,s])
        rec=dict(x=x.astype('float32'),own=own.astype('float32'),context=context.astype('float32'),y=noisy[t,s]/100.,truth=ref[t,s]/100.,t=t,s=np.full(len(t),s),client=np.full(len(t),groups[s]))
        for name in splits:
            m=valid.copy()
            if name=='spatial': m &= (s in held)&(t>=bounds[2])
            else:
                m &= (s in seen)
                if name=='train': m &= (t<bounds[1])&np.isfinite(rec['y'])
                elif name=='validation': m &= (t>=bounds[1])&(t<bounds[2])
                else: m &= t>=bounds[2]
            if m.any(): splits[name].append({key:val[m] for key,val in rec.items()})
    for name,parts in splits.items():
        if not parts: raise ValueError(f'Empty split {name}')
        splits[name]={key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    # Random nonmember holdout from training era; never optimized on or used for selection.
    rng=np.random.default_rng(seed+901)
    train=splits['train']; reserve=rng.random(len(train['y']))<.1
    splits['nonmember']={key:v[reserve] for key,v in train.items()}
    splits['train']={key:v[~reserve] for key,v in train.items()}
    for g in range(k):
        idx=np.where((groups==g)&np.isin(np.arange(len(groups)),seen))[0]
        err=noisy[:bounds[0],idx]-ref[:bounds[0],idx]
        rmse=float(np.sqrt(np.nanmean(err**2)))
        profiles[g]['calibration_rmse']=rmse
        profiles[g]['confidence']=max(float(np.exp(-rmse/35.)),.05)
        profiles[g]['stations']=[str(stations.index[s]) for s in idx]
    audit.update(dict(stations=len(stations),clients=k,heldout_stations=[str(stations.index[s]) for s in held],seen_stations=[str(stations.index[s]) for s in seen],lookback_hours=lookback,stride_hours=stride,split_boundaries=[str(times[b]) for b in bounds],sample_counts={a:len(b['y']) for a,b in splits.items()},profiles=profiles,first_time=str(times[0]),last_time=str(times[-1])))
    return dict(splits=splits,stations=stations,groups=groups,profiles=profiles,audit=audit,reference=ref,noisy=noisy,filled=filled,xy=xy,distance=distance,times=times,donors=donors,bounds=bounds)
