from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .data import ROOT, prepare
from .models import SpatialGraph

app=FastAPI(title='AirTrust Delhi',version='1.0.0')
RESULTS=ROOT/'results'
app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')

@app.get('/')
def index(): return FileResponse(ROOT/'web'/'index.html')

def records(path):
    if not path.exists(): return []
    return json.loads(pd.read_csv(path).replace([np.inf,-np.inf],np.nan).to_json(orient='records'))

@lru_cache(maxsize=3)
def get_data(seed=17): return prepare(seed)

@app.get('/api/overview')
def overview():
    if not (RESULTS/'run.json').exists(): raise HTTPException(503,'Experiments are not complete. Run scripts/run_experiments.py first.')
    audit=json.loads((RESULTS/'audit_17.json').read_text())
    return dict(audit=audit,run=json.loads((RESULTS/'run.json').read_text()),summary=records(RESULTS/'summary.csv'),privacy=records(RESULTS/'privacy.csv'),communication=records(RESULTS/'communication.csv'),robustness=records(RESULTS/'robustness.csv'),datasets=json.loads((ROOT/'datasets'/'manifest.json').read_text()))

@app.get('/api/history')
def history(): return json.loads((RESULTS/'history.json').read_text())

@app.get('/api/extended')
def extended():
    path=ROOT/'datasets'/'expansion.json'
    cv=RESULTS/'spatial_cv'
    return dict(expansion=json.loads(path.read_text()) if path.exists() else None,cv=records(cv/'summary.csv'),calibration=records(RESULTS/'calibration'/'summary.csv'),sensitivity=records(RESULTS/'sensitivity'/'summary.csv'),completed_cv_fits=len(list(cv.glob('metrics_*.csv'))),expected_cv_fits=120)

@app.get('/api/quality')
def quality():
    data=get_data(); rows=[]
    for j,(sid,station) in enumerate(data['stations'].iterrows()):
        values=data['reference'][:,j]; good=np.isfinite(values)
        rows.append(dict(station_id=sid,name=station.station_name,valid=int(good.sum()),missing=int((~good).sum()),coverage=float(good.mean()),mean=float(np.nanmean(values)),p95=float(np.nanpercentile(values,95)),client=int(data['groups'][j])))
    return rows

@app.get('/api/context/{parameter}')
def context_pollutant(parameter:str):
    if parameter not in ['pm10','no2']: raise HTTPException(400,'Choose pm10 or no2')
    path=ROOT/'datasets'/'raw'/f'delhi_{parameter}_hourly_2024.csv'
    if not path.exists(): raise HTTPException(503,'Dataset download is incomplete')
    frame=pd.read_csv(path); frame['date']=pd.to_datetime(frame.period_start).dt.strftime('%Y-%m')
    frame['mean']=pd.to_numeric(frame['mean'],errors='coerce'); frame=frame[frame['mean']>=0]
    return json.loads(frame.groupby('date')['mean'].agg(['mean','count']).reset_index().to_json(orient='records'))

@app.get('/api/scenario')
def scenario(station_id:str,bias:float=Query(20,ge=-100,le=100),gain:float=Query(1.1,ge=.5,le=2),drift:float=Query(20,ge=-100,le=100),noise:float=Query(10,ge=0,le=50),dropout:float=Query(.05,ge=0,le=.5)):
    data=get_data()
    if station_id not in data['stations'].index: raise HTTPException(404,'Unknown station')
    j=data['stations'].index.get_loc(station_id); ref=data['reference'][:,j].astype(float); rng=np.random.default_rng(17)
    raw=np.maximum(ref*gain+bias+np.linspace(0,1,len(ref))*drift+rng.normal(0,noise,len(ref)),0)
    raw[rng.random(len(ref))<dropout]=np.nan
    cut=data['bounds'][0]; good=np.isfinite(ref[:cut])&np.isfinite(raw[:cut])
    if good.sum()<30: raise HTTPException(422,'Insufficient calibration pairs')
    slope,intercept=np.linalg.lstsq(np.column_stack((raw[:cut][good],np.ones(good.sum()))),ref[:cut][good],rcond=None)[0]
    calibrated=np.maximum(slope*raw+intercept,0); test=np.arange(len(ref))>=data['bounds'][2]; valid=test&np.isfinite(ref)&np.isfinite(raw)
    def score(v): return float(np.sqrt(np.mean((v[valid]-ref[valid])**2)))
    daily=pd.DataFrame(dict(reference=ref,simulated=raw,calibrated=calibrated),index=data['times']).resample('MS').mean().reset_index(names='date')
    return dict(raw_rmse=score(raw),calibrated_rmse=score(calibrated),test_pairs=int(valid.sum()),calibration_pairs=int(good.sum()),slope=float(slope),intercept=float(intercept),series=json.loads(daily.to_json(orient='records',date_format='iso')),note='Fixed seed 17; synthetic scenario, not a retrained FL model. Linear correction is fitted only on first 10% co-located pairs. Later drift can invalidate calibration.')

@app.get('/api/stations')
def stations():
    d=get_data(); s=d['stations'].reset_index(); s['client']=d['groups']; s['heldout']=s.station_id.isin(d['audit']['heldout_stations'])
    return json.loads(s.to_json(orient='records'))

@app.get('/api/series/{station_id}')
def series(station_id:str):
    d=get_data()
    if station_id not in d['stations'].index: raise HTTPException(404,'Unknown station')
    j=d['stations'].index.get_loc(station_id)
    pred=pd.read_csv(RESULTS/'predictions.csv.gz')
    pred=pred[(pred.station_id==station_id)&(pred.seed==17)&(pred.model=='graph')&(pred.method=='trust')]
    # Only test predictions; daily means are for compact display and are explicitly labelled.
    obs=pd.DataFrame(dict(time=d['times'],reference=d['reference'][:,j],simulated=d['noisy'][:,j])).set_index('time').resample('D').mean()
    if len(pred):
        pred['time']=pd.to_datetime(pred.timestamp)
        obs=obs.join(pred.set_index('time').prediction.resample('D').mean())
    return json.loads(obs.reset_index().to_json(orient='records',date_format='iso'))

@app.get('/api/map')
def pollution_map(hour:int=8000,method:str='trust',seed:int=17):
    if method not in ['trust','fedavg','centralized','median']: raise HTTPException(400,'Unsupported method')
    if seed not in [17,29,43]: raise HTTPException(400,'Unsupported seed')
    d=get_data(seed); low=d['bounds'][2]
    if hour<low or hour>=len(d['times']): raise HTTPException(400,f'Hour must be between {low} and {len(d["times"])-1}')
    path=RESULTS/'models'/f'graph_{method}_{seed}.pt'
    if not path.exists(): raise HTTPException(404,'Model is unavailable')
    model=SpatialGraph(); model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); model.eval()
    lat=np.linspace(d['stations'].latitude.min()-.01,d['stations'].latitude.max()+.01,28)
    lon=np.linspace(d['stations'].longitude.min()-.01,d['stations'].longitude.max()+.01,28)
    points=np.array([(a,b) for a in lat for b in lon])
    xy=np.column_stack(((points[:,1]-77.2)*97.5,(points[:,0]-28.65)*111.))
    seen=[i for i,s in enumerate(d['stations'].index) if s in d['audit']['seen_stations']]
    train_xy=d['xy'][seen]; dist=np.linalg.norm(xy[:,None]-train_xy[None],axis=-1)
    clients=d['groups'][np.array(seen)[dist.argmin(1)]]
    values=[]
    when=d['times'][hour]; h=when.hour; day=when.dayofyear
    context=np.array([[np.sin(h*2*np.pi/24),np.cos(h*2*np.pi/24),np.sin(day*2*np.pi/366),np.cos(day*2*np.pi/366)]],dtype='float32')
    for i in range(len(points)):
        candidates=[s for s in seen if d['groups'][s]==clients[i]]
        candidates=sorted(candidates,key=lambda s:np.linalg.norm(d['xy'][s]-xy[i]))
        ids=[]
        for s in candidates:
            if np.isfinite(d['filled'][hour-6:hour,s]).all(): ids.append(s)
            if len(ids)==3: break
        if len(ids)<2: values.append(None); continue
        while len(ids)<3: ids.append(ids[-1])
        x=np.concatenate((d['filled'][hour-6:hour,ids].T/100.,(d['xy'][ids]-xy[i])/50.),axis=1).astype('float32')
        with torch.no_grad(): value=float(model(torch.from_numpy(x[None]),torch.from_numpy(context))[0])*100
        values.append(max(value,0))
    stations=d['stations'].reset_index()
    stations['observed']=d['reference'][hour]; stations['simulated']=d['noisy'][hour]
    stations['heldout']=stations.station_id.isin(d['audit']['heldout_stations'])
    return dict(timestamp=str(when),hour=hour,method=method,points=[dict(latitude=float(p[0]),longitude=float(p[1]),pm25=v,nearest_station_km=float(dist[i].min())) for i,(p,v) in enumerate(zip(points,values))],stations=json.loads(stations.to_json(orient='records')),note='Historical one-hour-ahead model surface. Grids have no observed ground truth. Local clients use their donor histories and return predictions; no live readings or certified street-level resolution.')

@app.get('/api/download/{name}')
def download(name:str):
    allowed={'metrics':'results/metrics.csv','summary':'results/summary.csv','privacy':'results/privacy.csv','communication':'results/communication.csv','robustness':'results/robustness.csv','predictions':'results/predictions.csv.gz','dataset':'datasets/raw/delhi_pm25_hourly_2024.csv','manifest':'datasets/manifest.json','report':'docs/IMPLEMENTATION_REPORT.md','datasets_zip':'submission/datasets_used.zip','project_zip':'submission/airtrust_project.zip'}
    if name not in allowed: raise HTTPException(404,'Unknown export')
    path=ROOT/allowed[name]
    if not path.exists(): raise HTTPException(404,'Export has not been generated')
    return FileResponse(path,filename=path.name)
