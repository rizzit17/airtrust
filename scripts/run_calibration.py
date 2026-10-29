"""Predeclared calibration ablation; no test-based parameter tuning."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from airtrust.data import ROOT,prepare
from airtrust.experiment import train_one
out=ROOT/'results'/'calibration';out.mkdir(exist_ok=True)
for seed in (17,29,43):
    data=prepare(seed,calibrate=True)
    (out/f'audit_{seed}.json').write_text(json.dumps(data['audit'],indent=2))
    for kind in ('graph','lstm'):
        for method in ('centralized','fedavg','trust'):
            target=out/f'metrics_{kind}_{method}_{seed}.csv'
            if target.exists():continue
            print(seed,kind,method,flush=True)
            rows,_,_,_,_=train_one(data,method,kind,20,12,seed)
            pd.DataFrame(rows).to_csv(target,index=False)
df=pd.concat([pd.read_csv(p) for p in out.glob('metrics_*.csv')])
df.to_csv(out/'metrics.csv',index=False)
df.groupby(['model','method','split']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),r2_mean=('r2','mean'),r2_sd=('r2','std')).reset_index().to_csv(out/'summary.csv',index=False)
