import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from airtrust.data import ROOT,prepare
from airtrust.experiment import train_one
out=ROOT/'results'/'sensitivity';out.mkdir(exist_ok=True)
for seed in (17,29,43):
    for scenario,methods in [('stress',['history_reputation']),('kanpur_relative',['centralized','fedavg','trust','median','history_reputation'])]:
        data=prepare(seed,sensor_model=scenario)
        (out/f'audit_{scenario}_{seed}.json').write_text(json.dumps(data['audit'],indent=2))
        for method in methods:
            target=out/f'metrics_{scenario}_{method}_{seed}.csv'
            if target.exists():continue
            print(seed,scenario,method,flush=True)
            rows,_,hist,_,_=train_one(data,method,'graph',20,12,seed)
            pd.DataFrame([dict(scenario=scenario,**r) for r in rows]).to_csv(target,index=False)
            (out/f'history_{scenario}_{method}_{seed}.json').write_text(json.dumps(hist))
df=pd.concat([pd.read_csv(p) for p in out.glob('metrics_*.csv')]);df.to_csv(out/'metrics.csv',index=False)
df.groupby(['scenario','method','split']).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),r2_mean=('r2','mean'),r2_sd=('r2','std')).reset_index().to_csv(out/'summary.csv',index=False)
