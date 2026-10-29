"""Five station folds x three seeds; each station held out exactly once per seed."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from airtrust.data import ROOT,prepare
from airtrust.experiment import train_one,baselines

def main():
    out=ROOT/'results'/'spatial_cv'; out.mkdir(exist_ok=True)
    for seed in (17,29,43):
        held=[]
        for fold in range(5):
            data=prepare(seed,fold=fold); held+=data['audit']['heldout_stations']
            (out/f'audit_{seed}_{fold}.json').write_text(json.dumps(data['audit'],indent=2))
            for method in ('baselines','centralized','fedavg','trust','median','trimmed','krum','reputation'):
                file=out/f'metrics_{seed}_{fold}_{method}.csv'
                if file.exists(): continue
                print(seed,fold,method,flush=True)
                if method=='baselines': rows,pred,_=baselines(data,seed)
                else: rows,pred,_,_,_=train_one(data,method,'graph',20,12,seed)
                pd.DataFrame([dict(fold=fold,**r) for r in rows]).to_csv(file,index=False)
                pred[pred.split=='spatial'].to_csv(out/f'predictions_{seed}_{fold}_{method}.csv.gz',index=False)
        assert len(held)==len(set(held))==len(data['stations']), 'Incomplete station coverage'
    df=pd.concat([pd.read_csv(p) for p in out.glob('metrics_*.csv')])
    df.to_csv(out/'metrics.csv',index=False)
    # Fold variability, not falsely described as independent sample-level confidence intervals.
    df.groupby(['model','method','split']).agg(mae_mean=('mae','mean'),mae_sd=('mae','std'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),r2_mean=('r2','mean'),r2_sd=('r2','std'),runs=('seed','size')).reset_index().to_csv(out/'summary.csv',index=False)
    print('Spatial CV complete',flush=True)

if __name__=='__main__': main()
