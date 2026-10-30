"""Additional seed replications for second-level manuscript review.

Preserves the original three-seed CV outputs. Reuses their metric files and
executes the same protocol for twelve further seeds. No hyperparameter tuning.
"""
import sys, json, shutil, hashlib, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from airtrust.data import ROOT, prepare
from airtrust.experiment import train_one, baselines

SEEDS = (17, 29, 43, 59, 71, 89, 101, 127, 149, 173, 197, 223, 251, 277, 307)
METHODS = ('baselines', 'centralized', 'fedavg', 'trust', 'median', 'trimmed', 'krum', 'reputation')
OUT = ROOT/'results'/'paper_validation'

def run_seed(seed):
    for fold in range(5):
        missing = [m for m in METHODS if not (OUT/f'metrics_{seed}_{fold}_{m}.csv').exists()]
        if not missing: continue
        data = prepare(seed, fold=fold)
        (OUT/f'audit_{seed}_{fold}.json').write_text(json.dumps(data['audit'], indent=2))
        for method in missing:
            print(f'Seed {seed}, fold {fold}, {method}', flush=True)
            if method == 'baselines': rows, _, _ = baselines(data, seed)
            else: rows, _, _, _, _ = train_one(data, method, 'graph', 20, 12, seed)
            pd.DataFrame([dict(fold=fold, **r) for r in rows]).to_csv(OUT/f'metrics_{seed}_{fold}_{method}.csv', index=False)
    return seed

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = ROOT/'results'/'spatial_cv'
    for seed in SEEDS[:3]:
        for file in list(old.glob(f'metrics_{seed}_*.csv')) + list(old.glob(f'audit_{seed}_*.json')):
            if not (OUT/file.name).exists(): shutil.copy2(file, OUT/file.name)
    pending=[s for s in SEEDS if any(not (OUT/f'metrics_{s}_{f}_{m}.csv').exists() for f in range(5) for m in METHODS)]
    if pending:
        with ProcessPoolExecutor(max_workers=3) as pool:
            for seed in pool.map(run_seed, pending): print(f'Completed seed {seed}', flush=True)
    df = pd.concat([pd.read_csv(OUT/f'metrics_{s}_{f}_{m}.csv') for s in SEEDS for f in range(5) for m in METHODS], ignore_index=True)
    df.to_csv(OUT/'metrics.csv', index=False)
    df.groupby(['model','method','split']).agg(mae_mean=('mae','mean'),mae_sd=('mae','std'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),r2_mean=('r2','mean'),r2_sd=('r2','std'),runs=('seed','size')).reset_index().to_csv(OUT/'summary.csv',index=False)
    for seed in SEEDS:
        audits=[json.loads((OUT/f'audit_{seed}_{fold}.json').read_text()) for fold in range(5)]
        held=[s for a in audits for s in a['heldout_stations']]
        assert len(held)==len(set(held))==36
        assert all(set(a['heldout_stations']).isdisjoint(a['seen_stations']) for a in audits)
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['airtrust/data.py','airtrust/experiment.py','airtrust/models.py','datasets/manifest.json']}
    import torch
    (OUT/'protocol.json').write_text(json.dumps(dict(seeds=SEEDS, folds=5, rounds=20, local_steps=12, original_results_reused=list(SEEDS[:3]), new_seeds=list(SEEDS[3:]), hyperparameters_changed=False, python=platform.python_version(),torch=torch.__version__,source_sha256=sources,scope='Same Delhi year and fixed station folds; seed replication varies initialization, minibatches, synthetic faults and reserved nonmember windows. Not independent geographic replication.'), indent=2))
    print('Fifteen-seed validation complete', flush=True)

if __name__ == '__main__':
    if '--additional-only' in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=3) as pool:
            for seed in pool.map(run_seed, SEEDS[10:]): print(f'Completed additional seed {seed}', flush=True)
    else:
        main()
