"""Build a compact, self-contained Vercel deployment from verified local outputs.

Run after prepare_local_data.py and package_submission.py. The generated directory
is ignored: it contains copies, not a second source of truth. No training runs here.
"""
import hashlib
import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from airtrust.data import prepare
from airtrust.server import context_pollutant

OUT = ROOT / 'deployment' / 'vercel'
OUT.mkdir(parents=True, exist_ok=True)


def copy(relative):
    destination = OUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, destination)


for name in ['__init__.py', 'data.py', 'server.py', 'inference.py']:
    copy('airtrust/' + name)
shutil.copytree(ROOT / 'web', OUT / 'web', dirs_exist_ok=True)
for name in ['manifest.json', 'expansion.json']:
    copy('datasets/' + name)
for relative in ['results/run.json', 'results/history.json', 'results/predictions.csv.gz']:
    copy(relative)
for folder in ['', 'spatial_cv', 'calibration', 'sensitivity']:
    for source in (ROOT / 'results' / folder).glob('*.csv'):
        copy(str(source.relative_to(ROOT)))
for seed in [17, 29, 43]:
    copy(f'results/audit_{seed}.json')
runtime = OUT / 'runtime'
(runtime / 'models').mkdir(parents=True, exist_ok=True)
for seed in [17, 29, 43]:
    data = prepare(seed)
    arrays = {key: data[key] for key in ['reference', 'noisy', 'filled', 'xy', 'groups']}
    np.savez_compressed(runtime / f'data_{seed}.npz', **arrays)
    metadata = dict(stations=data['stations'].reset_index().to_dict(orient='records'),
                    times=[str(t) for t in data['times']], bounds=data['bounds'], audit=data['audit'])
    (runtime / f'data_{seed}.json').write_text(json.dumps(metadata), encoding='utf-8')
    for method in ['trust', 'fedavg', 'centralized', 'median']:
        state = torch.load(ROOT / 'results' / 'models' / f'graph_{method}_{seed}.pt',
                           map_location='cpu', weights_only=True)
        np.savez_compressed(runtime / 'models' / f'graph_{method}_{seed}.npz',
                            **{key: value.numpy() for key, value in state.items()})
for parameter in ['pm10', 'no2']:
    (runtime / f'context_{parameter}.json').write_text(json.dumps(context_pollutant(parameter)), encoding='utf-8')

exports = {'metrics':'results/metrics.csv', 'summary':'results/summary.csv',
           'privacy':'results/privacy.csv', 'communication':'results/communication.csv',
           'robustness':'results/robustness.csv', 'predictions':'results/predictions.csv.gz',
           'dataset':'datasets/raw/delhi_pm25_hourly_2024.csv', 'manifest':'datasets/manifest.json',
           'report':'docs/IMPLEMENTATION_REPORT.md', 'datasets_zip':'submission/datasets_used.zip',
           'project_zip':'submission/airtrust_project.zip'}
public = OUT / 'public' / 'downloads'
public.mkdir(parents=True, exist_ok=True)
downloads = {}
for key, relative in exports.items():
    source = ROOT / relative
    if not source.exists(): raise FileNotFoundError(f'Build submission packages first: {source}')
    filename=source.name
    if key == 'dataset':
        filename += '.gz'
        with source.open('rb') as src, gzip.open(public / filename, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        # Remove only this generated uncompressed duplicate from earlier builds.
        duplicate=public/source.name
        if duplicate.exists(): duplicate.unlink()
    else:
        shutil.copy2(source, public / filename)
    downloads[key] = filename
(runtime / 'downloads.json').write_text(json.dumps(downloads), encoding='utf-8')
(OUT / 'app.py').write_text("import os\nfrom pathlib import Path\nos.environ['AIRTRUST_RUNTIME_BUNDLE'] = str(Path(__file__).resolve().parent / 'runtime')\nfrom airtrust.server import app\n", encoding='utf-8')
(OUT / 'requirements.txt').write_text('numpy==2.4.3\npandas==2.3.3\nfastapi==0.117.1\nuvicorn>=0.34,<1\n', encoding='utf-8')
(OUT / '.python-version').write_text('3.13\n', encoding='utf-8')
(OUT / 'vercel.json').write_text(json.dumps({'$schema':'https://openapi.vercel.sh/vercel.json',
    'framework':'fastapi', 'functions':{'app.py':{'maxDuration':60}},
    'headers':[{'source':'/api/(.*)', 'headers':[{'key':'Cache-Control','value':'no-store'}]}]}, indent=2), encoding='utf-8')
(OUT / '.vercelignore').write_text('__pycache__\n*.pyc\n.vercel\n', encoding='utf-8')
manifest = {str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.rglob('*') if p.is_file() and '.vercel' not in p.parts and p.name != 'hosting_manifest.json'}
(OUT / 'hosting_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f'Hosting package: {OUT}; {len(manifest)} files, {sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())/1e6:.1f} MB')
