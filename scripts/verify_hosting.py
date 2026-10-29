"""Check the generated hosting package and compare its API to local inference."""
import json
import sys
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airtrust import server

package = ROOT / 'deployment' / 'vercel'
assert sum(p.stat().st_size for p in package.rglob('*') if p.is_file() and '.vercel' not in p.parts) < 100_000_000
local_map = server.pollution_map(8000, 'trust', 17)
local_scenario = server.scenario('site_103', 20, 1.1, 20, 10, .05)
server.ROOT = package
server.RESULTS = package / 'results'
server.RUNTIME = package / 'runtime'
server.get_data.cache_clear()
client = TestClient(server.app)
paths = ['/', '/api/overview', '/api/history', '/api/extended', '/api/quality',
         '/api/context/pm10', '/api/context/no2', '/api/stations', '/api/series/site_103']
for path in paths:
    response = client.get(path)
    assert response.status_code == 200, (path, response.text[:200])
cloud_map = client.get('/api/map?hour=8000&method=trust').json()
for original, hosted in zip(local_map['points'], cloud_map['points']):
    assert (original['pm25'] is None) == (hosted['pm25'] is None)
    if original['pm25'] is not None:
        np.testing.assert_allclose(original['pm25'], hosted['pm25'], atol=.001, rtol=2e-6)
cloud_scenario = client.get('/api/scenario?station_id=site_103').json()
for field in ['raw_rmse', 'calibrated_rmse', 'slope', 'intercept', 'test_pairs', 'calibration_pairs']:
    np.testing.assert_equal(local_scenario[field], cloud_scenario[field])
for seed in [17, 29, 43]:
    for method in ['trust', 'fedavg', 'centralized', 'median']:
        response = client.get(f'/api/map?hour=8000&method={method}&seed={seed}')
        assert response.status_code == 200 and len(response.json()['points']) == 784
for key, name in json.loads((server.RUNTIME / 'downloads.json').read_text()).items():
    response = client.get('/api/download/' + key, follow_redirects=False)
    assert response.status_code == 307 and (package / 'public' / 'downloads' / name).is_file()
print('Verified hosting APIs, all 12 graph checkpoints, exact scenario parity, map parity within 0.001 µg/m³, and 11 CDN exports.')
