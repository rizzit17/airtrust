import numpy as np
from fastapi.testclient import TestClient
from airtrust.data import prepare
from airtrust.server import app

def test_all_stations_held_out_exactly_once():
    held=[]
    for fold in range(5):
        d=prepare(17,fold=fold)
        ids=d['audit']['heldout_stations'];held.extend(ids)
        assert set(ids).isdisjoint(d['audit']['seen_stations'])
        used={str(d['stations'].index[j]) for donors in d['donors'].values() for j in donors}
        assert set(ids).isdisjoint(used)
        assert set(d['splits']['train']['s']).isdisjoint(d['splits']['spatial']['s'])
    assert len(held)==len(set(held))==36

def test_calibration_excludes_unseen_locations():
    d=prepare(17,calibrate=True)
    assert set(d['audit']['calibration_coefficients'])==set(d['audit']['seen_stations'])
    assert all(np.isfinite(c['slope']) for c in d['audit']['calibration_coefficients'].values())

def test_scenario_identity_and_bounds():
    c=TestClient(app);sid=c.get('/api/stations').json()[0]['station_id']
    r=c.get('/api/scenario',params=dict(station_id=sid,bias=0,gain=1,drift=0,noise=0,dropout=0))
    assert r.status_code==200
    assert r.json()['raw_rmse']==0
    assert r.json()['calibrated_rmse']<1e-5
    assert c.get('/api/scenario',params=dict(station_id=sid,dropout=2)).status_code==422
    assert c.get('/api/context/bad').status_code==400
    assert len(c.get('/api/quality').json())==36
