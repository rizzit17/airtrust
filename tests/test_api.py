from fastapi.testclient import TestClient
from airtrust.server import app

client=TestClient(app)

def test_health_and_measured_results():
    assert client.get('/').status_code==200
    r=client.get('/api/overview'); assert r.status_code==200
    d=r.json(); assert len(d['run']['seeds'])>=3
    assert d['audit']['heldout_stations']
    assert len(d['summary'])>10

def test_local_map_inference_and_bounds():
    r=client.get('/api/map?hour=8000&method=trust'); assert r.status_code==200
    d=r.json(); assert len(d['points'])==784
    assert any(p['pm25'] is not None for p in d['points'])
    assert all(p['pm25'] is None or p['pm25']>=0 for p in d['points'])
    assert client.get('/api/map?hour=1').status_code==400
    assert client.get('/api/map?method=unknown').status_code==400

def test_download_allowlist_and_station_lookup():
    assert client.get('/api/download/summary').status_code==200
    assert client.get('/api/download/unknown').status_code==404
    assert client.get('/api/series/unknown').status_code==404
    stations=client.get('/api/stations').json()
    d=client.get('/api/series/'+stations[0]['station_id']).json()
    assert len(d)==366
