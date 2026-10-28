import copy
import numpy as np
import pytest
import torch
from airtrust.data import prepare
from airtrust.models import aggregate, SpatialGraph

@pytest.fixture(scope='module')
def data(): return prepare(17)

def test_sample_weight_and_trust_weight_equations():
    states=[{'w':torch.tensor([2.,4.])},{'w':torch.tensor([10.,12.])}]
    assert torch.allclose(aggregate(states,[3,1])['w'],torch.tensor([4.,6.]))
    assert torch.allclose(aggregate(states,[3*.1,1*.9])['w'],torch.tensor([8.,10.]))
    with pytest.raises(ValueError): aggregate(states,[0,0])

def test_robust_aggregators_and_krum_validity():
    states=[{'w':torch.tensor([v])} for v in [1.,2.,3.,4.,100.]]
    assert aggregate(states,[1]*5,'median')['w'].item()==3
    assert aggregate(states,[1]*5,'trimmed')['w'].item()==3
    assert aggregate(states,[1]*5,'krum',f=1)['w'].item()<100
    with pytest.raises(ValueError): aggregate(states[:4],[1]*4,'krum',f=1)

def test_unseen_stations_never_used_in_training_or_donors(data):
    held={i for i,s in enumerate(data['stations'].index) if s in data['audit']['heldout_stations']}
    assert held
    for name in ['train','validation','test','nonmember']:
        assert not held.intersection(set(data['splits'][name]['s']))
    assert set(data['splits']['spatial']['s'])==held
    for s,ids in data['donors'].items():
        assert s not in ids
        assert not held.intersection(ids)
        assert all(data['groups'][j]==data['groups'][s] for j in ids)

def test_chronology_and_membership_holdout(data):
    split=data['splits']; calibration,train_end,test_start=data['bounds']
    assert split['train']['t'].min()>=calibration
    assert split['train']['t'].max()<train_end
    assert split['validation']['t'].min()>=train_end
    assert split['validation']['t'].max()<test_start
    assert split['spatial']['t'].min()>=test_start
    members=set(zip(split['train']['s'],split['train']['t']))
    nonmembers=set(zip(split['nonmember']['s'],split['nonmember']['t']))
    assert not members.intersection(nonmembers)

def test_heldout_truth_perturbation_cannot_change_training(monkeypatch,data):
    import airtrust.data as module
    p,stations,audit=module.load_delhi()
    p.loc[:,data['audit']['heldout_stations']]+=500.
    monkeypatch.setattr(module,'load_delhi',lambda:(p,stations,audit))
    other=module.prepare(17)
    for key in ['x','own','context','y','truth','t','s','client']:
        np.testing.assert_allclose(data['splits']['train'][key],other['splits']['train'][key],equal_nan=True)
    np.testing.assert_allclose([p['confidence'] for p in data['profiles']],[p['confidence'] for p in other['profiles']])

def test_model_is_inductive_and_donor_order_invariant():
    torch.manual_seed(12); model=SpatialGraph().eval()
    x=torch.rand(7,3,8); c=torch.rand(7,4)
    with torch.no_grad():
        a=model(x,c); b=model(x[:,[2,0,1]],c)
    assert a.shape==(7,)
    assert torch.allclose(a,b,atol=1e-6)

def test_clean_labels_are_original_unimputed_observations(data):
    for name in ['train','validation','test','spatial']:
        d=data['splits'][name]
        assert np.isfinite(d['x']).all()
        truth=data['reference'][d['t'],d['s']]/100.
        np.testing.assert_allclose(d['truth'],truth)
        assert np.isfinite(truth).all()

def test_distinct_station_locations(data):
    dist=data['distance'].copy(); np.fill_diagonal(dist,np.inf)
    assert dist.min()>=.05

def test_uci_hour_grid_preserves_missing_hours():
    from airtrust.data import load_beijing
    p=load_beijing()
    assert p.shape==(35064,12)
    assert len(p)*len(p.columns)==420768
    assert np.all(np.diff(p.index.asi8)==3600*10**9)

def test_saved_model_reproduces_exported_predictions(data):
    import pandas as pd
    from airtrust.data import ROOT
    from airtrust.experiment import tensors,predict
    model=SpatialGraph()
    model.load_state_dict(torch.load(ROOT/'results'/'models'/'graph_trust_17.pt',weights_only=True,map_location='cpu'))
    d,x,c,_=tensors(data['splits']['spatial'],'graph')
    predictions=predict(model,x[:100],c[:100])*100
    exported=pd.read_csv(ROOT/'results'/'predictions.csv.gz')
    exported=exported[(exported.seed==17)&(exported.model=='graph')&(exported.method=='trust')&(exported.split=='spatial')]
    np.testing.assert_allclose(np.maximum(predictions,0),exported.prediction.to_numpy()[:100],rtol=1e-5,atol=1e-3)
