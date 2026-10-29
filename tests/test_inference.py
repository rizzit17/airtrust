"""Protect inference equivalence when deploying without the training framework."""
from pathlib import Path
import numpy as np
import pytest
import torch
from airtrust.inference import NumpySpatialGraph
from airtrust.models import SpatialGraph


@pytest.mark.parametrize('seed', [17, 29, 43])
@pytest.mark.parametrize('method', ['trust', 'fedavg', 'centralized', 'median'])
def test_numpy_export_matches_trained_pytorch_checkpoint(tmp_path, seed, method):
    checkpoint = Path(__file__).resolve().parents[1] / 'results' / 'models' / f'graph_{method}_{seed}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = SpatialGraph().eval()
    model.load_state_dict(state)
    exported = tmp_path / 'weights.npz'
    np.savez_compressed(exported, **{key: value.numpy() for key, value in state.items()})
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 8, (128, 3, 8)).astype('float32')
    x[:, :, -2:] = rng.uniform(-1, 1, (128, 3, 2))
    context = rng.uniform(-1, 1, (128, 4)).astype('float32')
    with torch.no_grad():
        expected = model(torch.from_numpy(x), torch.from_numpy(context)).numpy()
    actual = NumpySpatialGraph(exported)(x, context)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
