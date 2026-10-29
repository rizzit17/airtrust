"""Inference-only export of SpatialGraph; training remains in models.py/PyTorch."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd


class NumpySpatialGraph:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.weights = {key: archive[key] for key in archive.files}

    def __call__(self, x, context):
        w = self.weights
        def linear(a, name):
            return a @ w[name + '.weight'].T + w[name + '.bias']
        msg = np.maximum(linear(x, 'message.0'), 0)
        msg = np.maximum(linear(msg, 'message.2'), 0)
        logits = linear(msg, 'attention')[..., 0] - np.linalg.norm(x[:, :, -2:], axis=-1)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        pooled = (msg * weights[:, :, None]).sum(axis=1)
        base = (x[:, :, 5] * weights).sum(axis=1)
        hidden = np.maximum(linear(np.concatenate((pooled, context), axis=1), 'head.0'), 0)
        return base + linear(hidden, 'head.2')[:, 0]


@lru_cache(maxsize=12)
def load_graph(path):
    return NumpySpatialGraph(path)


def load_snapshot(directory, seed):
    directory = Path(directory)
    metadata = json.loads((directory / f'data_{seed}.json').read_text())
    with np.load(directory / f'data_{seed}.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data.update(stations=pd.DataFrame(metadata['stations']).set_index('station_id'),
                times=pd.DatetimeIndex(metadata['times']),
                bounds=metadata['bounds'], audit=metadata['audit'])
    return data
