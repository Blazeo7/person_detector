import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import Iterator

from dataset import Sample


def make_session_groups(samples: list[Sample]) -> np.ndarray:
    keys = [f"{s.person_id}_{s.session_id}" for s in samples]
    unique_keys = sorted(set(keys))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    return np.array([key_to_idx[k] for k in keys])


def session_kfold_splits(samples: list[Sample], n_splits: int = 5) -> Iterator[tuple]:
    groups = make_session_groups(samples)

    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    dummy_X = np.zeros((len(samples), 1))
    y = np.array([s.label.value for s in samples])

    for train_idx, val_idx in sgkf.split(dummy_X, y, groups):
        yield train_idx, val_idx
