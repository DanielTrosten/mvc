import numpy as np
import torch as th

import config


def _load_npz(name):
    return np.load(config.DATA_DIR / "processed" / f"{name}.npz")


def _fix_labels(l):
    uniq = np.unique(l)[None, :]
    new = (l[:, None] == uniq).argmax(axis=1)
    return new


def load_dataset(name, n_samples=None, select_views=None, select_labels=None, label_counts=None, noise_sd=None,
                 noise_views=None, to_dataset=True, **kwargs):
    npz = _load_npz(name)
    labels = npz["labels"]
    views = [npz[f"view_{i}"] for i in range(npz["n_views"])]

    if select_labels is not None:
        mask = np.isin(labels, select_labels)
        labels = labels[mask]
        views = [v[mask] for v in views]
        labels = _fix_labels(labels)

    if label_counts is not None:
        idx = []
        unique_labels = np.unique(labels)
        assert len(unique_labels) == len(label_counts)
        for l, n in zip(unique_labels, label_counts):
            _idx = np.random.choice(np.where(labels == l)[0], size=n, replace=False)
            idx.append(_idx)

        idx = np.concatenate(idx, axis=0)
        labels = labels[idx]
        views = [v[idx] for v in views]

    if n_samples is not None:
        idx = np.random.choice(labels.shape[0], size=min(labels.shape[0], int(n_samples)), replace=False)
        labels = labels[idx]
        views = [v[idx] for v in views]

    if select_views is not None:
        if not isinstance(select_views, (list, tuple)):
            select_views = [select_views]
        views = [views[i] for i in select_views]

    if noise_sd is not None:
        assert noise_views is not None, "'noise_views' has to be specified when 'noise_sd' is not None."
        if not isinstance(noise_views, (list, tuple)):
            noise_views = [int(noise_views)]
        for v in noise_views:
            views[v] += np.random.normal(loc=0, scale=float(noise_sd), size=views[v].shape)

    views = [v.astype(np.float32) for v in views]
    if to_dataset:
        dataset = th.utils.data.TensorDataset(*[th.Tensor(v).to(config.DEVICE, non_blocking=True) for v in views],
                                              th.Tensor(labels).to(config.DEVICE, non_blocking=True))
    else:
        dataset = (views, labels)
    return dataset
