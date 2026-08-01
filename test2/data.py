"""Datasets for Test 2. Each returns (train_loader, test_loader, meta).

  * cifar10   -- torchvision, standard augmentation
  * covtype   -- sklearn forest cover-type tabular benchmark (54 feat, 7 class)
  * agnews    -- HuggingFace AG News, char-level (language task); falls back to a
                 synthetic associative-recall sequence task if offline
"""
from __future__ import annotations
import os, string
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_ROOT = os.environ.get("TEST2_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


def cifar10(batch=256, workers=2, subset=None):
    import torchvision
    import torchvision.transforms as T
    mean = (0.4914, 0.4822, 0.4465); std = (0.247, 0.243, 0.261)
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    root = os.path.join(DATA_ROOT, "cifar")
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te)
    if subset:
        train = torch.utils.data.Subset(train, range(subset))
    return (DataLoader(train, batch, shuffle=True, num_workers=workers, drop_last=True),
            DataLoader(test, 512, shuffle=False, num_workers=workers),
            {"in_dim": 3 * 32 * 32, "n_classes": 10, "kind": "image"})


def covtype(batch=512, subset=60000, seed=0):
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = fetch_covtype(return_X_y=True, data_home=os.path.join(DATA_ROOT, "sklearn"))
    y = y - 1
    rng = np.random.default_rng(seed)
    if subset and subset < len(X):
        idx = rng.choice(len(X), subset, replace=False)
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    def ds(X, y):
        return TensorDataset(torch.tensor(X, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.long))
    return (DataLoader(ds(Xtr, ytr), batch, shuffle=True, drop_last=True),
            DataLoader(ds(Xte, yte), 1024, shuffle=False),
            {"in_dim": X.shape[1], "n_classes": 7, "kind": "tabular"})


_VOCAB = ["<pad>"] + list(string.printable)
_CH2I = {c: i for i, c in enumerate(_VOCAB)}
SEQ_LEN = 128


def _encode(s, seq_len=SEQ_LEN):
    ids = [_CH2I.get(c, 0) for c in s[:seq_len]]
    ids += [0] * (seq_len - len(ids))
    return ids


def agnews(batch=128, subset=20000, seed=0):
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news")
        rng = np.random.default_rng(seed)
        def build(split, n):
            texts = split["text"]; labels = split["label"]
            idx = rng.choice(len(texts), min(n, len(texts)), replace=False)
            X = torch.tensor([_encode(texts[i]) for i in idx], dtype=torch.long)
            y = torch.tensor([labels[i] for i in idx], dtype=torch.long)
            return TensorDataset(X, y)
        train = build(ds["train"], subset)
        test = build(ds["test"], subset // 3)
        meta = {"vocab": len(_VOCAB), "seq_len": SEQ_LEN, "n_classes": 4,
                "kind": "text", "task": "agnews"}
        return (DataLoader(train, batch, shuffle=True, drop_last=True),
                DataLoader(test, 256, shuffle=False), meta)
    except Exception as e:
        print(f"[data] AG News unavailable ({e}); using synthetic associative recall")
        return assoc_recall(batch, subset, seed)


def assoc_recall(batch=128, subset=20000, seed=0, n_pairs=8, n_symbols=20):
    """Synthetic sequence task: [k1 v1 k2 v2 ... kP vP ? kq] -> predict v for kq.
    Tests the transformer's FFN capacity under different activations."""
    rng = np.random.default_rng(seed)
    seq_len = 2 * n_pairs + 2
    def gen(n):
        X = np.zeros((n, seq_len), dtype=np.int64)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            keys = rng.permutation(n_symbols)[:n_pairs]
            vals = rng.integers(0, n_symbols, n_pairs)
            seq = []
            for k, v in zip(keys, vals):
                seq += [k + 1, v + 1 + n_symbols]     # offset key/val ranges
            q = rng.integers(0, n_pairs)
            seq += [0, keys[q] + 1]                     # query marker + key
            X[i] = seq; y[i] = vals[q]
        return X, y
    Xtr, ytr = gen(subset); Xte, yte = gen(subset // 4)
    vocab = 2 * n_symbols + 2
    def ds(X, y):
        return TensorDataset(torch.tensor(X), torch.tensor(y))
    meta = {"vocab": vocab, "seq_len": seq_len, "n_classes": n_symbols,
            "kind": "text", "task": "assoc_recall"}
    return (DataLoader(ds(Xtr, ytr), batch, shuffle=True, drop_last=True),
            DataLoader(ds(Xte, yte), 256, shuffle=False), meta)


def mnist(batch=256, workers=2, subset=None):
    """Flattened MNIST as a clean, well-separated MLP benchmark (standard testbed
    in optical-NN papers). Signed, zero-mean standardised inputs."""
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    root = os.path.join(DATA_ROOT, "mnist")
    train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf)
    if subset:
        train = torch.utils.data.Subset(train, range(subset))
    return (DataLoader(train, batch, shuffle=True, num_workers=workers, drop_last=True),
            DataLoader(test, 512, shuffle=False, num_workers=workers),
            {"in_dim": 28 * 28, "n_classes": 10, "kind": "image"})


LOADERS = {"cifar10": cifar10, "covtype": covtype, "agnews": agnews,
           "assoc_recall": assoc_recall, "mnist": mnist}
