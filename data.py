import numpy as np
from datasets import load_dataset


def load(split, n=None, seed=0):
    """Return ids, prompts, and labels from a ToxicChat split."""
    d = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
    idx = np.arange(len(d))
    if n is not None:
        idx = np.sort(np.random.RandomState(seed).choice(len(d), n, replace=False))
    x = d[idx.tolist()]
    ids = ["{}-{}".format(split, i) for i in idx]
    return ids, x["user_input"], np.array(x["toxicity"], dtype=int)
