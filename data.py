import json

import numpy as np
from datasets import load_dataset


def minors(s):
    """Return the sexual/minors score from an openai_moderation JSON string."""
    return dict(json.loads(s))["sexual/minors"]


def load(split, n=None, seed=0, keep_all=False):
    """Return ids, prompts, and labels from a ToxicChat split, dropping sexual/minors > 0.1 unless keep_all."""
    d = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
    idx = np.arange(len(d))
    if not keep_all:
        idx = idx[[minors(s) <= 0.1 for s in d["openai_moderation"]]]
    if n is not None:
        idx = np.sort(np.random.RandomState(seed).choice(idx, n, replace=False))
    x = d[idx.tolist()]
    ids = ["{}-{}".format(split, i) for i in idx]
    return ids, x["user_input"], np.array(x["toxicity"], dtype=int)
