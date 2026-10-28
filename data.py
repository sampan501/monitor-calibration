import json
import os

import numpy as np
from datasets import load_dataset

DATASETS = ("toxicchat", "wildguardtest")
# WildGuardMix keeps its test and train sets in separate configs.
WG = {"test": ("wildguardtest", "test", "wgt"), "train": ("wildguardtrain", "train", "wgtrain")}


def minors(s):
    """Return the sexual/minors score from an openai_moderation JSON string."""
    return dict(json.loads(s))["sexual/minors"]


def cache_path(split):
    """Return the moderation cache path for a WildGuardMix split."""
    return os.path.join("data", "{}_moderation.json".format(WG[split][0]))


def wildguard(split):
    """Return the WildGuardMix split with unlabeled prompts dropped, and its row indices."""
    d = load_dataset("allenai/wildguardmix", WG[split][0], split=WG[split][1])
    idx = np.array([i for i, h in enumerate(d["prompt_harm_label"]) if h is not None])
    return d, idx


def load(split, n=None, seed=0, keep_all=False, dataset="toxicchat"):
    """Return ids, prompts, labels, and per-prompt extras, dropping sexual/minors > 0.1 unless keep_all."""
    if dataset == "wildguardtest":
        return load_wg(split, n, seed, keep_all)
    d = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
    idx = np.arange(len(d))
    if not keep_all:
        idx = idx[[minors(s) <= 0.1 for s in d["openai_moderation"]]]
    if n is not None:
        idx = np.sort(np.random.RandomState(seed).choice(idx, n, replace=False))
    x = d[idx.tolist()]
    ids = ["{}-{}".format(split, i) for i in idx]
    return ids, x["user_input"], np.array(x["toxicity"], dtype=int), [{}] * len(ids)


def load_wg(split, n=None, seed=0, keep_all=False):
    """Return a WildGuardMix split, filtered with the moderation cache from moderate.py."""
    d, idx = wildguard(split)
    tag = WG[split][2]
    y = np.array([h == "harmful" for h in d["prompt_harm_label"]], dtype=int)
    if not keep_all:
        path = cache_path(split)
        if not os.path.exists(path):
            raise SystemExit("no {}; run moderate.py --split {} first".format(path, split))
        m = json.load(open(path))
        miss = [i for i in idx if "{}-{}".format(tag, i) not in m]
        if miss:
            raise SystemExit("{} prompts missing from {}; rerun moderate.py".format(len(miss), path))
        drop = np.array([m["{}-{}".format(tag, i)] > 0.1 for i in idx])
        print("minors filter removed {} prompts, {} harmful".format(drop.sum(), y[idx[drop]].sum()))
        idx = idx[~drop]
    if n is not None:
        idx = np.sort(np.random.RandomState(seed).choice(idx, n, replace=False))
    x = d[idx.tolist()]
    y, adv = y[idx], np.array(x["adversarial"], dtype=bool)
    print(
        "{} rows, {} harmful ({} adversarial, {} plain)".format(
            len(idx), y.sum(), y[adv].sum(), y[~adv].sum()
        )
    )
    ids = ["{}-{}".format(tag, i) for i in idx]
    extra = [{"adv": bool(a), "sub": s} for a, s in zip(adv, x["subcategory"])]
    return ids, x["prompt"], y, extra
