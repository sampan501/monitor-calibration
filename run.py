import argparse
import codecs
import json
import os

import numpy as np

import scorers
from data import load


def build(a):
    """Return the scorer named on the command line."""
    if a.scorer == "baseline":
        return scorers.Baseline()
    if a.scorer == "hf":
        return scorers.HF(a.model, yes=a.yes, no=a.no, raw=a.raw, prefix=a.prefix)
    if a.scorer == "openai":
        return scorers.OpenAICompat(a.model, a.base_url, a.api_key_env or "OPENAI_API_KEY", a.mode)
    if a.scorer == "anthropic":
        return scorers.Anthropic(a.model, a.api_key_env or "ANTHROPIC_API_KEY")
    return scorers.Jev()


def done(path):
    """Return the (id, rep) pairs already in a results file."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["id"], r["rep"]) for r in map(json.loads, f) if r}


def main():
    ap = argparse.ArgumentParser(description="Score ToxicChat test prompts and append P(harmful) to results.")
    ap.add_argument("--scorer", required=True, choices=["baseline", "hf", "openai", "anthropic", "jev"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key-env")
    ap.add_argument("--mode", default="logprob", choices=["logprob", "verbal"])
    ap.add_argument("--yes", nargs="+", default=["Yes", "yes"])
    ap.add_argument("--no", nargs="+", default=["No", "no"])
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    a.prefix = codecs.decode(a.prefix, "unicode_escape")
    if a.scorer in ("hf", "openai", "anthropic") and not a.model:
        ap.error("--model is required for --scorer {}".format(a.scorer))

    ids, x, y = load("test", a.n, a.seed)
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "{}.jsonl".format(a.name))
    skip = done(path)
    s = build(a)
    with open(path, "a") as f:
        for i in range(len(ids)):
            for r in range(a.repeats):
                if (ids[i], r) in skip:
                    continue
                try:
                    p = s.score(x[i])
                except Exception as e:
                    print("{} rep {}: {}".format(ids[i], r, e))
                    continue
                if np.isnan(p):
                    print("{} rep {}: nan".format(ids[i], r))
                    continue
                f.write(json.dumps({"id": ids[i], "rep": r, "p": p, "y": int(y[i])}) + "\n")
                f.flush()
            if (i + 1) % 100 == 0:
                print("{}/{}".format(i + 1, len(ids)))


if __name__ == "__main__":
    main()
