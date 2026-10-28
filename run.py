import argparse
import codecs
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np

import scorers
from data import DATASETS, load


def build(a):
    """Return the scorer named on the command line."""
    if a.scorer == "baseline":
        return scorers.Baseline(a.dataset)
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


def record(a, s):
    """Add the run's model ID, scorer, mode, dataset, and first run date to models.json."""
    path = os.path.join("results", a.dataset, "models.json")
    m = json.load(open(path)) if os.path.exists(path) else {}
    if a.name in m:
        return
    model = {"baseline": "tfidf-logreg", "jev": "jev-latest"}.get(a.scorer, a.model)
    if a.scorer == "hf":
        model += "@" + (getattr(s.m.config, "_commit_hash", None) or "unknown")[:7]
    mode = {"baseline": "native", "hf": "token", "anthropic": "stated", "jev": "native"}.get(
        a.scorer, "token" if a.mode == "logprob" else "stated"
    )
    m[a.name] = {
        "model": model,
        "scorer": a.scorer,
        "mode": mode,
        "dataset": a.dataset,
        "date": str(date.today()),
    }
    with open(path, "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Score test prompts and append P(harmful) to results/<dataset>.")
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
    ap.add_argument("--dataset", default="toxicchat", choices=DATASETS)
    ap.add_argument(
        "--n", type=int, help="prompts to sample (default 2000 for toxicchat, all for wildguardtest)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16, help="parallel requests for API scorers")
    ap.add_argument(
        "--keep-all", action="store_true", help="keep prompts with sexual/minors moderation > 0.1"
    )
    a = ap.parse_args()
    a.prefix = codecs.decode(a.prefix, "unicode_escape")
    if a.scorer in ("hf", "openai", "anthropic") and not a.model:
        ap.error("--model is required for --scorer {}".format(a.scorer))

    if a.n is None and a.dataset == "toxicchat":
        a.n = 2000
    ids, x, y, extra = load("test", a.n, a.seed, a.keep_all, a.dataset)
    os.makedirs(os.path.join("results", a.dataset), exist_ok=True)
    path = os.path.join("results", a.dataset, "{}.jsonl".format(a.name))
    skip = done(path)
    s = build(a)
    todo = [(i, r) for i in range(len(ids)) for r in range(a.repeats) if (ids[i], r) not in skip]
    record(a, s)
    # Local models share one device, so only API scorers run in parallel. Workers only score; this
    # thread does every write, so lines can't interleave.
    workers = 1 if a.scorer in ("baseline", "hf") else a.workers
    with open(path, "a") as f, ThreadPoolExecutor(workers) as ex:
        jobs = {ex.submit(s.score, x[i]): (i, r) for i, r in todo}
        for k, job in enumerate(as_completed(jobs)):
            if (k + 1) % 100 == 0:
                print("{}/{}".format(k + 1, len(todo)))
            i, r = jobs[job]
            try:
                p = job.result()
            except Exception as e:
                print("{} rep {}: {}".format(ids[i], r, e))
                continue
            rec = {"id": ids[i], "rep": r, "p": p, "y": int(y[i])}
            if np.isnan(p):
                print("{} rep {}: no probability, recorded as a non-answer".format(ids[i], r))
                rec = {"id": ids[i], "rep": r, "p": None, "refused": True, "y": int(y[i])}
            rec.update(extra[i])
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
