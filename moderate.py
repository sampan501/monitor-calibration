import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from data import WG, cache_path, wildguard


def score(client, texts):
    """Return the sexual/minors score of each text from omni-moderation-latest."""
    r = client.moderations.create(model="omni-moderation-latest", input=texts)
    return [c.category_scores.sexual_minors for c in r.results]


def save(m, path):
    """Write the cache of id to score."""
    with open(path, "w") as f:
        json.dump(m, f, indent=0, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Cache OpenAI sexual/minors scores for WildGuardMix prompts.")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32, help="prompts per moderation request")
    a = ap.parse_args()

    d, idx = wildguard(a.split)
    path = cache_path(a.split)
    os.makedirs("data", exist_ok=True)
    m = json.load(open(path)) if os.path.exists(path) else {}
    # The cache holds scores only, keyed by id; repeated prompt texts are sent once.
    ids = {}
    for i, x in zip(idx, d[idx.tolist()]["prompt"]):
        k = "{}-{}".format(WG[a.split][2], i)
        if k not in m:
            ids.setdefault(x, []).append(k)
    texts = list(ids)
    batches = [texts[i : i + a.batch] for i in range(0, len(texts), a.batch)]
    print("{} prompts to score in {} requests".format(len(texts), len(batches)))
    client = OpenAI(max_retries=8)
    with ThreadPoolExecutor(a.workers) as ex:
        jobs = {ex.submit(score, client, b): b for b in batches}
        for k, job in enumerate(as_completed(jobs)):
            try:
                for x, s in zip(jobs[job], job.result()):
                    m.update({i: s for i in ids[x]})
            except Exception as e:
                print("request failed: {}".format(e))
            if (k + 1) % 100 == 0:
                print("{}/{}".format(k + 1, len(batches)))
                save(m, path)
    save(m, path)
    print("{} ids cached in {}".format(len(m), path))


if __name__ == "__main__":
    main()
