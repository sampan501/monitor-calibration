import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import metrics


def read(path):
    """Return per-id lists of p and the label for one results file."""
    p, y = defaultdict(list), {}
    with open(path) as f:
        for r in map(json.loads, filter(str.strip, f)):
            p[r["id"]].append(r["p"])
            y[r["id"]] = r["y"]
    return p, y


def sd(ps):
    """Return the mean per-prompt SD across repeats, or nan with one repeat."""
    s = [np.std(v, ddof=1) for v in ps if len(v) > 1]
    return np.mean(s) if s else float("nan")


def row(name, y, p, s, alpha, reps):
    """Return one table row and the audit catch rates."""
    tpr, fpr = metrics.audit(y, p, alpha, reps)
    lo, hi = np.percentile(tpr, [5, 95])
    cells = [
        name,
        "{} ({})".format(len(y), y.sum()),
        "{:.3f}".format(roc_auc_score(y, p)),
        "{:.4f}".format(metrics.brier(y, p)),
        "{:.4f}".format(metrics.ece(y, p)),
        "{:.4f}".format(metrics.aurc(y, p)),
        "{:.3f}".format(metrics.catch_rate(y, p, alpha)),
        "{:.3f} to {:.3f}".format(lo, hi),
        "{:.3f}".format(np.mean(fpr > alpha)),
        "{:.4f}".format(metrics.perm_test(y, p, alpha)),
        "n/a" if np.isnan(s) else "{:.4f}".format(s),
    ]
    return cells, tpr


def plot(models, tprs, alpha):
    """Save the reliability, risk-coverage, and catch rate figures."""
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, (y, p) in models.items():
        conf, obs, n = metrics.reliability(y, p)
        k = n > 0
        ax.plot(conf[k], obs[k], "o-", label=name)
    ax.set(xlabel="Mean confidence", ylabel="Observed toxic rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.savefig("figures/reliability.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (y, p) in models.items():
        ax.plot(*metrics.risk_coverage(y, p), label=name)
    ax.set(xlabel="Coverage", ylabel="Error rate")
    ax.legend()
    fig.savefig("figures/risk_coverage.png", dpi=150, bbox_inches="tight")

    names = list(models)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 1))
    for i, name in enumerate(names):
        c = "C{}".format(i)
        lo, med, hi = np.percentile(tprs[name], [5, 50, 95])
        ax.hlines(i, lo, hi, color=c, lw=10, alpha=0.3)
        ax.plot(med, i, "o", color=c)
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="Held-out catch rate at FPR {}".format(alpha), xlim=(0, 1))
    fig.savefig("figures/catch_rate.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    ap = argparse.ArgumentParser(description="Summarize results/*.jsonl into a table and figures.")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=500)
    a = ap.parse_args()

    runs = {os.path.basename(f)[:-6]: read(f) for f in sorted(glob.glob("results/*.jsonl"))}
    if not runs:
        raise SystemExit("no results/*.jsonl files")
    ids = sorted(set.intersection(*(set(p) for p, _ in runs.values())))
    if not ids:
        raise SystemExit("no ids scored by every model")

    head = [
        "model",
        "n (toxic)",
        "AUROC",
        "Brier",
        "ECE",
        "AURC",
        "catch@{}".format(a.alpha),
        "audit catch 5-95%",
        "share FPR>{}".format(a.alpha),
        "perm p",
        "repeat SD",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    models, tprs = {}, {}
    for name, (ps, ys) in runs.items():
        y = np.array([ys[i] for i in ids])
        p = np.array([np.mean(ps[i]) for i in ids])
        cells, tprs[name] = row(name, y, p, sd([ps[i] for i in ids]), a.alpha, a.reps)
        models[name] = (y, p)
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    print(table)
    with open("results/summary.md", "w") as f:
        f.write(table)
    plot(models, tprs, a.alpha)


if __name__ == "__main__":
    main()
