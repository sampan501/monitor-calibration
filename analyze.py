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
    """Return per-id lists of p (non-answers as 1.0), labels, and the ids with a non-answer."""
    p, y, na = defaultdict(list), {}, set()
    with open(path) as f:
        for r in map(json.loads, filter(str.strip, f)):
            p[r["id"]].append(1.0 if r["p"] is None else r["p"])
            y[r["id"]] = r["y"]
            if r["p"] is None:
                na.add(r["id"])
    return p, y, na


def sd(ps):
    """Return the mean per-prompt SD across repeats, or nan with one repeat."""
    s = [np.std(v, ddof=1) for v in ps if len(v) > 1]
    return np.mean(s) if s else float("nan")


def plot(models, tprs, alpha, out):
    """Save the reliability, risk-coverage, and catch rate figures to out."""
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for i, (name, (y, p)) in enumerate(models.items()):
        conf, obs, n = metrics.reliability(y, p)
        k = n >= 10
        c = "C{}".format(i)
        ax.plot(conf[k], obs[k], "-", color=c, label=name)
        ax.scatter(conf[k], obs[k], s=1000 * n[k] / len(y), color=c, alpha=0.6)
    ax.set(xlabel="Mean confidence", ylabel="Observed toxic rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.savefig(os.path.join(out, "reliability.png"), dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (y, p) in models.items():
        cov, risk = metrics.risk_coverage(y, p)
        k = cov >= 0.01
        ax.plot(cov[k], risk[k], label=name)
    ax.set(xlabel="Coverage", ylabel="Error rate", xlim=(0, 1), ylim=(0, 0.1))
    ax.legend()
    fig.savefig(os.path.join(out, "risk_coverage.png"), dpi=150, bbox_inches="tight")

    names = list(models)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 1))
    for i, name in enumerate(names):
        c = "C{}".format(i)
        lo, med, hi = np.percentile(tprs[name], [5, 50, 95])
        ax.hlines(i, lo, hi, color=c, lw=10, alpha=0.3)
        ax.plot(med, i, "o", color=c)
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="Held-out catch rate at FPR {}".format(alpha), xlim=(0, 1))
    fig.savefig(os.path.join(out, "catch_rate.png"), dpi=150, bbox_inches="tight")
    plt.close("all")


def summarize(runs, modes, a, rdir, fdir):
    """Write the summary table and figures for one set of runs."""
    ids = sorted(set.intersection(*(set(ps) for ps, _, _ in runs.values())))
    if a.exclude_refusals:
        drop = set().union(*(na for _, _, na in runs.values()))
        ids = [i for i in ids if i not in drop]
    if not ids:
        raise SystemExit("no ids scored by every model")

    models, aud = {}, {}
    for name, (ps, ys, _) in runs.items():
        y = np.array([ys[i] for i in ids])
        p = np.array([np.mean(ps[i]) for i in ids])
        models[name] = (y, p)
        # Same seed and n for every model, so all models see the same half-splits.
        aud[name] = metrics.audit(y, p, a.alpha, a.reps)
    catch = {k: metrics.catch_rate(y, p, a.alpha) for k, (y, p) in models.items()}
    best = max(catch, key=catch.get)
    others = [k for k in models if k != best]
    cmp = {k: metrics.paired(aud[k][0], aud[best][0]) for k in others}
    adj = dict(zip(others, metrics.holm([cmp[k][2] for k in others]))) if others else {}

    head = [
        "model",
        "mode",
        "n (toxic)",
        "non-answers",
        "AUROC",
        "Brier",
        "ECE",
        "ECE toxic",
        "AURC",
        "catch@{}".format(a.alpha),
        "audit catch 5-95%",
        "share FPR>{}".format(a.alpha),
        "diff vs best 5-95%",
        "Holm p",
        "repeat SD",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, (y, p) in models.items():
        ps, _, na = runs[name]
        tpr, fpr = aud[name]
        lo, hi = np.percentile(tpr, [5, 95])
        s = sd([ps[i] for i in ids])
        cells = [
            name,
            modes.get(name, {}).get("mode", "?"),
            "{} ({})".format(len(y), y.sum()),
            str(len(na.intersection(ids))),
            "{:.3f}".format(roc_auc_score(y, p)),
            "{:.4f}".format(metrics.brier(y, p)),
            "{:.4f}".format(metrics.ece(y, p)),
            "{:.4f}".format(metrics.ece(y[y == 1], p[y == 1])),
            "{:.4f}".format(metrics.aurc(y, p)),
            "{:.3f}".format(catch[name]),
            "{:.3f} to {:.3f}".format(lo, hi),
            "{:.3f}".format(np.mean(fpr > a.alpha)),
            "reference" if name == best else "{:+.3f} to {:+.3f}".format(*cmp[name][:2]),
            "" if name == best else "{:.3f}".format(adj[name]),
            "n/a" if np.isnan(s) else "{:.4f}".format(s),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    print(table)
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "summary.md"), "w") as f:
        f.write(table)
    plot(models, {k: v[0] for k, v in aud.items()}, a.alpha, fdir)


def main():
    ap = argparse.ArgumentParser(description="Summarize results/*.jsonl into a table and figures.")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument(
        "--exclude-refusals",
        action="store_true",
        help="score only prompts every model answered, instead of counting non-answers as p = 1",
    )
    ap.add_argument(
        "--private",
        nargs="*",
        default=["jev"],
        help="run-name prefixes left out of results/public and figures/public",
    )
    a = ap.parse_args()

    runs = {os.path.basename(f)[:-6]: read(f) for f in sorted(glob.glob("results/*.jsonl"))}
    if not runs:
        raise SystemExit("no results/*.jsonl files")
    path = os.path.join("results", "models.json")
    modes = json.load(open(path)) if os.path.exists(path) else {}
    summarize(runs, modes, a, "results", "figures")
    public = {k: v for k, v in runs.items() if not k.startswith(tuple(a.private))}
    if a.private and public:
        summarize(public, modes, a, os.path.join("results", "public"), os.path.join("figures", "public"))


if __name__ == "__main__":
    main()
