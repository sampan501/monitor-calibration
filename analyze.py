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
import tests_hyppo
from data import DATASETS


def read(path):
    """Return per-id lists of p (non-answers as 1.0), labels, the ids with a non-answer, and extras."""
    p, y, na, ex = defaultdict(list), {}, set(), {}
    with open(path) as f:
        for r in map(json.loads, filter(str.strip, f)):
            p[r["id"]].append(1.0 if r["p"] is None else r["p"])
            y[r["id"]] = r["y"]
            ex[r["id"]] = (r.get("adv"), r.get("sub"))
            if r["p"] is None:
                na.add(r["id"])
    return p, y, na, ex


def sd(ps):
    """Return the mean per-prompt SD across repeats, or nan with one repeat."""
    s = [np.std(v, ddof=1) for v in ps if len(v) > 1]
    return np.mean(s) if s else float("nan")


def plot(models, tprs, alpha, out):
    """Save the reliability, risk-coverage, and catch rate figures to out."""
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for i, (name, (y, p, na)) in enumerate(models.items()):
        conf, obs, n = metrics.reliability(y[~na], p[~na])
        k = n >= 10
        c = "C{}".format(i)
        ax.plot(conf[k], obs[k], "-", color=c, label=name)
        ax.scatter(conf[k], obs[k], s=1000 * n[k] / n.sum(), color=c, alpha=0.6)
    ax.set(xlabel="Mean confidence", ylabel="Observed toxic rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.savefig(os.path.join(out, "reliability.png"), dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (y, p, na) in models.items():
        cov, risk = metrics.risk_coverage(y, p, na)
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


def subsets(models, adv, bt, alpha, pairs, names):
    """Return markdown lines for pairwise differences in catch rate on adversarial prompts."""
    adv_catch = []
    for y, p, _ in models.values():
        f = p[y == 1] > metrics.conformal_threshold(p[y == 0], alpha)
        adv_catch.append(f[adv[y == 1]].mean())
    pairs = [(u, v) if adv_catch[u] >= adv_catch[v] else (v, u) for u, v in pairs]
    st = [metrics.paired(bt["adv"][:, u] - bt["adv"][:, v]) for u, v in pairs]
    q = metrics.holm([s[2] for s in st])
    rows = [
        [
            names[u],
            names[v],
            "{:+.3f}".format(adv_catch[u] - adv_catch[v]),
            "{:+.3f} to {:+.3f}".format(lo, hi),
            "{:.3f}".format(h),
        ]
        for (u, v), (lo, hi, _), h in sorted(zip(pairs, st, q), key=lambda r: r[2])
    ]
    head = ["model A", "model B", "adversarial catch A - B", "95% interval", "Holm p"]
    return [
        "",
        "Catch rate on adversarial prompts only, threshold set on all benign prompts",
        "",
    ] + tests_hyppo.md(head, rows)


def summarize(runs, modes, a, rdir, fdir):
    """Write the summary table and figures for one set of runs."""
    ids = sorted(set.intersection(*(set(r[0]) for r in runs.values())))
    if a.exclude_refusals:
        drop = set().union(*(r[2] for r in runs.values()))
        ids = [i for i in ids if i not in drop]
    if not ids:
        raise SystemExit("no ids scored by every model")

    # p counts a non-answer as 1.0 (flagged); na marks those prompts so calibration can leave them out.
    models, aud = {}, {}
    for name, (ps, ys, na, _) in runs.items():
        y = np.array([ys[i] for i in ids])
        p = np.array([np.mean(ps[i]) for i in ids])
        models[name] = (y, p, np.array([i in na for i in ids]))
        aud[name] = metrics.audit(y, p, a.alpha, a.reps)
    names = list(models)
    full = {
        "catch": [metrics.catch_rate(y, p, a.alpha) for y, p, _ in models.values()],
        "ece": [metrics.ece(y[~na], p[~na]) for y, p, na in models.values()],
        "brier": [metrics.brier(y[~na], p[~na]) for y, p, na in models.values()],
    }
    catch = dict(zip(names, full["catch"]))
    # Every model is scored on the same resampled prompts; each pair lists the higher catch rate first.
    y0 = models[names[0]][0]
    wg = a.dataset == "wildguardtest"
    groups = {}
    if wg:
        ex = runs[names[0]][3]
        adv = np.array([bool(ex[i][0]) for i in ids])
        sub = np.array([ex[i][1] for i in ids])
        groups = {"adv": adv, "plain": ~adv}
        groups.update({"sub:" + s: sub == s for s in tests_hyppo.subcats(y0, sub)})
    boot = metrics.boot(
        y0, [m[1] for m in models.values()], [m[2] for m in models.values()], a.alpha, a.boot, groups=groups
    )
    pairs = [
        (i, j) if full["catch"][i] >= full["catch"][j] else (j, i)
        for i in range(len(names))
        for j in range(i)
    ]
    cmp = {}
    for k in full:
        st = [metrics.paired(boot[k][:, u] - boot[k][:, v]) for u, v in pairs]
        cmp[k] = [
            (full[k][u] - full[k][v], lo, hi, q)
            for (u, v), (lo, hi, _), q in zip(pairs, st, metrics.holm([s[2] for s in st]))
        ]

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
        "repeat SD",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, (y, p, na) in models.items():
        ps = runs[name][0]
        tpr, fpr = aud[name]
        lo, hi = np.percentile(tpr, [5, 95])
        s = sd([ps[i] for i in ids])
        yk, pk = y[~na], p[~na]
        cells = [
            name,
            modes.get(name, {}).get("mode", "?"),
            "{} ({})".format(len(y), y.sum()),
            str(na.sum()),
            "{:.3f}".format(roc_auc_score(yk, pk)),
            "{:.4f}".format(metrics.brier(yk, pk)),
            "{:.4f}".format(metrics.ece(yk, pk)),
            "{:.4f}".format(metrics.ece(yk[yk == 1], pk[yk == 1])),
            "{:.4f}".format(metrics.aurc(y, p, na)),
            "{:.3f}".format(catch[name]),
            "{:.3f} to {:.3f}".format(lo, hi),
            "{:.3f}".format(np.mean(fpr > a.alpha)),
            "n/a" if np.isnan(s) else "{:.4f}".format(s),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    head = ["model A", "model B"]
    for k in ("catch", "ECE", "Brier"):
        head += ["{} A - B".format(k), "95% interval", "Holm p"]
    lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in sorted(range(len(pairs)), key=lambda n: cmp["catch"][n][3]):
        u, v = pairs[n]
        cells = [names[u], names[v]]
        for k, f in (("catch", "{:+.3f}"), ("ece", "{:+.4f}"), ("brier", "{:+.4f}")):
            d, lo, hi, q = cmp[k][n]
            cells += [f.format(d), "{} to {}".format(f.format(lo), f.format(hi)), "{:.3f}".format(q)]
        lines.append("| " + " | ".join(cells) + " |")
    if wg:
        lines += subsets(models, adv, boot, a.alpha, pairs, names)
        lines += tests_hyppo.report(models, adv, sub, boot, a.alpha, fdir)
    table = "\n".join(lines) + "\n"
    print(table)
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "summary.md"), "w") as f:
        f.write(table)
    plot(models, {k: v[0] for k, v in aud.items()}, a.alpha, fdir)


def main():
    ap = argparse.ArgumentParser(description="Summarize results/<dataset>/*.jsonl into tables and figures.")
    ap.add_argument("--dataset", default="toxicchat", choices=DATASETS)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--boot", type=int, default=2000, help="paired bootstrap resamples")
    ap.add_argument(
        "--exclude-refusals",
        action="store_true",
        help="score only prompts every model answered, instead of counting non-answers as p = 1",
    )
    ap.add_argument(
        "--private",
        nargs="*",
        default=[],
        help="run-name prefixes left out of results/<dataset>/public and figures/<dataset>/public",
    )
    a = ap.parse_args()

    rdir, fdir = os.path.join("results", a.dataset), os.path.join("figures", a.dataset)
    runs = {os.path.basename(f)[:-6]: read(f) for f in sorted(glob.glob(os.path.join(rdir, "*.jsonl")))}
    if not runs:
        raise SystemExit("no {}/*.jsonl files".format(rdir))
    path = os.path.join(rdir, "models.json")
    modes = json.load(open(path)) if os.path.exists(path) else {}
    summarize(runs, modes, a, rdir, fdir)
    public = {k: v for k, v in runs.items() if not k.startswith(tuple(a.private))}
    if a.private and public:
        summarize(public, modes, a, os.path.join(rdir, "public"), os.path.join(fdir, "public"))


if __name__ == "__main__":
    main()
