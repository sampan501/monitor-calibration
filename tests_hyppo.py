import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hyppo.independence import Dcorr
from hyppo.ksample import KSample
from scipy.stats import pearsonr

import metrics

REPS, SEED = 1000, 0


def col(p):
    """Return scores as an n x 1 array."""
    return np.asarray(p, dtype=float).reshape(-1, 1)


def md(head, rows):
    """Return a markdown table."""
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def subcats(y, sub, least=20):
    """Return the subcategories with at least `least` harmful prompts."""
    s, n = np.unique(sub[y == 1], return_counts=True)
    return list(s[n >= least])


def ksample(*xs):
    """Return the Dcorr k-sample statistic and permutation p-value."""
    # auto=False keeps the permutation test; with auto=True hyppo switches to a chi-square
    # approximation for n > 20 and ignores reps and random_state.
    return KSample("Dcorr").test(*map(col, xs), reps=REPS, auto=False, random_state=SEED)


def dcorr(x, y):
    """Return the distance correlation of two score vectors and its permutation p-value."""
    return Dcorr().test(col(x), col(y), reps=REPS, auto=False, random_state=SEED)


def holm_rows(names, res):
    """Return table rows of name, statistic, p, and Holm p across names."""
    q = metrics.holm([r[1] for r in res])
    return [
        [n, "{:.4f}".format(s), "{:.3f}".format(p), "{:.3f}".format(h)] for n, (s, p), h in zip(names, res, q)
    ]


def report(models, adv, sub, bt, alpha, fdir):
    """Return markdown lines for the hyppo tests and save the redundancy heatmap to fdir."""
    names = list(models)
    y = next(iter(models.values()))[0]
    h = y == 1
    keep = subcats(y, sub)
    out = ["", "Do harmful prompts in different subcategories get different scores? (Dcorr k-sample)", ""]
    res = [ksample(*(p[h & (sub == s)] for s in keep)) for _, p, _ in models.values()]
    out += md(["model", "statistic", "p", "Holm p"], holm_rows(names, res))

    out += ["", "Catch rate at FPR {} by subcategory, 95% bootstrap interval".format(alpha), ""]
    rows = []
    for s in keep:
        cells = ["{} ({})".format(s, np.sum(h & (sub == s)))]
        for j, (_, p, _) in enumerate(models.values()):
            f = p[h] > metrics.conformal_threshold(p[~h], alpha)
            lo, hi = np.percentile(bt["sub:" + s][:, j], [2.5, 97.5])
            cells.append("{:.3f} ({:.3f} to {:.3f})".format(f[sub[h] == s].mean(), lo, hi))
        rows.append(cells)
    out += md(["subcategory (harmful)"] + names, rows)

    for label, m in (("harmful", h), ("benign", ~h)):
        out += [
            "",
            "Do adversarial and plain {} prompts get different scores? (Dcorr two-sample)".format(label),
            "",
        ]
        res = [ksample(p[m & adv], p[m & ~adv]) for _, p, _ in models.values()]
        out += md(["model", "statistic", "p", "Holm p"], holm_rows(names, res))

    out += ["", "How much do two monitors' scores move together on harmful prompts? (Dcorr)", ""]
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    ps = [m[1][h] for m in models.values()]
    res = [dcorr(ps[i], ps[j]) for i, j in pairs]
    q = metrics.holm([r[1] for r in res])
    rows = [
        [
            names[i],
            names[j],
            "{:.3f}".format(s),
            "{:.3f}".format(b),
            "{:+.3f}".format(pearsonr(ps[i], ps[j])[0]),
        ]
        for (i, j), (s, _), b in zip(pairs, res, q)
    ]
    out += md(["model A", "model B", "Dcorr", "Holm p", "Pearson r"], rows)

    d = np.eye(len(names))
    for (i, j), (s, _) in zip(pairs, res):
        d[i, j] = d[j, i] = s
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(d, vmin=0, vmax=1, cmap="viridis")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j, i, "{:.2f}".format(d[i, j]), ha="center", va="center", color="w" if d[i, j] < 0.6 else "k"
            )
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(im, label="Distance correlation on harmful prompts")
    os.makedirs(fdir, exist_ok=True)
    fig.savefig(os.path.join(fdir, "redundancy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
