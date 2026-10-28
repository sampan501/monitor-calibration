import numpy as np


def reliability(y, p, bins=10):
    """Return mean confidence, observed rate, and count in equal-width bins."""
    b = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    n = np.bincount(b, minlength=bins)
    with np.errstate(invalid="ignore"):
        conf = np.bincount(b, p, bins) / n
        obs = np.bincount(b, y, bins) / n
    return conf, obs, n


def ece(y, p, bins=10):
    """Return expected calibration error."""
    conf, obs, n = reliability(y, p, bins)
    k = n > 0
    return np.sum(n[k] * np.abs(conf[k] - obs[k])) / n.sum()


def brier(y, p):
    """Return the Brier score."""
    return np.mean((p - y) ** 2)


def risk_coverage(y, p):
    """Return coverage and cumulative error rate, most confident first."""
    i = np.argsort(-np.maximum(p, 1 - p), kind="stable")
    err = ((p >= 0.5).astype(int) != y)[i]
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(err) / k


def aurc(y, p):
    """Return the area under the risk-coverage curve."""
    return risk_coverage(y, p)[1].mean()


def conformal_threshold(s, alpha):
    """Return the split conformal threshold on benign scores s; flag scores above it."""
    s = np.sort(s)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


def audit(y, p, alpha=0.02, reps=500, seed=0):
    """Return held-out catch rates and false positive rates over random half splits."""
    rs = np.random.RandomState(seed)
    n = len(y)
    tpr, fpr = np.empty(reps), np.empty(reps)
    for r in range(reps):
        i = rs.permutation(n)
        a, b = i[: n // 2], i[n // 2 :]
        t = conformal_threshold(p[a][y[a] == 0], alpha)
        f = p[b] > t
        tpr[r], fpr[r] = f[y[b] == 1].mean(), f[y[b] == 0].mean()
    return tpr, fpr


def catch_rate(y, p, alpha):
    """Return the share of harmful prompts above the threshold set on all benign scores."""
    return np.mean(p[y == 1] > conformal_threshold(p[y == 0], alpha))


def perm_test(y, p, alpha, reps=1000, seed=0):
    """Return the permutation p-value for catch_rate against chance."""
    rs = np.random.RandomState(seed)
    obs = catch_rate(y, p, alpha)
    null = np.array([catch_rate(rs.permutation(y), p, alpha) for _ in range(reps)])
    return (1 + np.sum(null >= obs)) / (1 + reps)
