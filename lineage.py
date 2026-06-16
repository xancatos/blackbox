# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
DIGITS demo -- the fast, dependency-light entry point to the attack lineage.

Reads ONLY the victim's top-1 label (hard-label) and walks the whole lineage of
black-box adversarial attacks from a blind baseline up to SQBA, on sklearn's 8x8
digits with a small MLP.  All attack logic lives in `attacks.py`; this file just
supplies the digits victim/surrogate and prints the comparison tables.

Run:  uv run lineage.py
For the higher-dimensional, paper-shaped version see cifar_lineage.py.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from attacks import (Ctx, loose_start, attack_random, attack_line, attack_boundary,
                     attack_opt, attack_sign_opt, attack_hsj, attack_sqba, attack_sqba_full)

rng = np.random.default_rng(0)

# ---- data + black-box victim (an MLP; we only ever read its label) ----------
digits = load_digits()
X = digits.images.reshape(len(digits.images), -1) / 16.0      # 64-dim, [0,1]
y = digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
victim = MLPClassifier(hidden_layer_sizes=(128,), max_iter=600, random_state=0).fit(Xtr, ytr)


def victim_label(x):                                  # flat x -> int label (uncounted)
    return int(victim.predict(x.reshape(1, -1))[0])


# ---- surrogate (white box) + the weaken knob --------------------------------
def build_surrogate(frac=1.0, hidden=96, seed=7, label_noise=0.0):
    idx = rng.choice(len(Xtr), max(20, int(frac * len(Xtr))), replace=False)
    Xs, ys = Xtr[idx].copy(), ytr[idx].copy()
    if label_noise:
        m = rng.random(len(ys)) < label_noise
        ys[m] = rng.integers(0, 10, m.sum())
    return MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=400, random_state=seed).fit(Xs, ys)


def mlp_sgrad(surr):
    """Analytic d(cross-entropy)/d(input) for a one-hidden-layer ReLU MLP, unit."""
    W1, W2 = surr.coefs_; b1, b2 = surr.intercepts_
    def sgrad(x, y0):
        a1 = x @ W1 + b1; h = np.maximum(a1, 0.0); a2 = h @ W2 + b2
        p = np.exp(a2 - a2.max()); p /= p.sum()
        onehot = np.zeros_like(p); onehot[y0] = 1.0
        g = (((p - onehot) @ W2.T) * (a1 > 0)) @ W1.T
        n = np.linalg.norm(g)
        return g / n if n else g
    return sgrad


def make_ctx(surr):
    return Ctx(victim_label, mlp_sgrad(surr), Xtr, ytr, rng, lo=0.0, hi=1.0)


# ============================================================================
# Driver: run the lineage on one image, then the weaken-surrogate sweep.
# ============================================================================
surr = build_surrogate()
ctx = make_ctx(surr)
x0 = Xte[0]; y0 = victim_label(x0)
loose = loose_start(ctx, x0, y0)


def run(fn):
    ctx.q = 0; ctx.budget = float("inf")
    x = fn()
    return x, ctx.q


plan = [
    ("white-box", "(see cifar)", "", "FGSM/PGD/transfer need pixel CNNs; shown in cifar_lineage.py", None),
    ("hard-label", "random",     " - ", "blind noise (scaffold)",                   lambda: attack_random(ctx, x0, y0)),
    ("hard-label", "line",       " - ", "aim at nearest class + binsearch (scaffold)", lambda: attack_line(ctx, x0, y0)),
    ("hard-label", "boundary",   "'18", "random walk along boundary",               lambda: attack_boundary(ctx, x0, y0, loose)),
    ("hard-label", "opt",        "'19", "minimize g(theta), zeroth-order",          lambda: attack_opt(ctx, x0, y0, loose)),
    ("hard-label", "sign-opt",   "'20", "g(theta) via SIGN of dir. deriv",          lambda: attack_sign_opt(ctx, x0, y0, loose)),
    ("hard-label", "hopskipjump","'20", "MC boundary-normal estimate",              lambda: attack_hsj(ctx, x0, y0, loose)),
    ("surrogate",  "biased-bdry","'19", "surrogate-biased boundary walk",           lambda: attack_boundary(ctx, x0, y0, loose, bias=0.5)),
]
results = []
for tier, name, year, idea, fn in plan:
    if fn is None:
        results.append((tier, name, year, "", "", "", idea)); continue
    x, q = run(fn)
    results.append((tier, name, year, q, np.linalg.norm(x - x0), ctx.fooled(x, y0), idea))
x, q = run(lambda: attack_sqba(ctx, x0, y0, loose)[0])
results.append(("surrogate", "sqba", "'24", q, np.linalg.norm(x - x0), ctx.fooled(x, y0), "teaching: single surrogate normal + fallback"))
x, q = run(lambda: attack_sqba_full(ctx, x0, y0)[0])
results.append(("surrogate", "sqba-full", "'24", q, np.linalg.norm(x - x0), ctx.fooled(x, y0), "paper Algo 1: multi-gradient + beta switch"))

print("=" * 86)
print("DIGITS ATTACK LINEAGE  (one image; victim is hard-label only)")
print("=" * 86)
print(f"{'attack':<13}{'yr':>4}{'victim_q':>10}{'L2':>8}{'win':>5}   key idea")
last = None
for tier, name, year, q, d, win, idea in results:
    if tier != last:
        print(f"-- {tier.upper()} " + "-" * (80 - len(tier))); last = tier
    if q == "":
        print(f"{name:<13}{year:>4}{'':>10}{'':>8}{'':>5}   {idea}")
    else:
        print(f"{name:<13}{year:>4}{q:>10}{d:>8.3f}{('Y' if win else 'n'):>5}   {idea}")

print("\n" + "=" * 86)
print("WEAKEN-SURROGATE SWEEP (SQBA decays toward query-based HopSkipJump; avg of 5 images)")
print("=" * 86)
imgs = list(Xte[:5])
print(f"{'surrogate':<10}{'agree%':>8}{'L2':>8}{'victim_q':>10}{'white_only':>12}{'fallbacks':>11}")
for nm, cfg in [("strong ", dict(frac=1.0, hidden=96, label_noise=0.0)),
                ("ok     ", dict(frac=0.3, hidden=32, label_noise=0.05)),
                ("weak   ", dict(frac=0.1, hidden=16, label_noise=0.25)),
                ("useless", dict(frac=0.03, hidden=8, label_noise=0.6))]:
    s = build_surrogate(**cfg)
    agree = np.mean(victim.predict(Xte) == s.predict(Xte)) * 100
    c = make_ctx(s)
    L2s, qs, wis, fbs = [], [], [], []
    for xi in imgs:
        yi = victim_label(xi)
        li = loose_start(c, xi, yi)
        c.q = 0; c.budget = float("inf")
        adv, wi, fb = attack_sqba(c, xi, yi, li)
        L2s.append(np.linalg.norm(adv - xi)); qs.append(c.q); wis.append(wi); fbs.append(fb)
    print(f"{nm:<10}{agree:>7.1f}{np.mean(L2s):>8.3f}{np.mean(qs):>10.0f}{np.mean(wis):>8.1f}/15{np.mean(fbs):>11.1f}")
