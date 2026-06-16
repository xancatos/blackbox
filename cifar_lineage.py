# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "torch"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
# ///
"""
CIFAR-10 demo -- the higher-dimensional, paper-shaped version of the lineage.

Same hard-label attack ladder as lineage.py (imported from attacks.py), but the
victim and surrogate are now small CONVOLUTIONAL networks on a 2-class slice of
CIFAR-10 (cat vs dog, 3x32x32 = 3072-dim).  This is much closer to the SQBA
paper's setting, and -- crucially -- in this high dimension SQBA's surrogate
guidance and the multi-gradient method finally pay off in a way the 64-dim
digits toy could not show.

Everything is CPU-only and small: two tiny CNNs (~a minute each to train), a
2-class/800-per-class subset, and results cached to disk so re-runs are fast.

Run:  uv run cifar_lineage.py        (first run downloads ~170MB CIFAR + CPU torch)
The CIFAR tar, the 2-class subset (.npz) and trained nets (.pt) are cached under
~/.cache/blackbox so subsequent runs skip the download and the training.

Reporting follows the paper: perturbation is the relative L2 budget
    rho(x_adv) = ||x_adv - x|| / ||x||,           success := rho <= 0.10,
and the headline is an ASR-vs-query-budget table (the shape of the paper's
Tables 3-6) showing SQBA's advantage at small budgets.
"""
import os
import pickle
import tarfile
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks import (Ctx, loose_start, attack_random, attack_line, attack_boundary,
                     attack_sign_opt, attack_hsj, attack_sqba, attack_sqba_full)

torch.manual_seed(0)
torch.set_num_threads(max(1, (os.cpu_count() or 2)))
rng = np.random.default_rng(0)
CACHE = os.path.expanduser("~/.cache/blackbox")
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CAT, DOG = 3, 5                                   # two visually-similar CIFAR-10 classes

# ============================================================================
# DATA -- a 2-class CIFAR-10 subset, scaled to [0,1], cached as a small .npz.
# ============================================================================
def _read_batch(tar, name):
    d = pickle.load(tar.extractfile(name), encoding="bytes")
    return d[b"data"], np.array(d[b"labels"])

def load_cifar2(c0=CAT, c1=DOG, n_train=800, n_test=50):
    npz = os.path.join(CACHE, f"cifar2_{c0}-{c1}_{n_train}_{n_test}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    os.makedirs(CACHE, exist_ok=True)
    tarp = os.path.join(CACHE, "cifar-10-python.tar.gz")
    if not os.path.exists(tarp):                  # urllib follows the 301 redirect
        urllib.request.urlretrieve(CIFAR_URL, tarp)
    with tarfile.open(tarp) as tar:
        names = tar.getnames()
        trd = [_read_batch(tar, n) for n in sorted(names) if "data_batch" in n]
        ted = [_read_batch(tar, n) for n in names if "test_batch" in n]
    Xtr_all = np.concatenate([d for d, _ in trd]); ytr_all = np.concatenate([l for _, l in trd])
    Xte_all = np.concatenate([d for d, _ in ted]); yte_all = np.concatenate([l for _, l in ted])

    def subset(X, y, n_per):
        outX, outY = [], []
        for new, c in enumerate((c0, c1)):        # relabel c0->0, c1->1
            idx = rng.permutation(np.where(y == c)[0])[:n_per]
            outX.append(X[idx]); outY.append(np.full(len(idx), new))
        X = np.concatenate(outX).astype(np.float32) / 255.0     # uint8 -> [0,1]
        y = np.concatenate(outY).astype(np.int64)
        p = rng.permutation(len(y))
        return X[p], y[p]

    Xtr, ytr = subset(Xtr_all, ytr_all, n_train)
    Xte, yte = subset(Xte_all, yte_all, n_test)
    np.savez(npz, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte

Xtr, ytr, Xte, yte = load_cifar2()

# ============================================================================
# MODELS -- two DIFFERENT tiny CNNs (real cross-architecture transfer).
# Normalisation is baked into forward(), so the attacks operate in plain [0,1]
# pixel space and the model handles the mean/std.
# ============================================================================
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

class VictimNet(nn.Module):                       # 3x3 kernels, widths 16 -> 32
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 2)
    def forward(self, x):
        x = (x - MEAN) / STD
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 32 -> 16
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 16 -> 8
        return self.fc(x.flatten(1))

class SurrogateNet(nn.Module):                    # 5x5 then 3x3, widths scaled by `width`
    def __init__(self, width=1.0):
        super().__init__()
        w1, w2 = max(2, int(8 * width)), max(2, int(24 * width))
        self.c1 = nn.Conv2d(3, w1, 5, padding=2)
        self.c2 = nn.Conv2d(w1, w2, 3, padding=1)
        self.fc = nn.Linear(w2 * 8 * 8, 2)
    def forward(self, x):
        x = (x - MEAN) / STD
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))

def train_net(net, X, y, epochs=5, bs=64, seed=0, tag=None):
    """Train (or load from cache) a net.  Inputs are [0,1]; the net normalises."""
    if tag:
        ckpt = os.path.join(CACHE, f"net_{tag}.pt")
        if os.path.exists(ckpt):
            net.load_state_dict(torch.load(ckpt)); net.eval(); return net
    torch.manual_seed(seed)
    Xt = torch.tensor(X.reshape(-1, 3, 32, 32), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    if tag:
        torch.save(net.state_dict(), os.path.join(CACHE, f"net_{tag}.pt"))
    return net

def accuracy(net, X, y):
    with torch.no_grad():
        pred = net(torch.tensor(X.reshape(-1, 3, 32, 32), dtype=torch.float32)).argmax(1).numpy()
    return float((pred == y).mean())

# ---- the victim's hard-label oracle + a surrogate's gradient (torch autograd) -
def make_label(net):
    def label(x):                                 # flat [0,1] -> int class (uncounted)
        with torch.no_grad():
            t = torch.tensor(x.reshape(1, 3, 32, 32), dtype=torch.float32)
            return int(net(t).argmax(1).item())
    return label

def make_sgrad(net):
    def sgrad(x, y0):                             # unit d(CE)/d(input) via autograd
        t = torch.tensor(x.reshape(1, 3, 32, 32), dtype=torch.float32, requires_grad=True)
        F.cross_entropy(net(t), torch.tensor([y0])).backward()
        g = t.grad.numpy().ravel().astype(np.float64)
        n = np.linalg.norm(g)
        return g / n if n else g
    return sgrad

# ---- the weaken knob: surrogates of controllable quality --------------------
def build_surrogate(frac=1.0, width=1.0, label_noise=0.0, epochs=5, seed=7, tag=None):
    idx = rng.choice(len(Xtr), max(64, int(frac * len(Xtr))), replace=False)
    Xs, ys = Xtr[idx].copy(), ytr[idx].copy()
    if label_noise:
        m = rng.random(len(ys)) < label_noise
        ys[m] = rng.integers(0, 2, m.sum())
    return train_net(SurrogateNet(width), Xs, ys, epochs=epochs, seed=seed, tag=tag)

# ============================================================================
# TIER 1-2 -- white-box (FGSM/PGD on the surrogate) + transfer.  Now that we
# have CNNs with real gradients, these rungs (omitted from the digits demo) work.
# ============================================================================
def pgd_on_surr(x0, y0, sgrad, eps, steps):
    x, alpha = x0.copy(), (eps / 4 if steps > 1 else eps)
    for _ in range(steps):
        x = x + alpha * np.sign(sgrad(x, y0))
        x = np.clip(np.clip(x, x0 - eps, x0 + eps), 0, 1)
    return x

def attack_fgsm(x0, y0, sgrad, eps=0.05):
    return pgd_on_surr(x0, y0, sgrad, eps, 1)

def attack_pgd(x0, y0, sgrad, eps=0.05):
    return pgd_on_surr(x0, y0, sgrad, eps, 20)

def attack_transfer(ctx, x0, y0, sgrad):
    for eps in np.linspace(0.01, 0.25, 16):       # smallest L-inf budget that transfers
        x = pgd_on_surr(x0, y0, sgrad, eps, 15)
        if ctx.is_adv(x, y0):
            return x
    return x

# ============================================================================
# TRAIN victim + strong surrogate.
# ============================================================================
print("training tiny CNNs (cached after first run) ...")
victim = train_net(VictimNet(), Xtr, ytr, epochs=6, seed=0, tag="victim")
surr = build_surrogate(tag="surr_strong")
victim_label = make_label(victim)
print(f"victim acc {accuracy(victim, Xte, yte):.3f} | surrogate acc {accuracy(surr, Xte, yte):.3f} | "
      f"victim/surrogate agree {np.mean([victim_label(x) == make_label(surr)(x) for x in Xte]):.3f}")

def make_ctx(s):
    return Ctx(victim_label, make_sgrad(s), Xtr, ytr, rng, lo=0.0, hi=1.0)

ctx = make_ctx(surr)
sgrad = make_sgrad(surr)
rho = lambda adv, x0: np.linalg.norm(adv - x0) / np.linalg.norm(x0)

# ============================================================================
# (A) Single-image lineage table.
# ============================================================================
# first test image the victim gets right, so success is meaningful
x0 = next(Xte[i].astype(np.float64) for i in range(len(Xte)) if victim_label(Xte[i]) == yte[i])
y0 = victim_label(x0)
loose = loose_start(ctx, x0, y0)

def run(fn):
    ctx.q = 0; ctx.budget = float("inf")
    return fn(), ctx.q

plan = [
    ("white-box", "fgsm",       "'14", "one gradient-sign step on surrogate",        lambda: attack_fgsm(x0, y0, sgrad)),
    ("white-box", "pgd",        "'17", "iterated FGSM on surrogate",                 lambda: attack_pgd(x0, y0, sgrad)),
    ("transfer",  "transfer",   "'16", "grow eps on surrogate til victim flips",     lambda: attack_transfer(ctx, x0, y0, sgrad)),
    ("hard-label","random",     " - ", "blind noise (scaffold)",                     lambda: attack_random(ctx, x0, y0, N=400, scale=0.2)),
    ("hard-label","line",       " - ", "nearest class + binsearch (scaffold)",       lambda: attack_line(ctx, x0, y0)),
    ("hard-label","boundary",   "'18", "random walk along boundary",                 lambda: attack_boundary(ctx, x0, y0, loose, steps=400)),
    ("hard-label","sign-opt",   "'20", "g(theta) via SIGN of dir. deriv",            lambda: attack_sign_opt(ctx, x0, y0, loose)),
    ("hard-label","hopskipjump","'20", "MC boundary-normal estimate",                lambda: attack_hsj(ctx, x0, y0, loose)),
    ("surrogate", "biased-bdry","'19", "surrogate-biased boundary walk",             lambda: attack_boundary(ctx, x0, y0, loose, bias=0.5, steps=400)),
]
rows = []
for tier, name, year, idea, fn in plan:
    x, q = run(fn)
    rows.append((tier, name, year, q, rho(x, x0), ctx.fooled(x, y0), idea))
x, q = run(lambda: attack_sqba(ctx, x0, y0, loose)[0])
rows.append(("surrogate", "sqba", "'24", q, rho(x, x0), ctx.fooled(x, y0), "teaching: single surrogate normal + fallback"))
x, q = run(lambda: attack_sqba_full(ctx, x0, y0)[0])
rows.append(("surrogate", "sqba-full", "'24", q, rho(x, x0), ctx.fooled(x, y0), "paper Algo 1: multi-gradient + beta switch"))

print("\n" + "=" * 88)
print(f"CIFAR-10 (cat vs dog) ATTACK LINEAGE  (one image; hard-label only; success = rho<=0.10)")
print("=" * 88)
print(f"{'attack':<13}{'yr':>4}{'victim_q':>10}{'rho':>8}{'win':>5}   key idea")
last = None
for tier, name, year, q, r, win, idea in rows:
    if tier != last:
        print(f"-- {tier.upper()} " + "-" * (82 - len(tier))); last = tier
    print(f"{name:<13}{year:>4}{q:>10}{r:>8.3f}{('Y' if win else 'n'):>5}   {idea}")

# ============================================================================
# (B) ASR vs query-budget -- the paper's headline table shape (Tables 3-6).
# success := the attack reaches rho<=0.10 within the query budget.
# ============================================================================
M = 8
asr_imgs = [Xte[i].astype(np.float64) for i in range(len(Xte)) if victim_label(Xte[i]) == yte[i]][:M]
budgets = [100, 250, 1000]
runners = {                                       # generous iters; the budget stops them
    "hopskipjump": lambda c, x, y: attack_hsj(c, x, y, loose_start(c, x, y), iters=100),
    "sign-opt":    lambda c, x, y: attack_sign_opt(c, x, y, loose_start(c, x, y), iters=100),
    "boundary":    lambda c, x, y: attack_boundary(c, x, y, loose_start(c, x, y), steps=100000),
    "biased-bdry": lambda c, x, y: attack_boundary(c, x, y, loose_start(c, x, y), bias=0.5, steps=100000),
    "sqba":        lambda c, x, y: attack_sqba(c, x, y, loose_start(c, x, y), iters=100)[0],
    "sqba-full":   lambda c, x, y: attack_sqba_full(c, x, y, iters=100)[0],
}
asr = {name: {b: 0 for b in budgets} for name in runners}
for xi in asr_imgs:
    yi = victim_label(xi)
    for name, fn in runners.items():
        for b in budgets:
            ctx.q = 0; ctx.budget = b
            adv = fn(ctx, xi, yi)
            if ctx.fooled(adv, yi) and rho(adv, xi) <= 0.10:
                asr[name][b] += 1

print("\n" + "=" * 88)
print(f"ASR vs QUERY BUDGET  (success = rho<=0.10 within budget; {M} images, strong surrogate)")
print("=" * 88)
print(f"{'attack':<13}" + "".join(f"{('Q=' + str(b)):>9}" for b in budgets))
for name in runners:
    print(f"{name:<13}" + "".join(f"{100 * asr[name][b] / M:>8.0f}%" for b in budgets))

# ============================================================================
# (C) Weaken-surrogate sweep -- SQBA decays toward query-based HopSkipJump.
# ============================================================================
print("\n" + "=" * 88)
print(f"WEAKEN-SURROGATE SWEEP (SQBA decays toward HopSkipJump; avg of {min(4, M)} images)")
print("=" * 88)
sweep_imgs = asr_imgs[:4]
print(f"{'surrogate':<10}{'agree%':>8}{'rho':>8}{'victim_q':>10}{'white_only':>12}{'fallbacks':>11}")
for nm, cfg in [("strong ", dict(tag="surr_strong")),
                ("weak   ", dict(frac=0.2, width=0.5, label_noise=0.2, epochs=2, seed=11, tag="surr_weak")),
                ("useless", dict(frac=0.05, width=0.3, label_noise=0.5, epochs=1, seed=23, tag="surr_useless"))]:
    s = surr if cfg.get("tag") == "surr_strong" else build_surrogate(**cfg)
    agree = np.mean([victim_label(x) == make_label(s)(x) for x in Xte]) * 100
    c = make_ctx(s)
    rs, qs, wis, fbs = [], [], [], []
    for xi in sweep_imgs:
        yi = victim_label(xi)
        li = loose_start(c, xi, yi)
        c.q = 0; c.budget = float("inf")
        adv, wi, fb = attack_sqba(c, xi, yi, li)
        rs.append(rho(adv, xi)); qs.append(c.q); wis.append(wi); fbs.append(fb)
    print(f"{nm:<10}{agree:>7.1f}{np.mean(rs):>8.3f}{np.mean(qs):>10.0f}{np.mean(wis):>8.1f}/15{np.mean(fbs):>11.1f}")
