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

============================================================================
ML PRIMER  -- read this if "model", "CNN", "gradient", or "loss" are new to you.
============================================================================
* MODEL / CLASSIFIER: a function that takes an image and returns a class. Ours
  returns "cat" or "dog". It has millions of internal numbers ("weights") that
  are tuned during TRAINING so it usually gets the answer right.
* TRAIN vs TEST data: we tune the weights on the TRAINING images, then measure
  quality on separate TEST images the model never saw (so we measure real skill,
  not memorisation).
* NEURAL NETWORK: the model is built from layers of simple math (multiply by a
  weight matrix, add a bias, apply a simple non-linear function). Stacking many
  layers lets it learn complicated patterns.
* CNN (Convolutional Neural Network): the kind of network used for images. A
  "convolution" slides a small learnable filter over the image to detect local
  patterns (edges, textures); "pooling" shrinks the image to summarise regions.
  Images here are 3 x 32 x 32 = 3 colour CHANNELS (red/green/blue), 32x32 pixels.
* FORWARD PASS: feed an image in, get out 2 numbers ("logits"), one per class.
  The bigger logit is the prediction (argmax). "softmax" turns the logits into
  probabilities; we never read those here -- the victim gives us only the label
  (that is what "hard-label" means).
* LOSS: a single number measuring how wrong the model is on an example
  (cross-entropy). Training nudges the weights to make the loss small.
* GRADIENT: the derivative of the loss -- a vector that points in the direction
  that INCREASES the loss fastest. Two uses:
    - during TRAINING we take the gradient w.r.t. the WEIGHTS and step the
      weights the opposite way (down the loss) -> the model learns.
    - for an ATTACK we take the gradient w.r.t. the INPUT PIXELS: it points in
      the pixel-direction that makes the image look "more wrong" to the model,
      i.e. the adversarial direction. PyTorch computes both automatically
      ("autograd"): you call .backward() and read the gradient.
* TENSOR: PyTorch's array type (like a numpy array but it can track gradients).
  We convert numpy <-> tensor at the boundaries.
============================================================================
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
# CIFAR-10 is 60,000 colour photos (32x32 pixels) in 10 classes. Each image is
# stored as a flat row of 3072 integers in 0..255: the first 1024 are the RED
# 32x32 plane, then GREEN, then BLUE. We keep only two classes (cat, dog) to stay
# small and fast. `reshape(-1, 3, 32, 32)` later turns each flat row back into the
# (channels, height, width) shape the CNN expects.
# ============================================================================
def _read_batch(tar, name):
    # CIFAR ships as pickled python dicts; b"data" is the pixel rows, b"labels"
    # the class index (0..9) of each row.
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
        # keep n_per images of each chosen class and RELABEL them 0 and 1, because
        # our 2-output network thinks in terms of "class 0" vs "class 1".
        outX, outY = [], []
        for new, c in enumerate((c0, c1)):        # c0 (cat)->0, c1 (dog)->1
            idx = rng.permutation(np.where(y == c)[0])[:n_per]
            outX.append(X[idx]); outY.append(np.full(len(idx), new))
        # pixels are integers 0..255; divide by 255 so every value is in [0,1].
        # Neural nets train better when inputs are on a small, consistent scale.
        X = np.concatenate(outX).astype(np.float32) / 255.0
        y = np.concatenate(outY).astype(np.int64)
        p = rng.permutation(len(y))               # shuffle so classes are interleaved
        return X[p], y[p]

    Xtr, ytr = subset(Xtr_all, ytr_all, n_train)
    Xte, yte = subset(Xte_all, yte_all, n_test)
    np.savez(npz, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte

Xtr, ytr, Xte, yte = load_cifar2()

# ============================================================================
# MODELS -- two small CNNs.  We build TWO different ones on purpose:
#   * VICTIM    -- the black box we attack (we pretend we can't see inside it).
#   * SURROGATE -- our OWN stand-in model. We CAN see inside it, so we can read
#                  its gradients. The hope ("transfer") is that a direction that
#                  fools the surrogate also fools the victim, even though they
#                  have different architectures. That is the whole basis of SQBA.
# A network in PyTorch is a class deriving from nn.Module: __init__ creates the
# layers, forward() defines how an input flows through them.
# ============================================================================

# NORMALISATION: before the first layer we subtract a per-colour average (MEAN)
# and divide by a per-colour spread (STD). This re-centres pixel values around 0,
# which helps networks train. We bake it INTO forward() so the rest of the code
# (and every attack) can work in plain [0,1] pixel space and not worry about it.
# The .view(1,3,1,1) shape lets one number per colour-channel broadcast over the
# whole 32x32 image.
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

class VictimNet(nn.Module):                       # 3x3 filters, widths 16 -> 32
    def __init__(self):
        super().__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size): a convolution layer.
        # It slides `out_channels` small learnable filters of size kernel x kernel
        # over the image; each filter detects one kind of local pattern (an edge,
        # a colour blob, a texture). padding=1 keeps the image the same size.
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)  # 3 colour inputs -> 16 feature maps
        self.c2 = nn.Conv2d(16, 32, 3, padding=1) # 16 -> 32 feature maps
        # nn.Linear(in, out): a plain "fully connected" layer -- multiply by a
        # weight matrix. Here it maps the flattened features to 2 numbers (one
        # score per class). After two poolings a 32x32 image is 8x8, with 32
        # feature maps, so the flattened size is 32*8*8.
        self.fc = nn.Linear(32 * 8 * 8, 2)
    def forward(self, x):                         # x: a batch of images, shape (N,3,32,32), in [0,1]
        x = (x - MEAN) / STD                      # normalise (see above)
        # relu(z) = max(z, 0): a simple non-linearity that lets the network learn
        # non-trivial patterns. max_pool2d(.,2) halves height & width by keeping the
        # max of each 2x2 block -> shrinks the image and keeps the strongest signal.
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 32x32 -> 16x16
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 16x16 -> 8x8
        # flatten(1) turns each image's (32,8,8) block into a flat vector; fc gives
        # the 2 class scores ("logits"). The bigger one is the prediction.
        return self.fc(x.flatten(1))

class SurrogateNet(nn.Module):                    # a DIFFERENT shape: 5x5 then 3x3 filters
    # Deliberately unlike VictimNet (bigger first filter, different widths) so the
    # demo shows transfer working BETWEEN DIFFERENT architectures. `width` scales
    # the number of filters -- the weaken-surrogate sweep shrinks it to make a
    # deliberately worse surrogate.
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
    """TRAINING: tune the net's weights so its predictions match the labels.

    The recipe (the standard supervised-learning loop):
      repeat for several `epochs` (full passes over the data):
        split the data into small `bs`-sized BATCHES; for each batch:
          1. forward pass: predict the batch
          2. LOSS: measure how wrong the predictions are (cross-entropy)
          3. backward pass: compute the gradient of the loss w.r.t. every weight
          4. OPTIMISER step: nudge each weight a little DOWN that gradient
        over many batches the loss drops and accuracy rises.
    We cache the trained weights to disk (`tag`.pt) so re-runs skip training.
    """
    if tag:
        ckpt = os.path.join(CACHE, f"net_{tag}.pt")
        if os.path.exists(ckpt):                  # already trained earlier -> just load it
            net.load_state_dict(torch.load(ckpt)); net.eval(); return net
    torch.manual_seed(seed)                       # reproducible weight init / shuffling
    # convert numpy data into PyTorch tensors (the type the network consumes)
    Xt = torch.tensor(X.reshape(-1, 3, 32, 32), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    # the OPTIMISER (Adam) is the rule for turning gradients into weight updates;
    # 1e-3 is the LEARNING RATE -- how big each step is.
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train()                                   # put the net in "training mode"
    for _ in range(epochs):                       # one epoch = one full sweep of the data
        perm = torch.randperm(len(Xt))            # reshuffle each epoch
        for i in range(0, len(Xt), bs):           # walk the data in batches of `bs`
            idx = perm[i:i + bs]
            opt.zero_grad()                        # clear gradients left from the last step
            # forward pass + cross-entropy loss, then .backward() fills in the
            # gradient of that loss w.r.t. every weight (this is "backpropagation").
            F.cross_entropy(net(Xt[idx]), yt[idx]).backward()
            opt.step()                             # nudge the weights down the gradient
    net.eval()                                    # switch to "evaluation mode" for use
    if tag:
        torch.save(net.state_dict(), os.path.join(CACHE, f"net_{tag}.pt"))
    return net

def accuracy(net, X, y):
    # fraction of images the net labels correctly. torch.no_grad() = "just predict,
    # don't track gradients" (faster). argmax(1) picks the higher of the 2 logits
    # = the predicted class.
    with torch.no_grad():
        pred = net(torch.tensor(X.reshape(-1, 3, 32, 32), dtype=torch.float32)).argmax(1).numpy()
    return float((pred == y).mean())

# ---- the two things the attacks need from the models -----------------------
def make_label(net):
    """Build the victim's HARD-LABEL oracle: image -> predicted class (an int).
    This is ALL the attack is allowed to see of the victim -- no probabilities,
    no gradients, just the winning class."""
    def label(x):                                 # flat [0,1] vector -> int class
        with torch.no_grad():                     # we only predict, no learning here
            t = torch.tensor(x.reshape(1, 3, 32, 32), dtype=torch.float32)
            return int(net(t).argmax(1).item())   # argmax = index of the bigger logit
    return label

def make_sgrad(net):
    """Build the SURROGATE'S INPUT GRADIENT -- the heart of why a surrogate helps.

    Normally gradients are used to train weights. Here we instead ask: "holding
    the weights fixed, which way should I change the PIXELS to raise the loss on
    the true class y0?" That direction is the adversarial direction -- it makes
    the image look more like the OTHER class to the model. Because we own the
    surrogate, PyTorch can compute this for free with autograd:
      1. mark the input tensor `requires_grad=True` so PyTorch tracks it,
      2. compute the loss, call .backward(),
      3. read t.grad -- the gradient of the loss w.r.t. each input pixel.
    We return it as a unit vector (length 1) so only the DIRECTION matters."""
    def sgrad(x, y0):
        t = torch.tensor(x.reshape(1, 3, 32, 32), dtype=torch.float32, requires_grad=True)
        F.cross_entropy(net(t), torch.tensor([y0])).backward()   # fill t.grad
        g = t.grad.numpy().ravel().astype(np.float64)            # back to a flat numpy vector
        n = np.linalg.norm(g)                                     # its length
        return g / n if n else g                                  # make it length 1 (unit direction)
    return sgrad

# ---- the weaken knob: build surrogates of controllable quality --------------
# To study "what if our surrogate is bad?", we can make it worse on purpose:
#   frac        -- train on only a fraction of the data (less data -> worse model)
#   width       -- fewer filters (smaller, weaker network)
#   label_noise -- randomly corrupt this fraction of the training labels (lies)
def build_surrogate(frac=1.0, width=1.0, label_noise=0.0, epochs=5, seed=7, tag=None):
    idx = rng.choice(len(Xtr), max(64, int(frac * len(Xtr))), replace=False)
    Xs, ys = Xtr[idx].copy(), ytr[idx].copy()
    if label_noise:                               # flip some labels to random classes
        m = rng.random(len(ys)) < label_noise
        ys[m] = rng.integers(0, 2, m.sum())
    return train_net(SurrogateNet(width), Xs, ys, epochs=epochs, seed=seed, tag=tag)

# ============================================================================
# TIER 1-2 -- the simplest "use a gradient" attacks, run on the SURROGATE.
# These are WHITE-BOX (they need the surrogate's gradient) and don't query the
# victim at all -- they just hope the result TRANSFERS to it. With real CNN
# gradients these finally work (the digits demo had no pixel gradients to show).
# ============================================================================
def pgd_on_surr(x0, y0, sgrad, eps, steps):
    """Walk the image in the adversarial direction, a bit at a time.
    `eps` is the BUDGET: no pixel may move more than eps from its original value
    (an "L-infinity ball" -- a small, imperceptible change). `np.sign(grad)` takes
    just the +/- direction of each pixel's gradient; we step `alpha` that way, then
    clip back into the budget and into the valid [0,1] pixel range."""
    x, alpha = x0.copy(), (eps / 4 if steps > 1 else eps)
    for _ in range(steps):
        x = x + alpha * np.sign(sgrad(x, y0))     # one step toward "more wrong"
        x = np.clip(np.clip(x, x0 - eps, x0 + eps), 0, 1)   # stay within eps and in [0,1]
    return x

def attack_fgsm(x0, y0, sgrad, eps=0.05):
    """FGSM (Goodfellow et al. 2014): take ONE full-size step along the sign of
    the gradient. The original, simplest gradient attack."""
    return pgd_on_surr(x0, y0, sgrad, eps, 1)

def attack_pgd(x0, y0, sgrad, eps=0.05):
    """PGD (Madry et al. 2017): FGSM repeated in many small steps -- stronger and
    more reliable."""
    return pgd_on_surr(x0, y0, sgrad, eps, 20)

def attack_transfer(ctx, x0, y0, sgrad):
    """TRANSFER attack (Papernot et al. 2016): craft on the surrogate, then check
    against the victim. Grow the budget eps until the victim's label flips. The
    only victim contact is that one label check per eps."""
    for eps in np.linspace(0.01, 0.25, 16):       # smallest budget that fools the victim
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
# rho = relative size of the perturbation: length of the change divided by length
# of the original image. The paper calls an attack successful if rho <= 0.10
# (a change small enough to be roughly imperceptible). Smaller rho = better attack.
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
