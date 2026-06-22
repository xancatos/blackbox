# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "torch", "scipy"]
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
CIFAR-10 Demo: A realistic, high-dimensional demonstration of adversarial attacks.

In this script, we use 3-channel color photos (Red, Green, Blue) of airplanes and automobiles from the CIFAR-10 dataset.
Each photo is 32x32 pixels, meaning it contains 3 * 32 * 32 = 3072 numbers (features).
This is much larger and more complex than the 64-number digits dataset in `lineage.py`, representing
a realistic scenario where adversarial attacks are deployed.

We train two Convolutional Neural Networks (CNNs) using PyTorch:
1. The Victim Model: The model we want to attack. We treat this as a black-box, meaning we cannot inspect
   its math, layers, or gradients. We only ask it for predictions.
2. The Surrogate Model: Our own helper model. We can inspect its inner workings and calculate its
   gradients for free.

We then:
- Run white-box attacks (which attack the surrogate directly and hope they transfer to the victim).
- Run black-box decision attacks (which only use label predictions from the victim).
- Run surrogate-assisted attacks (like SQBA) to show how much query savings a surrogate helper provides.
- Run sweeps showing the Attack Success Rate (ASR) under different query budgets.
- Run sweeps showing how SQBA gracefully decays to HopSkipJump when the surrogate becomes useless.

To run this file:
    uv run cifar_lineage.py
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
                     attack_sign_opt, attack_hsj, attack_triangle, attack_sqba, attack_sqba_full)

# Set random seed for reproducibility (ensures network initializations are consistent across runs)
torch.manual_seed(0)
# Use all available CPU cores to make PyTorch operations fast
torch.set_num_threads(max(1, (os.cpu_count() or 2)))
rng = np.random.default_rng(0)
CACHE = os.path.expanduser("~/.cache/blackbox")
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
AIRPLANE, AUTOMOBILE = 0, 1                       # Class indices for airplane and automobile in CIFAR-10

# ============================================================================
# DATASET LOADING
# CIFAR-10 contains 60,000 images across 10 classes. We filter for only 2 classes
# (airplanes and automobiles) to keep the demo quick and lightweight.
# ============================================================================
def _read_batch(tar, name):
    """
    Reads a single pickle batch file from the CIFAR-10 compressed tarball.

    Parameters:
    -----------
    tar : tarfile.TarFile
        The opened tar file pointer.
    name : str
        The member name of the batch file in the tarball.

    Returns:
    --------
    tuple (np.ndarray, np.ndarray)
        - Data array of shape (batch_size, 3072) representing pixel values.
        - Labels array of shape (batch_size,) representing correct class index.
    """
    d = pickle.load(tar.extractfile(name), encoding="bytes")
    return d[b"data"], np.array(d[b"labels"])

def load_cifar2(c0=AIRPLANE, c1=AUTOMOBILE, n_train=3000, n_test=200):
    """
    Downloads CIFAR-10 if not already present, extracts airplane and automobile images,
    normalizes pixel values to [0.0, 1.0], and caches the result for fast re-runs.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `255.0`: The maximum integer value of a color pixel (8-bit representation).
      Dividing by 255.0 normalizes values to [0.0, 1.0]. This standard scaling prevents
      exploding gradients during neural network training.
    - `3000` (n_train) & `200` (n_test): The subset size. This is chosen to be large enough
      to train standard CNNs to high accuracy (>=90%) in seconds without requiring GPU acceleration.

    Parameters:
    -----------
    c0 : int
        First class category to select (Airplane = 0).
    c1 : int
        Second class category to select (Automobile = 1).
    n_train : int
        Number of training images to sample.
    n_test : int
        Number of test images to sample.

    Returns:
    --------
    tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        - Xtr: Training images (n_train, 3072).
        - ytr: Training labels (n_train,).
        - Xte: Test images (n_test, 3072).
        - yte: Test labels (n_test,).
        """
    npz = os.path.join(CACHE, f"cifar2_{c0}-{c1}_{n_train}_{n_test}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    os.makedirs(CACHE, exist_ok=True)
    tarp = os.path.join(CACHE, "cifar-10-python.tar.gz")
    if not os.path.exists(tarp):                  
        urllib.request.urlretrieve(CIFAR_URL, tarp)
    with tarfile.open(tarp) as tar:
        names = tar.getnames()
        trd = [_read_batch(tar, n) for n in sorted(names) if "data_batch" in n]
        ted = [_read_batch(tar, n) for n in names if "test_batch" in n]
    Xtr_all = np.concatenate([d for d, _ in trd]); ytr_all = np.concatenate([l for _, l in trd])
    Xte_all = np.concatenate([d for d, _ in ted]); yte_all = np.concatenate([l for _, l in ted])

    def subset(X, y, n_per):
        # Extracts `n_per` images for each class and maps them to labels 0 and 1.
        outX, outY = [], []
        for new, c in enumerate((c0, c1)):        # c0 -> 0, c1 -> 1
            idx = rng.permutation(np.where(y == c)[0])[:n_per]
            outX.append(X[idx]); outY.append(np.full(len(idx), new))
        # Normalize pixel integers in 0..255 to floats in 0.0..1.0
        X = np.concatenate(outX).astype(np.float32) / 255.0
        y = np.concatenate(outY).astype(np.int64)
        p = rng.permutation(len(y))               # Shuffle the training batch
        return X[p], y[p]

    Xtr, ytr = subset(Xtr_all, ytr_all, n_train // 2)
    Xte, yte = subset(Xte_all, yte_all, n_test // 2)
    np.savez(npz, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte

Xtr, ytr, Xte, yte = load_cifar2()

# ============================================================================
# NEURAL NETWORKS (CNNs)
# Convolutional Neural Networks slide small matrices (filters) across images
# to learn local patterns like edges, shapes, and features.
# ============================================================================

# Before passing pixels to a network, we subtract the mean color and divide by
# standard deviation. This centres the values around 0, which makes training much faster.
# We do this normalization inside the model's `forward` function so our attacks can
# work directly with plain [0,1] images.
# These values are standard statistics computed across the entire CIFAR-10 dataset.
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

class VictimNet(nn.Module):                       
    """
    The Victim CNN model.
    It uses 3x3 pixel filters. The first layer outputs 32 feature maps, the second 64, and the third 64.
    
    Explanation of PyTorch Layers for Beginners:
    -------------------------------------------
    - `nn.Conv2d(in_channels, out_channels, kernel_size)`: Performs a 2D convolution.
      Slides a small matrix of weights (filters) of size `kernel_size x kernel_size` across
      the image's channels to extract local visual patterns like edges.
    - `F.relu()`: Rectified Linear Unit. A non-linear activation function that replaces
      all negative numbers with 0.0. This allows the network to learn complex patterns.
    - `F.max_pool2d(x, 2)`: Slides a 2x2 window and keeps only the maximum value.
      This divides both the width and height of the image by 2, shrinking spatial size.
    - `nn.Linear(in_features, out_features)`: A fully connected layer that multiplies the
      flattened pixel representation by a weight matrix to produce classification scores (logits).
    """
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 2)
    def forward(self, x):                         
        x = (x - MEAN) / STD                      
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   
        x = F.max_pool2d(F.relu(self.c3(x)), 2)   
        return self.fc(x.flatten(1))

class SurrogateNet(nn.Module):                    
    """
    The Surrogate CNN helper model.
    It is deliberately designed with a different architecture (5x5 filters then 3x3 filters,
    and adjustable width of filters) so we can test "cross-architecture transferability":
    whether an attack designed on one architecture can fool a completely different model.
    """
    def __init__(self, width=1.0):
        super().__init__()
        w1, w2 = max(2, int(16 * width)), max(2, int(48 * width))
        self.c1 = nn.Conv2d(3, w1, 5, padding=2)
        self.c2 = nn.Conv2d(w1, w2, 3, padding=1)
        self.fc = nn.Linear(w2 * 8 * 8, 2)
    def forward(self, x):
        x = (x - MEAN) / STD
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))

def train_net(net, X, y, epochs=5, bs=64, seed=0, tag=None):
    """
    Standard neural network training loop.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `epochs=12` (passed from driver): Number of complete runs through the dataset.
      Chosen to reach >=94% test accuracy for our CIFAR-10 classifier.
    - `bs=64` (batch size): The number of images processed at once to compute gradient updates.
    - `1e-3` (learning rate): The step size coefficient used by the Adam optimizer.

    Parameters:
    -----------
    net : nn.Module
        The network to train.
    X : np.ndarray
        Input training images.
    y : np.ndarray
        Correct label indices.
    epochs : int, default=5
        Number of complete training passes.
    bs : int, default=64
        Batch size.
    seed : int, default=0
        Random seed for parameter initialization.
    tag : str, optional
        Checkpoint file name.

    Returns:
    --------
    nn.Module
        The trained network.
    """
    if tag:
        ckpt = os.path.join(CACHE, f"net_{tag}.pt")
        if os.path.exists(ckpt):                  # Load weights if already trained
            net.load_state_dict(torch.load(ckpt)); net.eval(); return net
    torch.manual_seed(seed)                       
    Xt = torch.tensor(X.reshape(-1, 3, 32, 32), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    net.train()                                   
    for _ in range(epochs):                       
        perm = torch.randperm(len(Xt))            # Shuffle data every epoch
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
    """
    Calculates the percentage of images the network classifies correctly.

    Parameters:
    -----------
    net : nn.Module
        The neural network.
    X : np.ndarray
        Images to classify.
    y : np.ndarray
        Correct ground-truth labels.

    Returns:
    --------
    float
        The accuracy score (0.0 to 1.0).
    """
    with torch.no_grad():
        pred = net(torch.tensor(X.reshape(-1, 3, 32, 32), dtype=torch.float32)).argmax(1).numpy()
    return float((pred == y).mean())

# ---- Interfaces for Attack Context ----
def make_label(net):
    """
    Wraps a PyTorch CNN model to provide a simple black-box hard-label prediction function.
    """
    def label(x):                                 
        with torch.no_grad():                     
            t = torch.tensor(x.reshape(1, 3, 32, 32), dtype=torch.float32)
            return int(net(t).argmax(1).item())   
    return label

def make_sgrad(net):
    """
    Wraps a model to calculate the normalized mathematical input gradient.
    Setting `requires_grad=True` allows tracking operations on `t`. `backward()`
    computes derivatives of the classification loss with respect to every pixel.
    
    Explanation of Math/Linear Algebra:
    ----------------------------------
    - `np.linalg.norm(g)`: The L2 norm of the gradient vector. We divide `g` by this norm
      to normalize it to unit length (length = 1.0). This isolates the direction of the change,
      which is the adversarial direction.
    """
    def sgrad(x, y0):
        t = torch.tensor(x.reshape(1, 3, 32, 32), dtype=torch.float32, requires_grad=True)
        F.cross_entropy(net(t), torch.tensor([y0])).backward()   
        g = t.grad.numpy().ravel().astype(np.float64)            
        n = np.linalg.norm(g)                                     
        return g / n if n else g                                  
    return sgrad

# ---- The surrogate quality knob ----
def build_surrogate(frac=1.0, width=1.0, label_noise=0.0, epochs=10, seed=7, tag=None):
    """
    Helper to construct surrogates with adjustable dataset sizes, widths, or noise.
    Used to sweep and test how SQBA degrades when surrogate quality decreases.
    """
    idx = rng.choice(len(Xtr), max(64, int(frac * len(Xtr))), replace=False)
    Xs, ys = Xtr[idx].copy(), ytr[idx].copy()
    if label_noise:                               
        m = rng.random(len(ys)) < label_noise
        ys[m] = rng.integers(0, 2, m.sum())
    return train_net(SurrogateNet(width), Xs, ys, epochs=epochs, seed=seed, tag=tag)

# ============================================================================
# WHITE-BOX GRADIENT ATTACKS (Crafted on the surrogate)
# Since we own the surrogate, we can perform gradient descent on its pixels.
# We then test if these crafted adversarial images successfully fool the victim.
# ============================================================================
def pgd_on_surr(x0, y0, sgrad, eps, steps):
    """
    Projected Gradient Descent (PGD):
    Steps along the direction of the gradient sign to maximize loss,
    clipping the pixels back into the valid [0,1] range and within the
    budget distance `eps` from the original image at each step.

    Explanation of Math/Logic:
    -------------------------
    - `np.sign(sgrad(x, y0))`: Takes the sign of every pixel gradient element (+1 for positive,
      -1 for negative, 0 for zero). Stepping along the signs maximizes the L-infinity perturbation.
    - `np.clip(x, x0 - eps, x0 + eps)`: Projects the updated image vector back onto the L-infinity
      sphere around `x0` by restricting the change of every pixel to be at most `eps`.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `eps / 4`: The step size `alpha`. A standard heuristic. Iterating 4 or more smaller steps of size
      `eps/4` allows PGD to converge cleanly without overshooting.
    """
    x, alpha = x0.copy(), (eps / 4 if steps > 1 else eps)
    for _ in range(steps):
        x = x + alpha * np.sign(sgrad(x, y0))     
        x = np.clip(np.clip(x, x0 - eps, x0 + eps), 0, 1)   
    return x

def attack_fgsm(x0, y0, sgrad, eps=0.05):
    """
    Fast Gradient Sign Method (FGSM):
    Takes exactly one step of size `eps` along the sign of the gradient.
    """
    return pgd_on_surr(x0, y0, sgrad, eps, 1)

def attack_pgd(x0, y0, sgrad, eps=0.05):
    """
    Projected Gradient Descent (PGD) attack:
    Iterates the FGSM step 20 times with smaller step sizes.
    """
    return pgd_on_surr(x0, y0, sgrad, eps, 20)

def attack_transfer(ctx, x0, y0, sgrad):
    """
    Transfer Attack:
    Crafts a PGD image on the surrogate, slowly increasing the budget `eps`
    until it successfully flips the label on the victim model.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `np.linspace(0.01, 0.25, 16)`: We sweep 16 levels of epsilon from 0.01 (almost invisible)
      up to 0.25 (the limit of success). This is done to find the smallest budget that succeeds.
    - `15`: The number of PGD iterations used at each budget check.
    """
    for eps in np.linspace(0.01, 0.25, 16):       
        x = pgd_on_surr(x0, y0, sgrad, eps, 15)
        if ctx.is_adv(x, y0):
            return x
    return x

# ============================================================================
# TRAIN / PREPARE
# ============================================================================
print("training tiny CNNs (cached after first run) ...")
victim = train_net(VictimNet(), Xtr, ytr, epochs=12, seed=0, tag="victim_air_auto")
# White-box replica surrogate (using the exact same seed and weights as the victim model to represent full parameter access)
surr_wb = train_net(VictimNet(), Xtr, ytr, epochs=12, seed=0, tag="victim_air_auto")
# Black-box generic surrogate (attacker has generic external helper model)
surr = build_surrogate(tag="surr_strong_air_auto")

victim_label = make_label(victim)
print(f"victim acc {accuracy(victim, Xte, yte):.3f} | "
      f"replica surrogate acc {accuracy(surr_wb, Xte, yte):.3f} | "
      f"generic surrogate acc {accuracy(surr, Xte, yte):.3f}")
print(f"victim/replica agreement {np.mean([victim_label(x) == make_label(surr_wb)(x) for x in Xte]):.3f} | "
      f"victim/generic agreement {np.mean([victim_label(x) == make_label(surr)(x) for x in Xte]):.3f}")

def make_ctx(s):
    return Ctx(victim_label, make_sgrad(s), Xtr, ytr, rng, lo=0.0, hi=1.0)

ctx = make_ctx(surr)
sgrad = make_sgrad(surr)
wb_grad = make_sgrad(surr_wb)

# Perturbation size metric (rho): L2 norm of the change divided by the L2 norm of the original image.
# An attack is considered successful if rho <= 0.25.
#
# Explanation of Math:
# --------------------
# - `np.linalg.norm(adv - x0)`: L2 distance of the perturbation (the difference).
# - `np.linalg.norm(x0)`: L2 distance of the original image.
# - Dividing them yields a relative scaling ratio (rho).
rho = lambda adv, x0: np.linalg.norm(adv - x0) / np.linalg.norm(x0)

# ============================================================================
# (A) Single-image lineage table
# ============================================================================
# Select the first test image that the victim classifies correctly.
x0 = next(Xte[i].astype(np.float64) for i in range(len(Xte)) if victim_label(Xte[i]) == yte[i])
y0 = victim_label(x0)
loose = loose_start(ctx, x0, y0)

def run(fn):
    ctx.q = 0; ctx.budget = float("inf")
    return fn(), ctx.q

plan = [
    ("white-box", "fgsm",       "'14", "one gradient-sign step on replica surr",     lambda: attack_fgsm(x0, y0, wb_grad)),
    ("white-box", "pgd",        "'17", "iterated FGSM on replica surr",              lambda: attack_pgd(x0, y0, wb_grad)),
    ("transfer",  "transfer",   "'16", "grow eps on generic surr til victim flips",  lambda: attack_transfer(ctx, x0, y0, sgrad)),
    ("hard-label","random",     " - ", "blind noise (scaffold)",                     lambda: attack_random(ctx, x0, y0, N=400, scale=0.2)),
    ("hard-label","line",       " - ", "nearest class + binsearch (scaffold)",       lambda: attack_line(ctx, x0, y0)),
    ("hard-label","boundary",   "'18", "random walk along boundary",                 lambda: attack_boundary(ctx, x0, y0, loose, steps=400)),
    ("hard-label","sign-opt",   "'20", "g(theta) via SIGN of dir. deriv",            lambda: attack_sign_opt(ctx, x0, y0, loose)),
    ("hard-label","hopskipjump","'20", "MC boundary-normal estimate",                lambda: attack_hsj(ctx, x0, y0, loose)),
    ("hard-label","triangle",   "'22", "geometric law of sines in DCT subspace",     lambda: attack_triangle(ctx, x0, y0, loose)),
    ("surrogate", "biased-bdry","'19", "surrogate-biased boundary walk",             lambda: attack_boundary(ctx, x0, y0, loose, bias=0.5, steps=400)),
]
rows = []
for tier, name, year, idea, fn in plan:
    x, q = run(fn)
    r_val = rho(x, x0)
    flip_val = ctx.fooled(x, y0)
    win_val = flip_val and r_val <= 0.25
    rows.append((tier, name, year, q, r_val, flip_val, win_val, idea))
x, q = run(lambda: attack_sqba(ctx, x0, y0, loose)[0])
r_val = rho(x, x0)
flip_val = ctx.fooled(x, y0)
win_val = flip_val and r_val <= 0.25
rows.append(("surrogate", "sqba", "'24", q, r_val, flip_val, win_val, "teaching: single surrogate normal + fallback"))
x, q = run(lambda: attack_sqba_full(ctx, x0, y0)[0])
r_val = rho(x, x0)
flip_val = ctx.fooled(x, y0)
win_val = flip_val and r_val <= 0.25
rows.append(("surrogate", "sqba-full", "'24", q, r_val, flip_val, win_val, "paper Algo 1: multi-gradient + beta switch"))

print("\n" + "=" * 88)
print(f"CIFAR-10 (airplane vs automobile) ATTACK LINEAGE  (one image; hard-label only; success = rho<=0.25)")
print("=" * 88)
print(f"{'attack':<13}{'yr':>4}{'victim_q':>10}{'rho':>8}{'flip':>6}{'win':>5}   key idea")
last = None
for tier, name, year, q, r, flip, win, idea in rows:
    if tier != last:
        print(f"-- {tier.upper()} " + "-" * (82 - len(tier))); last = tier
    print(f"{name:<13}{year:>4}{q:>10}{r:>8.3f}{('Y' if flip else 'n'):>6}{('Y' if win else 'n'):>5}   {idea}")

# ============================================================================
# (B) ASR vs query-budget
# ============================================================================
M = 8
asr_imgs = [Xte[i].astype(np.float64) for i in range(len(Xte)) if victim_label(Xte[i]) == yte[i]][:M]
budgets = [100, 250, 1000]
runners = {                                       
    "boundary":    lambda c, x, y: attack_boundary(c, x, y, loose_start(c, x, y), steps=100000),
    "sign-opt":    lambda c, x, y: attack_sign_opt(c, x, y, loose_start(c, x, y), iters=100),
    "hopskipjump": lambda c, x, y: attack_hsj(c, x, y, loose_start(c, x, y), iters=100),
    "triangle":    lambda c, x, y: attack_triangle(c, x, y, loose_start(c, x, y), iters=200),
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
            if ctx.fooled(adv, yi) and rho(adv, xi) <= 0.25:
                asr[name][b] += 1

print("\n" + "=" * 88)
print(f"ASR vs QUERY BUDGET  (success = rho<=0.25 within budget; {M} images, strong surrogate)")
print("=" * 88)
print(f"{'attack':<13}" + "".join(f"{('Q=' + str(b)):>9}" for b in budgets))
for name in runners:
    print(f"{name:<13}" + "".join(f"{100 * asr[name][b] / M:>8.0f}%" for b in budgets))

# ============================================================================
# (C) Weaken-surrogate sweep
# ============================================================================
print("\n" + "=" * 88)
print(f"WEAKEN-SURROGATE SWEEP (SQBA decays toward HopSkipJump; avg of {min(4, M)} images)")
print("=" * 88)
sweep_imgs = asr_imgs[:4]
print(f"{'surrogate':<10}{'agree%':>8}{'rho':>8}{'victim_q':>10}{'white_only':>12}{'fallbacks':>11}")
for nm, cfg in [("strong ", dict(tag="surr_strong_air_auto")),
                ("weak   ", dict(frac=0.2, width=0.5, label_noise=0.2, epochs=2, seed=11, tag="surr_weak_air_auto")),
                ("useless", dict(frac=0.05, width=0.3, label_noise=0.5, epochs=1, seed=23, tag="surr_useless_air_auto"))]:
    s = surr if cfg.get("tag") == "surr_strong_air_auto" else build_surrogate(**cfg)
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
