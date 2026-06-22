# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
DIGITS Demo: A fast and lightweight introduction to adversarial attacks.

In this demo, we run a sequence of attacks on a machine learning model trained to recognize 
handwritten digits (from 0 to 9). Each digit image is very small: 8 pixels by 8 pixels (a total of 64 pixels).
The model we are attacking is a simple neural network (called `MLPClassifier` from scikit-learn).
The attacks will try to slightly tweak these 64 pixel values to fool the classifier.

All the actual attack algorithms live in `attacks.py`. This file handles:
1. Loading the dataset of digits.
2. Training the "victim" model (which we want to fool) and the "surrogate" model (our helper model).
3. Defining the surrogate's gradient calculation (which tells us how to nudge pixels).
4. Running each attack on a digit image and printing a nice comparison table.

To run this file:
    uv run lineage.py
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from attacks import (Ctx, loose_start, attack_random, attack_line, attack_boundary,
                     attack_opt, attack_sign_opt, attack_hsj, attack_triangle, attack_sqba, attack_sqba_full)

rng = np.random.default_rng(0)

# ---- 1. Load data and train the black-box victim model ----
# We use scikit-learn's built-in digits dataset.
digits = load_digits()
# Flatten each 8x8 image into a 1D vector of 64 numbers, and scale values from [0, 16] to [0, 1] for better training.
X = digits.images.reshape(len(digits.images), -1) / 16.0      
y = digits.target
# Split the dataset: 80% for training the model, 20% for testing the attacks.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
# Train our victim model. This is the model we want to fool.
# It has one hidden layer of 128 neurons.
victim = MLPClassifier(hidden_layer_sizes=(128,), max_iter=600, random_state=0).fit(Xtr, ytr)


def victim_label(x):                                  
    # Helper function: asks the victim model to guess the digit (label) for a flat input image.
    return int(victim.predict(x.reshape(1, -1))[0])


# ---- 2. Build the surrogate helper model ----
def build_surrogate(frac=1.0, hidden=96, seed=7, label_noise=0.0):
    """
    Builds the surrogate model. This is a helper model we train ourselves.
    Since we own it, we can calculate its math gradients.
    
    We can tweak these parameters to make the surrogate weaker (to see how SQBA handles bad surrogates):
    - `frac`: fraction of training data to use (less data -> worse model).
    - `hidden`: number of neurons in the hidden layer (fewer neurons -> weaker model).
    - `label_noise`: percentage of training labels to randomly mess up.
    """
    idx = rng.choice(len(Xtr), max(20, int(frac * len(Xtr))), replace=False)
    Xs, ys = Xtr[idx].copy(), ytr[idx].copy()
    if label_noise:
        m = rng.random(len(ys)) < label_noise
        ys[m] = rng.integers(0, 10, m.sum())
    return MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=400, random_state=seed).fit(Xs, ys)


def mlp_sgrad(surr):
    """
    Calculates the input gradient of our surrogate model by hand.
    
    A gradient is a direction in pixel space. If we move the pixels in this direction,
    the model's loss (how wrong it is) increases the fastest. This is the adversarial direction!
    
    For a beginner, here is how a simple one-hidden-layer network calculates its output:
    1. First Layer: Multiply the input pixels `x` by a weights matrix `W1` and add bias `b1`.
       Formula: a1 = x * W1 + b1
    2. Activation (ReLU): Replace all negative numbers with zero.
       Formula: h = max(a1, 0)
    3. Second Layer: Multiply by weights `W2` and add bias `b2`.
       Formula: a2 = h * W2 + b2 (this gives one score per class/digit)
    4. Probabilities (Softmax): Convert scores into probabilities that sum up to 1.
       Formula: p = softmax(a2)
    
    To find how to change the pixels to increase loss, we apply the "chain rule" (calculus)
    in reverse order (this is called "backpropagation"):
    - Step A: Find how changes in class scores affect loss. Result is `p - onehot(y0)`.
    - Step B: Propagate back through the second layer weights: `that @ W2.T`.
    - Step C: Propagate through the ReLU (only pass gradients where the value was positive): `* (a1 > 0)`.
    - Step D: Propagate back through the first layer weights: `@ W1.T`.
    
    We return the final pixel-direction as a unit vector (length 1) so only the direction matters.
    """
    W1, W2 = surr.coefs_; b1, b2 = surr.intercepts_      
    def sgrad(x, y0):
        # 1. Forward pass: compute outputs of each layer
        a1 = x @ W1 + b1; h = np.maximum(a1, 0.0); a2 = h @ W2 + b2   
        # 2. Softmax: convert output scores to probabilities
        p = np.exp(a2 - a2.max()); p /= p.sum()                       
        # 3. Create a one-hot vector representing the correct class label
        onehot = np.zeros_like(p); onehot[y0] = 1.0                   
        # 4. Backpropagation: compute the derivative of the loss with respect to the input pixels
        g = (((p - onehot) @ W2.T) * (a1 > 0)) @ W1.T                 
        n = np.linalg.norm(g)
        return g / n if n else g                                      
    return sgrad


def make_ctx(surr):
    # Create the Context object for the attacks, specifying coordinate bounds [0.0, 1.0] for pixels.
    return Ctx(victim_label, mlp_sgrad(surr), Xtr, ytr, rng, lo=0.0, hi=1.0)


# ============================================================================
# Driver: run the lineage on one image, then the weaken-surrogate sweep.
# ============================================================================
surr = build_surrogate()
ctx = make_ctx(surr)
# Pick the first test image, and find its correct label from the victim.
x0 = Xte[0]; y0 = victim_label(x0)
# Generate a deliberately loose (far-away) starting adversarial point.
loose = loose_start(ctx, x0, y0)


def run(fn):
    # Reset queries to 0 and clear any budgets before running each attack.
    ctx.q = 0; ctx.budget = float("inf")
    x = fn()
    return x, ctx.q


# Define our attack checklist.
plan = [
    ("white-box", "(see cifar)", "", "FGSM/PGD/transfer need pixel CNNs; shown in cifar_lineage.py", None),
    ("hard-label", "random",     " - ", "blind noise (scaffold)",                   lambda: attack_random(ctx, x0, y0)),
    ("hard-label", "line",       " - ", "aim at nearest class + binsearch (scaffold)", lambda: attack_line(ctx, x0, y0)),
    ("hard-label", "boundary",   "'18", "random walk along boundary",               lambda: attack_boundary(ctx, x0, y0, loose)),
    ("hard-label", "opt",        "'19", "minimize g(theta), zeroth-order",          lambda: attack_opt(ctx, x0, y0, loose)),
    ("hard-label", "sign-opt",   "'20", "g(theta) via SIGN of dir. deriv",          lambda: attack_sign_opt(ctx, x0, y0, loose)),
    ("hard-label", "hopskipjump","'20", "MC boundary-normal estimate",              lambda: attack_hsj(ctx, x0, y0, loose)),
    ("hard-label", "triangle",   "'22", "geometric law of sines in DCT subspace",   lambda: attack_triangle(ctx, x0, y0, loose)),
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
