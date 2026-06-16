# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
================================================================================
THE RESEARCH LINEAGE OF BLACK-BOX ADVERSARIAL ATTACKS  --  a runnable ladder
that ends at SQBA (Hard-label based Small Query Black-box Adversarial Attack).
================================================================================

WHAT THIS IS
------------
A single, CPU-only script that implements eleven adversarial attacks on the SAME
victim model and the SAME image, so you can read the history of the field as a
table.  The attacks are ordered the way the research actually progressed: by
THREAT MODEL -- i.e. by *how little* the attacker is allowed to know.  Each tier
relaxes one assumption of the tier above it.

    TIER                     ASSUMES                          ATTACKS
    ----------------------   ------------------------------   ----------------------
    white-box                full gradients (of a surrogate)  FGSM '14, PGD '17
    transfer                 a surrogate + ~0 victim queries  Papernot '16
    hard-label / decision    only the victim's top-1 LABEL    Boundary '18, OPT '19,
                                                              Sign-OPT '20, HSJA '20
    surrogate-assisted       label + a surrogate's gradient   Biased Boundary '19,
                                                              SQBA '24

WHY "HARD-LABEL" IS THE HARD CASE
---------------------------------
A "hard-label" oracle returns ONLY the predicted class -- no probabilities, no
logits.  The model's output is therefore a step function: flat almost
everywhere, with a jump at the decision boundary.  You cannot differentiate it,
and naive finite differences return zero.  Every hard-label attack below is, at
heart, a way to extract a usable signal from that 0/1 "did the label change?"
feedback.

THE TWO IDEAS THAT EVERYTHING REUSES
------------------------------------
 1. BINARY SEARCH TO THE BOUNDARY (`binary_search`): given an adversarial point
    and the clean image, bisect the segment between them to land exactly on the
    boundary -- the closest adversarial point along that line.  This turns a
    discrete label into a real-valued distance you can minimize.
 2. THE BOUNDARY NORMAL: near the boundary, the direction that most quickly
    changes the label is the boundary's normal vector.  Decision-based attacks
    differ mainly in HOW they estimate that normal:
        - Boundary Attack : don't; random-walk instead.
        - HopSkipJump     : estimate it by Monte-Carlo querying (costly).
        - SQBA            : read it for FREE from a surrogate's gradient.

THE PUNCHLINE (SQBA)
--------------------
SQBA fuses the two historical branches: it runs a decision-based boundary search
(the hard-label branch) but takes its boundary-normal estimate from a white-box
surrogate's gradient (the transfer branch).  That makes it "small query": the
expensive victim queries that HopSkipJump spends ESTIMATING the normal are
replaced by a free surrogate backward pass.  When the surrogate stops helping,
SQBA falls back to the query-based estimate -- gracefully decaying into
HopSkipJump.  The weaken-surrogate sweep at the bottom demonstrates exactly that.

THREAT-MODEL HONESTY
--------------------
Every attack here reads ONLY the victim's top-1 label (hard-label).  The
white-box and transfer rungs use gradients of a SEPARATE SURROGATE model that
the attacker trained themselves -- never the victim's gradients.  `random` and
`line` are pedagogical scaffolding, not published attacks; they are labelled as
such.

RUN
---
    uv run lineage.py
(`uv` reads the PEP-723 header above and provisions numpy + scikit-learn in an
ephemeral environment; nothing is installed globally.)

See README.md for the expected output, the per-rung paper mapping, and caveats.
================================================================================
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# One global RNG, fixed seed, so every run is reproducible.  (Note: numpy's
# Generator is used everywhere instead of the legacy np.random.* functions.)
rng = np.random.default_rng(0)

# ============================================================================
# DATA + BLACK-BOX VICTIM
# ----------------------------------------------------------------------------
# We use sklearn's `digits` dataset: 1797 images of 8x8 handwritten digits, so
# every input is a 64-dimensional vector.  Pixel values are 0..16; we scale to
# [0, 1] so perturbation sizes are easy to read.  The dataset ships with sklearn
# (no download), and the victim trains in ~1 second on a CPU.
#
# The victim is an MLP.  Conceptually it is a BLACK BOX: the attacks below never
# look at its weights and never read its probabilities -- they only ever call
# `predict()`, which returns a single class label.
# ============================================================================
digits = load_digits()
X = digits.images.reshape(len(digits.images), -1) / 16.0      # (1797, 64) in [0,1]
y = digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
victim = MLPClassifier(hidden_layer_sizes=(128,), max_iter=600,
                       random_state=0).fit(Xtr, ytr)

# `victim_q` counts EVERY query to the victim.  It is the resource each attack is
# trying to spend as little of as possible -- the whole point of "small query".
# We reset it to 0 before timing each attack (see `run()`).
victim_q = 0
def predict(x):
    """The ONLY channel to the victim.  Returns the top-1 class as an int and
    increments the global query counter.  Inputs are clipped to the valid image
    range [0,1] first, because an out-of-range 'image' is not a real query."""
    global victim_q
    victim_q += 1
    return int(victim.predict(np.clip(x, 0, 1).reshape(1, -1))[0])

def is_adv(x, y0):
    """True iff `x` is misclassified, i.e. the victim's label != the true label
    `y0`.  This single boolean is the entire feedback signal a hard-label attack
    receives.  Costs one victim query."""
    return predict(x) != y0

def fooled(x, y0):
    """Same test as `is_adv`, but does NOT increment the counter.  Used only for
    REPORTING the final outcome of an attack, so the report itself doesn't
    pollute the query budget we are measuring."""
    return int(victim.predict(np.clip(x, 0, 1).reshape(1, -1))[0]) != y0

def binary_search(x_adv, x0, y0, tol=1e-2):
    """BOUNDARY-PROJECTION PRIMITIVE (reused by rungs 2-5 and 'line').

    Given an adversarial point `x_adv` (wrong label) and the clean image `x0`
    (correct label), the decision boundary lies somewhere on the segment between
    them.  Bisect that segment until the two ends are within `tol`, always
    keeping `lo` non-adversarial and `hi` adversarial.  Return `hi`: the closest
    adversarial point to `x0` along this line -- i.e. the smallest perturbation
    in this particular direction.  Each bisection step costs one victim query."""
    lo, hi = x0.copy(), x_adv.copy()                 # lo = clean side, hi = adv side
    while np.linalg.norm(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if is_adv(mid, y0):
            hi = mid                                  # midpoint still adversarial -> shrink toward x0
        else:
            lo = mid                                  # midpoint safe -> push outward
    return hi

# ============================================================================
# SURROGATE (WHITE BOX) + THE WEAKEN KNOB
# ----------------------------------------------------------------------------
# A surrogate is a stand-in model the attacker trains themselves and therefore
# has full white-box access to.  The hope (transferability) is that adversarial
# directions for the surrogate are also adversarial for the victim.
#
# `build_surrogate` is the experiment's KNOB.  Three parameters control how well
# the surrogate mimics the victim, letting us watch SQBA degrade:
#     frac        : fraction of the training set the surrogate sees (less = worse)
#     hidden      : width of its single hidden layer (smaller = weaker)
#     label_noise : fraction of training labels randomly corrupted (more = worse)
# A different architecture/seed/data split from the victim makes transfer
# IMPERFECT, which is the realistic case.
# ============================================================================
def build_surrogate(frac=1.0, hidden=96, seed=7, label_noise=0.0):
    idx = rng.choice(len(Xtr), max(20, int(frac * len(Xtr))), replace=False)
    Xs, ys = Xtr[idx].copy(), ytr[idx].copy()
    if label_noise:                                   # corrupt some labels -> worse mimic
        m = rng.random(len(ys)) < label_noise
        ys[m] = rng.integers(0, 10, m.sum())
    return MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=400,
                         random_state=seed).fit(Xs, ys)

def surrogate_grad(x, y0, surr):
    """Gradient of the surrogate's cross-entropy loss (on the true class y0) with
    respect to the INPUT pixels, computed analytically from the surrogate's
    weights and returned as a UNIT vector.

    This is the free, white-box signal that the transfer and surrogate-assisted
    attacks rely on.  For a one-hidden-layer ReLU MLP with softmax output the
    backward pass is just four lines:
        forward : a1 = W1.x + b1 ; h = relu(a1) ; a2 = W2.h + b2 ; p = softmax(a2)
        backward: dL/da2 = p - onehot(y0)
                  dL/dh  = (p - onehot) . W2^T
                  dL/da1 = dL/dh * 1[a1 > 0]            (ReLU derivative)
                  dL/dx  = dL/da1 . W1^T
    The vector points in the direction that INCREASES loss on the true class,
    i.e. that makes `x` MORE adversarial.  (Costs zero victim queries -- it only
    touches the surrogate, which the attacker owns.)"""
    W1, W2 = surr.coefs_                              # weight matrices  (64,H), (H,10)
    b1, b2 = surr.intercepts_                         # biases           (H,),   (10,)
    a1 = x @ W1 + b1
    h = np.maximum(a1, 0.0)                           # ReLU
    a2 = h @ W2 + b2
    p = np.exp(a2 - a2.max()); p /= p.sum()           # softmax (shifted for stability)
    onehot = np.zeros_like(p); onehot[y0] = 1.0
    g = (((p - onehot) @ W2.T) * (a1 > 0)) @ W1.T     # the four backward lines, fused
    n = np.linalg.norm(g)
    return g / n if n else g                          # unit vector (or 0 if degenerate)

# ============================================================================
# TIER 1 -- WHITE-BOX gradient attacks.
# ----------------------------------------------------------------------------
# These need real gradients, so they are run against the SURROGATE, not the
# victim.  They spend ZERO victim queries; they "win" only if the surrogate-
# crafted example happens to TRANSFER to the victim.  Historically these are the
# foundation: FGSM (Goodfellow et al. 2014) introduced the one-step gradient-sign
# attack; PGD (Madry et al. 2017) iterated it into the canonical white-box attack.
# ============================================================================
def pgd_on_surrogate(x0, y0, surr, eps, steps):
    """L-infinity Projected Gradient Descent that ASCENDS the surrogate's loss on
    the true class y0 (thereby pushing the prediction away from y0).
        - take a step of size `alpha` along the SIGN of the gradient,
        - project back into the L-inf ball of radius `eps` around x0,
        - clip to the valid image range.
    `steps == 1` reduces this to FGSM (one full-size step)."""
    x, alpha = x0.copy(), eps / 4 if steps > 1 else eps
    for _ in range(steps):
        x = x + alpha * np.sign(surrogate_grad(x, y0, surr))   # gradient-SIGN step
        x = np.clip(np.clip(x, x0 - eps, x0 + eps), 0, 1)      # project to L-inf ball, then to [0,1]
    return x

def attack_fgsm(x0, y0, surr, eps=0.30):
    """Goodfellow et al. 2014 -- one gradient-sign step."""
    return pgd_on_surrogate(x0, y0, surr, eps, steps=1)

def attack_pgd(x0, y0, surr, eps=0.30):
    """Madry et al. 2017 -- iterated FGSM.  Stronger/more reliable than FGSM; at a
    fixed L-inf radius its L2 size is similar (both saturate the same ball), so on
    this easy task its advantage over FGSM is reliability rather than a smaller L2."""
    return pgd_on_surrogate(x0, y0, surr, eps, steps=20)

# ============================================================================
# TIER 2 -- TRANSFER attack (Papernot et al. 2016).
# ----------------------------------------------------------------------------
# The first true BLACK-BOX attack: craft on a surrogate, apply to the victim.
# Here we grow the L-inf budget `eps` and re-craft until the VICTIM's label
# flips.  The only victim contact is the one label check per `eps` value, so a
# successful transfer costs only a handful of queries -- but it gives no control
# over perturbation size and no guarantee of success if transfer is poor.  This
# is the conceptual seed of SQBA's use of a surrogate.
# ============================================================================
def attack_transfer(x0, y0, surr):
    for eps in np.linspace(0.05, 0.8, 16):           # smallest budget that transfers
        x = pgd_on_surrogate(x0, y0, surr, eps, steps=15)
        if is_adv(x, y0):                            # one victim query per eps
            return x
    return x                                          # (return the largest attempt if none transfer)

# ============================================================================
# TIER 3 -- HARD-LABEL / DECISION-BASED attacks (top-1 label only).
# ============================================================================

def attack_random(x0, y0, N=1500, scale=0.5):
    """SCAFFOLD (not a paper): blind baseline.  Sample Gaussian noise, keep the
    smallest perturbation that happens to flip the label.  Establishes how bad
    'no optimization' is -- huge L2, thousands of queries."""
    best, best_d = None, np.inf
    for _ in range(N):
        x = np.clip(x0 + rng.standard_normal(x0.shape) * scale, 0, 1)
        if is_adv(x, y0) and np.linalg.norm(x - x0) < best_d:
            best, best_d = x, np.linalg.norm(x - x0)
    return best if best is not None else x0

def attack_line(x0, y0):
    """SCAFFOLD (not a paper): aim straight at the NEAREST other-class training
    sample (which is guaranteed adversarial -- it IS another class) and binary-
    search to the boundary.  One direction, no refinement, ~8 queries.  This is
    really the *initialization* step used inside Boundary Attack and OPT."""
    others = Xtr[ytr != y0]
    xt = others[np.argmin(np.linalg.norm(others - x0, axis=1))]
    return binary_search(xt, x0, y0)

def loose_start(x0, y0):
    """A deliberately POOR adversarial starting point: aim at the FARTHEST other-
    class sample.  Rungs 3-5 all start from this same loose point (~L2 1.9) so
    you can watch each refinement drive the perturbation DOWN from an identical,
    unflattering start -- making the optimization, not the initialization, the
    thing being compared."""
    others = Xtr[ytr != y0]
    xt = others[np.argmax(np.linalg.norm(others - x0, axis=1))]
    return binary_search(xt, x0, y0)

def attack_boundary(x0, y0, x_init, surr=None, bias=0.0, steps=500):
    """Boundary Attack (Brendel, Rauber & Bethge 2018) -- the FIRST decision-based
    attack and the origin of the whole hard-label family.

    Idea: stay exactly on the boundary and RANDOM-WALK along it toward x0.
    Each step has two parts:
        - an ORTHOGONAL move (perpendicular to the radius x_b - x0) that explores
          the boundary while keeping the distance to x0 roughly fixed, and
        - a small PULL toward x0 that shrinks the perturbation.
    A proposal is accepted only if it stays adversarial.  Step sizes adapt to the
    acceptance rate (classic Boundary Attack behaviour).

    With `bias > 0` and a surrogate supplied, the orthogonal move is steered
    toward the surrogate gradient -- this single change turns the plain Boundary
    Attack into the BIASED BOUNDARY ATTACK (Brunner et al. 2019, 'Guessing
    Smart'), the direct precursor to SQBA.  (The `plan` table calls this rung
    once with bias=0 -> 'boundary' and once with bias=0.5 -> 'biased-bdry'.)"""
    x_b = x_init.copy()
    sph, src, acc = 0.10, 0.05, []                   # spherical-step, source-step, accept log
    for t in range(steps):
        r = x_b - x0
        d = np.linalg.norm(r)
        eta = rng.standard_normal(x0.shape)
        eta /= np.linalg.norm(eta)
        if bias > 0:                                 # <-- the ONLY Biased-Boundary change
            g = surrogate_grad(x_b, y0, surr)
            eta = (1 - bias) * eta + bias * g        # blend random direction with surrogate gradient
        eta -= (eta @ r) / (d * d) * r               # make the step orthogonal to the radius
        eta *= sph * d / np.linalg.norm(eta)         # scale relative to current distance
        cand = x_b + eta
        cand = x0 + (cand - x0) / np.linalg.norm(cand - x0) * d   # re-project onto the sphere of radius d
        cand = np.clip(cand - src * (cand - x0), 0, 1)           # pull toward x0 (shrink), clip
        ok = is_adv(cand, y0)
        if ok:
            x_b = cand                               # accept only adversarial proposals
        acc.append(ok)
        if len(acc) >= 30:                           # adapt step sizes to keep acceptance ~20-50%
            rate = np.mean(acc[-30:])
            sph, src = (sph * 0.9, src * 0.9) if rate < 0.2 else \
                       ((sph * 1.1, src * 1.1) if rate > 0.5 else (sph, src))
    return x_b

# --- OPT family: reformulate the attack as minimizing a DISTANCE function -----
# OPT (Cheng et al. 2019) reframes the hard-label attack as a continuous
# optimization over the DIRECTION theta:
#       g(theta) = distance from x0 to the boundary along theta
# The closest adversarial example is the theta that minimizes g.  g is not
# directly observable, but `g_dist` computes it from labels alone (expand to
# bracket the boundary, then binary-search).  Crucially g is SMOOTH in theta even
# though the model's output is a step function -- so we can do gradient descent on
# it with a zeroth-order (finite-difference) gradient estimate.
def g_dist(x0, y0, theta, guess=1.0, tol=4e-2):
    """Return g(theta): the distance to the boundary along unit direction theta.
    `guess` warm-starts the search near a previous distance (the single biggest
    query saver).  Returns inf if this direction never crosses the boundary."""
    theta = theta / np.linalg.norm(theta)
    if predict(x0 + guess * theta) == y0:            # guess is too SHORT -> grow outward
        lo = hi = guess
        while predict(x0 + hi * theta) == y0:
            hi *= 1.5
            if hi > 50:
                return np.inf                        # direction never crosses within range
    else:                                            # guess is too LONG -> shrink to bracket
        hi = guess; lo = guess * 0.5
        while predict(x0 + lo * theta) != y0 and lo > 1e-3:
            lo *= 0.5
    while hi - lo > tol:                             # binary-search the bracketed boundary
        mid = 0.5 * (lo + hi)
        hi, lo = (mid, lo) if predict(x0 + mid * theta) != y0 else (hi, mid)
    return hi

def attack_opt(x0, y0, x_init, iters=14, q=6, beta=0.02, alpha=0.3):
    """OPT attack (Cheng et al. 2019).  Minimize g(theta) by zeroth-order gradient
    descent: estimate the gradient of g from `q` random probe directions using
    finite differences, step downhill, keep the step only if g actually
    decreased.  Query-hungry, because EACH probe runs a full `g_dist` search."""
    theta = (x_init - x0) / np.linalg.norm(x_init - x0)          # init direction from the loose start
    g_theta = g_dist(x0, y0, theta, guess=np.linalg.norm(x_init - x0))
    for t in range(iters):
        grad = np.zeros_like(theta)
        for _ in range(q):                                       # finite-difference gradient of g
            u = rng.standard_normal(theta.shape); u /= np.linalg.norm(u)
            grad += (g_dist(x0, y0, theta + beta * u, guess=g_theta) - g_theta) / beta * u
        grad /= q
        nt = theta - alpha * grad; nt /= np.linalg.norm(nt)      # descend: -grad reduces g
        gn = g_dist(x0, y0, nt, guess=g_theta)
        if gn < g_theta:                                         # accept only improvements (monotone)
            theta, g_theta = nt, gn
    return np.clip(x0 + g_theta * theta, 0, 1)

def attack_sign_opt(x0, y0, x_init, iters=14, q=12, beta=0.02, alpha=0.3):
    """Sign-OPT (Cheng et al. 2020).  Same reformulation as OPT, but it does not
    need the MAGNITUDE of the directional derivative -- only its SIGN.  That sign
    is a single query: x0 + g_theta*(theta+beta*u) is adversarial iff the boundary
    along (theta+beta*u) is CLOSER than the current g_theta, i.e. iff g decreased
    in direction u.  So each probe costs ONE query instead of a full g_dist
    search, making Sign-OPT markedly cheaper than OPT for similar quality.

    Sign convention: s=+1 marks directions u that REDUCE g(theta).  grad = mean of
    s*u therefore points toward smaller g, so we ASCEND it (theta + alpha*grad)."""
    theta = (x_init - x0) / np.linalg.norm(x_init - x0)
    g_theta = g_dist(x0, y0, theta, guess=np.linalg.norm(x_init - x0))
    for t in range(iters):
        grad = np.zeros_like(theta)
        for _ in range(q):
            u = rng.standard_normal(theta.shape); u /= np.linalg.norm(u)
            nt = theta + beta * u; nt /= np.linalg.norm(nt)
            s = 1.0 if is_adv(x0 + g_theta * nt, y0) else -1.0   # ONE query: did g decrease along u?
            grad += s * u
        grad /= q
        nt = theta + alpha * grad; nt /= np.linalg.norm(nt)      # ascend toward smaller g
        gn = g_dist(x0, y0, nt, guess=g_theta)
        if gn < g_theta:
            theta, g_theta = nt, gn
    return np.clip(x0 + g_theta * theta, 0, 1)

# --- HopSkipJump: estimate the boundary NORMAL by Monte-Carlo querying --------
def mc_normal(x_b, y0, B=20, delta=0.01):
    """Estimate the boundary normal at a boundary point `x_b` (the core of
    HopSkipJump, Chen et al. 2020).  Probe `B` random unit directions a small
    distance `delta`; label each +1 if it stayed adversarial, -1 if not.  The
    weighted average of the directions approximates the boundary normal (the
    direction along which the label changes fastest).  Costs B victim queries --
    this is exactly the expense SQBA avoids by reading the normal off a
    surrogate instead."""
    U = rng.standard_normal((B, x_b.size)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    phi = np.array([1.0 if is_adv(x_b + delta * u, y0) else -1.0 for u in U])
    g = (phi[:, None] * U).mean(0); n = np.linalg.norm(g)
    return g / n if n else U[0]

def step_and_project(x_b, x0, y0, v, xi, best_d):
    """One HopSkipJump/SQBA geometric update: step `xi` along normal estimate `v`
    (deeper into the adversarial region), then binary-search back onto the
    boundary toward x0.  Because `v` has a tangential component, re-projecting
    can land CLOSER to x0 than where we started.  Returns (new boundary point,
    its distance capped at best_d, success flag).  `False` means the step left
    the adversarial region -- the normal estimate was bad."""
    x_step = np.clip(x_b + xi * v, 0, 1)
    if not is_adv(x_step, y0):
        return x_b, best_d, False
    x_b = binary_search(x_step, x0, y0)
    d = np.linalg.norm(x_b - x0)
    return x_b, min(d, best_d), True

def attack_hsj(x0, y0, x_init, iters=15, B=20):
    """HopSkipJumpAttack (Chen, Jordan & Wainwright 2020), a.k.a. Boundary
    Attack++.  Replaces Boundary Attack's random walk with a DIRECTED step: at
    each boundary point estimate the normal by querying (`mc_normal`), step along
    it with a decaying size, and re-project.  Far better per-iteration than the
    random walk -- but each normal estimate costs B victim queries."""
    x_b = x_init.copy(); best, best_d = x_b, np.linalg.norm(x_b - x0)
    for t in range(iters):
        v = mc_normal(x_b, y0, B)                    # B victim queries per iteration
        xi = best_d / np.sqrt(t + 1)                 # decaying step size (geometric schedule)
        x_b2, nd, ok = step_and_project(x_b, x0, y0, v, xi, best_d)
        if ok and nd < best_d:                       # keep the best (monotone)
            x_b, best, best_d = x_b2, x_b2, nd
    return best

# ============================================================================
# TIER 4 -- SQBA (Park, Miller & McLaughlin, "Hard-label based Small Query
#           Black-box Adversarial Attack" / Small-Query Black-Box Attack, WACV 2024).
# ----------------------------------------------------------------------------
# WHAT THE PAPER DOES.  SQBA estimates TWO gradients each iteration and blends
# them with a BOOLEAN switch beta in {0, 1} (paper Eq 7, Algorithm 1):
#     g_t = beta * dHw/||dHw||  +  (1 - beta) * dHb/||dHb||
#   - dHw : the WHITE-BOX surrogate gradient (paper Eq 5).  beta starts at 1, so
#           the surrogate gradient alone drives the search.
#   - dHb : the BLACK-BOX query-based gradient -- HopSkipJump's Monte-Carlo
#           estimate (paper Eq 6).  When dHw stalls in a local minimum, beta
#           switches to 0 and SQBA falls back to dHb.
# Each step is a line search along g_t that stays adversarial (Eq 8), followed by
# a binary search back toward the clean image (Eq 9).  The "small query" win is
# that while beta=1 the boundary normal is FREE (a surrogate backward pass)
# instead of costing B victim queries as in HopSkipJump.
#
# HOW THIS FUNCTION MAPS TO THAT.  `attack_sqba` is a faithful *simplification*:
#   - `surrogate_grad(...)`  plays the role of dHw (beta=1, "white_only" iters).
#   - `mc_normal(...)`        plays the role of dHb (beta=0, "fallback" iters).
#   - the switch: trust the surrogate while its step SHRINKS the perturbation;
#     otherwise fall back to mc_normal.  So a useless surrogate makes SQBA decay
#     smoothly into HopSkipJump -- exactly the paper's beta -> 0 behaviour.
#   - `step_and_project(...)` is the Eq-8 step + Eq-9 binary search.
#
# WHERE IT DIFFERS FROM THE PAPER (read before treating this as a reference impl):
#   1. NO MULTI-GRADIENT METHOD (the paper's core novelty, Section 4.1 / Eq 10-11).
#      The paper computes dHw by evaluating surrogate gradients at SEVERAL points
#      x + eta*v_tilde along the perturbation path and selecting the one (eta>=0.2)
#      whose direction best reduces distance -- exploiting that the surrogate
#      gradient rotates from parallel-to-perturbation toward the optimal
#      perpendicular direction as eta grows (paper Fig 2).  Here we just take ONE
#      surrogate gradient at the current boundary point.
#   2. INITIALISATION differs.  The paper starts from sign(surrogate gradient)
#      (FGSM-like) then binary-searches (Algorithm 1).  Here we start from the
#      shared `loose` point so every ladder rung has the same origin to improve.
#   3. The paper uses p_t = 10*sqrt(t+1) MC samples, an l-inf DGM gradient, and
#      delta_t = 0.01*D(x,x').  Here: fixed B=20, a unit L2 cross-entropy
#      gradient, and a fixed step schedule.  (See README "Alignment with paper".)
#
# (Aside on the trust rule -- why "did it make progress?" and not a geometric
# test: two stricter tests -- "did the step stay adversarial?" and "does the
# normal locally separate the classes?" -- both fail to discriminate here,
# because from a boundary point almost any large step stays adversarial, and ANY
# classifier pushing away from the true class passes a local separation test.)
# ============================================================================
def attack_sqba(x0, y0, surr, x_init, iters=15):
    x_b = x_init.copy(); best, best_d = x_b, np.linalg.norm(x_b - x0)
    white_iters, fallbacks = 0, 0                    # bookkeeping: free steps vs paid steps
    for t in range(iters):
        xi = best_d / np.sqrt(t + 1)
        v = surrogate_grad(x_b, y0, surr)            # FREE boundary-normal estimate
        x_b2, nd, ok = step_and_project(x_b, x0, y0, v, xi, best_d)
        if ok and nd < best_d - 1e-3:                # surrogate made progress -> trust it
            white_iters += 1; x_b, best, best_d = x_b2, x_b2, nd
        else:                                        # surrogate unproductive -> pay for a query estimate
            fallbacks += 1
            # RETROFIT from the paper: the fallback's probe size now follows Eq 6's
            # delta_t = 0.01 * D(x, x') (distance-scaled) instead of a fixed 0.01.
            # (The paper's p_t = 10*sqrt(t+1) sample schedule is left to the
            # paper-faithful `attack_sqba_full`; the teaching version keeps a fixed
            # B so its weaken-sweep trend stays clean to read.)
            v = mc_normal(x_b, y0, delta=1e-2 * best_d)
            x_b2, nd, ok = step_and_project(x_b, x0, y0, v, xi, best_d)
            if ok and nd < best_d:
                x_b, best, best_d = x_b2, x_b2, nd
    return best, white_iters, fallbacks

# ----------------------------------------------------------------------------
# PAPER-FAITHFUL SQBA.  `attack_sqba_full` implements the actual Algorithm 1 from
# Park et al. (2024), including the parts the teaching version omits:
#   * the MULTI-GRADIENT white-box estimate dHw (Section 4.1, Eqs 10-11),
#   * the HopSkipJump Monte-Carlo black-box estimate dHb (Eq 6),
#   * the boolean beta switch between them (Eq 7),
#   * the sign(surrogate-gradient) initialisation (Algorithm 1).
# Helper functions below map one-to-one onto the paper's equations.
# ----------------------------------------------------------------------------
def grad_hw_multigrad(x0, y0, x_b, surr, delta, n=5):
    """dHw via the MULTI-GRADIENT METHOD (paper Section 4.1, Eqs 10-11).

    The surrogate gradient at the clean image points nearly PARALLEL to the
    perturbation, which is a poor search direction; many works show the optimal
    hard-label direction is PERPENDICULAR to it, and the paper observes (Fig 2)
    that the surrogate gradient rotates toward that perpendicular as you move
    along the perturbation path.  So instead of one gradient, sample the
    surrogate gradient at several points x'_i = x + eta_i*(x_b - x) along the
    path (eta_i >= 0.2, the saturated regime) and KEEP the candidate whose small
    probe step stays adversarial and lands closest to x:
        dHw = mu_i,  i = argmin_i D(x, x_b + 2*delta*mu_i)  s.t. H(x_b + 2*delta*mu_i)=1
    Costs n victim queries (one adversariality probe per candidate)."""
    v_tilde = x_b - x0                                # current perturbation direction
    best_mu, best_d = None, np.inf
    for eta in np.linspace(0.2, 1.0, n):             # eta >= 0.2 (Eq 11 constraint)
        x_i = np.clip(x0 + eta * v_tilde, 0, 1)
        mu = surrogate_grad(x_i, y0, surr)           # candidate gradient (Eq 10, free)
        probe = np.clip(x_b + 2 * delta * mu, 0, 1)
        if is_adv(probe, y0):                        # H(x_b + 2*delta*mu) == 1  (1 victim query)
            d = np.linalg.norm(x0 - probe)           # D(x, x_b + 2*delta*mu)
            if d < best_d:
                best_d, best_mu = d, mu
    if best_mu is None:                              # no candidate stayed adversarial
        best_mu = surrogate_grad(x_b, y0, surr)
    n_ = np.linalg.norm(best_mu)
    return best_mu / n_ if n_ else best_mu

def grad_hb_mc(x_b, y0, delta, t):
    """dHb via HopSkipJump's Monte-Carlo estimate (paper Eq 6), with the paper's
    schedule p_t = 10*sqrt(t+1) random samples mu ~ N(0, I).  Smaller than
    HopSkipJump's own schedule because dHw has already advanced the search."""
    p_t = int(np.ceil(10 * np.sqrt(t + 1)))
    U = rng.standard_normal((p_t, x_b.size))         # mu ~ N(0, I)  (NOT unit-normalised, per Eq 6)
    phi = np.array([1.0 if is_adv(np.clip(x_b + delta * u, 0, 1), y0) else -1.0 for u in U])
    g = (phi[:, None] * U).mean(0)                   # (1/p_t) sum H(x_b + delta*mu) mu
    n = np.linalg.norm(g)
    return g / n if n else U[0]

def eq8_eq9_update(x_b, x0, y0, g, alpha_cap=1.0):
    """Eq 8 line search + Eq 9 binary search.
       Eq 8: x_dot = x_b + alpha*g with the largest alpha (<= alpha_cap, paper
             uses alpha <= 1.0) that keeps the point adversarial.
       Eq 9: binary-search x_dot back toward x so the next iterate sits on the
             boundary, closest to the clean image.
    Returns (new boundary point, its distance to x, the alpha used).  A small
    alpha signals a local minimum and is what flips beta in Eq 7."""
    a, x_dot, used = min(alpha_cap, max(np.linalg.norm(x_b - x0), 1e-3)), x_b, 0.0
    for _ in range(6):                               # line search: halve until adversarial
        cand = np.clip(x_b + a * g, 0, 1)
        if is_adv(cand, y0):
            x_dot, used = cand, a
            break
        a *= 0.5
    if used == 0.0:
        return x_b, np.linalg.norm(x_b - x0), 0.0    # could not step -> hard local minimum
    x_new = binary_search(x_dot, x0, y0)             # Eq 9
    return x_new, np.linalg.norm(x_new - x0), used

def attack_sqba_full(x0, y0, surr, iters=15):
    """SQBA, faithful to Algorithm 1 (Park et al., WACV 2024)."""
    # --- init: x'_0 = sign(surrogate gradient), then binary search (Eq 9) ------
    x_adv = np.clip(x0 + np.sign(surrogate_grad(x0, y0, surr)), 0, 1)   # large sign step
    if not is_adv(x_adv, y0):                        # safety: if it didn't transfer
        x_adv = loose_start(x0, y0)
    x_b = binary_search(x_adv, x0, y0)
    best, best_d = x_b, np.linalg.norm(x_b - x0)

    beta = 1                                          # start with the surrogate gradient (Eq 7)
    white_iters, fallbacks = 0, 0
    for t in range(1, iters + 1):
        delta = 1e-2 * best_d                         # delta_t = 10^-2 * D(x, x'_t)
        if beta == 1:
            g = grad_hw_multigrad(x0, y0, x_b, surr, delta)   # dHw  (Eq 5 / 10-11)
        else:
            g = grad_hb_mc(x_b, y0, delta, t)                 # dHb  (Eq 6)
        x_new, nd, used_alpha = eq8_eq9_update(x_b, x0, y0, g)
        improved = nd < best_d - 1e-4
        if improved:
            x_b, best, best_d = x_new, x_new, nd
        if beta == 1 and improved:
            white_iters += 1
        elif beta == 0:
            fallbacks += 1
        # Eq 7 beta switch: a small step or no progress => dHw is in a local
        # minimum => switch to the query-based dHb; otherwise keep using dHw.
        beta = 0 if (used_alpha < 0.25 or not improved) else 1
    return best, white_iters, fallbacks

# ============================================================================
# DRIVER -- run the whole lineage on ONE image and print the comparison table.
# ============================================================================
x0 = Xte[0]; y0 = predict(x0)                        # the image under attack and its true label
surr = build_surrogate()                             # a strong surrogate for the main table
loose = loose_start(x0, y0)                           # shared loose start for rungs 3-5

def run(fn):
    """Run an attack with a fresh victim-query counter; return (result, queries)."""
    global victim_q
    victim_q = 0
    x = fn()
    return x, victim_q

# Each entry: (tier, display name, year, one-line idea, the attack thunk).
plan = [
    ("white-box", "fgsm",      "'14", "one gradient-sign step on surrogate",          lambda: attack_fgsm(x0, y0, surr)),
    ("white-box", "pgd",       "'17", "iterated FGSM on surrogate",                   lambda: attack_pgd(x0, y0, surr)),
    ("transfer",  "transfer",  "'16", "grow eps on surrogate til victim flips",       lambda: attack_transfer(x0, y0, surr)),
    ("hard-label","random",    " - ", "blind noise (scaffold)",                       lambda: attack_random(x0, y0)),
    ("hard-label","line",      " - ", "aim at nearest class + binsearch (scaffold)",  lambda: attack_line(x0, y0)),
    ("hard-label","boundary",  "'18", "random walk along boundary",                   lambda: attack_boundary(x0, y0, loose)),
    ("hard-label","opt",       "'19", "minimize g(theta), zeroth-order",              lambda: attack_opt(x0, y0, loose)),
    ("hard-label","sign-opt",  "'20", "g(theta) via SIGN of dir. deriv",              lambda: attack_sign_opt(x0, y0, loose)),
    ("hard-label","hopskipjump","'20","MC boundary-normal estimate",                  lambda: attack_hsj(x0, y0, loose)),
    ("surrogate", "biased-bdry","'19","surrogate-biased boundary walk",               lambda: attack_boundary(x0, y0, loose, surr, bias=0.5)),
]
results = []
for tier, name, year, idea, fn in plan:
    x, q = run(fn)
    results.append((tier, name, year, q, np.linalg.norm(x - x0), fooled(x, y0), idea))
# SQBA returns extra bookkeeping, so it is run separately.  Two variants:
#   sqba       -- the simplified teaching version (single surrogate gradient).
#   sqba-full  -- paper-faithful Algorithm 1 (multi-gradient + beta switch +
#                 sign-gradient init); uses its OWN init, not the shared loose start.
x, q = run(lambda: attack_sqba(x0, y0, surr, loose)[0])
results.append(("surrogate", "sqba", "'24", q, np.linalg.norm(x - x0), fooled(x, y0), "teaching: single surrogate normal + fallback"))
x, q = run(lambda: attack_sqba_full(x0, y0, surr)[0])
results.append(("surrogate", "sqba-full", "'24", q, np.linalg.norm(x - x0), fooled(x, y0), "paper Algo 1: multi-gradient + beta switch"))

print("=" * 86)
print("BLACK-BOX ADVERSARIAL ATTACK LINEAGE  (one image; victim is hard-label only)")
print("=" * 86)
print(f"{'attack':<13}{'yr':>4}{'victim_q':>10}{'L2':>8}{'win':>5}   key idea")
last = None
for tier, name, year, q, d, win, idea in results:
    if tier != last:
        print(f"-- {tier.upper()} " + "-" * (80 - len(tier)))
        last = tier
    print(f"{name:<13}{year:>4}{q:>10}{d:>8.3f}{('Y' if win else 'n'):>5}   {idea}")

# ============================================================================
# WEAKEN-SURROGATE SWEEP -- the quantitative demonstration of SQBA's fallback.
# ----------------------------------------------------------------------------
# Run SQBA with surrogates of decreasing quality, averaged over 5 images (one
# image is too noisy to read a trend).  As the surrogate degrades you should see:
#   white_only  fall   (fewer free steps work)
#   fallbacks   rise   (more paid query-based steps)
#   victim_q    rise   (SQBA is paying for what the surrogate no longer gives)
# i.e. SQBA smoothly decays into the pure query-based HopSkipJump attack.
# ============================================================================
print("\n" + "=" * 86)
print("WEAKEN-SURROGATE SWEEP (SQBA decays toward query-based HopSkipJump; avg of 5 images)")
print("=" * 86)
imgs = [xi for xi in Xte[:8] if fooled(xi, -1)][:5]  # 5 test images (the -1 'true label' makes fooled() always True -> just a filter to take 5)
print(f"{'surrogate':<10}{'agree%':>8}{'L2':>8}{'victim_q':>10}{'white_only':>12}{'fallbacks':>11}")
for nm, cfg in [("strong ", dict(frac=1.0, hidden=96, label_noise=0.0)),
                ("ok     ", dict(frac=0.3, hidden=32, label_noise=0.05)),
                ("weak   ", dict(frac=0.1, hidden=16, label_noise=0.25)),
                ("useless", dict(frac=0.03, hidden=8, label_noise=0.6))]:
    s = build_surrogate(**cfg)
    agree = np.mean(victim.predict(Xte) == s.predict(Xte)) * 100   # victim/surrogate label agreement
    L2s, qs, wis, fbs = [], [], [], []
    for xi in imgs:
        yi = int(victim.predict(xi.reshape(1, -1))[0])
        li = loose_start(xi, yi)
        victim_q = 0
        adv, wi, fb = attack_sqba(xi, yi, s, li)
        L2s.append(np.linalg.norm(adv - xi)); qs.append(victim_q); wis.append(wi); fbs.append(fb)
    print(f"{nm:<10}{agree:>7.1f}{np.mean(L2s):>8.3f}{np.mean(qs):>10.0f}"
          f"{np.mean(wis):>8.1f}/15{np.mean(fbs):>11.1f}")
