"""
Shared attack ladder for the black-box adversarial lineage demos.

Every attack here is model- and data-agnostic: it touches the world ONLY through
a small context object `ctx` that provides
    ctx.predict(x)      -> victim's top-1 label   (hard label; counts one query)
    ctx.is_adv(x, y0)   -> predict(x) != y0
    ctx.sgrad(x, y0)    -> unit gradient of a SURROGATE's loss wrt the input
    ctx.pool_X, pool_y  -> a pool of labelled examples (to pick a starting point)
    ctx.rng             -> a numpy Generator
    ctx.lo, ctx.hi      -> valid input range (clip bounds)
    ctx.budget          -> max victim queries before attacks stop early
So `lineage.py` (sklearn MLP on digits) and `cifar_lineage.py` (PyTorch CNN on
CIFAR-10) reuse this file unchanged -- only their `ctx` differs.

See README.md for the full per-rung paper mapping and the SQBA alignment notes.

----------------------------------------------------------------------------
PRIMER FOR NEWCOMERS -- the few ideas every attack here is built on:
* A classifier carves the space of all possible images into regions, one per
  class. The surface separating two classes is the DECISION BOUNDARY. An image
  is "adversarial" if a tiny move pushes it across that boundary into the wrong
  region while still looking unchanged to a human.
* "HARD LABEL" / "decision-based": the victim tells us ONLY which region an image
  landed in (the class), not how confident it was. So our only feedback is the
  yes/no question is_adv(x): "did the label change?". Everything below is a clever
  way to find the closest boundary using only that yes/no signal.
* A GRADIENT is the direction (a vector over the pixels) that changes a model's
  loss fastest -- effectively "which way is the boundary?". We can't get the
  victim's gradient (it's a black box), so SQBA borrows one from a SURROGATE
  model we built ourselves (ctx.sgrad). That borrowed direction is what makes it
  fast. Attacks that lack a surrogate must instead ESTIMATE the boundary
  direction by probing the victim many times, which costs many queries.
* "victim queries" (ctx.q) is the cost we are minimising: every call to the real
  victim. The whole game is: flip the label with the fewest queries.
----------------------------------------------------------------------------
"""
import numpy as np


class Ctx:
    """Everything the attacks need from one (victim, surrogate, data) setup.

    `label_fn(x_flat) -> int` is the victim's UNCOUNTED label function; `predict`
    wraps it to count queries.  `sgrad_fn(x_flat, y0) -> unit vector` is the
    surrogate gradient.  Inputs are flat float vectors in [lo, hi]."""
    def __init__(self, label_fn, sgrad_fn, pool_X, pool_y, rng, lo=0.0, hi=1.0):
        self._label = label_fn
        self._sgrad = sgrad_fn
        self.pool_X, self.pool_y = pool_X, pool_y
        self.rng = rng
        self.lo, self.hi = lo, hi
        self.q = 0                     # victim-query counter (reset per attack)
        self.budget = float("inf")     # query budget; attacks break when reached

    def clip(self, x):
        return np.clip(x, self.lo, self.hi)

    def predict(self, x):
        self.q += 1
        return self._label(self.clip(x))

    def is_adv(self, x, y0):
        return self.predict(x) != y0

    def fooled(self, x, y0):           # UNCOUNTED -- for final reporting only
        return self._label(self.clip(x)) != y0

    def sgrad(self, x, y0):
        return self._sgrad(self.clip(x), y0)

    def over(self):
        return self.q >= self.budget


# ----------------------------------------------------------------------------
# Boundary-projection primitive, reused by rungs 2-5.
# ----------------------------------------------------------------------------
def binary_search(ctx, x_adv, x0, y0, tol=1e-2):
    """Bisect the segment between an adversarial point and the clean image until
    within `tol`, keeping `lo` non-adversarial and `hi` adversarial.  Returns the
    closest adversarial point to x0 along this line."""
    lo, hi = x0.copy(), x_adv.copy()
    while np.linalg.norm(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if ctx.is_adv(mid, y0):
            hi = mid
        else:
            lo = mid
    return hi


# ============================================================================
# TIER 3 -- HARD-LABEL / DECISION-BASED (top-1 label only).
# ============================================================================
def attack_random(ctx, x0, y0, N=1500, scale=0.5):
    """SCAFFOLD: blind baseline -- sample noise, keep the smallest flip."""
    best, best_d = None, np.inf
    for _ in range(N):
        if ctx.over():
            break
        x = ctx.clip(x0 + ctx.rng.standard_normal(x0.shape) * scale)
        if ctx.is_adv(x, y0) and np.linalg.norm(x - x0) < best_d:
            best, best_d = x, np.linalg.norm(x - x0)
    return best if best is not None else x0


def _other_class_sample(ctx, x0, y0, farthest):
    others = ctx.pool_X[ctx.pool_y != y0]
    d = np.linalg.norm(others - x0, axis=1)
    return others[np.argmax(d) if farthest else np.argmin(d)]


def attack_line(ctx, x0, y0):
    """SCAFFOLD: aim at the NEAREST other-class sample, binary-search to boundary.
    Really the initialisation step used inside Boundary Attack and OPT."""
    return binary_search(ctx, _other_class_sample(ctx, x0, y0, False), x0, y0)


def loose_start(ctx, x0, y0):
    """A deliberately-loose adversarial start (FARTHEST other-class sample) shared
    by rungs 3-5 so refinement has something to improve."""
    return binary_search(ctx, _other_class_sample(ctx, x0, y0, True), x0, y0)


def attack_boundary(ctx, x0, y0, x_init, bias=0.0, steps=500):
    """Boundary Attack (Brendel et al. 2018): random-walk along the boundary --
    an orthogonal exploration step + a pull toward x0, accepting only adversarial
    proposals, with step sizes adapting to the acceptance rate.

    With `bias > 0` the orthogonal step is steered toward the surrogate gradient,
    turning this into the BIASED BOUNDARY ATTACK (Brunner et al. 2019)."""
    x_b = x_init.copy()
    sph, src, acc = 0.10, 0.05, []
    for t in range(steps):
        if ctx.over():
            break
        r = x_b - x0
        d = np.linalg.norm(r)
        eta = ctx.rng.standard_normal(x0.shape)
        eta /= np.linalg.norm(eta)
        if bias > 0:                                   # <-- Biased Boundary change
            eta = (1 - bias) * eta + bias * ctx.sgrad(x_b, y0)
        eta -= (eta @ r) / (d * d) * r                 # orthogonal to the radius
        eta *= sph * d / np.linalg.norm(eta)
        cand = x_b + eta
        cand = x0 + (cand - x0) / np.linalg.norm(cand - x0) * d   # back to sphere
        cand = ctx.clip(cand - src * (cand - x0))                # pull toward x0
        ok = ctx.is_adv(cand, y0)
        if ok:
            x_b = cand
        acc.append(ok)
        if len(acc) >= 30:                             # keep acceptance ~20-50%
            rate = np.mean(acc[-30:])
            sph, src = (sph * 0.9, src * 0.9) if rate < 0.2 else \
                       ((sph * 1.1, src * 1.1) if rate > 0.5 else (sph, src))
    return x_b


# --- OPT family: minimise g(theta) = distance to boundary along direction theta -
def g_dist(ctx, x0, y0, theta, guess=1.0, tol=4e-2):
    """Distance to the boundary along unit direction theta, from labels alone.
    `guess` warm-starts the search.  inf if theta never crosses the boundary."""
    theta = theta / np.linalg.norm(theta)
    if ctx.predict(x0 + guess * theta) == y0:
        lo = hi = guess
        while ctx.predict(x0 + hi * theta) == y0:
            hi *= 1.5
            if hi > 50:
                return np.inf
    else:
        hi = guess; lo = guess * 0.5
        while ctx.predict(x0 + lo * theta) != y0 and lo > 1e-3:
            lo *= 0.5
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        hi, lo = (mid, lo) if ctx.predict(x0 + mid * theta) != y0 else (hi, mid)
    return hi


def attack_opt(ctx, x0, y0, x_init, iters=14, q=6, beta=0.02, alpha=0.3):
    """OPT attack (Cheng et al. 2019): zeroth-order gradient descent on g(theta).
    Query-hungry -- each probe runs a full g_dist search."""
    theta = (x_init - x0) / np.linalg.norm(x_init - x0)
    g_theta = g_dist(ctx, x0, y0, theta, guess=np.linalg.norm(x_init - x0))
    for t in range(iters):
        if ctx.over():
            break
        grad = np.zeros_like(theta)
        for _ in range(q):
            u = ctx.rng.standard_normal(theta.shape); u /= np.linalg.norm(u)
            grad += (g_dist(ctx, x0, y0, theta + beta * u, guess=g_theta) - g_theta) / beta * u
        grad /= q
        nt = theta - alpha * grad; nt /= np.linalg.norm(nt)
        gn = g_dist(ctx, x0, y0, nt, guess=g_theta)
        if gn < g_theta:
            theta, g_theta = nt, gn
    return ctx.clip(x0 + g_theta * theta)


def attack_sign_opt(ctx, x0, y0, x_init, iters=14, q=12, beta=0.02, alpha=0.3):
    """Sign-OPT (Cheng et al. 2020): like OPT, but each probe needs only the SIGN
    of the directional derivative -- one query instead of a full g_dist search.
    s=+1 marks directions u that REDUCE g(theta), so we ascend +grad."""
    theta = (x_init - x0) / np.linalg.norm(x_init - x0)
    g_theta = g_dist(ctx, x0, y0, theta, guess=np.linalg.norm(x_init - x0))
    for t in range(iters):
        if ctx.over():
            break
        grad = np.zeros_like(theta)
        for _ in range(q):
            u = ctx.rng.standard_normal(theta.shape); u /= np.linalg.norm(u)
            nt = theta + beta * u; nt /= np.linalg.norm(nt)
            s = 1.0 if ctx.is_adv(x0 + g_theta * nt, y0) else -1.0   # one query/probe
            grad += s * u
        grad /= q
        nt = theta + alpha * grad; nt /= np.linalg.norm(nt)
        gn = g_dist(ctx, x0, y0, nt, guess=g_theta)
        if gn < g_theta:
            theta, g_theta = nt, gn
    return ctx.clip(x0 + g_theta * theta)


# --- HopSkipJump: estimate the boundary normal by Monte-Carlo querying ---------
def mc_normal(ctx, x_b, y0, B=20, delta=0.01):
    """Estimate the boundary normal at x_b (HopSkipJump, Chen et al. 2020): probe
    B random directions; the weighted average of +1/-1 (stayed adversarial?) ~ the
    normal.  Costs B victim queries -- the expense SQBA replaces with a surrogate."""
    U = ctx.rng.standard_normal((B, x_b.size)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    phi = np.array([1.0 if ctx.is_adv(x_b + delta * u, y0) else -1.0 for u in U])
    g = (phi[:, None] * U).mean(0); n = np.linalg.norm(g)
    return g / n if n else U[0]


def step_and_project(ctx, x_b, x0, y0, v, xi, best_d):
    """Step xi along normal v (deeper adversarial), then binary-search back toward
    x0.  Returns (new boundary point, distance capped at best_d, success flag)."""
    x_step = ctx.clip(x_b + xi * v)
    if not ctx.is_adv(x_step, y0):
        return x_b, best_d, False
    x_b = binary_search(ctx, x_step, x0, y0)
    d = np.linalg.norm(x_b - x0)
    return x_b, min(d, best_d), True


def attack_hsj(ctx, x0, y0, x_init, iters=15, B=20):
    """HopSkipJumpAttack (Chen, Jordan & Wainwright 2020)."""
    x_b = x_init.copy(); best, best_d = x_b, np.linalg.norm(x_b - x0)
    for t in range(iters):
        if ctx.over():
            break
        v = mc_normal(ctx, x_b, y0, B)
        x_b2, nd, ok = step_and_project(ctx, x_b, x0, y0, v, best_d / np.sqrt(t + 1), best_d)
        if ok and nd < best_d:
            x_b, best, best_d = x_b2, x_b2, nd
    return best


# ============================================================================
# TIER 4 -- SQBA (Park, Miller & McLaughlin, WACV 2024).  Two versions:
#   attack_sqba       -- simplified teaching version (single surrogate gradient).
#   attack_sqba_full  -- paper-faithful Algorithm 1 (multi-gradient + beta switch).
# See README "Alignment with the SQBA paper" for the equation mapping.
# ============================================================================
def attack_sqba(ctx, x0, y0, x_init, iters=15):
    """Teaching SQBA: trust the FREE surrogate gradient while it shrinks the
    perturbation; otherwise fall back to the costly query-based mc_normal -- so a
    useless surrogate decays the attack into HopSkipJump (the paper's beta -> 0)."""
    x_b = x_init.copy(); best, best_d = x_b, np.linalg.norm(x_b - x0)
    white_iters, fallbacks = 0, 0
    for t in range(iters):
        if ctx.over():
            break
        xi = best_d / np.sqrt(t + 1)
        v = ctx.sgrad(x_b, y0)                          # free surrogate normal
        x_b2, nd, ok = step_and_project(ctx, x_b, x0, y0, v, xi, best_d)
        if ok and nd < best_d - 1e-3:
            white_iters += 1; x_b, best, best_d = x_b2, x_b2, nd
        else:
            fallbacks += 1
            v = mc_normal(ctx, x_b, y0, delta=1e-2 * best_d)   # Eq-6 distance-scaled probe
            x_b2, nd, ok = step_and_project(ctx, x_b, x0, y0, v, xi, best_d)
            if ok and nd < best_d:
                x_b, best, best_d = x_b2, x_b2, nd
    return best, white_iters, fallbacks


def grad_hw_multigrad(ctx, x0, y0, x_b, delta, n=5):
    """dHw via the MULTI-GRADIENT METHOD (paper Section 4.1, Eqs 10-11): sample the
    surrogate gradient at points x + eta*(x_b - x) along the perturbation path
    (eta >= 0.2, where the gradient has rotated toward the optimal perpendicular
    direction, paper Fig 2) and keep the candidate whose small probe stays
    adversarial and lands closest to x.  Costs n victim queries."""
    v_tilde = x_b - x0
    best_mu, best_d = None, np.inf
    for eta in np.linspace(0.2, 1.0, n):
        x_i = ctx.clip(x0 + eta * v_tilde)
        mu = ctx.sgrad(x_i, y0)                         # candidate gradient (Eq 10, free)
        probe = ctx.clip(x_b + 2 * delta * mu)
        if ctx.is_adv(probe, y0):                       # H(x_b + 2*delta*mu) == 1
            d = np.linalg.norm(x0 - probe)
            if d < best_d:
                best_d, best_mu = d, mu
    if best_mu is None:
        best_mu = ctx.sgrad(x_b, y0)
    nrm = np.linalg.norm(best_mu)
    return best_mu / nrm if nrm else best_mu


def grad_hb_mc(ctx, x_b, y0, delta, t):
    """dHb via HopSkipJump's MC estimate (Eq 6) with the paper's p_t=10*sqrt(t+1)
    sample schedule and mu ~ N(0, I)."""
    p_t = int(np.ceil(10 * np.sqrt(t + 1)))
    U = ctx.rng.standard_normal((p_t, x_b.size))
    phi = np.array([1.0 if ctx.is_adv(ctx.clip(x_b + delta * u), y0) else -1.0 for u in U])
    g = (phi[:, None] * U).mean(0); n = np.linalg.norm(g)
    return g / n if n else U[0]


def eq8_eq9_update(ctx, x_b, x0, y0, g, alpha_cap=1.0):
    """Eq 8 (largest alpha<=alpha_cap keeping adversarial) + Eq 9 (binary search
    back toward x0).  Returns (new point, distance, alpha used).  Small alpha
    signals a local minimum, which flips beta in Eq 7."""
    a = min(alpha_cap, max(np.linalg.norm(x_b - x0), 1e-3))
    x_dot, used = x_b, 0.0
    for _ in range(6):
        cand = ctx.clip(x_b + a * g)
        if ctx.is_adv(cand, y0):
            x_dot, used = cand, a
            break
        a *= 0.5
    if used == 0.0:
        return x_b, np.linalg.norm(x_b - x0), 0.0
    x_new = binary_search(ctx, x_dot, x0, y0)
    return x_new, np.linalg.norm(x_new - x0), used


def attack_sqba_full(ctx, x0, y0, iters=15):
    """Paper-faithful SQBA, Algorithm 1 (Park et al. 2024): dHw multi-gradient
    (Eqs 5/10-11) + dHb MC (Eq 6) combined by a boolean beta switch (Eq 7), with
    sign(surrogate-gradient) init.  Uses its OWN init, not the shared loose start."""
    x_adv = ctx.clip(x0 + np.sign(ctx.sgrad(x0, y0)))   # sign-gradient init
    if not ctx.is_adv(x_adv, y0):
        x_adv = loose_start(ctx, x0, y0)
    x_b = binary_search(ctx, x_adv, x0, y0)
    best, best_d = x_b, np.linalg.norm(x_b - x0)
    beta = 1
    white_iters, fallbacks = 0, 0
    for t in range(1, iters + 1):
        if ctx.over():
            break
        delta = 1e-2 * best_d                           # delta_t = 10^-2 * D(x, x'_t)
        if beta == 1:
            g = grad_hw_multigrad(ctx, x0, y0, x_b, delta)   # dHw
        else:
            g = grad_hb_mc(ctx, x_b, y0, delta, t)           # dHb
        x_new, nd, used_alpha = eq8_eq9_update(ctx, x_b, x0, y0, g)
        improved = nd < best_d - 1e-4
        if improved:
            x_b, best, best_d = x_new, x_new, nd
        if beta == 1 and improved:
            white_iters += 1
        elif beta == 0:
            fallbacks += 1
        beta = 0 if (used_alpha < 0.25 or not improved) else 1   # Eq 7 switch
    return best, white_iters, fallbacks
