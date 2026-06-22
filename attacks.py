"""
Shared attack ladder for the black-box adversarial lineage demos.

Welcome! This file contains the "adversarial attacks" code. 

If you are new to machine learning, here is what this is about:
1. What is an Adversarial Attack?
   Imagine we have a computer program (a "classifier" or "model") that looks at pictures
   and guesses what they are (for example, "airplane" or "automobile"). An adversarial attack is a
   way to make a tiny, almost invisible change to a picture of an airplane so that the computer
   guesses "automobile" instead, even though it still looks like an airplane to us.

2. What is a "Black-Box" Attack?
   A black-box attack means we do not know how the model works on the inside. We cannot see
   its neural network layers, its mathematical formulas, or its confidence score. We can only
   send it an image, and it returns a single word/label: "airplane" or "automobile" (this is called a "hard label").
   Our goal is to fool this model using as few questions (or "queries") as possible, because querying
   a real model can be slow or expensive.

3. The Helper Object: Context (`ctx`)
   To keep the attack code simple, all communication with the model happens through a `ctx` object.
   It provides:
   - `ctx.predict(x)`: Asks the model to guess the label of image `x` (this costs 1 query).
   - `ctx.is_adv(x, y0)`: Checks if the model's guess is different from the correct label `y0`.
   - `ctx.sgrad(x, y0)`: Asks our own helper model (a "surrogate" model that we *can* look inside)
     for the direction of the quickest change.
   - `ctx.lo` and `ctx.hi`: The minimum and maximum pixel values allowed (like 0.0 for black and 1.0 for white).

This file is shared. It is used by `lineage.py` (which attacks a simple text/digits classifier)
and `cifar_lineage.py` (which attacks a convolutional neural network that classifies airplane vs automobile images).
"""
import numpy as np


class Ctx:
    """
    This class is the "Context" that wraps around our target model (the victim),
    our helper model (the surrogate), and our dataset. It helps us run and keep track
    of the attack.

    Perceptual Guidance via Space Transformation (Weighted L2):
    To implement weighted L2 guidance (e.g., hiding perturbations in textured regions),
    we do not need to modify the internal Euclidean geometry of the attacks. Instead,
    we can run the attacks in a transformed coordinate space:
      z = W * x   ==>   x = z / W
    By passing a prediction function `predict_z(z) = predict(z / W)` and gradient
    `sgrad_z(z) = sgrad(z / W) / W` to Ctx, any standard L2 attack running inside Ctx
    will naturally minimize the weighted L2 distance:
      ||z - z0||_2 = ||W * (x - x0)||_2
    The clipping bounds inside Ctx are scaled to `lo = 0.0` and `hi = W`.
    """
    def __init__(self, label_fn, sgrad_fn, pool_X, pool_y, rng, lo=0.0, hi=1.0):
        # `label_fn` is a function that asks the victim model for its guess (without counting it).
        # We wrap this to count every time an attack asks the victim model a question.
        self._label = label_fn
        # `sgrad_fn` is a function that returns the gradient (change direction) from our surrogate helper model.
        self._sgrad = sgrad_fn
        # A pool of clean images we can use to find starting points.
        self.pool_X, self.pool_y = pool_X, pool_y
        # A random number generator to help us add random noise when we need to explore.
        self.rng = rng
        # The allowed range for pixel values (usually 0.0 to 1.0).
        self.lo, self.hi = lo, hi
        # A counter for how many times we queried the victim model.
        self.q = 0                     
        # The maximum number of queries allowed. If we go over this, the attack stops early.
        self.budget = float("inf")     

    def clip(self, x):
        # Clip ensures that all pixel values stay in the valid range [lo, hi] (e.g., between 0.0 and 1.0).
        return np.clip(x, self.lo, self.hi)

    def predict(self, x):
        # Ask the victim model for its prediction and increment the query counter by 1.
        self.q += 1
        return self._label(self.clip(x))

    def is_adv(self, x, y0):
        # Returns True if the model's guess for `x` is NOT the correct label `y0` (meaning the model is fooled).
        return self.predict(x) != y0

    def fooled(self, x, y0):           
        # Same as `is_adv`, but does NOT count as a query. We use this only for printing the final summary.
        return self._label(self.clip(x)) != y0

    def sgrad(self, x, y0):
        # Gets the gradient (best direction to change the image) from our surrogate helper model.
        return self._sgrad(self.clip(x), y0)

    def over(self):
        # Checks if we have run out of our query budget.
        return self.q >= self.budget


# ----------------------------------------------------------------------------
# Boundary-projection primitive, reused by rungs 2-5.
# ----------------------------------------------------------------------------
def binary_search(ctx, x_adv, x0, y0, tol=1e-2):
    """
    Finds the boundary where the model changes its mind by searching the line between
    a clean image (`x0`) and an adversarial image (`x_adv`).
    
    Imagine a slider between the clean image and the adversarial image. 
    At 0% (clean image), the model is correct. At 100% (adversarial image), the model is fooled.
    We want to find the exact percentage (say, 42%) where the model just starts to get fooled.
    We do this by:
    1. Finding the midpoint between the correct point (`lo`) and the incorrect point (`hi`).
    2. Asking the model: "Does this midpoint fool you?"
    3. If yes, the boundary is closer to the clean image, so we update `hi` to be this midpoint.
    4. If no, the boundary is farther away, so we update `lo` to be this midpoint.
    5. We repeat this until the distance between `lo` and `hi` is smaller than `tol` (tolerance).
    """
    lo, hi = x0.copy(), x_adv.copy()
    while np.linalg.norm(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if ctx.is_adv(mid, y0):
            hi = mid
        else:
            lo = mid
    return hi


# ============================================================================
# TIER 3 -- HARD-LABEL / DECISION-BASED ATTACKS
# These attacks can ONLY ask the victim model for its guess (the "label").
# They have no access to the model's inner math or gradients.
# ============================================================================
def attack_random(ctx, x0, y0, N=1500, scale=0.5):
    """
    A simple baseline attack that tries random noise.
    
    It generates random changes (noise) to the image, checks if they fool the model,
    and keeps the one that fools the model with the smallest change.
    - `N`: how many random guesses to try.
    - `scale`: how large the random changes can be.
    """
    best, best_d = None, np.inf
    for _ in range(N):
        if ctx.over():
            break
        # Generate random noise and add it to the original image x0.
        x = ctx.clip(x0 + ctx.rng.standard_normal(x0.shape) * scale)
        # If it successfully fools the model, and is closer to the original image than our best so far:
        if ctx.is_adv(x, y0) and np.linalg.norm(x - x0) < best_d:
            best, best_d = x, np.linalg.norm(x - x0)
    return best if best is not None else x0


def _other_class_sample(ctx, x0, y0, farthest):
    # Helper function: Finds an image from our pool that belongs to a DIFFERENT class
    # than the original image's class (`y0`).
    # If `farthest` is True, it returns the image that is most different from ours.
    # If `farthest` is False, it returns the image that is closest to ours.
    others = ctx.pool_X[ctx.pool_y != y0]
    d = np.linalg.norm(others - x0, axis=1)
    return others[np.argmax(d) if farthest else np.argmin(d)]


def attack_line(ctx, x0, y0):
    """
    A simple attack that takes another image (like a dog if ours is a cat),
    draws a line between them, and uses binary search to find where they cross the boundary.
    """
    return binary_search(ctx, _other_class_sample(ctx, x0, y0, False), x0, y0)


def loose_start(ctx, x0, y0):
    """
    Finds a starting point that is adversarial but quite far from our original image.
    We do this by choosing the farthest image of a different class, and searching the boundary on that line.
    This gives our optimization attacks a starting point that they can then try to improve.
    """
    return binary_search(ctx, _other_class_sample(ctx, x0, y0, True), x0, y0)


def attack_boundary(ctx, x0, y0, x_init, bias=0.0, steps=500):
    """
    Boundary Attack: Walks along the decision boundary, trying to get closer to the original image.
    
    Imagine walking along a fence (the decision boundary). One side is "cat" (correct), the other is "dog" (adversarial).
    We start on the "dog" side. We want to take steps that:
    1. Stay on the "dog" side (so we keep fooling the model).
    2. Move us closer to our original cat image (`x0`).
    
    How it works:
    - We take a step in a random direction that is tangent to the sphere around `x0` (so we don't change the distance to `x0` yet).
    - If `bias > 0` (Biased Boundary Attack), instead of a completely random direction, we bias our step direction towards
      the gradient from our surrogate helper model.
    - Then, we take a small step towards the original image (`x0`).
    - We ask the model if this new point is still adversarial. If yes, we accept the step. If no, we try again.
    - We adjust how big our steps are based on how often our proposals are accepted (aiming for 20% to 50% acceptance).
    """
    x_b = x_init.copy()
    sph, src, acc = 0.10, 0.05, []
    for t in range(steps):
        if ctx.over():
            break
        r = x_b - x0
        d = np.linalg.norm(r)
        # Generate random direction
        eta = ctx.rng.standard_normal(x0.shape)
        eta /= np.linalg.norm(eta)
        
        # If we have a surrogate helper, steer the search direction towards the surrogate's gradient.
        if bias > 0:                                   
            eta = (1 - bias) * eta + bias * ctx.sgrad(x_b, y0)
            
        # Make the step orthogonal (perpendicular) to the vector from the original image to our current boundary point.
        eta -= (eta @ r) / (d * d) * r                 
        eta *= sph * d / np.linalg.norm(eta)
        
        # Propose the new point on the sphere
        cand = x_b + eta
        cand = x0 + (cand - x0) / np.linalg.norm(cand - x0) * d   
        # Pull the proposal a tiny bit closer to the original image x0
        cand = ctx.clip(cand - src * (cand - x0))                
        
        # Check if the proposal still fools the model
        ok = ctx.is_adv(cand, y0)
        if ok:
            x_b = cand
        acc.append(ok)
        
        # Adjust step sizes dynamically to maintain a good acceptance rate
        if len(acc) >= 30:                             
            rate = np.mean(acc[-30:])
            sph, src = (sph * 0.9, src * 0.9) if rate < 0.2 else \
                       ((sph * 1.1, src * 1.1) if rate > 0.5 else (sph, src))
    return x_b


# --- OPT Attack Family ---
# These attacks think of the boundary in terms of angles/directions (theta) from the original image.
# For any direction theta, we define g(theta) as: "How far do we have to walk in direction theta before the model flips?"
# Our goal is to find the direction theta that makes this distance g(theta) as small as possible.

def g_dist(ctx, x0, y0, theta, guess=1.0, tol=4e-2):
    """
    Finds how far we have to travel along a direction `theta` to hit the decision boundary.
    It uses binary search to find this distance.
    """
    theta = theta / np.linalg.norm(theta)
    # If our guess is not adversarial, we increase our step size until we find a point that is.
    if ctx.predict(x0 + guess * theta) == y0:
        lo = hi = guess
        while ctx.predict(x0 + hi * theta) == y0:
            hi *= 1.5
            if hi > 50:
                return np.inf
    # If our guess is already adversarial, we decrease our step size until we find a point that is not.
    else:
        hi = guess; lo = guess * 0.5
        while ctx.predict(x0 + lo * theta) != y0 and lo > 1e-3:
            lo *= 0.5
    # Do binary search between lo and hi to find the exact boundary distance
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        hi, lo = (mid, lo) if ctx.predict(x0 + mid * theta) != y0 else (hi, mid)
    return hi


def attack_opt(ctx, x0, y0, x_init, iters=14, q=6, beta=0.02, alpha=0.3):
    """
    OPT Attack: Minimizes the distance to the boundary by estimating gradients.
    
    Since we cannot compute math gradients directly from the black-box model, we estimate them:
    1. We pick `q` random directions.
    2. We measure the boundary distance `g` for each random direction.
    3. We average these measurements to estimate which way the boundary gets closer.
    4. We take a step in that direction (using step size `alpha`).
    
    This is called "zeroth-order optimization" because we only use function values, not true derivatives.
    Note: It is very query-hungry because each measurement requires a full binary search (`g_dist`).
    """
    theta = (x_init - x0) / np.linalg.norm(x_init - x0)
    g_theta = g_dist(ctx, x0, y0, theta, guess=np.linalg.norm(x_init - x0))
    for t in range(iters):
        if ctx.over():
            break
        grad = np.zeros_like(theta)
        # Estimate the gradient by probing around our current direction
        for _ in range(q):
            u = ctx.rng.standard_normal(theta.shape); u /= np.linalg.norm(u)
            # Measure distance in a slightly nudged direction
            grad += (g_dist(ctx, x0, y0, theta + beta * u, guess=g_theta) - g_theta) / beta * u
        grad /= q
        # Move our direction to reduce the boundary distance
        nt = theta - alpha * grad; nt /= np.linalg.norm(nt)
        gn = g_dist(ctx, x0, y0, nt, guess=g_theta)
        if gn < g_theta:
            theta, g_theta = nt, gn
    return ctx.clip(x0 + g_theta * theta)


def attack_sign_opt(ctx, x0, y0, x_init, iters=14, q=12, beta=0.02, alpha=0.3):
    """
    Sign-OPT Attack: A much faster version of the OPT attack.
    
    Instead of performing a full, slow binary search (`g_dist`) for every random direction,
    Sign-OPT asks a single yes/no question: "If I nudge my direction by a tiny amount,
    am I still adversarial at the current distance?"
    - If yes: the boundary distance in this direction must be smaller (this is a good direction, sign = +1).
    - If no: the boundary distance in this direction is larger (this is a bad direction, sign = -1).
    By using just the sign (+1 or -1) of the direction, it can estimate the gradient with way fewer queries.
    """
    theta = (x_init - x0) / np.linalg.norm(x_init - x0)
    g_theta = g_dist(ctx, x0, y0, theta, guess=np.linalg.norm(x_init - x0))
    for t in range(iters):
        if ctx.over():
            break
        grad = np.zeros_like(theta)
        # Probe random directions
        for _ in range(q):
            u = ctx.rng.standard_normal(theta.shape); u /= np.linalg.norm(u)
            nt = theta + beta * u; nt /= np.linalg.norm(nt)
            # Ask the model if this nudged direction is adversarial (only 1 query!)
            s = 1.0 if ctx.is_adv(x0 + g_theta * nt, y0) else -1.0   
            grad += s * u
        grad /= q
        # Move in the estimated direction that reduces distance
        nt = theta + alpha * grad; nt /= np.linalg.norm(nt)
        gn = g_dist(ctx, x0, y0, nt, guess=g_theta)
        if gn < g_theta:
            theta, g_theta = nt, gn
    return ctx.clip(x0 + g_theta * theta)


# --- HopSkipJump Attack ---
# HopSkipJump finds the direction perpendicular to the boundary (the "boundary normal") at our current point.
# Moving perpendicular to a boundary is the fastest way to get away from it (or toward the clean image).

def mc_normal(ctx, x_b, y0, B=20, delta=0.01):
    """
    Estimates the boundary normal direction using a Monte Carlo (sampling) approach.
    
    Imagine standing right on the boundary. We want to know which direction is perpendicular to the boundary line.
    We do this by:
    1. Scattering `B` random directions around us.
    2. For each direction, we step a tiny amount (`delta`) and check if we are still adversarial (yes/no).
    3. We take a weighted average of these directions (where directions that remained adversarial get positive weight,
       and directions that became correct get negative weight).
    The average direction points directly perpendicular (normal) to the boundary.
    """
    U = ctx.rng.standard_normal((B, x_b.size)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    # Check each random probe direction
    phi = np.array([1.0 if ctx.is_adv(x_b + delta * u, y0) else -1.0 for u in U])
    # Average them to estimate the normal vector
    g = (phi[:, None] * U).mean(0); n = np.linalg.norm(g)
    return g / n if n else U[0]


def step_and_project(ctx, x_b, x0, y0, v, xi, best_d):
    """
    Takes a step along the direction `v` (into the adversarial region),
    and then binary searches back to find the new boundary point.
    
    If the step size `xi` is too large and fails to be adversarial, we reject the step.
    Returns: (new boundary point, new distance, success flag)
    """
    x_step = ctx.clip(x_b + xi * v)
    if not ctx.is_adv(x_step, y0):
        return x_b, best_d, False
    # Project back to the boundary using binary search
    x_b = binary_search(ctx, x_step, x0, y0)
    d = np.linalg.norm(x_b - x0)
    return x_b, min(d, best_d), True


def attack_hsj(ctx, x0, y0, x_init, iters=15, B=20):
    """
    HopSkipJump Attack: Repeats the process of estimating the normal vector and stepping/projecting.
    
    This is one of the strongest pure black-box attacks, but it is query-heavy because we have to
    estimate the normal direction by probing the model `B` times at every single step.
    """
    x_b = x_init.copy(); best, best_d = x_b, np.linalg.norm(x_b - x0)
    for t in range(iters):
        if ctx.over():
            break
        # 1. Estimate normal direction
        v = mc_normal(ctx, x_b, y0, B)
        # 2. Step in that direction and project back to boundary
        x_b2, nd, ok = step_and_project(ctx, x_b, x0, y0, v, best_d / np.sqrt(t + 1), best_d)
        if ok and nd < best_d:
            x_b, best, best_d = x_b2, x_b2, nd
    return best


# --- Triangle Attack (ECCV 2022) ---
def attack_triangle(ctx, x0, y0, x_init, iters=40, N=2):
    """
    Triangle Attack (Wang et al., ECCV 2022):
    
    A highly query-efficient decision-based attack that does NOT require estimating gradients.
    Instead, it uses the geometric properties of a triangle (the law of sines) to guide the search.
    It operates in the low-frequency DCT (Discrete Cosine Transform) subspace to perform effective
    dimensionality reduction.
    
    Explanation for Beginners:
    Imagine we want to find a path that gets our adversarial image as close to the original image as possible.
    At each step:
    1. We define a 2D plane (subspace) spanned by:
       - The vector from the original image to our current adversarial image.
       - A random direction in the low-frequency space (since low-frequency changes are more natural and effective).
    2. In this 2D plane, the original image, current adversarial image, and a new candidate image form a triangle.
    3. According to the "Law of Sines", the side lengths of a triangle depend on its angles.
       By carefully choosing the angles (alpha and beta), we can guarantee that the new candidate is closer to the original image.
    4. We adjust the search angle alpha adaptively: if we successfully find an adversarial candidate, we make the search harder
       (larger alpha) to get a smaller perturbation. If we fail, we make it easier (smaller alpha).
    """
    from scipy.fftpack import dct, idct
    
    D = x0.size
    x_adv = x_init.copy()
    alpha = np.pi / 2
    gamma = 0.01
    lamb = 0.05
    tau = 0.1
    d = 3
    beta_limit = np.pi / 16
    
    # Low-frequency range (first 10% of DCT coefficients)
    L = max(1, int(0.10 * D))
    
    for t in range(iters):
        if ctx.over():
            break
            
        u_dir = x_adv - x0
        delta_t = np.linalg.norm(u_dir)
        if delta_t == 0:
            break
        u_dir = u_dir / delta_t
        
        # 1. Sample a random direction in the low-frequency DCT subspace
        v_dct = np.zeros(D)
        # Randomly pick d frequency components
        chosen_indices = ctx.rng.choice(L, min(d, L), replace=False)
        v_dct[chosen_indices] = ctx.rng.standard_normal(len(chosen_indices))
        # Transform back to spatial domain
        v_spatial = idct(v_dct, norm='ortho')
        # Orthogonalize v with respect to u
        v_dir = v_spatial - (v_spatial @ u_dir) * u_dir
        norm_v = np.linalg.norm(v_dir)
        if norm_v == 0:
            continue
        v_dir = v_dir / norm_v
        
        # 2. Check the initial candidate angle beta
        beta = max(np.pi - 2 * alpha, beta_limit)
        
        # Try the positive and negative directions for the triangle vertex
        # delta_new is calculated using the Law of Sines: delta_new = delta_t * sin(alpha + beta) / sin(alpha)
        # We clamp the denominator sin(alpha) to avoid division by zero
        sin_alpha = max(np.sin(alpha), 1e-5)
        delta_new = delta_t * np.sin(alpha + beta) / sin_alpha
        
        cand_pos = ctx.clip(x0 + delta_new * (u_dir * np.cos(beta) + v_dir * np.sin(beta)))
        cand_neg = ctx.clip(x0 + delta_new * (u_dir * np.cos(beta) - v_dir * np.sin(beta)))
        
        best_cand = None
        
        # Check if the positive candidate is NOT adversarial (returns original label y0)
        if ctx.predict(cand_pos) == y0:
            # Failure: decrease alpha
            alpha = max(alpha - lamb * gamma, np.pi / 2 - tau)
            # Check the symmetric negative candidate
            if ctx.predict(cand_neg) == y0:
                # Failure: decrease alpha and give up this subspace
                alpha = max(alpha - lamb * gamma, np.pi / 2 - tau)
                continue
            else:
                # Negative candidate is adversarial
                alpha = min(alpha + gamma, np.pi / 2 + tau)
                best_cand = cand_neg
        else:
            # Positive candidate is adversarial
            alpha = min(alpha + gamma, np.pi / 2 + tau)
            best_cand = cand_pos
            
        # 3. Binary search for the optimal angle beta (to minimize perturbation further)
        beta_lower = beta
        beta_upper = min(np.pi / 2, np.pi - alpha)
        
        for i in range(N):
            beta_mid = (beta_lower + beta_upper) / 2
            sin_alpha = max(np.sin(alpha), 1e-5)
            delta_mid = delta_t * np.sin(alpha + beta_mid) / sin_alpha
            
            cand_pos_mid = ctx.clip(x0 + delta_mid * (u_dir * np.cos(beta_mid) + v_dir * np.sin(beta_mid)))
            cand_neg_mid = ctx.clip(x0 + delta_mid * (u_dir * np.cos(beta_mid) - v_dir * np.sin(beta_mid)))
            
            if ctx.predict(cand_pos_mid) == y0:
                alpha = max(alpha - lamb * gamma, np.pi / 2 - tau)
                if ctx.predict(cand_neg_mid) == y0:
                    alpha = max(alpha - lamb * gamma, np.pi / 2 - tau)
                    # Midpoint was not adversarial, search smaller beta
                    beta_upper = beta_mid
                else:
                    alpha = min(alpha + gamma, np.pi / 2 + tau)
                    beta_lower = beta_mid
                    best_cand = cand_neg_mid
            else:
                alpha = min(alpha + gamma, np.pi / 2 + tau)
                beta_lower = beta_mid
                best_cand = cand_pos_mid
                
        if best_cand is not None:
            x_adv = best_cand
            
    return x_adv


# ============================================================================
# TIER 4 -- SQBA (Small-Query Black-Box Attack)
# This attack combines our black-box queries with a white-box "surrogate" model.
# A surrogate is a helper model we trained ourselves. Since we own the surrogate,
# we can compute its gradients (its normal directions) FOR FREE without any queries!
#
# Crucial idea:
# The surrogate's normal direction is usually similar to the victim's.
# We try using the surrogate's free gradient direction first.
# If it successfully moves us closer to the original image, we save queries!
# If it fails (meaning the surrogate's gradient is incorrect), we fall back
# to the slow, query-based Monte Carlo estimation (`mc_normal`) from HopSkipJump.
# ============================================================================

def attack_sqba(ctx, x0, y0, x_init, iters=15):
    """
    A simplified "teaching" version of SQBA.
    
    It tries using the surrogate's gradient as the step direction.
    If the step succeeds, we accept it (0 queries spent estimating).
    If it fails, we fall back to HopSkipJump's query-based normal estimation (`mc_normal`).
    If our surrogate is completely useless, SQBA automatically behaves like HopSkipJump.
    """
    x_b = x_init.copy(); best, best_d = x_b, np.linalg.norm(x_b - x0)
    white_iters, fallbacks = 0, 0
    for t in range(iters):
        if ctx.over():
            break
        xi = best_d / np.sqrt(t + 1)
        # Try the free surrogate gradient direction
        v = ctx.sgrad(x_b, y0)                          
        x_b2, nd, ok = step_and_project(ctx, x_b, x0, y0, v, xi, best_d)
        if ok and nd < best_d - 1e-3:
            # Success! No queries were spent estimating the normal direction.
            white_iters += 1; x_b, best, best_d = x_b2, x_b2, nd
        else:
            # Failure. Fall back to estimating the normal by querying the victim.
            fallbacks += 1
            v = mc_normal(ctx, x_b, y0, delta=1e-2 * best_d)   
            x_b2, nd, ok = step_and_project(ctx, x_b, x0, y0, v, xi, best_d)
            if ok and nd < best_d:
                x_b, best, best_d = x_b2, x_b2, nd
    return best, white_iters, fallbacks


def grad_hw_multigrad(ctx, x0, y0, x_b, delta, n=5):
    """
    The Multi-Gradient method (from Section 4.1 of the SQBA paper).
    
    As we move from the clean image to the boundary, the gradient directions of the surrogate
    rotate. To find the most effective direction, this function:
    1. Samples the surrogate's gradient at `n` different points along the path.
    2. Tests each candidate gradient direction with a tiny probe.
    3. Keeps the candidate direction that stays adversarial and gets us closest to the original image.
    This costs `n` queries to the victim but finds a much better direction in high-dimensional spaces.
    """
    v_tilde = x_b - x0
    best_mu, best_d = None, np.inf
    # Sample n points along the line segment
    for eta in np.linspace(0.2, 1.0, n):
        x_i = ctx.clip(x0 + eta * v_tilde)
        mu = ctx.sgrad(x_i, y0)                         # Free surrogate gradient at intermediate point
        probe = ctx.clip(x_b + 2 * delta * mu)
        if ctx.is_adv(probe, y0):                       # Check if probe is still adversarial
            d = np.linalg.norm(x0 - probe)
            if d < best_d:
                best_d, best_mu = d, mu
    if best_mu is None:
        # If no intermediate point worked, fall back to the gradient at the boundary point.
        best_mu = ctx.sgrad(x_b, y0)
    nrm = np.linalg.norm(best_mu)
    return best_mu / nrm if nrm else best_mu


def grad_hb_mc(ctx, x_b, y0, delta, t):
    """
    Estimates the victim's boundary normal using the Monte Carlo method (similar to HopSkipJump).
    It adjusts the number of random directions it tries based on the current step number `t`.
    """
    p_t = int(np.ceil(10 * np.sqrt(t + 1)))
    U = ctx.rng.standard_normal((p_t, x_b.size))
    phi = np.array([1.0 if ctx.is_adv(ctx.clip(x_b + delta * u), y0) else -1.0 for u in U])
    g = (phi[:, None] * U).mean(0); n = np.linalg.norm(g)
    return g / n if n else U[0]


def eq8_eq9_update(ctx, x_b, x0, y0, g, alpha_cap=1.0):
    """
    Updates the boundary point along direction `g` (Equations 8 and 9 in the paper).
    
    1. Line Search (Eq 8): Finds the largest step size `alpha` that keeps the point adversarial.
       We start with a guess and halve it up to 6 times until we find an adversarial point.
    2. Projection (Eq 9): Performs a binary search from that point back to the boundary.
    """
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
    """
    The full, paper-faithful implementation of SQBA (Algorithm 1 from the WACV 2024 paper).
    
    It uses:
    1. A custom initialization: starting from the sign of the surrogate gradient.
    2. A boolean "beta switch" (Eq 7):
       - If beta = 1: we trust our surrogate and use the Multi-Gradient method.
       - If beta = 0: we are stuck, so we fall back to the Monte Carlo estimate.
       - We switch beta to 0 if the step size used was too small, or if we didn't improve.
    """
    # Initialize using the sign (+1 or -1) of the surrogate gradient
    x_adv = ctx.clip(x0 + np.sign(ctx.sgrad(x0, y0)))   
    if not ctx.is_adv(x_adv, y0):
        x_adv = loose_start(ctx, x0, y0)
    x_b = binary_search(ctx, x_adv, x0, y0)
    best, best_d = x_b, np.linalg.norm(x_b - x0)
    beta = 1
    white_iters, fallbacks = 0, 0
    for t in range(1, iters + 1):
        if ctx.over():
            break
        delta = 1e-2 * best_d                           
        if beta == 1:
            # Use surrogate gradient (Multi-Gradient method)
            g = grad_hw_multigrad(ctx, x0, y0, x_b, delta)   
        else:
            # Use query-based Monte Carlo estimation
            g = grad_hb_mc(ctx, x_b, y0, delta, t)           
        # Step and project back to boundary
        x_new, nd, used_alpha = eq8_eq9_update(ctx, x_b, x0, y0, g)
        improved = nd < best_d - 1e-4
        if improved:
            x_b, best, best_d = x_new, x_new, nd
        if beta == 1 and improved:
            white_iters += 1
        elif beta == 0:
            fallbacks += 1
        # Update the beta switch: if we didn't make good progress, switch to query estimation (beta = 0)
        beta = 0 if (used_alpha < 0.25 or not improved) else 1   
    return best, white_iters, fallbacks
