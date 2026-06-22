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
        """
        Initializes the Context helper object.

        Parameters:
        -----------
        label_fn : callable
            A function that asks the victim model for its predicted label.
        sgrad_fn : callable
            A function that calculates the surrogate model's normalized input gradient.
        pool_X : np.ndarray
            A database of clean dataset images used to search for starting points.
        pool_y : np.ndarray
            The correct label indices corresponding to pool_X.
        rng : np.random.Generator
            A random number generator for reproducibility.
        lo : float or np.ndarray, default=0.0
            The minimum valid pixel boundary (0.0 for normal images).
        hi : float or np.ndarray, default=1.0
            The maximum valid pixel boundary (1.0 for normal images).
        """
        self._label = label_fn
        self._sgrad = sgrad_fn
        self.pool_X = pool_X
        self.pool_y = pool_y
        self.rng = rng
        self.lo = lo
        self.hi = hi
        self.q = 0                     
        self.budget = float("inf")     

    def clip(self, x):
        """
        Ensures that all pixels in image `x` stay within the valid boundaries [lo, hi].

        Parameters:
        -----------
        x : np.ndarray
            The input image vector.

        Returns:
        --------
        np.ndarray
            The clipped image vector.
        """
        return np.clip(x, self.lo, self.hi)

    def predict(self, x):
        """
        Asks the victim model for a prediction and increments the query counter.

        Parameters:
        -----------
        x : np.ndarray
            The input image vector to classify.

        Returns:
        --------
        int
            The predicted label index from the victim model.
        """
        self.q += 1
        return self._label(self.clip(x))

    def is_adv(self, x, y0):
        """
        Checks if the victim classifies `x` as adversarial (a class other than `y0`).
        This counts as 1 query.

        Parameters:
        -----------
        x : np.ndarray
            The candidate image vector.
        y0 : int
            The correct label index.

        Returns:
        --------
        bool
            True if the model is fooled, False otherwise.
        """
        return self.predict(x) != y0

    def fooled(self, x, y0):           
        """
        Checks if the model classifies `x` as adversarial WITHOUT incrementing the query counter.
        Used only for printing final diagnostic summaries.

        Parameters:
        -----------
        x : np.ndarray
            The candidate image vector.
        y0 : int
            The correct label index.

        Returns:
        --------
        bool
            True if the model is fooled, False otherwise.
        """
        return self._label(self.clip(x)) != y0

    def sgrad(self, x, y0):
        """
        Asks the surrogate helper model for the normalized gradient direction.
        Since we own the surrogate, this requires 0 queries to the victim.

        Parameters:
        -----------
        x : np.ndarray
            The input image vector.
        y0 : int
            The correct label index.

        Returns:
        --------
        np.ndarray
            The normalized gradient direction vector.
        """
        return self._sgrad(self.clip(x), y0)

    def over(self):
        """
        Checks if the queries made so far have exceeded the specified budget.

        Returns:
        --------
        bool
            True if out of budget, False otherwise.
        """
        return self.q >= self.budget


# ----------------------------------------------------------------------------
# Boundary-projection primitive, reused by rungs 2-5.
# ----------------------------------------------------------------------------
def binary_search(ctx, x_adv, x0, y0, tol=1e-2):
    """
    Finds the boundary where the model changes its mind by searching the line between
    a clean image (`x0`) and an adversarial image (`x_adv`).
    
    Explanation of Math/Linear Algebra:
    ----------------------------------
    - `np.linalg.norm(hi - lo)`: Calculates the Euclidean distance (also called the L2 norm)
      between the two image vectors `hi` and `lo`. Conceptually, it takes the difference of
      every corresponding pixel value, squares them, adds them all up, and takes the square root:
      distance = sqrt(sum((hi_i - lo_i)^2)).
    - `0.5 * (lo + hi)`: Linearly averages the two image vectors to find the exact halfway midpoint.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `tol=1e-2` (0.01): The convergence threshold. Once the Euclidean distance between
      the benign point `lo` and the adversarial point `hi` is less than 0.01, we stop searching.
      This value is chosen because it yields high spatial precision while keeping the number of
      binary search steps low (typically log2(1 / 0.01) ≈ 7 steps).

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x_adv : np.ndarray
        An adversarial starting image vector.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label of the clean image.
    tol : float, default=1e-2
        The distance tolerance at which to stop search.

    Returns:
    --------
    np.ndarray
        The boundary image vector (which classifies as adversarial but is extremely close to the clean space).
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
    
    Explanation of Math/Linear Algebra:
    ----------------------------------
    - `ctx.rng.standard_normal(x0.shape)`: Generates a vector of the same shape as `x0` filled with
      random numbers drawn from a Gaussian distribution (bell curve) with a mean of 0.0 and a standard
      deviation of 1.0. This scatters changes randomly in all dimensions.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `N=1500`: The budget of random directions to try. Chosen because in low-dimensional spaces (like 64 pixels),
      1500 trials are enough to occasionally find a random flip. In high-dimensional spaces (like CIFAR-10's 3072 features),
      we override this to `N=400` in the demo to keep execution fast.
    - `scale=0.5` (or `0.2` in CIFAR-10): The standard deviation scaling factor for the Gaussian noise.
      If it is too small (e.g., 0.01), we will never flip the label. If it is too large (e.g., 2.0), the changes will be
      massive and highly visible, resulting in a poor perturbation score.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label index.
    N : int, default=1500
        Number of iterations.
    scale : float, default=0.5
        Standard deviation scale for random noise.

    Returns:
    --------
    np.ndarray
        The best adversarial image vector found, or the original image x0 if no flip was achieved.
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
    """
    Finds an image from our database pool that belongs to a class other than `y0`.

    Explanation of Math/Linear Algebra:
    ----------------------------------
    - `np.linalg.norm(others - x0, axis=1)`: Subtracts `x0` from every row in the `others` array, and calculates
      the L2 norm (length) of the difference vector for each image. This returns a 1D array of distances.
    - `np.argmax(d)` / `np.argmin(d)`: Returns the index of the largest/smallest distance value in the array.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label index.
    farthest : bool
        If True, returns the sample that is farthest from x0. If False, returns the closest.

    Returns:
    --------
    np.ndarray
        The selected image vector from a different class.
    """
    others = ctx.pool_X[ctx.pool_y != y0]
    d = np.linalg.norm(others - x0, axis=1)
    return others[np.argmax(d) if farthest else np.argmin(d)]


def attack_line(ctx, x0, y0):
    """
    A simple attack that takes another image (like a dog if ours is a cat),
    draws a line between them, and uses binary search to find where they cross the boundary.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label index.

    Returns:
    --------
    np.ndarray
        The boundary image vector found on the line segment.
    """
    return binary_search(ctx, _other_class_sample(ctx, x0, y0, False), x0, y0)


def loose_start(ctx, x0, y0):
    """
    Finds a starting point that is adversarial but quite far from our original image.
    We do this by choosing the farthest image of a different class, and searching the boundary on that line.
    This gives our optimization attacks a starting point that they can then try to improve.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label index.

    Returns:
    --------
    np.ndarray
        A guaranteed adversarial boundary starting point.
    """
    return binary_search(ctx, _other_class_sample(ctx, x0, y0, True), x0, y0)


def attack_boundary(ctx, x0, y0, x_init, bias=0.0, steps=500):
    """
    Boundary Attack: Walks along the decision boundary, trying to get closer to the original image.
    
    Explanation of Math/Linear Algebra:
    ----------------------------------
    - `eta - (eta @ r) / (d * d) * r`: Projects the random vector `eta` onto the tangent plane of the sphere
      around `x0` at the point `x_b`. The dot product `@` measures how much `eta` points along the radial vector `r`.
      Subtracting this component makes the step orthogonal (perpendicular) to `r`, which keeps the distance to `x0` constant.
    - `cand - src * (cand - x0)`: Takes the candidate and pulls it direct line-of-sight towards `x0` by a fraction `src`.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `sph = 0.10`: Initial orthogonal step size. Chosen as a small fraction (10%) of the current distance to explore the boundary.
    - `src = 0.05`: Initial radial step size. Chosen as 5% to pull the candidate closer to `x0`.
    - `30`: The period for adjusting step sizes. After 30 steps, we evaluate the success rate.
    - `0.2` and `0.5`: Target acceptance rates. If the success rate is < 20% (too many steps failed), we shrink step sizes by `0.9` to stay on the boundary.
      If success rate is > 50% (too easy), we expand them by `1.1` to speed up optimization.
    - `0.5` (bias): When `bias=0.5` (Biased Boundary Attack), we blend 50% random walk direction with 50% surrogate gradient direction.
      This is a balanced value that guides the search using the surrogate model without getting stuck in local surrogate minima.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label index.
    x_init : np.ndarray
        The starting adversarial image vector (on the boundary).
    bias : float, default=0.0
        The surrogate gradient steer bias.
    steps : int, default=500
        The maximum number of walk iterations.

    Returns:
    --------
    np.ndarray
        The optimized adversarial image vector closer to x0.
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

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `tol=4e-2` (0.04): The distance tolerance for the binary search. Since this function is called repeatedly
      inside the gradient estimation loop, we use a looser tolerance (0.04 instead of 0.01) to save queries.
    - `1.5`: The multiplier to increase the step size if the initial step isn't adversarial yet (exponential search).
    - `50.0`: The maximum distance safety limit. If we go beyond this distance without finding a flip, we declare
      the direction as non-adversarial (infinity) to avoid infinite loops.
    - `0.5`: The halving factor to shrink search range.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label index.
    theta : np.ndarray
        The direction unit vector.
    guess : float, default=1.0
        Initial distance guess.
    tol : float, default=4e-2
        Tolerance threshold.

    Returns:
    --------
    float
        The distance along theta to the boundary (or inf if out of range).
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
    
    Explanation of Math/Linear Algebra:
    ----------------------------------
    - `(g_dist(ctx, x0, y0, theta + beta * u, guess=g_theta) - g_theta) / beta * u`: Estimates the directional
      derivative of the boundary function `g` along the random direction `u` using finite differences.
      Averaging these over `q` probes yields a zeroth-order approximation of the gradient.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `iters=14`: Number of optimization updates. Chosen to fit within query limitations of standard tests.
    - `q=6`: Number of random directions probed per step. A small number to conserve queries (as each probe is a full binary search).
    - `beta=0.02` (2%): The finite difference perturbation size. Small enough to remain a local derivative estimate,
      but large enough to not get wiped out by numerical precision limits.
    - `alpha=0.3` (30%): The learning rate (gradient descent step size) used to update the search direction theta.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label index.
    x_init : np.ndarray
        Initial boundary point.
    iters : int, default=14
        Number of steps.
    q : int, default=6
        Number of directions to sample.
    beta : float, default=0.02
        Finite difference step.
    alpha : float, default=0.3
        Learning rate.

    Returns:
    --------
    np.ndarray
        The optimized adversarial image vector.
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
    
    Explanation of Math/Logic:
    -------------------------
    - `s = 1.0 if ctx.is_adv(x0 + g_theta * nt, y0) else -1.0`: Instead of measuring the exact boundary distance,
      we check if nudging the direction unit vector keeps the image adversarial (+1) or makes it clean (-1).
      This sign acts as a binary indicator, and the average `s * u` estimates the gradient direction with a single query per direction.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `q=12`: We probe 12 random directions. Since each direction only costs 1 query (instead of ~7 queries for a full binary search),
      we can afford to double the directions probed compared to OPT (`q=6`) to get a more accurate gradient, while still using 70% fewer queries!
    - Other parameters (`beta=0.02`, `alpha=0.3`, `iters=14`) match OPT's default tuning to maintain stable convergence properties.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image vector.
    y0 : int
        The correct label index.
    x_init : np.ndarray
        Initial boundary point.
    iters : int, default=14
        Number of steps.
    q : int, default=12
        Number of sign directions to sample.
    beta : float, default=0.02
        Direction nudge scale.
    alpha : float, default=0.3
        Learning rate.

    Returns:
    --------
    np.ndarray
        The optimized adversarial image vector.
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
    
    Explanation of Math/Logic:
    -------------------------
    - `phi[:, None] * U`: Weights each random direction row in `U` by its classification sign `phi` (+1 or -1).
      The average vector `g` will point in the direction where the space is adversarial.
      Since the current point `x_b` lies exactly on the boundary, this average vector points directly perpendicular
      (normal) to the boundary plane.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `B=20`: Number of random sample directions (Monte Carlo probes). 20 probes provide a solid balance between
      query count and high directional accuracy in normal estimation.
    - `delta=0.01` (1%): The size of the Monte Carlo probe step. It must be small enough to stay in the local
      vicinity of the boundary point to represent a true local mathematical normal.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x_b : np.ndarray
        The current boundary image vector.
    y0 : int
        The correct label index.
    B : int, default=20
        Number of probes.
    delta : float, default=0.01
        Probe step scale.

    Returns:
    --------
    np.ndarray
        The estimated boundary normal direction unit vector.
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

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x_b : np.ndarray
        Current boundary point.
    x0 : np.ndarray
        Clean target image vector.
    y0 : int
        Correct label index.
    v : np.ndarray
        Search direction vector.
    xi : float
        Step size.
    best_d : float
        The current best L2 distance.

    Returns:
    --------
    tuple (np.ndarray, float, bool)
        - The updated boundary point.
        - The new L2 distance to x0.
        - True if the step was accepted, False if rejected (not adversarial).
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
    
    Explanation of Math/Logic:
    -------------------------
    - `best_d / np.sqrt(t + 1)`: The step size `xi` dynamically decays as iterations progress.
      As we converge closer to the target image `x0`, the decision boundary becomes more curved,
      so we must take smaller steps to avoid overshooting or stepping out of bounds.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `iters=15` and `B=20`: HSJ defaults that ensure we get a highly optimized perturbation within a total budget
      of around 300 to 500 queries.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image.
    y0 : int
        The correct label.
    x_init : np.ndarray
        The initial boundary starting point.
    iters : int, default=15
        Number of update iterations.
    B : int, default=20
        Monte Carlo sample size.

    Returns:
    --------
    np.ndarray
        The optimized adversarial image vector.
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
    
    Explanation of Math/Logic:
    -------------------------
    - `dct` / `idct`: Discrete Cosine Transform. Projects spatial pixel representations into the frequency domain.
    - `v_dir = v_spatial - (v_spatial @ u_dir) * u_dir`: Standard Gram-Schmidt orthogonalization.
      Ensures the random direction vector `v_dir` is mathematically perpendicular to `u_dir` (the vector to `x0`).
    - `delta_t * np.sin(alpha + beta) / sin_alpha`: The Law of Sines formula. In the 2D plane, if we form a triangle
      with vertices `x0`, `x_t` and `x_{t+1}`, this formula calculates the new distance `delta_{t+1}`.
      By keeping `alpha` close to 90 degrees and choosing a small `beta`, it guarantees `delta_{t+1} < delta_t`.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `iters=40`: Number of updates. Since Triangle Attack uses only 2 queries per step (extreme efficiency),
      we can afford 40 updates while spending fewer than 100 queries overall!
    - `N=2`: Number of binary search iterations to find optimal `beta`. 2 iterations are sufficient to get a good step.
    - `0.10 * D` (L): Low-frequency range (10% of total dimensions). Higher frequencies represent rapid, noisy pixel transitions,
      whereas low frequencies represent smooth changes. Restricting changes to low frequencies yields much more natural,
      hard-to-detect adversarial perturbations.
    - `alpha = np.pi / 2` (90 degrees): The initial search vertex angle.
    - `gamma = 0.01` & `lamb = 0.05` & `tau = 0.1`: Geometric scaling hyperparameters tuned by the paper authors to ensure
      stable angle updates.
    - `d = 3`: Number of random frequency components modified at a time to keep updates localized.
    - `beta_limit = np.pi / 16` (11.25 degrees): Prevents the search angle from collapsing to zero (which would yield no progress).

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image.
    y0 : int
        The correct label.
    x_init : np.ndarray
        Initial boundary point.
    iters : int, default=40
        Number of steps.
    N : int, default=2
        Number of search iterations for beta.

    Returns:
    --------
    np.ndarray
        The optimized adversarial image vector.
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
    
    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `1e-3`: A small improvement threshold. If the surrogate step does not improve the L2 distance
      by at least 0.001, we reject it and execute the Monte Carlo fallback.
    - `delta = 1e-2 * best_d` (1% of distance): The scale of Monte Carlo probes used in the fallback.
      Dynamically scales down as we get closer to `x0`.
    - `iters=15`: The default iterations for direct comparison against HopSkipJump.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image.
    y0 : int
        The correct label index.
    x_init : np.ndarray
        Initial boundary point.
    iters : int, default=15
        Number of steps.

    Returns:
    --------
    tuple (np.ndarray, int, int)
        - The optimized adversarial image.
        - Number of successful surrogate (white-box) steps.
        - Number of query-based Monte Carlo fallback steps.
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
    
    Explanation of Math/Logic:
    -------------------------
    - `np.linspace(0.2, 1.0, n)`: Probes the surrogate's gradient at multiple points along the path
      connecting the clean image `x0` and the boundary `x_b` (at 20%, 40%, 60%, 80%, and 100% of the segment).
      This handles gradient rotation in high-dimensional spaces to find the direction that stays adversarial the longest.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `n=5`: Number of path samples. Proving 5 points balances query efficiency with directional stability.
    - `2 * delta`: A small testing step size to check if a candidate gradient direction `mu` is adversarial.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image.
    y0 : int
        The correct label index.
    x_b : np.ndarray
        The current boundary point.
    delta : float
        Nudge step size.
    n : int, default=5
        Number of multi-gradient samples.

    Returns:
    --------
    np.ndarray
        The selected optimal normalized surrogate gradient vector.
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

    Explanation of Math/Hardcoded Numbers:
    --------------------------------------
    - `int(np.ceil(10 * np.sqrt(t + 1)))`: Dynamic probe size `p_t`. As iteration `t` increases, the search
      gets closer to `x0` and the boundary details get finer. To maintain high accuracy, we increase the number
      of Monte Carlo directions probed dynamically over time.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x_b : np.ndarray
        Current boundary point.
    y0 : int
        Correct label.
    delta : float
        Probe step size.
    t : int
        The current iteration number.

    Returns:
    --------
    np.ndarray
        The estimated normal unit vector.
    """
    p_t = int(np.ceil(10 * np.sqrt(t + 1)))
    U = ctx.rng.standard_normal((p_t, x_b.size))
    phi = np.array([1.0 if ctx.is_adv(ctx.clip(x_b + delta * u), y0) else -1.0 for u in U])
    g = (phi[:, None] * U).mean(0); n = np.linalg.norm(g)
    return g / n if n else U[0]


def eq8_eq9_update(ctx, x_b, x0, y0, g, alpha_cap=1.0):
    """
    Updates the boundary point along direction `g` (Equations 8 and 9 in the paper).
    
    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `6`: Number of line search halving attempts. If we cannot find an adversarial step after halving the
      learning rate 6 times (shrunk to 1/64 of the original size), we reject the update to save queries and prevent oscillations.
    - `0.5`: The halving factor to shrink step sizes.
    - `1e-3`: Minimum distance threshold.

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x_b : np.ndarray
        Current boundary point.
    x0 : np.ndarray
        Clean target image.
    y0 : int
        Correct label.
    g : np.ndarray
        Search direction.
    alpha_cap : float, default=1.0
        Maximum allowed step cap.

    Returns:
    --------
    tuple (np.ndarray, float, float)
        - The updated boundary image vector.
        - The new L2 distance to x0.
        - The actual step size (alpha) successfully used.
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
    
    Explanation of Math/Logic:
    -------------------------
    - **Beta Switch**: $\beta$ controls whether we use the free surrogate gradient ($\beta=1$) or run query-heavy
      Monte Carlo normal probing ($\beta=0$). If the surrogate gradient fails to yield a successful step, we switch
      $\beta$ to 0 to rely on queries. Once a query-based step succeeds, we switch $\beta$ back to 1 to try the surrogate again.
    - `np.sign(ctx.sgrad(x0, y0))`: Custom initialization. Stepping along the sign (+1 or -1) of the surrogate gradient at the clean image.

    Explanation of Hardcoded Numbers:
    ---------------------------------
    - `1e-4` (improvement threshold): If the update doesn't reduce the L2 distance by at least 0.0001, we consider the update a failure.
    - `0.25`: The step size threshold. If the line search was forced to shrink the step size below 25% of the distance,
      the surrogate's geometry is unaligned, triggering the Beta Switch (`beta = 0`).

    Parameters:
    -----------
    ctx : Ctx
        The attack context object.
    x0 : np.ndarray
        The clean target image.
    y0 : int
        The correct label index.
    iters : int, default=15
        Number of steps.

    Returns:
    --------
    tuple (np.ndarray, int, int)
        - The optimized adversarial image.
        - Number of successful surrogate steps.
        - Number of query-based Monte Carlo steps.
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
