"""Slide 14 - Biased Boundary: The Hiker with a Guide Dog.
Blend the random sidestep with the surrogate gradient: eta = (1-bias)*eta + bias*sgrad.
The accepted point x_b' stays ON the wall, closer to x0; the surrogate bias finds it faster."""
import numpy as np
from matplotlib.patches import Circle
from style import (canvas, finish, save, arrow, dot, star, label, badges, legend_dots,
                   boundary_scene, INK, MUTE, BENIGN, ADV, AMBER, CYAN, VIOLET, STAR, MINT)

fig, ax = canvas()
fb = boundary_scene(ax, fb=lambda y: 6.2 + 0.15 * np.sin(0.5 * y),
                    benign_xy=(1.0, 6.7), adv_xy=(9.6, 6.7), name_xy=(6.3, 0.7))
is_adv = lambda p: p[0] > fb(p[1])

x0 = np.array([3.0, 4.0])
x_b = np.array([fb(6.6), 6.6])
x_bp = np.array([fb(4.7), 4.7])                    # accepted point: ON the wall, closer to x0
d, dp = np.linalg.norm(x_b - x0), np.linalg.norm(x_bp - x0)
u = (x_bp - x0) / dp
cand = x0 + d * u                                  # on the radius-d circle, past the wall -> adversarial
assert is_adv(cand) and dp < d
r_hat = (x_b - x0) / d
tan = np.array([-r_hat[1], r_hat[0]])
fp = 0.15 * 0.5 * np.cos(0.5 * x_b[1])             # boundary slope -> surrogate gradient ~ normal
nrm = np.array([1.0, -fp]); nrm /= np.linalg.norm(nrm)

ax.add_patch(Circle(x0, d, fill=False, ec=MUTE, lw=1.2, ls=(0, (5, 4)), alpha=.4, zorder=3))
ax.add_patch(Circle(x0, dp, fill=False, ec=MINT, lw=1.2, ls=(0, (5, 4)), alpha=.45, zorder=3))

# unbiased random sidesteps -- many "fall off the wall" (become clean = wasted query)
rng = np.random.default_rng(2)
for _ in range(8):
    a = rng.uniform(-1.0, 1.0)
    c = x_b + 1.5 * (np.cos(a) * tan + np.sin(a) * (-r_hat))
    arrow(ax, x_b, c, BENIGN if not is_adv(c) else MUTE, lw=1.3, scale=8, z=3, alpha=.5)

# surrogate gradient = directional prior (boundary normal, into adversarial) -- distinct from the step
arrow(ax, x_b, x_b + 1.4 * nrm, AMBER, lw=2.6, scale=15, z=5)
label(ax, x_b + 1.4 * nrm, "surrogate\ngradient (prior)", AMBER, fs=10.5, dx=0.28, dy=0.0, ha="left")

# blended step lands on the circle (adversarial), then re-projects radially onto the wall at x_b'
arrow(ax, x_b, cand, CYAN, lw=3.4, scale=20, z=6)
arrow(ax, cand, x_bp, VIOLET, lw=2.6, scale=18, z=6)

star(ax, x0); label(ax, x0, r"$x_0$", STAR, fs=14, dx=-0.2, dy=-0.5, ha="right")
dot(ax, x_b, INK); label(ax, x_b, r"$x_b$", INK, fs=14, dx=-0.34, dy=0.1, ha="right")
dot(ax, cand, ADV, s=140); label(ax, cand, "blended step\n(adversarial)", ADV, fs=10, dx=0.25, dy=0.05, ha="left")
dot(ax, x_bp, MINT, s=170); label(ax, x_bp, r"$x_b'$  (closer, on the wall)", MINT, fs=11, dx=-0.35, dy=-0.55, ha="center")

ax.text(4.7, 8.2, "a surrogate 'guide dog' steers the walk — far fewer steps fall off the wall",
        color=INK, fontsize=12, ha="center", style="italic", alpha=.9)
ax.text(0.5, 2.0, r"$\eta = (1-b)\,\eta_{rand} + b\,\nabla_{surrogate}$", color=CYAN, fontsize=14, ha="left")
badges(ax, [(1, r"unbiased random sidesteps mostly fall to clean (wasted) in 3072-D", BENIGN),
            (2, r"blend 50% random + 50% surrogate gradient as a directional prior", AMBER),
            (3, r"the accepted  $x_b'$  stays on the wall, closer than  $x_b$", MINT)], 0.6, 1.45)
legend_dots(ax, [(BENIGN, "random step fell to clean (failed)")], 6.7, 2.0)

finish(ax, (-1.6, 11.0), (-0.4, 8.6))
save(fig, "14_biased_boundary")
