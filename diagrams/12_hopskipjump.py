"""Slide 12 - HopSkipJump: The Three-Step Dance.
Hop (estimate normal from +/-1 probes) -> Skip (step along normal) -> Jump (binary-search back to x0)."""
import numpy as np
from matplotlib.patches import Circle
from style import (canvas, finish, save, arrow, dot, star, label, badges, legend_dots,
                   boundary_scene, INK, MUTE, BENIGN, ADV, AMBER, CYAN, VIOLET, STAR, MINT, BG)

fig, ax = canvas()
fb = boundary_scene(ax, name_xy=(5.05, 0.7))

x0 = np.array([1.35, 3.2])
yb = 5.2; x_b = np.array([fb(yb), yb])
fp = 0.7 * 0.45 * np.cos(0.45 * yb + 0.3)
n = np.array([1.0, -fp]); n /= np.linalg.norm(n)
x_step = x_b + 2.1 * n
t = np.linspace(0, 1, 800)
P = x_step[None, :] + t[:, None] * (x0 - x_step)[None, :]
x_bnew = P[np.argmin(np.abs(P[:, 0] - fb(P[:, 1])))]

# distance guides: old vs new
ax.plot([x0[0], x_b[0]], [x0[1], x_b[1]], color=MUTE, lw=1.1, ls=(0, (2, 3)), alpha=.5, zorder=2)
ax.plot([x0[0], x_bnew[0]], [x0[1], x_bnew[1]], color=MINT, lw=1.3, ls=(0, (2, 3)), alpha=.6, zorder=2)

# 1. HOP: probes around x_b
ax.add_patch(Circle(x_b, 0.78, fill=False, ec=MUTE, lw=1.0, ls=(0, (2, 3)), alpha=.45, zorder=3))
rng = np.random.default_rng(7)
for _ in range(14):
    a = rng.uniform(0, 2 * np.pi); p = x_b + 0.78 * np.array([np.cos(a), np.sin(a)])
    arrow(ax, x_b, p, ADV if p[0] > fb(p[1]) else BENIGN, lw=1.3, scale=7, z=4, alpha=.9)
# Hop->Skip two-tone arrow
x_mid = x_b + 0.95 * n
arrow(ax, x_b, x_mid, AMBER, lw=3.4, scale=16, z=5)
arrow(ax, x_mid, x_step, CYAN, lw=3.0, scale=20, z=5)
ax.scatter(*x_mid, s=42, c=BG, edgecolors=AMBER, linewidths=1.6, zorder=6)
label(ax, x_mid, r"$\hat g$", AMBER, fs=15, dy=0.42)

# 3. JUMP: binary search back to x0
arrow(ax, x_step, x0, VIOLET, lw=2.6, scale=18, ls=(0, (6, 4)), z=4)
for tb in np.linspace(0.14, 0.66, 4):
    ax.scatter(*(x_step + tb * (x0 - x_step)), s=20, c=VIOLET, zorder=4, alpha=.85)

star(ax, x0); label(ax, x0, r"$x_0$  (clean target)", STAR, dy=-0.6)
dot(ax, x_step, CYAN, s=150); label(ax, x_step, r"$x_{step}$", CYAN, dx=0.22, dy=0.16, ha="left")
dot(ax, x_b, INK); label(ax, x_b, r"$x_b$", INK, fs=14, dx=0.34, dy=0.30, ha="left")
dot(ax, x_bnew, MINT); label(ax, x_bnew, r"$x_b'$" + "\ncloser to $x_0$", MINT, fs=11, dx=-0.55, dy=-0.42, va="top")

badges(ax, [(1, r"HOP  ·  average  ±1  probes  →  boundary normal  $\hat g$", AMBER),
            (2, r"SKIP  ·  step along  $\hat g$  into the adversarial region", CYAN),
            (3, r"JUMP  ·  binary-search back toward  $x_0$", VIOLET)], 6.5, 1.45)
legend_dots(ax, [(ADV, "probe still adversarial (+1)"), (BENIGN, "probe fell to clean (−1)")], 8.7, 4.95)

finish(ax, (-0.2, 10.6), (-0.1, 8.2))
save(fig, "12_hopskipjump")
