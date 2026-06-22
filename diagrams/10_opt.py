"""Slide 10 - OPT Attack: The Angle-Based Distance Measurer.
For each direction theta, g(theta) = distance to the boundary. Minimize g over theta."""
import numpy as np
from style import (canvas, finish, save, arrow, dot, star, label, badges, legend_dots,
                   boundary_scene, INK, MUTE, BENIGN, ADV, AMBER, STAR, MINT)

fig, ax = canvas()
fb = boundary_scene(ax, name_xy=(5.05, 0.7))

x0 = np.array([1.7, 4.0])

def hit(theta):
    "distance from x0 to boundary along unit direction theta (positive x component)."
    u = np.array([np.cos(theta), np.sin(theta)])
    ts = np.linspace(0.1, 12, 1200)
    pts = x0[None, :] + ts[:, None] * u[None, :]
    k = np.argmin(np.abs(pts[:, 0] - fb(pts[:, 1])))
    return ts[k], pts[k]

angles = np.deg2rad([-26, -13, 0, 13, 26, 38])
gs = [(a, *hit(a)) for a in angles]
best = min(gs, key=lambda z: z[1])

for a, g, p in gs:
    is_best = (a == best[0])
    col = AMBER if is_best else MUTE
    ax.plot([x0[0], p[0]], [x0[1], p[1]], color=col, lw=3.0 if is_best else 1.4,
            alpha=1 if is_best else .55, zorder=4 if is_best else 3)
    dot(ax, p, col, s=130 if is_best else 70, z=5)
    if is_best:
        label(ax, (x0 + p) / 2, r"$g(\theta^\star)$", AMBER, fs=12, dy=0.45)

# probe nudges around the best direction (gradient estimation)
ab = best[0]
for du in (-0.16, 0.16):
    _, pp = hit(ab + du)
    ax.plot([x0[0], pp[0]], [x0[1], pp[1]], color=MINT, lw=1.2, ls=(0, (3, 3)), alpha=.7, zorder=3)

star(ax, x0); label(ax, x0, r"$x_0$", STAR, fs=14, dx=-0.15, dy=-0.5, ha="right")
ax.text(1.7, 7.6, "view the attack as choosing a direction θ, not a point",
        color=INK, fontsize=12, ha="left", style="italic", alpha=.9)
badges(ax, [(1, r"for each direction θ, binary-search the distance  $g(θ)$  to the boundary", MUTE),
            (2, r"probe nudges  θ±u  to estimate which way lowers  $g$  (zeroth-order)", MINT),
            (3, r"turn θ to minimize  $g(θ)$  →  the closest adversarial point", AMBER)], 4.7, 1.5)

finish(ax, (-0.2, 10.8), (-0.1, 8.2))
save(fig, "10_opt")
