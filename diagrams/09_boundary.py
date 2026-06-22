"""Slide 9 - Boundary Attack: Walking Along the Wall.
Orthogonal sidestep along a constant-distance circle, then a radial pull -> a NEW boundary point closer to x0.
(Every accepted point stays ON the boundary / adversarial side.)"""
import numpy as np
from matplotlib.patches import Circle
from style import (canvas, finish, save, arrow, dot, star, label, badges,
                   boundary_scene, INK, MUTE, BENIGN, ADV, AMBER, CYAN, STAR, MINT)

fig, ax = canvas()
# gentle boundary so distance-to-x0 varies along it (a near-circular wall would not)
fb = boundary_scene(ax, fb=lambda y: 6.2 + 0.15 * np.sin(0.5 * y),
                    benign_xy=(1.0, 6.7), adv_xy=(9.6, 6.7), name_xy=(6.3, 0.7))
is_adv = lambda p: p[0] > fb(p[1])

x0 = np.array([3.0, 4.0])
x_b = np.array([fb(6.8), 6.8])                     # current boundary point (far up the wall)
x_bp = np.array([fb(4.6), 4.6])                    # a boundary point CLOSER to x0 (the accepted result)
d, dp = np.linalg.norm(x_b - x0), np.linalg.norm(x_bp - x0)
u = (x_bp - x0) / dp
cand = x0 + d * u                                  # on the radius-d circle, but past the wall -> adversarial
assert is_adv(cand) and dp < d, (is_adv(cand), dp, d)

ax.add_patch(Circle(x0, d, fill=False, ec=MUTE, lw=1.3, ls=(0, (5, 4)), alpha=.5, zorder=3))
ax.add_patch(Circle(x0, dp, fill=False, ec=MINT, lw=1.3, ls=(0, (5, 4)), alpha=.5, zorder=3))
ax.plot([x0[0], x_b[0]], [x0[1], x_b[1]], color=MUTE, lw=1.0, ls=(0, (2, 3)), alpha=.45, zorder=2)

# 1. orthogonal sidestep along the constant-distance circle (x_b -> cand, both at distance d)
arrow(ax, x_b, cand, CYAN, lw=3.4, scale=22, z=6)
# 2. radial pull toward x0 -> lands on the wall at x_b' (distance dp < d)
arrow(ax, cand, x_bp, AMBER, lw=3.4, scale=22, z=6)

star(ax, x0); label(ax, x0, r"$x_0$  (clean target)", STAR, dy=-0.6)
dot(ax, x_b, INK); label(ax, x_b, r"$x_b$", INK, fs=14, dx=-0.36, dy=0.1, ha="right")
dot(ax, cand, ADV, s=150); label(ax, cand, "sidestep\n(still adversarial)", ADV, fs=10.5, dx=0.26, dy=0.0, ha="left")
dot(ax, x_bp, MINT, s=175); label(ax, x_bp, r"$x_b'$  (closer, on the wall)", MINT, fs=11.5, dx=-0.4, dy=-0.55, ha="center")

ax.text(4.7, 8.2, "pure black-box: only the label — every accepted point stays on the wall",
        color=INK, fontsize=12, ha="center", style="italic", alpha=.9)
badges(ax, [(1, r"SIDESTEP  ·  move along the constant-distance circle (explore the wall)", CYAN),
            (2, r"PULL  ·  step radially toward  $x_0$ , re-projecting onto the wall", AMBER),
            (3, r"keep it only if still adversarial  →  $x_b'$  is closer than  $x_b$", MINT)], 0.6, 1.5)

finish(ax, (-1.6, 11.0), (-0.4, 8.6))
save(fig, "09_boundary")
