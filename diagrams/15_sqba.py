"""Slide 15 - SQBA: The Best of Both Worlds.
Read the boundary normal FREE from the surrogate (instead of paying HSJ for it); verify; fall back only on failure."""
import numpy as np
from matplotlib.patches import Circle
from style import (canvas, finish, save, arrow, dot, star, label, badges,
                   boundary_scene, INK, MUTE, BENIGN, ADV, AMBER, CYAN, VIOLET, STAR, MINT, BG)

fig, ax = canvas()
fb = boundary_scene(ax, benign_xy=(1.3, 7.3), adv_xy=(9.3, 7.3), name_xy=(5.05, 0.7))

x0 = np.array([1.6, 3.4])
yb = 5.7; x_b = np.array([fb(yb), yb])
fp = 0.7 * 0.45 * np.cos(0.45 * yb + 0.3)
n = np.array([1.0, -fp]); n /= np.linalg.norm(n)

ax.plot([x0[0], x_b[0]], [x0[1], x_b[1]], color=MUTE, lw=1.0, ls=(0, (2, 3)), alpha=.4, zorder=2)

# --- what HSJ PAYS for: a faded Monte-Carlo probe fan (the thing SQBA avoids) ---
ax.add_patch(Circle(x_b, 0.7, fill=False, ec=MUTE, lw=1.0, ls=(0, (2, 3)), alpha=.3, zorder=3))
rng = np.random.default_rng(5)
for _ in range(12):
    a = rng.uniform(0, 2 * np.pi); p = x_b + 0.7 * np.array([np.cos(a), np.sin(a)])
    arrow(ax, x_b, p, (ADV if p[0] > fb(p[1]) else BENIGN), lw=1.0, scale=6, z=3, alpha=.32)
ax.text(x_b[0] + 0.15, x_b[1] - 1.15, "HopSkipJump would\nquery ~20 probes here", color=MUTE, fontsize=10, ha="center")

# --- what SQBA does: read the normal FREE from the surrogate, step into adversarial, project back ---
is_adv = lambda p: p[0] > fb(p[1])
x_step = x_b + 1.75 * n
assert is_adv(x_step)
arrow(ax, x_b, x_step, AMBER, lw=3.4, scale=18, z=6)
ax.scatter(*x_step, marker="o", s=120, c=BG, edgecolors=AMBER, linewidths=2, zorder=6)
label(ax, x_step, r"$\hat g$", AMBER, fs=14, dx=0.0, dy=0.4)

# success: binary-search back toward x0 -> a NEW boundary point closer than x_b (on the wall)
t = np.linspace(0, 1, 800)
P = x_step[None, :] + t[:, None] * (x0 - x_step)[None, :]
x_new = P[np.argmin(np.abs(P[:, 0] - fb(P[:, 1])))]
assert np.linalg.norm(x_new - x0) < np.linalg.norm(x_b - x0)
arrow(ax, x_step, x_new, CYAN, lw=2.8, scale=16, ls=(0, (6, 4)), z=5)
dot(ax, x_new, MINT, s=170, z=7); label(ax, x_new, r"$x_b'$", MINT, fs=13, dx=0.2, dy=-0.35, ha="left")

# clean step-stack in the open adversarial region
tx, ty = 7.5, 6.7
ax.text(tx, ty,        "FREE  ·  surrogate normal  (0 q)", color=AMBER,  fontsize=11.5, weight="bold")
ax.text(tx, ty - 0.58, "verify  ·  1 victim query",        color=INK,    fontsize=11.5)
ax.text(tx, ty - 1.16, "✓  accept → 0-query step",          color=BENIGN, fontsize=11.5, weight="bold")
ax.text(tx, ty - 1.74, "✗  fail → HSJ probes (paid)",       color=ADV,    fontsize=11.5, weight="bold")

star(ax, x0); label(ax, x0, r"$x_0$", STAR, fs=14, dx=-0.18, dy=-0.5, ha="right")
dot(ax, x_b, INK); label(ax, x_b, r"$x_b$", INK, fs=14, dx=-0.36, dy=0.12, ha="right")

ax.text(5.3, 8.05, "trust the free surrogate normal — verify — pay only when it fails",
        color=INK, fontsize=12, ha="center", style="italic", alpha=.92)
badges(ax, [(1, r"read the boundary normal from the surrogate gradient — FREE (0 queries)", AMBER),
            (2, r"step + 1 victim query: still adversarial?  ✓ accept, saving ~20 queries", CYAN),
            (3, r"✗ → fall back to HopSkipJump probing.  Decays gracefully to HSJ", VIOLET)], 1.6, 1.5)

finish(ax, (-0.2, 11.0), (-0.1, 8.4))
save(fig, "15_sqba")
