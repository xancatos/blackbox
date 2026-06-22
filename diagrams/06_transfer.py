"""Slide 6 - Transfer Attack: The Black-Box Sweep.
Craft PGD on a SURROGATE boundary (white-box), grow eps until the L-inf box corner crosses the VICTIM boundary."""
import numpy as np
from matplotlib.patches import Rectangle
from style import (canvas, finish, save, arrow, dot, star, label,
                   INK, MUTE, BENIGN, ADV, AMBER, CYAN, STAR, BOUND, MINT)

fig, ax = canvas(11.5, 7.2)

ys = np.linspace(-1, 9, 200)
surr_b = lambda y: 5.0 + 0.35 * np.sin(0.5 * y)
vict_b = lambda y: 5.8 + 0.30 * np.sin(0.5 * y + 1.0)
ax.fill_betweenx(ys, -3, vict_b(ys), color=BENIGN, alpha=0.06)
ax.fill_betweenx(ys, vict_b(ys), 13, color=ADV, alpha=0.06)
ax.plot(surr_b(ys), ys, color=CYAN, lw=2.2, ls=(0, (6, 4)), alpha=.9, zorder=3)
ax.plot(vict_b(ys), ys, color=BOUND, lw=2.6, zorder=3)
label(ax, (surr_b(6.6) - 0.15, 6.9), "surrogate\nboundary", CYAN, fs=10.5, ha="right")
label(ax, (vict_b(7.3) + 0.2, 7.4), "victim\nboundary", BOUND, fs=10.5, ha="left")

x0 = np.array([2.5, 2.7])
sg = np.array([1.0, 1.0])                            # sign(grad): +-eps per pixel -> diagonal corner
boxes = [1.2, 2.1, 3.0]
hit_e = next((e for e in boxes if (x0 + e * sg)[0] > vict_b((x0 + e * sg)[1])), boxes[-1])
for e in boxes:
    crossed = e >= hit_e
    c = ADV if crossed else MUTE
    ax.add_patch(Rectangle((x0[0] - e, x0[1] - e), 2 * e, 2 * e, fill=False, ec=c,
                 lw=1.6 if crossed else 1.2, ls=(0, (4, 4)), alpha=.75 if crossed else .45, zorder=3))
    corner = x0 + e * sg
    dot(ax, corner, c, s=150 if crossed else 70, z=5)

corner = x0 + hit_e * sg
arrow(ax, x0, corner, CYAN, lw=3.0, scale=18, z=6)
label(ax, corner, "ε-box corner\ncrosses victim ✓", ADV, fs=11, dx=0.28, dy=0.2, ha="left")
star(ax, x0); label(ax, x0, r"$x_0$", STAR, fs=14, dx=-0.2, dy=-0.45, ha="right")

ax.text(0.2, 8.55, "no victim gradients — craft on the surrogate, grow ε until it transfers",
        color=INK, fontsize=12.5, ha="left", style="italic", alpha=.92)
tx, ty = 7.6, 5.3
ax.text(tx, ty,        "① PGD on the surrogate", color=CYAN, fontsize=12, weight="bold")
ax.text(tx, ty - 0.55, "    white-box · 0 victim queries", color=MUTE, fontsize=10.5)
ax.text(tx, ty - 1.35, "② grow the budget ε", color=AMBER, fontsize=12, weight="bold")
ax.text(tx, ty - 1.90, "    0.01 … 0.25  ($L_\\infty$ boxes)", color=MUTE, fontsize=10.5)
ax.text(tx, ty - 2.70, "③ stop at the smallest ε", color=ADV, fontsize=12, weight="bold")
ax.text(tx, ty - 3.25, "    that flips the victim", color=MUTE, fontsize=10.5)

finish(ax, (-1.4, 11.6), (-0.6, 9.0))
save(fig, "06_transfer")
