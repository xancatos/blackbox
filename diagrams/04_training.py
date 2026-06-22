"""Slide 4 - Foundations of Training: Gradients & Cross-Entropy.
Gradient DESCENT on the weights theta: theta <- theta - eta * grad_theta L."""
import numpy as np
from style import (canvas, finish, save, arrow, star, label, INK, MUTE, AMBER, CYAN, STAR, BENIGN)

fig, ax = canvas(11.0, 7.0)

# loss bowl f(a,b) and its contours
A, B = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-3.5, 3.5, 300))
F = 0.22 * (A**2 + 2.0 * B**2) + 0.4
ax.contourf(A, B, F, levels=14, cmap="cividis", alpha=0.55, zorder=1)
ax.contour(A, B, F, levels=10, colors=MUTE, linewidths=0.6, alpha=.5, zorder=2)

# gradient-descent path toward the minimum (0,0)
p = np.array([-4.2, 2.7]); pts = [p.copy()]
for _ in range(7):
    g = np.array([0.44 * p[0], 0.88 * p[1]])     # grad of F
    p = p - 0.55 * g; pts.append(p.copy())
pts = np.array(pts)
for a, b in zip(pts[:-1], pts[1:]):
    arrow(ax, a, b, AMBER, lw=2.6, scale=15, z=5)
ax.scatter(pts[:, 0], pts[:, 1], s=45, c=AMBER, zorder=6)

star(ax, (0, 0), STAR, s=520); label(ax, (0, 0), "min loss", STAR, fs=12, dy=-0.5)
label(ax, (pts[0, 0], pts[0, 1]), "start  θ", INK, fs=12, dx=0.1, dy=0.45)

label(ax, (0, 3.05), "the loss landscape over the weights  θ  (58,370-D, shown as 2-D)", INK, fs=12.5)
ax.text(-4.9, -3.05, r"$\theta \;\leftarrow\; \theta \;-\; \eta\,\nabla_\theta L$",
        color=AMBER, fontsize=17, ha="left")
ax.text(0.7, -3.0, "follow  −gradient  downhill → lower error.  (weights move, image is fixed)",
        color=MUTE, fontsize=11, ha="left")

finish(ax, (-5, 5), (-3.5, 3.5), equal=False)
save(fig, "04_training")
