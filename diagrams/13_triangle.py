"""Slide 13 - Triangle Attack: Geometric Query Efficiency.
Law of sines on the triangle (x0, x_t, x_{t+1}) guarantees delta_{t+1} < delta_t -- no gradients."""
import numpy as np
from matplotlib.patches import Arc
from style import (canvas, finish, save, arrow, dot, star, label, badges,
                   INK, MUTE, BENIGN, ADV, BOUND, AMBER, CYAN, STAR, MINT)

fig, ax = canvas()

x0 = np.array([2.0, 3.0])
x_t = np.array([7.0, 5.0])
x_t1 = np.array([6.3, 3.4])
d_t = np.linalg.norm(x_t - x0); d_t1 = np.linalg.norm(x_t1 - x0)

# faint boundary line through x_t and x_t1, shaded half-planes
m = (x_t[1] - x_t1[1]) / (x_t[0] - x_t1[0])
fbx = lambda y: x_t[0] + (y - x_t[1]) / m
ys = np.linspace(-0.2, 8, 200); xs = fbx(ys)
ax.fill_betweenx(ys, -2, xs, color=BENIGN, alpha=0.06)
ax.fill_betweenx(ys, xs, 13, color=ADV, alpha=0.06)
ax.plot(xs, ys, color=BOUND, lw=2.2, alpha=.8, zorder=2)
ax.text(8.5, 6.6, "ADVERSARIAL", color=ADV, fontsize=11, weight="bold", alpha=.85)
ax.text(0.2, 6.6, "BENIGN", color=BENIGN, fontsize=11, weight="bold", alpha=.85)
ax.text(2.0, 7.5, "a 2-D plane is sampled in low-frequency DCT space",
        color=INK, fontsize=12, ha="left", style="italic", alpha=.9)

# triangle sides
ax.plot([x0[0], x_t[0]], [x0[1], x_t[1]], color=ADV, lw=3.0, zorder=4)
ax.plot([x0[0], x_t1[0]], [x0[1], x_t1[1]], color=MINT, lw=3.0, zorder=4)
ax.plot([x_t[0], x_t1[0]], [x_t[1], x_t1[1]], color=CYAN, lw=2.4, ls=(0, (5, 3)), zorder=4)

label(ax, (x0 + x_t) / 2, r"$\delta_t$", ADV, fs=15, dx=-0.05, dy=0.45)
label(ax, (x0 + x_t1) / 2, r"$\delta_{t+1}\;<\;\delta_t$", MINT, fs=14, dx=0.1, dy=-0.5)
label(ax, (x_t + x_t1) / 2, "rotate by ±β", CYAN, fs=11, dx=0.55, dy=0.0, ha="left")

# angle arcs (beta at x0, alpha at x_t1)
b1, b2 = np.degrees(np.arctan2(*(x_t1 - x0)[::-1])), np.degrees(np.arctan2(*(x_t - x0)[::-1]))
ax.add_patch(Arc(x0, 1.7, 1.7, theta1=b1, theta2=b2, color=AMBER, lw=2.2, zorder=5))
label(ax, x0 + 1.15 * (np.array([np.cos(np.radians((b1+b2)/2)), np.sin(np.radians((b1+b2)/2))])),
      r"$\beta$", AMBER, fs=13)
a1 = np.degrees(np.arctan2(*(x0 - x_t1)[::-1])); a2 = np.degrees(np.arctan2(*(x_t - x_t1)[::-1]))
ax.add_patch(Arc(x_t1, 1.5, 1.5, theta1=a2, theta2=a1, color=AMBER, lw=2.2, zorder=5))
label(ax, x_t1 + 1.0 * np.array([np.cos(np.radians((a1+a2)/2)), np.sin(np.radians((a1+a2)/2))]),
      r"$\alpha$", AMBER, fs=13)

star(ax, x0); label(ax, x0, r"$x_0$  (benign)", STAR, dx=-0.2, dy=-0.5, ha="right")
dot(ax, x_t, ADV, s=160); label(ax, x_t, r"$x_t^{adv}$", ADV, fs=13, dx=0.28, dy=0.18, ha="left")
dot(ax, x_t1, MINT, s=170); label(ax, x_t1, r"$x_{t+1}^{adv}$", MINT, fs=13, dx=0.28, dy=-0.05, ha="left")

ax.text(5.6, 2.05, r"$\delta_{t+1} = \delta_t \cdot \dfrac{\sin(\alpha+\beta)}{\sin(\alpha)}$",
        color=INK, fontsize=15, ha="left")
badges(ax, [(1, r"build a triangle from two adversarial points near the boundary", CYAN),
            (2, r"law of sines  ⇒  the new side  $\delta_{t+1}$  is provably shorter", MINT),
            (3, r"only 2 queries per step — no gradient estimation at all", AMBER)], 2.0, 1.35)

finish(ax, (-0.4, 10.8), (-0.2, 8.2))
save(fig, "13_triangle")
