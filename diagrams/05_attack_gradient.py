"""Slide 5 - Foundations of Attack: Weaponizing the Gradient (FGSM).
Now weights are FIXED and we ASCEND the loss in INPUT space: x_adv = x + eps*sign(grad_x L)."""
import numpy as np
from matplotlib.patches import Rectangle
from style import (canvas, finish, save, arrow, dot, star, label, badges,
                   boundary_scene, INK, MUTE, BENIGN, ADV, AMBER, CYAN, STAR)

fig, ax = canvas(11.0, 7.2)
fb = boundary_scene(ax, fb=lambda y: 5.0 + 0.4 * np.sin(0.5 * y + 0.5),
                    benign_xy=(1.2, 6.9), adv_xy=(8.6, 6.9), name_xy=(5.15, 0.7))
is_adv = lambda p: p[0] > fb(p[1])

x0 = np.array([3.5, 4.0])
eps = 1.8
# epsilon L-infinity ball (a box) around x0
ax.add_patch(Rectangle((x0[0] - eps, x0[1] - eps), 2 * eps, 2 * eps,
             fill=False, ec=MUTE, lw=1.4, ls=(0, (5, 4)), alpha=.6, zorder=3))
label(ax, (x0[0] - eps + 0.1, x0[1] + eps - 0.3), r"$\epsilon$  ball ($L_\infty$)", MUTE, fs=11, ha="left")

# raw loss gradient (smooth) vs its SIGN (snaps to a box corner)
g = np.array([0.86, 0.5]); g = g / np.linalg.norm(g)
arrow(ax, x0, x0 + 1.55 * g, AMBER, lw=2.6, scale=15, z=5)
label(ax, x0 + 1.55 * g, r"$\nabla_x L$", AMBER, fs=13, dx=-0.35, dy=0.32, ha="right")

x_adv = x0 + eps * np.sign(g)                    # FGSM lands at the box corner
assert is_adv(x_adv), "x_adv must be across the boundary"
arrow(ax, x0, x_adv, CYAN, lw=3.4, scale=20, z=6)
dot(ax, x_adv, ADV, s=190, z=7)
label(ax, x_adv, "$x_{adv}$\n(misclassified)", ADV, fs=11, dx=0.28, dy=0.0, ha="left")

star(ax, x0); label(ax, x0, r"$x_0$  (clean, correct)", STAR, dy=-0.55)
ax.text(5.3, 7.85, "weights frozen — we move the IMAGE to maximize loss",
        color=INK, fontsize=12, ha="center", style="italic", alpha=.92)
ax.text(0.6, 1.95, r"$x_{adv} = x_0 + \epsilon\,\mathrm{sign}(\nabla_x L)$", color=CYAN, fontsize=15, ha="left")
badges(ax, [(1, r"ascend the loss in INPUT space (mirror of training)", AMBER),
            (2, r"sign(·) snaps the step to an  $\epsilon$-box corner  (one big move)", CYAN),
            (3, r"land across the boundary → confident misclassification", ADV)], 0.6, 1.3)

finish(ax, (-0.2, 10.6), (-0.1, 8.2))
save(fig, "05_attack_gradient")
