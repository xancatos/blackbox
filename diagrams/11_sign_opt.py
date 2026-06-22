"""Slide 11 - Sign-OPT: The 'Yes or No' Shortcut.
At the current distance g(theta), one query asks 'still adversarial if I nudge theta?' -> sign +-1.
The boundary is oblique to theta so the two nudges straddle it (one +1, one -1)."""
import numpy as np
from matplotlib.patches import Arc
from style import (canvas, finish, save, arrow, dot, star, label, badges, legend_dots,
                   boundary_scene, INK, MUTE, BENIGN, ADV, AMBER, STAR)

fig, ax = canvas()
fb = boundary_scene(ax, fb=lambda y: 7.0 - 0.9 * (y - 1.0),
                    benign_xy=(1.7, 1.1), adv_xy=(8.6, 5.2))
ax.text(3.45, 5.6, "decision boundary", color="#fb7185", fontsize=11, rotation=-48, alpha=.9)
is_adv = lambda p: p[0] > fb(p[1])

x0 = np.array([1.8, 5.0])
pb = np.array([fb(3.0), 3.0])                       # an OBLIQUE hit, far from the nearest point
v = pb - x0; g = np.linalg.norm(v); th = np.arctan2(v[1], v[0])

# arc of radius g around x0 (the fixed test distance)
ax.add_patch(Arc(x0, 2 * g, 2 * g, angle=0,
                 theta1=np.degrees(th) - 26, theta2=np.degrees(th) + 26,
                 color=MUTE, lw=1.2, ls=(0, (4, 4)), alpha=.5, zorder=3))
arrow(ax, x0, pb, AMBER, lw=3.0, scale=18, z=5)
label(ax, x0 + 0.55 * v, r"$\theta$,  dist $g(\theta)$", AMBER, fs=12, dx=0.1, dy=0.45)

signs = {}
for dth, key in [(+0.34, "+"), (-0.34, "-")]:
    un = np.array([np.cos(th + dth), np.sin(th + dth)])
    tp = x0 + g * un; adv = is_adv(tp); signs[key] = adv
    col = ADV if adv else BENIGN
    arrow(ax, x0, tp, col, lw=2.1, scale=14, z=4, alpha=.92)
    dot(ax, tp, col, s=160, z=6)
    label(ax, tp, ("+1" if adv else "−1"), col, fs=15, dx=0.32, dy=0.0, ha="left")
assert signs["+"] != signs["-"], f"need opposite signs, got {signs}"

dot(ax, pb, INK, s=120); label(ax, pb, "boundary @ θ", INK, fs=10.5, dx=0.3, dy=-0.25, ha="left")
star(ax, x0); label(ax, x0, r"$x_0$", STAR, fs=14, dx=-0.2, dy=0.0, ha="right")

ax.text(0.2, 7.8, "one query replaces a full binary search per probe",
        color=INK, fontsize=12, ha="left", style="italic", alpha=.92)
badges(ax, [(1, r"step the fixed distance  $g(θ)$  along a nudged direction  θ±u", AMBER),
            (2, r"still adversarial?  Yes → boundary is closer that way  (sign +1)", ADV),
            (3, r"No → it is farther  (sign −1).    Average the signs = gradient of  $g$", BENIGN)], 4.6, 1.35)
legend_dots(ax, [(ADV, "test point adversarial (+1)"), (BENIGN, "test point clean (−1)")], 7.6, 7.5)

finish(ax, (-0.4, 10.8), (-0.2, 8.2))
save(fig, "11_sign_opt")
