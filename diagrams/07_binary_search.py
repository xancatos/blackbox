"""Slide 7 - Finding the Initial Seed: Binary Search.
Bisect the line between the clean image x0 and a different-class image x_other to land on the boundary."""
import numpy as np
from style import (canvas, finish, save, arrow, dot, star, label, badges, legend_dots,
                   boundary_scene, INK, MUTE, BENIGN, ADV, VIOLET, STAR, MINT)

fig, ax = canvas()
fb = boundary_scene(ax, name_xy=(5.05, 0.7))

x0 = np.array([1.5, 4.2])
x_oth = np.array([9.6, 4.9])
is_adv = lambda p: p[0] > fb(p[1])

# the search line
ax.plot([x0[0], x_oth[0]], [x0[1], x_oth[1]], color=MUTE, lw=1.6, alpha=.6, zorder=2)

# real bisection -- lo starts at the CLEAN image, hi at the OTHER-CLASS image; we return hi
lo, hi, mids = x0.copy(), x_oth.copy(), []
for _ in range(7):
    m = 0.5 * (lo + hi); mids.append(m.copy())
    if is_adv(m): hi = m         # adversarial midpoint -> shrink the bracket from the adversarial side
    else: lo = m
seed = hi                        # the returned seed is the ADVERSARIAL endpoint (closest adversarial point)

for i, m in enumerate(mids):
    c = ADV if is_adv(m) else BENIGN
    dot(ax, m, c, s=120 - i * 9, z=5)
    label(ax, m, str(i + 1), INK, fs=9, dy=0.34)

star(ax, x0); label(ax, x0, r"$x_0$  (clean,  the  lo  end)", STAR, dy=-0.62)
dot(ax, x_oth, ADV, s=150); label(ax, x_oth, "other-class image\n(the  hi  end)", ADV, fs=10.5, dy=-0.78)
dot(ax, seed, MINT, s=185)
label(ax, seed, "seed = hi\nclosest adversarial point", MINT, fs=10.5, dx=0.15, dy=0.82)

ax.text(5.5, 6.6, "a 3072-D problem collapses to a 1-D line search",
        color=INK, fontsize=12.5, ha="center", style="italic", alpha=.92)
badges(ax, [(1, r"anchor  hi  at a different-class image (guaranteed adversarial)", ADV),
            (2, r"bisect:  midpoint adversarial?  pull  hi  in,  else pull  lo  in", VIOLET),
            (3, r"return  hi  →  the closest adversarial point on the boundary", MINT)], 5.5, 1.5)
legend_dots(ax, [(ADV, "midpoint adversarial (→ hi)"), (BENIGN, "midpoint still clean (→ lo)")], 1.4, 2.2)

finish(ax, (-0.2, 10.8), (-0.1, 8.2))
save(fig, "07_binary_search")
