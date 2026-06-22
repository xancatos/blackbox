"""Slide 3 - CIFAR-10 Binary Classification: VictimNet Architecture.
Input 32x32x3 -> 3 conv+pool blocks (32,64,64) -> flatten 1024 -> FC -> 2 logits."""
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from style import (canvas, finish, save, label, INK, MUTE, BENIGN, ADV, AMBER, CYAN, STAR, BG, BOUND)

fig, ax = canvas(12.5, 6.6)

# conv volumes: (x_center, spatial, channels, label)
vols = [(0.9, 32, 3, "32×32×3"), (3.4, 16, 32, "16×16×32"),
        (5.7, 8, 64, "8×8×64"), (7.7, 4, 64, "4×4×64")]
ymid = 3.6
def h_of(s): return 0.9 + s / 32 * 3.2
COL = CYAN

for xc, s, c, txt in vols:
    h = h_of(s); w = 0.5
    # depth illusion: a few offset slabs
    for k in range(2, -1, -1):
        off = k * 0.14
        ax.add_patch(Rectangle((xc - w / 2 + off, ymid - h / 2 + off), w, h,
                     facecolor=COL, alpha=0.18 + 0.16 * (2 - k), edgecolor=COL, lw=1.4, zorder=4 + (2 - k)))
    label(ax, (xc, ymid - h / 2 - 0.42), txt, INK, fs=11)
    label(ax, (xc, ymid + h / 2 + 0.34), ("input" if c == 3 else "%d ch" % c), MUTE, fs=10)

# arrows + op captions between conv volumes
ops = ["Conv 3×3\nReLU · ↓2", "Conv 3×3\nReLU · ↓2", "Conv 3×3\nReLU · ↓2"]
for (x1, *_), (x2, *_), op in zip(vols, vols[1:], ops):
    ax.add_patch(FancyArrowPatch((x1 + 0.55, ymid), (x2 - 0.55, ymid), arrowstyle="-|>",
                 mutation_scale=16, color=MUTE, lw=1.8, zorder=3))
    label(ax, ((x1 + x2) / 2, ymid + 0.62), op, AMBER, fs=9.5)

# flatten -> 1024 vector
xf = 9.3
ax.add_patch(Rectangle((xf, ymid - 2.1), 0.32, 4.2, facecolor=AMBER, alpha=.28, edgecolor=AMBER, lw=1.4, zorder=4))
ax.add_patch(FancyArrowPatch((8.2, ymid), (xf - 0.05, ymid), arrowstyle="-|>", mutation_scale=16, color=MUTE, lw=1.8, zorder=3))
label(ax, (xf + 0.16, ymid - 2.45), "flatten\n1024", AMBER, fs=10)

# FC -> 2 logits
ax.add_patch(FancyArrowPatch((xf + 0.32, ymid), (10.6, ymid), arrowstyle="-|>", mutation_scale=16, color=MUTE, lw=1.8, zorder=3))
label(ax, (10.0, ymid + 0.5), "Linear\n1024→2", AMBER, fs=9.5)
for i, (name, col, val) in enumerate([("Airplane", BENIGN, 0.86), ("Automobile", ADV, 0.32)]):
    yy = ymid + 0.55 - i * 1.1
    ax.add_patch(Rectangle((10.7, yy - 0.22), 1.5 * val, 0.44, facecolor=col, alpha=.85, edgecolor=col, zorder=5))
    label(ax, (10.7, yy + 0.5), name, col, fs=10.5, ha="left")

label(ax, (6.2, 6.05), "VictimNet — a small CNN  (58,370 trainable weights)", INK, fs=13.5)
label(ax, (6.2, 0.5), "spatial size halves at each pool · channel depth grows · 2-class output (airplane vs automobile)",
      MUTE, fs=11)

finish(ax, (-0.2, 12.8), (0.0, 6.4), equal=False)
save(fig, "03_architecture")
