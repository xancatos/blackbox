"""Slide 8 - Evolutionary Overview: The Lineage of ML Attacks.
Seven tiers, FGSM -> SQBA, as a snake flow: gradient access -> query efficiency -> hybrid."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from style import canvas, finish, save, INK, MUTE, BENIGN, ADV, BOUND, AMBER, CYAN, MINT, STAR, BG

fig, ax = canvas(13.0, 6.6)

W, H = 2.7, 1.65
def box(cx, cy, title, body, col):
    ax.add_patch(FancyBboxPatch((cx - W / 2, cy - H / 2), W, H, boxstyle="round,pad=0.06,rounding_size=0.12",
                 facecolor=col, alpha=0.14, edgecolor=col, lw=2.0, zorder=4))
    ax.text(cx, cy + H / 2 - 0.30, title, color=col, fontsize=11.5, ha="center", weight="bold", zorder=5)
    ax.text(cx, cy - 0.18, body, color=INK, fontsize=9.6, ha="center", va="center", zorder=5)

HL = BOUND
tiers = [
    (0.0, 4.7, "TIER 1 · WHITE-BOX", "FGSM '14 · PGD '17\nfull surrogate gradients", CYAN),
    (3.3, 4.7, "TIER 2 · TRANSFER", "Papernot '16\nsurrogate · ~0 queries", AMBER),
    (6.6, 4.7, "TIER 3 · LOCAL SEARCH", "Boundary Attack '18\nrandom walk on the wall", HL),
    (9.9, 4.7, "TIER 4 · PATH OPT", "OPT '19 · Sign-OPT '20\ndirectional · sign probes", HL),
    (9.9, 1.7, "TIER 5 · NORMAL EST.", "HopSkipJump '20\nMonte-Carlo normal", HL),
    (6.6, 1.7, "TIER 6 · SUBSPACE", "Triangle '22\nlaw of sines · DCT", HL),
    (3.3, 1.7, "TIER 7 · HYBRID FUSION", "Biased '19 → SQBA '24\nsurrogate + query fallback", MINT),
]
for cx, cy, t, b, c in tiers:
    box(cx, cy, t, b, c)

def link(a, b, col):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, color=col, lw=2.4, zorder=3))
link((0.0 + W/2, 4.7), (3.3 - W/2, 4.7), MUTE)
link((3.3 + W/2, 4.7), (6.6 - W/2, 4.7), MUTE)
link((6.6 + W/2, 4.7), (9.9 - W/2, 4.7), MUTE)
link((9.9, 4.7 - H/2), (9.9, 1.7 + H/2), MUTE)              # turn down
link((9.9 - W/2, 1.7), (6.6 + W/2, 1.7), MUTE)
link((6.6 - W/2, 1.7), (3.3 + W/2, 1.7), MUTE)

# the destination
ax.scatter(3.3, 1.7 + H/2 + 0.0, s=0)  # noop keep layout
ax.text(6.5, 6.05, "The Lineage of ML Attacks — each tier relaxes one assumption of the tier above",
        color=INK, fontsize=13.5, ha="center")
ax.text(6.5, 0.45, "GRADIENT ACCESS   →   QUERY EFFICIENCY   →   HYBRID ADAPTATION",
        color=STAR, fontsize=12, ha="center", weight="bold", alpha=.95)

finish(ax, (-1.6, 11.7), (-0.1, 6.4), equal=False)
save(fig, "08_lineage")
