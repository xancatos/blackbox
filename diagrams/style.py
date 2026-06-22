"""Shared dark-theme styling for the attack-lineage slide diagrams.
Every diagram imports from here so the deck stays visually consistent.
Deliverables per figure: transparent PNG (primary) + SVG (vector) + dark-bg preview."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

# ---- palette ----
BG     = "#0b1020"   # dark preview background
INK    = "#e5e7eb"   # primary text / neutral points
MUTE   = "#94a3b8"   # secondary text / guides
BENIGN = "#34d399"   # green  (clean / -1 / benign region)
ADV    = "#f87171"   # red    (adversarial / +1 / adv region)
BOUND  = "#fb7185"   # decision boundary
AMBER  = "#fbbf24"   # gradient / normal / surrogate (the "free" signal)
CYAN   = "#38bdf8"   # taken step
VIOLET = "#c084fc"   # binary search / projection
STAR   = "#67e8f9"   # x0 (clean target)
MINT   = "#a7f3d0"   # improved / new point
GOLD   = "#fde68a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK})


def canvas(w=11.5, h=7.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor("none")
    return fig, ax


def finish(ax, xlim, ylim, equal=True):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if equal:
        ax.set_aspect("equal")


def arrow(ax, a, b, color, lw=2.8, scale=18, ls="-", z=5, alpha=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale,
                 color=color, lw=lw, linestyle=ls, zorder=z, alpha=alpha))


def dot(ax, p, color, s=170, z=6, ec=BG, lw=1.5):
    ax.scatter(*p, s=s, c=color, zorder=z, edgecolors=ec, linewidths=lw)


def star(ax, p, color=STAR, s=640, z=7):
    ax.scatter(*p, marker="*", s=s, c=color, zorder=z, edgecolors=BG, linewidths=1.5)


def label(ax, p, text, color=INK, fs=12.5, ha="center", va="center", weight="bold", dx=0, dy=0):
    ax.text(p[0] + dx, p[1] + dy, text, color=color, ha=ha, va=va, fontsize=fs, weight=weight)


def badges(ax, rows, x, y, dy=0.62):
    """rows = [(num, 'TEXT', color), ...] -> numbered caption block."""
    for i, (num, text, color) in enumerate(rows):
        yy = y - i * dy
        ax.scatter(x, yy, s=560, c=color, zorder=8, edgecolors=BG, linewidths=2)
        ax.text(x, yy, str(num), color=BG, ha="center", va="center", fontsize=14, weight="bold", zorder=9)
        ax.text(x + 0.42, yy, text, color=color, va="center", fontsize=12.5, weight="bold")


def legend_dots(ax, items, x, y, dy=0.42, fs=10.5):
    """items = [(color, 'label'), ...]"""
    for i, (color, text) in enumerate(items):
        yy = y - i * dy
        ax.scatter(x, yy, s=80, c=color)
        ax.text(x + 0.2, yy, text, color=MUTE, fontsize=fs, va="center")


def boundary_scene(ax, fb=None, benign_xy=(1.4, 7.5), adv_xy=(9.2, 7.5), name_xy=None):
    """Draw the shared backdrop: shaded benign/adversarial half-planes + the curved boundary.
    Returns the boundary function x = fb(y)."""
    if fb is None:
        fb = lambda y: 5.0 + 0.7 * np.sin(0.45 * y + 0.3)
    ys = np.linspace(0.2, 8.0, 400); xs = fb(ys)
    ax.fill_betweenx(ys, -2, xs, color=BENIGN, alpha=0.07)
    ax.fill_betweenx(ys, xs, 13, color=ADV, alpha=0.07)
    ax.plot(xs, ys, color=BOUND, lw=9, alpha=0.12, zorder=2)
    ax.plot(xs, ys, color=BOUND, lw=2.6, zorder=3)
    if benign_xy:
        ax.text(*benign_xy, "BENIGN\nregion", color=BENIGN, fontsize=12, ha="center", weight="bold", alpha=.9)
    if adv_xy:
        ax.text(*adv_xy, "ADVERSARIAL\nregion", color=ADV, fontsize=12, ha="center", weight="bold", alpha=.9)
    if name_xy:
        ax.text(*name_xy, "decision boundary", color=BOUND, fontsize=11, rotation=78, alpha=.9)
    return fb


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"diagrams/{name}.png", facecolor=BG, dpi=200, bbox_inches="tight")
    fig.savefig(f"diagrams/{name}.svg", transparent=True, bbox_inches="tight")
    fig.savefig(f"diagrams/{name}_transparent.png", transparent=True, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}: .png (dark) + .svg + _transparent.png")
