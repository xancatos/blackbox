# Slide diagrams (dark theme)

Dark-theme figures for the attack-lineage deck (`lineage.pdf`, slides 3–15). Each diagram exports
three files:

- `<name>.png` — baked dark background (quick preview / GitHub)
- `<name>.svg` — vector, scales crisply to any slide size
- `<name>_transparent.png` — transparent background, drops onto any dark slide

Conceptual schematics (not run on real data), but the geometry is checked against `../attacks.py`
and several scripts `assert` their key facts (e.g. the accepted point lands on the boundary, the
Sign-OPT probes get opposite signs).

## Render with uv

```bash
uv run diagrams/render_all.py                                   # render all 13
uv run --with numpy,matplotlib python3 diagrams/12_hopskipjump.py   # render just one
```

`render_all.py` carries a PEP-723 header, so uv provisions numpy + matplotlib automatically.
Shared styling (palette, boundary scene, arrow/badge helpers) lives in `style.py` — edit it once to
restyle every figure.

## Map to slides

| script | slide |
|---|---|
| `03_architecture` | VictimNet Architecture |
| `04_training` | Gradients & Cross-Entropy (descent on weights) |
| `05_attack_gradient` | Weaponizing the Gradient (FGSM) |
| `06_transfer` | Transfer: the Black-Box Sweep |
| `07_binary_search` | Finding the Initial Seed |
| `08_lineage` | Evolutionary Overview (the 7 tiers) |
| `09_boundary` | Boundary Attack: Walking Along the Wall |
| `10_opt` | OPT: the Angle-Based Distance Measurer |
| `11_sign_opt` | Sign-OPT: the Yes/No Shortcut |
| `12_hopskipjump` | HopSkipJump: the Three-Step Dance |
| `13_triangle` | Triangle Attack: Geometric Query Efficiency |
| `14_biased_boundary` | Biased Boundary: the Hiker with a Guide Dog |
| `15_sqba` | SQBA: the Best of Both Worlds |
