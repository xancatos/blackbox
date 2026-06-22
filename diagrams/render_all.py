# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Render every attack-lineage slide diagram.

Each script writes three files to diagrams/:  <name>.png (dark background),
<name>.svg (vector), <name>_transparent.png (drop onto any dark slide).

Run from anywhere:
    uv run diagrams/render_all.py
"""
import sys
import os
import glob
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))   # diagrams/
ROOT = os.path.dirname(HERE)                         # repo root
sys.path.insert(0, HERE)                             # so each script's `import style` resolves
os.chdir(ROOT)                                       # so save() writes diagrams/<name>.png at the repo root

scripts = sorted(glob.glob(os.path.join(HERE, "[0-9]*.py")))
for s in scripts:
    print(f"==> {os.path.basename(s)}")
    runpy.run_path(s, run_name="__main__")
print(f"done: rendered {len(scripts)} diagrams into diagrams/")
