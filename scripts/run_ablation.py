"""
scripts/run_ablation.py
Print ablation study results table.
Displays REPORTED reference values from the research paper.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 55)
print("ABLATION STUDY — LATERAL DEVIATION (REPORTED VALUES)")
print("=" * 55)
print("NOTE: These are reported values from the research paper.")
print("      They are NOT reproduced by this local prototype.\n")

rows = [
    ("A. RL-Only (Baseline)",  0.3924, "–"),
    ("B. Ensemble RL",         0.3825, "2.5%"),
    ("C. Ensemble RL + SRL",   0.2873, "26.8%"),
]
print(f"{'Configuration':<28} {'Lat. Dev (m)':>14} {'Improvement':>12}")
print("-" * 55)
for name, val, imp in rows:
    print(f"{name:<28} {val:>14.4f} {imp:>12}")
print("-" * 55)
print("Source: Research paper (reported, not reproduced locally)\n")
