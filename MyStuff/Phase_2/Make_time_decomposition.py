#!/usr/bin/env python3
"""
Regenerate the Scout+Refine time-decomposition figure for the thesis.

Output: Time_Decomposition_Scout_Refine.png

All values are transcribed directly from the thesis tables:
  - Baseline           <- tab:time_savings_summary, "Baseline" column
  - Scout (Phase 1)    <- tab:time_decomposition,  "Phase 1 (Scout)" column
  - Refine only        <- tab:time_decomposition,  "Refine only" column
The annotated percentage is the REFINE-ONLY reduction against the Baseline,
i.e. the wall-clock-to-final-result, NOT the total-CPU saving.

Run:  python make_time_decomposition.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTFILE = "Time_Decomposition_Scout_Refine.png"

# ---------------------------------------------------------------- CONFIG ----
# d_L (Mpc), approx network SNR, Baseline (h), Scout/Phase 1 (h), Refine only (h)
DATA = [
    (10,   790, 8.49, 2.61, 4.24),
    (20,   394, 7.92, 1.96, 3.31),
    (40,   197, 7.78, 1.56, 2.79),
    (80,    98, 6.36, 1.66, 2.48),
    (150,   52, 5.22, 1.17, 2.09),
    (300,   26, 3.77, 0.94, 1.71),
    (600,   13, 2.79, 0.78, 1.46),
    (1300,   6, 2.20, 0.59, 1.05),
    (2500,   3, 1.45, 0.46, 0.63),
    (5000,   2, 1.20, 0.01, 0.48),
]

C_BASELINE = "#9a9a9a"   # neutral grey: the reference arm
C_REFINE = "#1f6cb0"     # blue: the cost that actually gates time-to-result
C_SCOUT = "#a8cbe8"      # light blue + hatch: cost hidden by calibration overlap
EDGE = "#1a1a1a"

dl = np.array([r[0] for r in DATA])
snr = np.array([r[1] for r in DATA])
base = np.array([r[2] for r in DATA], dtype=float)
scout = np.array([r[3] for r in DATA], dtype=float)
refine = np.array([r[4] for r in DATA], dtype=float)

pct = (1.0 - refine / base) * 100.0   # refine-only reduction vs baseline

x = np.arange(len(DATA))
w = 0.38

fig, ax = plt.subplots(figsize=(12.5, 5.6))

ax.bar(x - w / 2, base, w, color=C_BASELINE, edgecolor=EDGE, linewidth=0.7,
       label="Baseline (blind final run)", zorder=3)
ax.bar(x + w / 2, refine, w, color=C_REFINE, edgecolor=EDGE, linewidth=0.7,
       label="Refine / Phase 2 (time-to-final-result)", zorder=3)
ax.bar(x + w / 2, scout, w, bottom=refine, color=C_SCOUT, edgecolor=EDGE,
       linewidth=0.7, hatch="///",
       label="Scout / Phase 1 (overlaps calibration — free)", zorder=3)

for xi, tot, p in zip(x, refine + scout, pct):
    ax.text(xi + w / 2, tot + 0.13, f"−{p:.0f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=C_REFINE, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels([f"{d}\n($\\rho \\approx {s}$)" for d, s in zip(dl, snr)],
                   fontsize=10)
ax.set_xlabel("Luminosity distance $d_L$ [Mpc]   (approx. network SNR)",
              fontsize=11.5, labelpad=8)
ax.set_ylabel("Wall-clock compute time [hours]", fontsize=11.5)
ax.set_title("Where the time is saved: Baseline vs. Scout + Refine",
             fontsize=13.5, pad=14)

ax.set_ylim(0, max(base.max(), (refine + scout).max()) * 1.14)
ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#555555")

ax.legend(loc="upper right", fontsize=10.5, framealpha=1.0, edgecolor="#bbbbbb")

fig.tight_layout()
fig.savefig(OUTFILE, dpi=200)
print(f"wrote {OUTFILE}")

print("\n d_L   Baseline  Scout  Refine   Total   Refine-only vs B   TotalCPU vs B")
for d, b, s, r in zip(dl, base, scout, refine):
    print(f"{d:>5}  {b:7.2f}  {s:5.2f}  {r:6.2f}  {s + r:6.2f}"
          f"        {(1 - r / b) * 100:5.1f}%          {(1 - (s + r) / b) * 100:5.1f}%")