#!/usr/bin/env python3
"""
make_overlap_corners.py
=======================
Regenerate the three posterior-overlap corner plots (40 / 150 / 300 Mpc) as a
*single consistent set*: same parameters, same colours, same legend labels,
same truth markers, same figure size and DPI.

Why this script exists
----------------------
The three corner plots currently in the thesis were produced by different
scripts, so they disagree on which parameters are shown, the colours
(orange vs. red), and the legend text ("Baseline (Blind)" vs. "Baseline").
This script produces all three from one code path, so they match.

The posterior data lives on a different machine, so this file only contains
the *code*. Edit the CONFIG block below to point at your bilby result files
on the data PC, then run:

    python make_overlap_corners.py

It writes:
    Posterior_Overlap_40Mpc.png
    Posterior_Overlap_150Mpc.png
    Posterior_Overlap_300Mpc.png

Dependencies (already present in your bilby environment):
    pip install bilby corner matplotlib numpy pandas
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# CONFIG  --  edit these paths/values to match the data PC
# ----------------------------------------------------------------------------

# Root directory that contains the result files on the data PC.
# Leave as "" and use absolute paths below, or set it and use relative names.
RESULT_ROOT = ""

# For each distance, give the Baseline and Phase-2 (Refine) result files.
# bilby writes these as <label>_result.json or <label>_result.hdf5.
RUNS = {
    40:  {"baseline": "baseline_40Mpc_result.json",
          "phase2":   "refine_40Mpc_result.json"},
    150: {"baseline": "baseline_150Mpc_result.json",
          "phase2":   "refine_150Mpc_result.json"},
    300: {"baseline": "baseline_300Mpc_result.json",
          "phase2":   "refine_300Mpc_result.json"},
}

# Parameters to show (bilby posterior column names) and their axis labels.
# Keep this identical across all three plots -> that is the whole point.
PARAMS = [
    ("chirp_mass",         r"$\mathcal{M}\ [M_\odot]$"),
    ("mass_ratio",         r"$q$"),
    ("luminosity_distance",r"$d_L$ [Mpc]"),
    ("theta_jn",           r"$\theta_{JN}$ [rad]"),
    ("phase",              r"$\phi$ [rad]"),
    ("geocent_time",       r"$t_c - t_0$ [s]"),
]

# Injected truth values (from the systematic SNR study). d_L is filled in
# per-distance automatically. If a result file stores injection_parameters,
# those are used instead and these act only as a fallback.
TRUTHS = {
    "chirp_mass": 30.0,
    "mass_ratio": 0.9,
    "theta_jn":   0.8,
    "phase":      1.0,
    "geocent_time": 1126259462.0,   # reference t0; plotted as t_c - t0
}

# Presentation: fixed colours / labels / style for every plot.
C_BASELINE = "#1f77b4"   # blue
C_PHASE2   = "#d62728"   # red
LABEL_BASELINE = "Baseline"
LABEL_PHASE2   = "Phase 2 (Scout + Refine)"
FIGSIZE_PER_PARAM = 2.1   # inches per parameter (controls overall size)
DPI = 200

# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------

def load_posterior(path):
    """Return (posterior_DataFrame, injection_dict) from a bilby result file.

    Tries bilby first (handles .json / .hdf5 / .h5). Falls back to a plain
    CSV/whitespace table of posterior samples if bilby cannot read it.
    """
    full = os.path.join(RESULT_ROOT, path) if RESULT_ROOT else path
    if not os.path.exists(full):
        raise FileNotFoundError(f"Result file not found: {full}")
    try:
        import bilby
        res = bilby.result.read_in_result(filename=full)
        inj = res.injection_parameters or {}
        return res.posterior, inj
    except Exception as e:
        print(f"  [warn] bilby could not read {full} ({e}); trying flat table")
        import pandas as pd
        try:
            df = pd.read_csv(full)
        except Exception:
            df = pd.read_csv(full, delim_whitespace=True)
        return df, {}


def get_columns(df, dL):
    """Extract the PARAMS columns as an (N, n_params) array, with geocent_time
    shifted to t_c - t0 for readability. Returns (array, truth_list)."""
    import numpy as np
    cols, truths = [], []
    for name, _label in PARAMS:
        if name not in df.columns:
            raise KeyError(
                f"Column '{name}' not in posterior (have: {list(df.columns)}). "
                f"Adjust PARAMS to match your bilby parameter names.")
        vals = np.asarray(df[name], dtype=float)
        if name == "geocent_time":
            vals = vals - TRUTHS["geocent_time"]
        cols.append(vals)
    arr = np.column_stack(cols)
    return arr


def truth_vector(inj, dL):
    """Build the truth list in PARAMS order, preferring injected values from
    the result file, then the CONFIG TRUTHS, with d_L set to this distance."""
    t = []
    for name, _ in PARAMS:
        if name == "luminosity_distance":
            t.append(float(dL))
        elif name == "geocent_time":
            v = inj.get(name, TRUTHS["geocent_time"])
            t.append(float(v) - TRUTHS["geocent_time"])
        else:
            t.append(float(inj.get(name, TRUTHS.get(name, np.nan))))
    return t

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def report_widths(dL, base_df, p2_df):
    """Print per-parameter posterior widths for Baseline vs Phase 2.

    Directly answers the "which posterior is wider?" / double-use-of-data
    question (thesis Fig. 12): the 1-sigma width is the 84th-minus-16th
    percentile range. ratio = width_P2 / width_Baseline. A ratio < 1 means
    Phase 2 is *narrower* than Baseline (possible over-confidence); ratio ~ 1
    means the widths match.
    """
    import numpy as np
    print(f"  posterior widths ({dL} Mpc)  [1-sigma = P84-P16]")
    print(f"    {'param':<20}{'Baseline':>12}{'Phase2':>12}{'ratio P2/B':>12}")
    for name, _ in PARAMS:
        if name not in base_df.columns or name not in p2_df.columns:
            continue
        b = np.asarray(base_df[name], float)
        p = np.asarray(p2_df[name], float)
        wb = np.subtract(*np.percentile(b, [84, 16]))
        wp = np.subtract(*np.percentile(p, [84, 16]))
        ratio = wp / wb if wb else float("nan")
        flag = "  <-- P2 narrower" if ratio < 0.9 else ("  <-- P2 wider" if ratio > 1.1 else "")
        print(f"    {name:<20}{wb:>12.4g}{wp:>12.4g}{ratio:>12.3f}{flag}")


def make_one(dL, files):
    import corner
    labels = [lab for _, lab in PARAMS]

    base_df, base_inj = load_posterior(files["baseline"])
    p2_df,   p2_inj   = load_posterior(files["phase2"])

    report_widths(dL, base_df, p2_df)

    base = get_columns(base_df, dL)
    p2   = get_columns(p2_df, dL)
    truths = truth_vector(base_inj or p2_inj, dL)

    ndim = len(PARAMS)
    size = FIGSIZE_PER_PARAM * ndim

    ck = dict(bins=40, smooth=0.9, plot_datapoints=False, fill_contours=True,
              levels=(0.5, 0.9), truths=truths, truth_color="black",
              labels=labels, label_kwargs={"fontsize": 12},
              hist_kwargs={"density": True})

    fig = corner.corner(base, color=C_BASELINE,
                        contour_kwargs={"colors": C_BASELINE},
                        hist_kwargs={"density": True, "color": C_BASELINE},
                        **{k: v for k, v in ck.items() if k != "hist_kwargs"})
    corner.corner(p2, color=C_PHASE2, fig=fig,
                  contour_kwargs={"colors": C_PHASE2},
                  hist_kwargs={"density": True, "color": C_PHASE2},
                  **{k: v for k, v in ck.items() if k != "hist_kwargs"})

    # consistent legend, top-right
    handles = [Line2D([0], [0], color=C_BASELINE, lw=3, label=LABEL_BASELINE),
               Line2D([0], [0], color=C_PHASE2,   lw=3, label=LABEL_PHASE2)]
    fig.legend(handles=handles, loc="upper right", fontsize=13,
               frameon=True, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(fr"Posterior Overlap $-$ {dL} Mpc", fontsize=15, y=1.02)

    fig.set_size_inches(size, size)
    out = f"Posterior_Overlap_{dL}Mpc.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    global RESULT_ROOT
    ap = argparse.ArgumentParser(description="Regenerate consistent overlap corner plots.")
    ap.add_argument("--root", default=RESULT_ROOT,
                    help="directory containing the result files (overrides RESULT_ROOT)")
    args = ap.parse_args()
    if args.root:
        RESULT_ROOT = args.root

    for dL, files in RUNS.items():
        print(f"[{dL} Mpc]")
        make_one(dL, files)
    print("done -- three consistent corner plots written.")


if __name__ == "__main__":
    main()