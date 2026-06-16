# Plan: Generate Posterior Overlap Corner Plots for 150 Mpc and 300 Mpc

## Goal
Generate `Posterior_Overlap_150Mpc.png` and `Posterior_Overlap_300Mpc.png` — overlay corner plots
showing Baseline (blue) vs Phase 2 Refine (red) posterior distributions, matching the style of the
existing `Posterior_Overlap_40Mpc.png` already used in the thesis.

## Output location
```
MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/Posterior_Overlap_150Mpc.png
MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/Posterior_Overlap_300Mpc.png
```

## Input files (use iteration 1 for each distance)
```
MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/dist_150_iter_1_baseline_result.json
MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/dist_150_iter_1_refine_result.json
MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/dist_300_iter_1_baseline_result.json
MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/dist_300_iter_1_refine_result.json
```

## Parameters to show in corner plot
The 6 free parameters used in all runs:
- `chirp_mass`
- `mass_ratio`
- `luminosity_distance`
- `theta_jn`
- `phase`
- `geocent_time`

## Injected values (for truth lines)
- chirp_mass = 30.0
- mass_ratio = 0.9
- luminosity_distance = 150.0 (or 300.0 depending on plot)
- theta_jn = 0.8
- phase = 1.0
- geocent_time = 1126259462.0

## Implementation

Write a single Python script `generate_overlay_plots.py` in this directory.

```python
import bilby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = "MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy"
PARAMS = ["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn", "phase", "geocent_time"]
LABELS = [r"$\mathcal{M}\ [M_\odot]$", r"$q$", r"$d_L\ \mathrm{[Mpc]}$",
          r"$\theta_{JN}\ \mathrm{[rad]}$", r"$\phi\ \mathrm{[rad]}$", r"$t_c\ \mathrm{[s]}$"]

INJECTION = {
    "chirp_mass": 30.0,
    "mass_ratio": 0.9,
    "theta_jn": 0.8,
    "phase": 1.0,
    "geocent_time": 1126259462.0,
}

configs = [
    {"dist": 150, "inj_dist": 150.0},
    {"dist": 300, "inj_dist": 300.0},
]

for cfg in configs:
    dist = cfg["dist"]
    inj_dist = cfg["inj_dist"]
    truth = {**INJECTION, "luminosity_distance": inj_dist}

    baseline_file = os.path.join(OUTDIR, f"dist_{dist}_iter_1_baseline_result.json")
    refine_file   = os.path.join(OUTDIR, f"dist_{dist}_iter_1_refine_result.json")

    result_baseline = bilby.core.result.read_in_result(filename=baseline_file)
    result_refine   = bilby.core.result.read_in_result(filename=refine_file)

    # Extract posterior samples for the 6 parameters only
    baseline_samples = result_baseline.posterior[PARAMS]
    refine_samples   = result_refine.posterior[PARAMS]

    import corner
    fig = corner.corner(
        baseline_samples.values,
        labels=LABELS,
        color="tab:blue",
        hist_kwargs={"density": True},
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.68, 0.95),
        smooth=1.0,
        truths=[truth[p] for p in PARAMS],
        truth_color="black",
    )
    corner.corner(
        refine_samples.values,
        fig=fig,
        color="tab:red",
        hist_kwargs={"density": True},
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.68, 0.95),
        smooth=1.0,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="tab:blue", alpha=0.7, label="Baseline"),
        Patch(facecolor="tab:red",  alpha=0.7, label="Phase 2 (Refine)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=12,
               bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(f"Posterior Overlap — {dist} Mpc", fontsize=14, y=1.01)

    outpath = os.path.join(OUTDIR, f"Posterior_Overlap_{dist}Mpc.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")
```

## Steps for Claude Code

1. Create the script `MyStuff/Phase_2/generate_overlay_plots.py` with the code above.
2. Run it from the repo root: `python MyStuff/Phase_2/generate_overlay_plots.py`
3. Verify both output files exist and are non-empty:
   - `ls -lh MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344_copy/Posterior_Overlap_*Mpc.png`
4. If `corner` is not installed: `pip install corner`
5. If bilby result loading fails (large JSON), fall back to loading with `bilby.core.result.Result`:
   ```python
   result = bilby.core.result.read_in_result(filename=baseline_file)
   ```

## Acceptance criteria
- Both PNG files exist with size > 100 KB
- The plots visually show blue (Baseline) and red (Phase 2) contours overlapping
- Injected truth values (black lines) are visible and fall within both posteriors

## Notes
- Use `iter_1` only (not an average across iterations) — consistent with the existing 40 Mpc plot
- The `geocent_time` axis will show values near 1126259462.0 — this is fine, do not subtract the mean
- If the corner plot looks cluttered due to geocent_time scale, subtract the injection time:
  add a column `geocent_time_residual = geocent_time - 1126259462.0` and replace in the DataFrame
  before plotting, updating the truth value to 0.0
