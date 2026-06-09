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
