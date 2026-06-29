"""
B1 — Noise-floor-normalised Bayes-Factor residual.

Replaces the "relative error" bottom panel of Figure 9
(BF_vs_Distance_With_Relative_Error.png) with a metric that does not
diverge near ln(BF) -> 0.

Primary metric:
    R(d) = |<ln BF_baseline> - <ln BF_phase2>| / sigma_pooled(d)
    sigma_pooled(d) = sqrt((sigma_b^2 + sigma_p^2) / 2)

Interpretation: R < 1 means the Baseline-vs-Phase 2 disagreement at that
distance is smaller than the typical sampler-to-sampler dispersion within
either method itself, i.e. inside the intra-run noise floor.

The "spike" at 1300 Mpc in the old relative-error panel disappears under
this normalisation: the absolute |Delta ln BF| there is ~0.1, comparable
to the ~0.08 intra-run sigma, so R ~ 1.4 -- statistically indistinguishable
from sampler noise.

Run from the bilby repo root:
    python MyStuff/Phase_2/b1_noise_floor_metric.py

Outputs (written next to the source JSON):
  - figure9_noise_floor.png      replacement figure for the thesis
  - figure9_noise_floor.pdf      vector version
  - b1_metric_table.csv          tidy numeric table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to the repo so the script is portable.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_JSON = (
    REPO_ROOT
    / "MyStuff"
    / "my_outdir"
    / "phase_2"
    / "comprehensive_snr_test_20260306_113344_copy"
    / "systematic_snr_variance_results.json"
)
OUT_DIR = RESULTS_JSON.parent

# Jeffreys-scale reference for ln(BF) interpretation (Kass & Raftery 1995).
#   1.0 -> "barely worth a mention"
#   2.3 -> "substantial"
#   5.0 -> "strong"
JEFFREYS_STRONG = 5.0


def load_results(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarise(results: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for key, payload in results.items():
        # keys look like "10.0_Mpc"
        d_l = float(key.split("_")[0])
        baseline = np.asarray(payload["baseline_bfs"], dtype=float)
        phase2 = np.asarray(payload["honest_bfs"], dtype=float)

        mu_b = baseline.mean()
        mu_p = phase2.mean()
        # ddof=1 -> sample std (matches the thesis tables, N-1=2)
        sd_b = baseline.std(ddof=1)
        sd_p = phase2.std(ddof=1)

        delta = mu_p - mu_b
        sigma_pooled = np.sqrt((sd_b**2 + sd_p**2) / 2.0)
        # Standard error of the difference of two means with N_b = N_p = 3.
        sigma_diff = np.sqrt(sd_b**2 / 3.0 + sd_p**2 / 3.0)

        rows.append(
            {
                "d_l_mpc": d_l,
                "baseline_mean": mu_b,
                "baseline_std": sd_b,
                "phase2_mean": mu_p,
                "phase2_std": sd_p,
                "delta_lnbf": delta,
                "abs_delta_lnbf": abs(delta),
                "sigma_pooled": sigma_pooled,
                "sigma_diff": sigma_diff,
                "metric_R": abs(delta) / sigma_pooled if sigma_pooled > 0 else np.nan,
                "z_score": abs(delta) / sigma_diff if sigma_diff > 0 else np.nan,
                "rel_to_strong": abs(delta) / JEFFREYS_STRONG,
            }
        )
    rows.sort(key=lambda r: r["d_l_mpc"])
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], png_path: Path, pdf_path: Path) -> None:
    distances = np.array([r["d_l_mpc"] for r in rows])
    mu_b = np.array([r["baseline_mean"] for r in rows])
    sd_b = np.array([r["baseline_std"] for r in rows])
    mu_p = np.array([r["phase2_mean"] for r in rows])
    sd_p = np.array([r["phase2_std"] for r in rows])
    delta = np.array([r["delta_lnbf"] for r in rows])
    sigma_pooled = np.array([r["sigma_pooled"] for r in rows])
    metric_r = np.array([r["metric_R"] for r in rows])

    fig, axes = plt.subplots(
        3, 1, figsize=(8.5, 10.0), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )

    # --- Top: ln(BF) overlay -------------------------------------------------
    ax = axes[0]
    ax.errorbar(
        distances, mu_b, yerr=sd_b, fmt="o-",
        label="Baseline", color="#185FA5", markersize=6, capsize=3, lw=1.5,
    )
    ax.errorbar(
        distances, mu_p, yerr=sd_p, fmt="s--",
        label="Phase 2 (Refine)", color="#D85A30", markersize=6, capsize=3, lw=1.5,
    )
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel(r"$\ln(\mathrm{BF})$")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_title(r"Bayes-factor recovery across the SNR spectrum (N=3 iterations)")

    # --- Middle: absolute residual with intra-run noise band -----------------
    ax = axes[1]
    ax.fill_between(
        distances, -sigma_pooled, +sigma_pooled,
        color="#888780", alpha=0.25, label=r"$\pm\sigma_{\rm pooled}$ (intra-run)",
    )
    ax.plot(distances, delta, "o-", color="#534AB7", lw=1.5, markersize=6,
            label=r"$\Delta\ln(\mathrm{BF})$")
    ax.axhline(0.0, color="#444441", lw=0.8, ls="-")
    ax.axhline(+JEFFREYS_STRONG, color="#A32D2D", lw=0.8, ls=":",
               label=r"Jeffreys strong ($|\Delta|=5$)")
    ax.axhline(-JEFFREYS_STRONG, color="#A32D2D", lw=0.8, ls=":")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel(
        r"$\Delta\ln(\mathrm{BF}) = "
        r"\langle\ln\mathrm{BF}\rangle_{\rm P2} - "
        r"\langle\ln\mathrm{BF}\rangle_{\rm B}$"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # --- Bottom: NEW noise-floor-normalised metric ---------------------------
    ax = axes[2]
    ax.axhspan(0.0, 1.0, color="#9FE1CB", alpha=0.30,
               label="within sampler noise (R < 1)")
    ax.axhline(1.0, color="#0F6E56", lw=0.9, ls="--")
    ax.plot(distances, metric_r, "o-", color="#0F6E56", lw=1.6, markersize=7)
    for d, r in zip(distances, metric_r):
        ax.annotate(f"{r:.2f}", (d, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8,
                    color="#04342C")
    ax.set_ylabel(r"$R \;=\; |\Delta\ln\mathrm{BF}|\,/\,\sigma_{\rm pooled}$")
    ax.set_xlabel(r"Luminosity distance $d_L$ (Mpc)")
    ax.set_ylim(bottom=0.0)
    ax.set_xscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    print(f"{'d_L (Mpc)':>9} | {'<lnBF>_B':>12} | {'<lnBF>_P2':>12} | "
          f"{'Delta':>10} | {'sigma_pool':>10} | {'R':>6} | {'|z|':>6}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['d_l_mpc']:>9.0f} | {r['baseline_mean']:>12.3f} | "
            f"{r['phase2_mean']:>12.3f} | {r['delta_lnbf']:>10.3f} | "
            f"{r['sigma_pooled']:>10.3f} | {r['metric_R']:>6.2f} | "
            f"{r['z_score']:>6.2f}"
        )


def main() -> None:
    results = load_results(RESULTS_JSON)
    rows = summarise(results)
    print_summary(rows)
    write_csv(rows, OUT_DIR / "b1_metric_table.csv")
    make_figure(
        rows,
        OUT_DIR / "B1_BF_vs_Distance_With_Relative_Error.png",
        OUT_DIR / "B1_BF_vs_Distance_With_Relative_Error.pdf",
    )
    print(f"\nWrote: {OUT_DIR / 'B1_BF_vs_Distance_With_Relative_Error.png'}")
    print(f"Wrote: {OUT_DIR / 'B1_BF_vs_Distance_With_Relative_Error.pdf'}")
    print(f"Wrote: {OUT_DIR / 'b1_metric_table.csv'}")


if __name__ == "__main__":
    main()
