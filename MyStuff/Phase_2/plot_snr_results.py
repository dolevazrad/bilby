#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import os

JSON_PATH = '/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344/systematic_snr_variance_results.json'

def Generate_Residual_Plot():
    if not os.path.exists(JSON_PATH):
        print(f"Error: Could not find {JSON_PATH}")
        return

    out_dir = os.path.dirname(JSON_PATH)

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    distances = []
    base_bf_means, base_bf_stds = [], []
    honest_bf_means, honest_bf_stds = [], []

    sorted_keys = sorted(data.keys(), key=lambda x: float(x.replace('_Mpc', '')))
    
    for key in sorted_keys:
        dist = float(key.replace('_Mpc', ''))
        distances.append(dist)
        stats = data[key]['stats']
        
        base_bf_means.append(stats['baseline_mean_bf'])
        base_bf_stds.append(stats['baseline_std_bf'])
        honest_bf_means.append(stats['honest_mean_bf'])
        honest_bf_stds.append(stats['honest_std_bf'])

    dist_array = np.array(distances)
    base_means = np.array(base_bf_means)
    base_stds = np.array(base_bf_stds)
    honest_means = np.array(honest_bf_means)
    honest_stds = np.array(honest_bf_stds)

    # Calculate Absolute Delta and Propagated Error
    delta_means = honest_means - base_means
    delta_stds = np.sqrt(base_stds**2 + honest_stds**2)

    # Calculate Relative Error [%] = (Delta / |Baseline|) * 100
    # Note: We use absolute value in the denominator to keep signs consistent
    rel_err_means = (delta_means / np.abs(base_means)) * 100
    # Simple error propagation for the relative error visually
    rel_err_stds = (delta_stds / np.abs(base_means)) * 100

    # Create figure with 3 subplots (Height ratio 3:1.2:1.2)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1.2, 1.2]})
    
    # --- Top Plot: Main BF ---
    ax1.errorbar(dist_array, base_means, yerr=base_stds, 
                 fmt='-o', color='blue', label='Baseline', capsize=4, markersize=6, alpha=0.8)
    ax1.errorbar(dist_array, honest_means, yerr=honest_stds, 
                 fmt='-s', color='red', label='Phase 2', capsize=4, markersize=6, alpha=0.8)

    ax1.set_yscale('symlog', linthresh=1.0)
    ax1.set_ylabel('$\ln(\mathrm{Bayes Factor})$')
    ax1.set_title('Log Bayes Factor vs. Luminosity Distance')
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend()

    # --- Middle Plot: Absolute Residuals (Delta) ---
    ax2.errorbar(dist_array, delta_means, yerr=delta_stds, 
                 fmt='-^', color='purple', capsize=4, markersize=6)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Absolute Error\n$\Delta \ln(\mathrm{BF})$')
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    
    # --- Bottom Plot: Relative Error [%] ---
    ax3.errorbar(dist_array, rel_err_means, yerr=rel_err_stds, 
                 fmt='-D', color='green', capsize=4, markersize=6)
    
    ax3.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xscale('log')
    ax3.set_xlabel('Luminosity Distance [Mpc]')
    ax3.set_ylabel('Relative Error [%]\n$(\Delta / |\mathrm{Baseline}|) \cdot 100$')
    
    # Limit the Y-axis of the relative error to make it readable (e.g., +/- 5%)
    # because crossing the zero-evidence line can distort the scale
    ax3.set_ylim([-5, 5])
    ax3.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    
    bf_plot_path = os.path.join(out_dir, 'BF_vs_Distance_With_Relative_Error.png')
    plt.savefig(bf_plot_path, dpi=300)
    print(f"Saved: {bf_plot_path}")

if __name__ == "__main__":
    Generate_Residual_Plot()