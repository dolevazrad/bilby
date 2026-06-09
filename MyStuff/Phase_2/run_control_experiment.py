#!/usr/bin/env python3
"""
Control experiment: nlive=1024, walks=50, dlogz=0.1, BROAD PRIORS (no prior compression).
Runs 3 iterations at 40 Mpc and 3 iterations at 150 Mpc.
Results saved with checkpointing after each iteration.

Purpose: isolate the contribution of prior compression from sampler budget reduction.
Compare against:
  - Baseline (2048, broad): to measure budget-only savings
  - Phase 2  (1024, compressed): to measure compression-only savings
"""

import os
import json
import time
import numpy as np
from datetime import datetime

import bilby
import numpy as np

from all_in_one_correct_phase2 import (
    find_asd_scenarios,
    run_pe,
    create_injection_parameters,
    Re_Weight_Posterior,
    OUTPUT_BASE,
)

def get_broad_priors(injection_params):
    """Broad priors identical to the Baseline configuration."""
    priors = bilby.gw.prior.BBHPriorDict()
    priors['chirp_mass'] = bilby.core.prior.Uniform(25.0, 35.0, name='chirp_mass')
    priors['mass_ratio'] = bilby.core.prior.Uniform(0.5, 1.0, name='mass_ratio')
    priors['luminosity_distance'] = bilby.core.prior.Uniform(1.0, 10000.0, name='luminosity_distance')
    priors['geocent_time'] = bilby.core.prior.Uniform(
        injection_params['geocent_time'] - 0.1,
        injection_params['geocent_time'] + 0.1,
        name='geocent_time'
    )
    priors['phase'] = bilby.core.prior.Uniform(0, 2 * np.pi, name='phase')
    priors['theta_jn'] = bilby.core.prior.Sine(name='theta_jn')
    priors['ra'] = bilby.core.prior.Uniform(0, 2 * np.pi, name='ra')
    priors['dec'] = bilby.core.prior.Cosine(name='dec')
    priors['psi'] = bilby.core.prior.Uniform(0, np.pi, name='psi')
    priors['a_1'] = bilby.core.prior.Uniform(0, 0.99, name='a_1')
    priors['a_2'] = bilby.core.prior.Uniform(0, 0.99, name='a_2')
    priors['tilt_1'] = bilby.core.prior.Sine(name='tilt_1')
    priors['tilt_2'] = bilby.core.prior.Sine(name='tilt_2')
    priors['phi_12'] = bilby.core.prior.Uniform(0, 2 * np.pi, name='phi_12', boundary='periodic')
    priors['phi_jl'] = bilby.core.prior.Uniform(0, 2 * np.pi, name='phi_jl', boundary='periodic')
    return priors


def main():
    print("=" * 70)
    print("CONTROL EXPERIMENT: nlive=1024, walks=50, BROAD PRIORS")
    print("Distances: 40 Mpc and 150 Mpc | 3 iterations each")
    print("=" * 70)

    full_files, half_files, _ = find_asd_scenarios()
    if not full_files:
        print("ERROR: Could not find ASD files. Aborting.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_BASE, f'control_run_{timestamp}')
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")

    json_path = os.path.join(outdir, 'control_experiment_results.json')

    # Existing results directory — used to load baseline and phase2 timing
    existing_dir = os.path.join(
        OUTPUT_BASE,
        'comprehensive_snr_test_20260306_113344_copy'
    )
    existing_json = os.path.join(existing_dir, 'systematic_snr_variance_results.json')

    # Load existing timing data
    with open(existing_json, 'r') as f:
        existing_results = json.load(f)

    distances_to_test = [40.0, 150.0]
    master_results = {}

    for dist in distances_to_test:
        dist_key = f"{int(dist)}_Mpc"
        existing_key = f"{dist}_Mpc"
        print(f"\n{'#' * 70}")
        print(f"DISTANCE: {dist} Mpc")
        print(f"{'#' * 70}")

        params = create_injection_parameters(distance=dist)
        broad_priors = get_broad_priors(params)

        control_times = []
        control_bfs = []

        for i in range(1, 4):
            print(f"\n--- {dist} Mpc | Iteration {i}/3 ---")

            # Label contains 'control' so run_pe will use nlive=1024, walks=50
            c_label = f'dist_{int(dist)}_iter_{i}_control'

            c_res, c_time = run_pe(
                full_files,          # same ASD as Baseline (not the scout half-files)
                c_label,
                outdir,
                informed_priors=None,  # NO prior compression — broad priors only
                custom_injection_params=params
            )

            c_bf = Re_Weight_Posterior(c_res, broad_priors)
            control_times.append(c_time)
            control_bfs.append(c_bf)
            print(f"  Control time: {c_time/3600:.2f} h | BF: {c_bf:.2f}")

            # Checkpoint after every iteration
            master_results[dist_key] = {
                'control_times_s': control_times,
                'control_bfs': control_bfs,
                'iterations_completed': i,
            }
            with open(json_path, 'w') as f:
                json.dump(master_results, f, indent=4)
            print(f"  [Checkpoint saved]")

        # --- Summary statistics for this distance ---
        ctrl_mean_h = np.mean(control_times) / 3600
        ctrl_std_h  = np.std(control_times)  / 3600
        ctrl_mean_bf = np.mean(control_bfs)

        # Pull existing baseline and phase2 times from the systematic results JSON
        dist_existing = existing_results.get(existing_key, {}).get('stats', {})
        baseline_mean_h = dist_existing.get('avg_baseline_time_h', None)
        phase2_mean_h   = dist_existing.get('avg_phase2_time_h',   None)

        budget_savings_pct      = None
        compression_savings_pct = None
        total_savings_pct       = None

        if baseline_mean_h:
            budget_savings_pct = (baseline_mean_h - ctrl_mean_h) / baseline_mean_h * 100
        if phase2_mean_h and ctrl_mean_h:
            compression_savings_pct = (ctrl_mean_h - phase2_mean_h) / ctrl_mean_h * 100
        if baseline_mean_h and phase2_mean_h:
            total_savings_pct = (baseline_mean_h - phase2_mean_h) / baseline_mean_h * 100

        master_results[dist_key].update({
            'control_mean_time_h':       round(ctrl_mean_h, 3),
            'control_std_time_h':        round(ctrl_std_h,  3),
            'control_mean_bf':           round(ctrl_mean_bf, 3),
            'baseline_mean_time_h':      round(baseline_mean_h, 3) if baseline_mean_h else None,
            'phase2_mean_time_h':        round(phase2_mean_h,   3) if phase2_mean_h   else None,
            'budget_savings_pct':        round(budget_savings_pct,      1) if budget_savings_pct      is not None else None,
            'compression_savings_pct':   round(compression_savings_pct, 1) if compression_savings_pct is not None else None,
            'total_savings_pct':         round(total_savings_pct,       1) if total_savings_pct       is not None else None,
        })

        with open(json_path, 'w') as f:
            json.dump(master_results, f, indent=4)

        print(f"\n  SUMMARY — {dist} Mpc")
        print(f"  Baseline:    {baseline_mean_h:.2f} h" if baseline_mean_h else "  Baseline:    N/A")
        print(f"  Control:     {ctrl_mean_h:.2f} h  (budget savings vs baseline: {budget_savings_pct:.1f}%)" if budget_savings_pct is not None else f"  Control: {ctrl_mean_h:.2f} h")
        print(f"  Phase 2:     {phase2_mean_h:.2f} h  (compression savings vs control: {compression_savings_pct:.1f}%)" if compression_savings_pct is not None else f"  Phase 2: {phase2_mean_h:.2f} h")
        print(f"  Total Phase2 savings vs Baseline: {total_savings_pct:.1f}%" if total_savings_pct is not None else "")

    # Final summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE")
    print(f"{'Distance':<12} {'Baseline':>10} {'Control':>10} {'Phase 2':>10} {'Budget%':>10} {'Compress%':>12}")
    print("-" * 70)
    for dist_key, data in master_results.items():
        b  = f"{data['baseline_mean_time_h']:.2f}h"  if data.get('baseline_mean_time_h')  else "N/A"
        c  = f"{data['control_mean_time_h']:.2f}h"
        p2 = f"{data['phase2_mean_time_h']:.2f}h"    if data.get('phase2_mean_time_h')    else "N/A"
        bs = f"{data['budget_savings_pct']:.1f}%"    if data.get('budget_savings_pct')    is not None else "N/A"
        cs = f"{data['compression_savings_pct']:.1f}%" if data.get('compression_savings_pct') is not None else "N/A"
        print(f"{dist_key:<12} {b:>10} {c:>10} {p2:>10} {bs:>10} {cs:>12}")
    print("=" * 70)
    print(f"\nFull results saved to: {json_path}")


if __name__ == "__main__":
    main()
