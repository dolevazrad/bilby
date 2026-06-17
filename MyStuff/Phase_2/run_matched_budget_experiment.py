#!/usr/bin/env python3
"""
B3 (thesis_fix_plan.md): Matched-Budget Control Experiment.

Run the Refine sampler with the BASELINE budget (nlive=2048, walks=100, dlogz=0.1)
and Scout-compressed informed priors, at d_L = 40 and 150 Mpc.

Purpose
-------
Phase 2 (existing) compresses the prior AND reduces the sampler budget; the
observed ~44% saving therefore mixes two effects. This script holds the budget
fixed at the Baseline value so the only remaining variable is the prior
compression. Comparing matched-budget Refine to the Baseline isolates the
contribution of prior compression alone.

  - If matched-budget Refine wall-clock == Baseline wall-clock
        -> compression contributes 0; all savings come from budget reduction.
  - If matched-budget Refine < Baseline
        -> compression contributes real, independent savings of that magnitude.

Re-uses Phase 1 (Scout) posteriors already on disk under
``comprehensive_snr_test_20260306_113344_copy``. No new Scout runs.

Layout
------
3 iterations x 2 distances = 6 Refine runs, each ~5-8 h on a single-pool
dynesty configuration. Checkpoints after every iteration so the script is
safely resumable.

Output JSON has the same compression/budget/total decomposition as the
sibling ``run_control_experiment.py`` so the two experiments report side-by-side.
"""

import json
import os
from datetime import datetime

import bilby
import numpy as np

from all_in_one_correct_phase2 import (
    OUTPUT_BASE,
    Re_Weight_Posterior,
    create_informed_priors,
    create_injection_parameters,
    find_asd_scenarios,
    run_pe,
)

# ---------------------------------------------------------------------------
# Existing comprehensive-SNR results directory. Scout JSONs and Baseline/Phase 2
# timing are read from here so we don't redo work.
# ---------------------------------------------------------------------------
EXISTING_RUN_DIR = os.path.join(
    OUTPUT_BASE, 'comprehensive_snr_test_20260306_113344_copy'
)
EXISTING_JSON = os.path.join(EXISTING_RUN_DIR, 'systematic_snr_variance_results.json')


def get_broad_priors(injection_params):
    """Replicate the Baseline broad priors used in the systematic SNR study."""
    priors = bilby.gw.prior.BBHPriorDict()
    priors['chirp_mass'] = bilby.core.prior.Uniform(25.0, 35.0, name='chirp_mass')
    priors['mass_ratio'] = bilby.core.prior.Uniform(0.5, 1.0, name='mass_ratio')
    priors['luminosity_distance'] = bilby.core.prior.Uniform(
        1.0, 10000.0, name='luminosity_distance'
    )
    priors['geocent_time'] = bilby.core.prior.Uniform(
        injection_params['geocent_time'] - 0.1,
        injection_params['geocent_time'] + 0.1,
        name='geocent_time',
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
    priors['phi_12'] = bilby.core.prior.Uniform(
        0, 2 * np.pi, name='phi_12', boundary='periodic'
    )
    priors['phi_jl'] = bilby.core.prior.Uniform(
        0, 2 * np.pi, name='phi_jl', boundary='periodic'
    )
    return priors


def load_scout_result(dist_int, iteration):
    """Load an existing Scout result JSON; return a bilby Result object."""
    fname = f'dist_{dist_int}_iter_{iteration}_scout_result.json'
    path = os.path.join(EXISTING_RUN_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scout result not found at {path}. "
            "B3 requires the existing comprehensive_snr_test_20260306_113344_copy "
            "directory to be present."
        )
    return bilby.result.read_in_result(path)


def deltas_vs_baseline(matched_res, baseline_res, params=('chirp_mass', 'mass_ratio')):
    """|median_matched - median_baseline| / std_baseline for each parameter."""
    deltas = {}
    for p in params:
        m_med = np.median(matched_res.posterior[p].values)
        b_med = np.median(baseline_res.posterior[p].values)
        b_std = np.std(baseline_res.posterior[p].values)
        deltas[p] = abs(m_med - b_med) / b_std if b_std > 0 else float('nan')
    return deltas


def main():
    print('=' * 70)
    print('B3 MATCHED-BUDGET EXPERIMENT: nlive=2048, walks=100, COMPRESSED PRIORS')
    print('Distances: 40 Mpc and 150 Mpc | 3 iterations each')
    print('=' * 70)

    full_files, _, _ = find_asd_scenarios()
    if not full_files:
        print('ERROR: Could not find ASD files. Aborting.')
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_BASE, f'matched_budget_run_{timestamp}')
    os.makedirs(outdir, exist_ok=True)
    print(f'Output directory: {outdir}')

    json_path = os.path.join(outdir, 'matched_budget_experiment_results.json')

    if not os.path.exists(EXISTING_JSON):
        raise FileNotFoundError(
            f'Existing systematic results JSON missing: {EXISTING_JSON}. '
            'B3 needs it for Baseline/Phase 2 timing reference.'
        )
    with open(EXISTING_JSON, 'r') as f:
        existing_results = json.load(f)

    distances_to_test = [40.0, 150.0]
    master_results = {}

    for dist in distances_to_test:
        dist_int = int(dist)
        dist_key = f'{dist_int}_Mpc'
        existing_key = f'{dist}_Mpc'
        print(f"\n{'#' * 70}")
        print(f'DISTANCE: {dist} Mpc')
        print(f"{'#' * 70}")

        params = create_injection_parameters(distance=dist)
        broad_priors = get_broad_priors(params)

        matched_times = []
        matched_bfs = []
        matched_deltas = []

        for i in range(1, 4):
            print(f'\n--- {dist} Mpc | Iteration {i}/3 ---')

            # 1. Re-use the existing Scout posterior; no new Scout run.
            scout_res = load_scout_result(dist_int, i)
            informed_priors = create_informed_priors(scout_res)

            # 2. Label must contain 'matched_budget' to trigger the
            #    nlive=2048, walks=100 branch in run_pe.
            m_label = f'dist_{dist_int}_iter_{i}_matched_budget'

            m_res, m_time = run_pe(
                full_files,
                m_label,
                outdir,
                informed_priors=informed_priors,
                custom_injection_params=params,
            )

            # 3. Honest BF via importance re-weighting back to the broad prior.
            m_bf = Re_Weight_Posterior(m_res, broad_priors)

            # 4. Parameter deltas against the existing Baseline iteration.
            baseline_path = os.path.join(
                EXISTING_RUN_DIR,
                f'dist_{dist_int}_iter_{i}_baseline_result.json',
            )
            try:
                baseline_res = bilby.result.read_in_result(baseline_path)
                deltas = deltas_vs_baseline(m_res, baseline_res)
            except Exception as exc:
                print(f'  WARNING: Could not load baseline for delta calc: {exc}')
                deltas = {'chirp_mass': None, 'mass_ratio': None}

            matched_times.append(m_time)
            matched_bfs.append(m_bf)
            matched_deltas.append(deltas)
            print(
                f'  Matched-budget time: {m_time / 3600:.2f} h | '
                f'BF: {m_bf:.2f} | deltas: {deltas}'
            )

            master_results[dist_key] = {
                'matched_times_s': matched_times,
                'matched_bfs': matched_bfs,
                'matched_deltas': matched_deltas,
                'iterations_completed': i,
            }
            with open(json_path, 'w') as f:
                json.dump(master_results, f, indent=4)
            print('  [Checkpoint saved]')

        # ---- Summary statistics for this distance ----
        matched_mean_h = float(np.mean(matched_times) / 3600)
        matched_std_h = float(np.std(matched_times) / 3600)
        matched_mean_bf = float(np.mean(matched_bfs))

        dist_existing = existing_results.get(existing_key, {}).get('stats', {})
        baseline_mean_h = dist_existing.get('avg_baseline_time_h')
        phase2_mean_h = dist_existing.get('avg_phase2_time_h')

        # Compression-only savings: Baseline -> Matched-budget Refine.
        # Sampler budget is identical to Baseline; the only difference is the
        # compressed prior.  Anything saved here is attributable to compression.
        compression_savings_pct = None
        if baseline_mean_h:
            compression_savings_pct = (
                (baseline_mean_h - matched_mean_h) / baseline_mean_h * 100
            )

        # Budget-only savings (residual): Matched-budget -> Phase 2.
        # Priors are identical (both compressed); the only difference is the
        # reduced sampler budget in Phase 2.
        budget_savings_pct = None
        if matched_mean_h and phase2_mean_h:
            budget_savings_pct = (
                (matched_mean_h - phase2_mean_h) / matched_mean_h * 100
            )

        total_savings_pct = None
        if baseline_mean_h and phase2_mean_h:
            total_savings_pct = (
                (baseline_mean_h - phase2_mean_h) / baseline_mean_h * 100
            )

        master_results[dist_key].update(
            {
                'matched_mean_time_h': round(matched_mean_h, 3),
                'matched_std_time_h': round(matched_std_h, 3),
                'matched_mean_bf': round(matched_mean_bf, 3),
                'baseline_mean_time_h': (
                    round(baseline_mean_h, 3) if baseline_mean_h else None
                ),
                'phase2_mean_time_h': (
                    round(phase2_mean_h, 3) if phase2_mean_h else None
                ),
                'compression_savings_pct': (
                    round(compression_savings_pct, 1)
                    if compression_savings_pct is not None
                    else None
                ),
                'budget_savings_pct': (
                    round(budget_savings_pct, 1)
                    if budget_savings_pct is not None
                    else None
                ),
                'total_savings_pct': (
                    round(total_savings_pct, 1)
                    if total_savings_pct is not None
                    else None
                ),
            }
        )

        with open(json_path, 'w') as f:
            json.dump(master_results, f, indent=4)

        print(f'\n  SUMMARY -- {dist} Mpc')
        if baseline_mean_h:
            print(f'  Baseline:        {baseline_mean_h:.2f} h')
        if compression_savings_pct is not None:
            print(
                f'  Matched-budget:  {matched_mean_h:.2f} h  '
                f'(compression savings vs Baseline: {compression_savings_pct:.1f}%)'
            )
        else:
            print(f'  Matched-budget:  {matched_mean_h:.2f} h')
        if budget_savings_pct is not None:
            print(
                f'  Phase 2:         {phase2_mean_h:.2f} h  '
                f'(budget savings vs Matched-budget: {budget_savings_pct:.1f}%)'
            )
        elif phase2_mean_h:
            print(f'  Phase 2:         {phase2_mean_h:.2f} h')
        if total_savings_pct is not None:
            print(
                f'  Total Phase 2 savings vs Baseline: {total_savings_pct:.1f}%'
            )

    # ---- Final printed table ----
    print('\n' + '=' * 70)
    print('FINAL SUMMARY TABLE (B3 matched-budget)')
    print(
        f"{'Distance':<10} {'Baseline':>10} {'Matched':>10} "
        f"{'Phase 2':>10} {'Compress%':>11} {'Budget%':>10}"
    )
    print('-' * 70)
    for dist_key, data in master_results.items():
        b = f"{data['baseline_mean_time_h']:.2f}h" if data.get('baseline_mean_time_h') else 'N/A'
        m = f"{data['matched_mean_time_h']:.2f}h"
        p2 = f"{data['phase2_mean_time_h']:.2f}h" if data.get('phase2_mean_time_h') else 'N/A'
        cs = (
            f"{data['compression_savings_pct']:.1f}%"
            if data.get('compression_savings_pct') is not None
            else 'N/A'
        )
        bs = (
            f"{data['budget_savings_pct']:.1f}%"
            if data.get('budget_savings_pct') is not None
            else 'N/A'
        )
        print(f'{dist_key:<10} {b:>10} {m:>10} {p2:>10} {cs:>11} {bs:>10}')
    print('=' * 70)
    print(f'\nFull results saved to: {json_path}')


if __name__ == '__main__':
    main()
