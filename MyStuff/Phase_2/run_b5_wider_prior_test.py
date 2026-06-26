#!/usr/bin/env python3
"""
B5 follow-up: wider-informed-prior test on the Scout-lock-in outlier.

Context
-------
The B5 + matched-budget follow-up identified two distinct failure modes:

  (i)  Reduced-budget Refine noise (e.g. inj_04): matched-budget Refine
       collapsed Delta_param from ~1.0 to ~0.3. Diagnosed cause: sampler.

  (ii) Scout-induced informed-prior lock-in (e.g. inj_05): matched-budget
       Refine made Delta_param *worse* (>1.0 for q, chi_1, chi_2) because
       more sampling budget converged more sharply on the Scout's
       slightly off-centre TruncatedGaussian. Diagnosed cause: prior.

This script tests the proposed fix for (ii): raise the informed-prior
sigma multiplier in ``create_informed_priors`` from 3.0 to 5.0 (or any
other value), keeping the production reduced-budget Refine sampler.
Wider truncation gives Refine room to escape the Scout's bias while
still benefiting from prior compression.

For each target injection we run two new Refines:

    wider_reduced   : safe_sigma=5*sigma, nlive=1024, walks=50   (production)
    wider_matched   : safe_sigma=5*sigma, nlive=2048, walks=100  (diagnostic)

so the resulting comparison spans all 2x2 combinations of {sigma=3, sigma=5}
and {reduced budget, matched budget} -- which is exactly the table needed to
make a clean recommendation in section 6.5.

Usage
-----
    python run_b5_wider_prior_test.py <b5_campaign_directory>

The default target is inj_05 (q=0.90, chi_eff=+0.39). Edit
``TARGET_INDICES`` and ``SIGMA_MULTIPLIER`` below to adjust.
"""

import json
import os
import sys
from datetime import datetime

import bilby
import numpy as np

from all_in_one_correct_phase2 import OUTPUT_BASE, find_asd_scenarios
from run_injection_campaign_b5 import (
    DELTA_PARAMS,
    build_injection_grid,
    delta_params as unweighted_delta_params,
    get_broad_priors,
    run_pe_aligned,
)
from recompute_b5_deltas_weighted import (
    importance_weights,
    weighted_delta_params,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_INDICES = [5]      # B5 injection indices to retest
SIGMA_MULTIPLIER = 5.0    # informed-prior width: safe_sigma = sigma * THIS
RUN_BOTH_BUDGETS = True   # if False, only the production (reduced-budget) leg


# ---------------------------------------------------------------------------
# Wider informed priors -- standalone copy so we don't have to modify the
# shared ``all_in_one_correct_phase2.create_informed_priors`` function.
# ---------------------------------------------------------------------------
def create_wider_informed_priors(posterior_result, sigma_multiplier=SIGMA_MULTIPLIER):
    """Mirror ``create_informed_priors`` but with configurable sigma factor."""
    print('\n' + '*' * 60)
    print(f'WIDER INFORMED PRIORS  (safe_sigma = {sigma_multiplier} * sigma)')
    print('*' * 60)
    informed = bilby.gw.prior.PriorDict()
    orig_priors = posterior_result.priors
    refine_keys = ('chirp_mass', 'mass_ratio', 'luminosity_distance', 'geocent_time')
    for param in refine_keys:
        if param not in posterior_result.posterior:
            continue
        samples = posterior_result.posterior[param].values
        mu = float(np.mean(samples))
        sigma = float(np.std(samples))
        safe_sigma = sigma * sigma_multiplier
        old_min = orig_priors[param].minimum
        old_max = orig_priors[param].maximum
        informed[param] = bilby.core.prior.TruncatedGaussian(
            mu=mu, sigma=safe_sigma, minimum=old_min, maximum=old_max, name=param,
        )
        print(f'  {param:<22s} mu={mu:.4f}  sigma={sigma:.4e}  safe_sigma={safe_sigma:.4e}')
    print('*' * 60 + '\n')
    return informed


# ---------------------------------------------------------------------------
# Pretty-print: row per parameter, one column per Refine variant tested
# ---------------------------------------------------------------------------
def print_table(rows, header_columns):
    cols = ['param'] + header_columns
    widths = [22] + [16] * len(header_columns)
    print(' '.join(f'{c:>{w}s}' for c, w in zip(cols, widths)))
    print('-' * (sum(widths) + len(widths) - 1))
    for p in DELTA_PARAMS:
        cells = [p]
        for col in header_columns:
            v = rows.get(col, {}).get(p)
            cells.append('-' if v is None else f'{v:.3f}')
        print(' '.join(f'{c:>{w}s}' for c, w in zip(cells, widths)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print('Usage: python run_b5_wider_prior_test.py <b5_campaign_directory>')
        sys.exit(1)
    campaign_dir = sys.argv[1]
    if not os.path.isdir(campaign_dir):
        print(f'ERROR: not a directory: {campaign_dir}')
        sys.exit(1)

    orig_json_path = os.path.join(campaign_dir, 'b5_campaign_results.json')
    if not os.path.exists(orig_json_path):
        raise FileNotFoundError(f'Missing campaign JSON at {orig_json_path}')
    with open(orig_json_path, 'r') as f:
        orig_master = json.load(f)

    full_files, _, _ = find_asd_scenarios()
    if not full_files:
        print('ERROR: could not locate ASD files.')
        sys.exit(1)

    # Look for an existing matched-budget outliers run so we can include those
    # numbers in the side-by-side table.
    mb_outliers = {}
    for sub in sorted(os.listdir(campaign_dir)):
        if sub.startswith('matched_budget_outliers_'):
            mb_json = os.path.join(campaign_dir, sub, 'matched_budget_outliers_results.json')
            if os.path.exists(mb_json):
                with open(mb_json, 'r') as f:
                    mb_outliers = json.load(f).get('injections', {})
                print(f'Loaded existing matched-budget results from: {mb_json}')
                break

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(campaign_dir, f'wider_prior_test_sigma{int(SIGMA_MULTIPLIER)}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'wider_prior_test_results.json')
    print(f'Output directory: {out_dir}')

    injections_grid = build_injection_grid(
        n=orig_master['config']['n_injections'],
        seed=orig_master['config']['random_seed'],
    )

    results = {
        'config': {
            'target_indices': TARGET_INDICES,
            'sigma_multiplier': SIGMA_MULTIPLIER,
            'parent_campaign_dir': campaign_dir,
            'run_both_budgets': RUN_BOTH_BUDGETS,
        },
        'injections': {},
    }

    for idx in TARGET_INDICES:
        key = f'inj_{idx:02d}'
        if idx >= len(injections_grid):
            print(f'  Skipping {key}: out of grid range (n={len(injections_grid)}).')
            continue
        inj = injections_grid[idx]

        print('\n' + '#' * 78)
        print(f'[{key}]  q={inj["mass_ratio"]:.3f}  chi_eff={inj["_chi_eff_target"]:+.3f}'
              f'   (sigma_multiplier = {SIGMA_MULTIPLIER})')
        print('#' * 78)

        scout_path = os.path.join(campaign_dir, f'{key}_scout_result.json')
        baseline_path = os.path.join(campaign_dir, f'{key}_baseline_result.json')
        if not (os.path.exists(scout_path) and os.path.exists(baseline_path)):
            print(f'  Missing Scout/Baseline for {key}; skipping.')
            continue

        scout_res = bilby.result.read_in_result(scout_path)
        baseline_res = bilby.result.read_in_result(baseline_path)
        wider = create_wider_informed_priors(scout_res, SIGMA_MULTIPLIER)
        broad = get_broad_priors(inj)

        per_inj = {
            'q_target': inj['mass_ratio'],
            'chi_eff_target': inj['_chi_eff_target'],
            'baseline_log_bf': float(baseline_res.log_bayes_factor),
            'orig_refine_deltas': orig_master['injections'].get(key, {}).get('delta_params'),
            'matched_budget_deltas': (mb_outliers.get(key, {}) or {})
                .get('delta_params_matched_budget'),
        }

        # ------------- Wider + reduced budget (production setting) -------------
        wr_label = f'{key}_wider_sigma{int(SIGMA_MULTIPLIER)}_reduced_refine'
        wr_path = os.path.join(out_dir, f'{wr_label}_result.json')
        if os.path.exists(wr_path):
            print(f'  Reusing existing wider+reduced result: {wr_path}')
            wr_res = bilby.result.read_in_result(wr_path)
            wr_time = float('nan')
        else:
            wr_res, wr_time = run_pe_aligned(full_files, wr_label, out_dir, inj,
                                             informed_priors=wider)
        ln_w_wr, ess_wr = importance_weights(wr_res, broad)
        per_inj['wider_reduced'] = {
            'time_s': float(wr_time),
            'log_bf_raw': float(wr_res.log_bayes_factor),
            'effective_sample_size': ess_wr,
            'posterior_size': int(len(wr_res.posterior)),
            'delta_params_unweighted': unweighted_delta_params(wr_res, baseline_res),
            'delta_params_weighted': weighted_delta_params(wr_res, baseline_res, ln_w_wr),
        }

        # ------------- Wider + matched budget (diagnostic, optional) -------------
        if RUN_BOTH_BUDGETS:
            wm_label = f'{key}_wider_sigma{int(SIGMA_MULTIPLIER)}_matched_budget_refine'
            wm_path = os.path.join(out_dir, f'{wm_label}_result.json')
            if os.path.exists(wm_path):
                print(f'  Reusing existing wider+matched result: {wm_path}')
                wm_res = bilby.result.read_in_result(wm_path)
                wm_time = float('nan')
            else:
                wm_res, wm_time = run_pe_aligned(full_files, wm_label, out_dir, inj,
                                                 informed_priors=wider)
            ln_w_wm, ess_wm = importance_weights(wm_res, broad)
            per_inj['wider_matched'] = {
                'time_s': float(wm_time),
                'log_bf_raw': float(wm_res.log_bayes_factor),
                'effective_sample_size': ess_wm,
                'posterior_size': int(len(wm_res.posterior)),
                'delta_params_unweighted': unweighted_delta_params(wm_res, baseline_res),
                'delta_params_weighted': weighted_delta_params(wm_res, baseline_res, ln_w_wm),
            }

        results['injections'][key] = per_inj
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  [Checkpoint saved: {out_json}]')

        # ------------- Per-injection report -------------
        cols = {
            'orig (sigma=3, reduced)': per_inj['orig_refine_deltas'] or {},
            'mb (sigma=3, matched)':   per_inj['matched_budget_deltas'] or {},
            'wider (sigma=5, reduced)': per_inj['wider_reduced']['delta_params_unweighted'],
        }
        if RUN_BOTH_BUDGETS:
            cols['wider (sigma=5, matched)'] = per_inj['wider_matched']['delta_params_unweighted']
        print(f'\nESS(wider+reduced) = {ess_wr:.0f}/{len(wr_res.posterior)}')
        if RUN_BOTH_BUDGETS:
            print(f'ESS(wider+matched) = {ess_wm:.0f}/{len(wm_res.posterior)}')
        print()
        print_table(cols, list(cols.keys()))

    print('\n' + '=' * 78)
    print('WIDER-PRIOR TEST -- POPULATION SYNTHESIS')
    print('=' * 78)
    for key, rec in results['injections'].items():
        print(f'\n[{key}]  q={rec["q_target"]:.2f}  chi_eff={rec["chi_eff_target"]:+.2f}')
        cols = {
            'orig (sigma=3, reduced)': rec['orig_refine_deltas'] or {},
            'mb (sigma=3, matched)':   rec['matched_budget_deltas'] or {},
            'wider (sigma=5, reduced)': rec['wider_reduced']['delta_params_unweighted'],
        }
        if RUN_BOTH_BUDGETS and 'wider_matched' in rec:
            cols['wider (sigma=5, matched)'] = rec['wider_matched']['delta_params_unweighted']
        print_table(cols, list(cols.keys()))

    print(f'\nFull results: {out_json}')
    print('\nInterpretation:')
    print('  - If "wider (sigma=5, reduced)" collapses Δ below ~0.4 for all params,')
    print('    the production fix is: change safe_sigma multiplier 3.0 -> 5.0.')
    print('    No sampler-budget change needed; 35%% saving preserved.')
    print('  - If only the matched-budget version works, the §6.5 recommendation')
    print('    is "wider prior + matched budget for the high-spin / equal-mass corner".')
    print('  - If neither works, the failure mode is intrinsic to the framework')
    print('    for this (q, chi_eff) region and that gets stated as such.')


if __name__ == '__main__':
    main()
