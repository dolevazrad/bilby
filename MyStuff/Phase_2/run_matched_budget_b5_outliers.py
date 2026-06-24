#!/usr/bin/env python3
"""
B5 follow-up: matched-budget Refine on the worst B5 outliers.

Purpose
-------
B5's reduced-budget Refine (nlive=1024, walks=50) showed a residual
~0.4-1.0 sigma Delta_param bias on weakly-constrained parameters for a
handful of injections (highest |chi_eff|, lowest q). Importance
reweighting (recompute_b5_deltas_weighted.py) confirmed this is NOT a
prior-truncation artifact -- the Refine and broad priors overlap well in
the posterior region (ESS ~ 85-95%). The remaining suspect is the
reduced sampler budget.

This script holds the prior compression fixed but raises the Refine
budget back to Baseline (nlive=2048, walks=100) for just the worst
outliers, so you can compare:

    Refine (1024/50, compressed priors)   -> existing run, on disk
    Matched-budget Refine (2048/100, compressed priors)  -> this script
    Baseline (2048/100, broad priors)     -> existing run, on disk

If the matched-budget Refine collapses Delta_param close to Baseline-level
noise (Delta ~ 0.05-0.15), the section 6.5 story becomes:

  "Compressed priors alone preserve fidelity; the reduced sampler budget
   contributes the bulk of the speedup but introduces a tolerable spin /
   inclination bias on a minority of injections."

Re-uses the existing Scout result JSONs already on disk; no new Scout
runs. Each matched-budget Refine costs ~ Baseline wall-clock (~ 4 h on
i7-10700KF), so two outliers ~ 8 wall-hours.

Usage
-----
    python run_matched_budget_b5_outliers.py <b5_campaign_directory>

Optional: edit OUTLIER_INDICES below to add / remove injections.
"""

import json
import os
import sys
from datetime import datetime

import bilby
import numpy as np

from all_in_one_correct_phase2 import OUTPUT_BASE, create_informed_priors, find_asd_scenarios
from run_injection_campaign_b5 import (
    DELTA_PARAMS,
    build_injection_grid,
    get_broad_priors,
    run_pe_aligned,
)
from recompute_b5_deltas_weighted import (
    importance_weights,
    weighted_delta_params,
)

# ---------------------------------------------------------------------------
# Which injections to rerun. Default: the two worst B5 outliers.
# ---------------------------------------------------------------------------
OUTLIER_INDICES = [4, 5]


# ---------------------------------------------------------------------------
# Δ helpers (unweighted, for direct comparison with the original B5 numbers)
# ---------------------------------------------------------------------------
def unweighted_delta_params(refine_result, baseline_result, params=DELTA_PARAMS):
    out = {}
    for p in params:
        if p not in baseline_result.posterior or p not in refine_result.posterior:
            out[p] = None
            continue
        b = baseline_result.posterior[p].values
        r = refine_result.posterior[p].values
        b_std = float(np.std(b))
        if b_std == 0:
            out[p] = None
        else:
            out[p] = float(abs(np.median(r) - np.median(b)) / b_std)
    return out


def print_side_by_side(orig_unweighted, mb_unweighted, mb_weighted):
    print(f"{'param':<22s} {'orig Refine':>13s} {'matched-budget':>16s} {'mb weighted':>13s}")
    print('-' * 70)
    for p in DELTA_PARAMS:
        o = orig_unweighted.get(p) if orig_unweighted else None
        m = mb_unweighted.get(p) if mb_unweighted else None
        mw = mb_weighted.get(p) if mb_weighted else None

        def fmt(v):
            return f'{v:>13.3f}' if v is not None else f'{"--":>13s}'

        print(f'{p:<22s} {fmt(o)} {fmt(m).rjust(16)} {fmt(mw)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print('Usage: python run_matched_budget_b5_outliers.py <b5_campaign_directory>')
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

    # Output goes in a subdirectory of the original campaign so all artefacts
    # for this injection set live in one place.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(campaign_dir, f'matched_budget_outliers_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'matched_budget_outliers_results.json')
    print(f'Matched-budget output directory: {out_dir}')

    injections_grid = build_injection_grid(
        n=orig_master['config']['n_injections'],
        seed=orig_master['config']['random_seed'],
    )

    results = {'config': {
        'outlier_indices': OUTLIER_INDICES,
        'parent_campaign_dir': campaign_dir,
        'sampler_settings': 'nlive=2048, walks=100 (matched to Baseline)',
        'priors': 'Compressed (informed Gaussians from existing Scout)',
    }, 'injections': {}}

    for idx in OUTLIER_INDICES:
        key = f'inj_{idx:02d}'
        if idx >= len(injections_grid):
            print(f'  Skipping {key}: out of grid range (n={len(injections_grid)}).')
            continue
        inj = injections_grid[idx]

        print('\n' + '#' * 78)
        print(f'[{key}]  q={inj["mass_ratio"]:.3f}  chi_eff={inj["_chi_eff_target"]:+.3f}'
              '   (matched-budget Refine)')
        print('#' * 78)

        scout_path = os.path.join(campaign_dir, f'{key}_scout_result.json')
        baseline_path = os.path.join(campaign_dir, f'{key}_baseline_result.json')
        if not os.path.exists(scout_path):
            print(f'  Missing Scout result: {scout_path}; skipping.')
            continue
        if not os.path.exists(baseline_path):
            print(f'  Missing Baseline result: {baseline_path}; skipping.')
            continue

        scout_res = bilby.result.read_in_result(scout_path)
        baseline_res = bilby.result.read_in_result(baseline_path)
        informed = create_informed_priors(scout_res)

        mb_label = f'{key}_matched_budget_refine'
        mb_path = os.path.join(out_dir, f'{mb_label}_result.json')

        if os.path.exists(mb_path):
            print(f'  Reusing existing matched-budget result: {mb_path}')
            mb_res = bilby.result.read_in_result(mb_path)
            mb_time = float('nan')
        else:
            mb_res, mb_time = run_pe_aligned(
                full_files, mb_label, out_dir, inj, informed_priors=informed
            )

        # Δ vs Baseline -- both unweighted and importance-weighted.
        broad = get_broad_priors(inj)
        ln_w, ess = importance_weights(mb_res, broad)
        mb_unweighted = unweighted_delta_params(mb_res, baseline_res)
        mb_weighted = weighted_delta_params(mb_res, baseline_res, ln_w)

        orig_unweighted = (orig_master['injections'].get(key, {}) or {}).get('delta_params')

        # Persist per-injection results.
        results['injections'][key] = {
            'q_target': inj['mass_ratio'],
            'chi_eff_target': inj['_chi_eff_target'],
            'matched_budget_refine_time_s': float(mb_time),
            'matched_budget_log_bf': float(mb_res.log_bayes_factor),
            'baseline_log_bf': float(baseline_res.log_bayes_factor),
            'effective_sample_size': ess,
            'posterior_size': int(len(mb_res.posterior)),
            'delta_params_orig_refine': orig_unweighted,
            'delta_params_matched_budget': mb_unweighted,
            'delta_params_matched_budget_weighted': mb_weighted,
        }

        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  [Checkpoint saved: {out_json}]')

        print()
        print(f'  ESS: {ess:.0f} / {len(mb_res.posterior)}'
              f'   (ln(BF): MB={mb_res.log_bayes_factor:.2f}, '
              f'Baseline={baseline_res.log_bayes_factor:.2f})')
        print_side_by_side(orig_unweighted, mb_unweighted, mb_weighted)

    # ------------------------------------------------------------------
    # Final synthesis across all reruns.
    # ------------------------------------------------------------------
    print('\n' + '=' * 78)
    print('MATCHED-BUDGET OUTLIER SUMMARY')
    print('=' * 78)
    for key, r in results['injections'].items():
        print(f'\n[{key}]  q={r["q_target"]:.2f}  chi_eff={r["chi_eff_target"]:+.2f}')
        print_side_by_side(
            r['delta_params_orig_refine'],
            r['delta_params_matched_budget'],
            r['delta_params_matched_budget_weighted'],
        )

    print(f'\nFull results: {out_json}')
    print('\nInterpretation: if "matched-budget" Δ columns are uniformly close to 0.1-0.2,')
    print('the residual B5 bias was due to the reduced sampler budget, not the priors.')
    print('In that case §6.5 can include a recommendation: matched-budget Refine for')
    print('users who need tight spin recovery; full reduced-budget Refine for those')
    print('who only need posterior shape + ln(BF).')


if __name__ == '__main__':
    main()
