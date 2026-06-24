#!/usr/bin/env python3
"""
Post-hoc importance reweighting of the B5 campaign Delta_param values.

Background
----------
``run_injection_campaign_b5.py`` records ln(BF) corrected to the broad prior
(via ``Re_Weight_Posterior``) but the Delta_param fidelity metric uses the
*raw* Refine posterior samples. Because Refine samples come from the
truncated prior (chirp_mass, mass_ratio, luminosity_distance, geocent_time
replaced with TruncatedGaussian), the resulting marginals for spin and
inclination carry a residual truncation bias even when the joint posterior
is recoverable.

This script fixes that by importance-reweighting each Refine sample back to
the broad prior, then recomputing Delta_param with weighted medians. The
underlying campaign data is untouched -- the weighted version is written to
new files alongside the originals.

Usage
-----
    python recompute_b5_deltas_weighted.py <campaign_directory>

Outputs (written into the campaign directory):

  - ``b5_campaign_results_weighted.json``  -- mirror of the original JSON
                                              with ``delta_params`` replaced
                                              by weighted values and a
                                              ``effective_sample_size`` field
                                              added per injection.
  - ``b5_delta_param_boxplot_weighted.pdf/.png`` -- updated section 6.5 figure.
  - ``b5_summary_table_weighted.tex``      -- updated LaTeX table.
"""

import json
import os
import sys

import bilby
import numpy as np
from scipy.special import logsumexp

# Re-use the broad prior definition from the campaign script so we don't drift.
from run_injection_campaign_b5 import DELTA_PARAMS, build_injection_grid, get_broad_priors

# Re-use the plotting routines.
from plot_b5_box import (
    PARAM_LABELS,  # noqa: F401  -- kept for symmetry; not used directly
    make_box_plot,
    make_grid_scatter,
    write_latex_table,
)


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------
def weighted_quantile(values, weights, q=0.5):
    """Weighted quantile via cumulative interpolation. ``q`` in [0, 1]."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float('nan')
    v = values[finite]
    w = weights[finite]
    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]
    cw = np.cumsum(w_sorted)
    cw /= cw[-1]
    return float(np.interp(q, cw, v_sorted))


def importance_weights(refine_result, broad_priors):
    """Return (ln_weights, effective_sample_size) for the Refine posterior."""
    scout_priors = refine_result.priors
    posterior = refine_result.posterior
    valid_keys = list(broad_priors.keys())
    n = len(posterior)
    ln_w = np.full(n, -np.inf)
    for i in range(n):
        raw = dict(posterior.iloc[i])
        sample = {k: raw[k] for k in valid_keys if k in raw}
        ln_p_broad = broad_priors.ln_prob(sample)
        ln_p_refine = scout_priors.ln_prob(sample)
        if np.isfinite(ln_p_broad) and np.isfinite(ln_p_refine):
            ln_w[i] = ln_p_broad - ln_p_refine
    # Effective sample size (Kish): (sum w)^2 / sum(w^2)
    if np.all(~np.isfinite(ln_w)):
        return ln_w, 0.0
    ln_w_shift = ln_w - np.max(ln_w[np.isfinite(ln_w)])
    w = np.exp(ln_w_shift)
    w[~np.isfinite(w)] = 0.0
    if w.sum() == 0:
        return ln_w, 0.0
    ess = float((w.sum() ** 2) / (w ** 2).sum())
    return ln_w, ess


def weighted_delta_params(refine_result, baseline_result, ln_w, params=DELTA_PARAMS):
    """|w_median_refine - median_baseline| / std_baseline per parameter."""
    # Convert log weights to normalised linear weights once.
    finite_mask = np.isfinite(ln_w)
    if not finite_mask.any():
        return {p: None for p in params}
    ln_w_shift = ln_w.copy()
    ln_w_shift[~finite_mask] = -np.inf
    ln_w_shift -= np.max(ln_w_shift[finite_mask])
    weights = np.exp(ln_w_shift)
    weights[~np.isfinite(weights)] = 0.0
    if weights.sum() == 0:
        return {p: None for p in params}

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
            continue
        b_med = float(np.median(b))
        r_med_weighted = weighted_quantile(r, weights, q=0.5)
        if not np.isfinite(r_med_weighted):
            out[p] = None
        else:
            out[p] = float(abs(r_med_weighted - b_med) / b_std)
    return out


# ---------------------------------------------------------------------------
# Campaign processing
# ---------------------------------------------------------------------------
def process(campaign_dir):
    json_path = os.path.join(campaign_dir, 'b5_campaign_results.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'No campaign JSON at {json_path}')

    with open(json_path, 'r') as f:
        master = json.load(f)

    # Re-derive injection grid so the broad prior matches what the run used.
    injections_grid = build_injection_grid(
        n=master['config']['n_injections'],
        seed=master['config']['random_seed'],
    )

    out = json.loads(json.dumps(master))  # deep copy
    out['config']['reweighted'] = True

    for idx, inj in enumerate(injections_grid):
        key = f'inj_{idx:02d}'
        rec = out['injections'].get(key)
        if rec is None or not rec.get('complete'):
            continue

        r_path = os.path.join(campaign_dir, f'{key}_refine_result.json')
        b_path = os.path.join(campaign_dir, f'{key}_baseline_result.json')
        if not (os.path.exists(r_path) and os.path.exists(b_path)):
            print(f'  [{key}] missing result files; leaving unweighted values.')
            continue

        r_res = bilby.result.read_in_result(r_path)
        b_res = bilby.result.read_in_result(b_path)
        broad = get_broad_priors(inj)

        ln_w, ess = importance_weights(r_res, broad)
        rec['delta_params_unweighted'] = rec.get('delta_params')
        rec['delta_params'] = weighted_delta_params(r_res, b_res, ln_w)
        rec['effective_sample_size'] = ess
        rec['posterior_size'] = int(len(r_res.posterior))

        print(
            f'  [{key}] q={rec["q_target"]:.2f} chi_eff={rec["chi_eff_target"]:+.2f} '
            f'ESS={ess:.0f}/{len(r_res.posterior)}'
        )
        for p in DELTA_PARAMS:
            u = (rec['delta_params_unweighted'] or {}).get(p)
            w = rec['delta_params'].get(p)
            if u is not None and w is not None:
                print(f'    {p:<22s} unweighted={u:.3f}  weighted={w:.3f}')

    # ------------------------------------------------------------------
    # Recompute population summary using weighted deltas.
    # ------------------------------------------------------------------
    completed = [r for r in out['injections'].values() if r.get('complete')]
    deltas_by_param = {p: [] for p in DELTA_PARAMS}
    for r in completed:
        for p in DELTA_PARAMS:
            v = (r.get('delta_params') or {}).get(p)
            if v is not None:
                deltas_by_param[p].append(v)

    summary = {}
    for p, values in deltas_by_param.items():
        if values:
            summary[p] = {
                'n': len(values),
                'median': float(np.median(values)),
                'p90': float(np.percentile(values, 90)),
                'max': float(np.max(values)),
            }
    time_savings = [r['time_savings_pct'] for r in completed if 'time_savings_pct' in r]
    if time_savings:
        summary['time_savings_pct'] = {
            'median': float(np.median(time_savings)),
            'p10': float(np.percentile(time_savings, 10)),
            'p90': float(np.percentile(time_savings, 90)),
        }
    out['summary'] = summary

    out_json = os.path.join(campaign_dir, 'b5_campaign_results_weighted.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote weighted JSON: {out_json}')

    # ------------------------------------------------------------------
    # Regenerate plots and LaTeX table from weighted Δ values.
    # ------------------------------------------------------------------
    deltas_for_plot = {p: vals for p, vals in deltas_by_param.items()}
    out_pdf_box = os.path.join(campaign_dir, 'b5_delta_param_boxplot_weighted.pdf')
    out_png_box = os.path.join(campaign_dir, 'b5_delta_param_boxplot_weighted.png')
    out_pdf_scatter = os.path.join(campaign_dir, 'b5_q_chi_eff_scatter_weighted.pdf')
    out_tex = os.path.join(campaign_dir, 'b5_summary_table_weighted.tex')

    make_box_plot(deltas_for_plot, out_pdf_box, out_png_box)
    make_grid_scatter(out, out_pdf_scatter)
    write_latex_table(deltas_for_plot, out, out_tex)

    print(f'Wrote weighted box plot:     {out_pdf_box}')
    print(f'Wrote weighted PNG preview:  {out_png_box}')
    print(f'Wrote weighted grid scatter: {out_pdf_scatter}')
    print(f'Wrote weighted LaTeX table:  {out_tex}')

    # ------------------------------------------------------------------
    # Print headline numbers side-by-side.
    # ------------------------------------------------------------------
    print('\n' + '=' * 78)
    print('WEIGHTED VS UNWEIGHTED SUMMARY (median | p90 | max)')
    print('=' * 78)
    print(f"{'param':<22s} {'unweighted':>32s}   {'weighted':>32s}")
    print('-' * 78)
    orig_summary = master.get('summary', {})
    for p in DELTA_PARAMS:
        if p not in summary:
            continue
        u = orig_summary.get(p, {})
        w = summary[p]
        u_str = (f"{u.get('median', float('nan')):.3f} | "
                 f"{u.get('p90', float('nan')):.3f} | "
                 f"{u.get('max', float('nan')):.3f}") if u else '--'
        w_str = f"{w['median']:.3f} | {w['p90']:.3f} | {w['max']:.3f}"
        print(f"{p:<22s} {u_str:>32s}   {w_str:>32s}")
    if 'time_savings_pct' in summary:
        ts = summary['time_savings_pct']
        print(f"\nTime savings (unchanged): median {ts['median']:.1f}%  "
              f"(p10-p90: {ts['p10']:.1f}%-{ts['p90']:.1f}%)")


def main():
    if len(sys.argv) < 2:
        print('Usage: python recompute_b5_deltas_weighted.py <campaign_directory>')
        sys.exit(1)
    process(sys.argv[1])


if __name__ == '__main__':
    main()
