#!/usr/bin/env python3
"""
B5 (thesis_fix_plan.md): Small injection campaign at 150 Mpc.

Twenty injections at d_L = 150 Mpc sampled on a quasi-random (Latin Hypercube)
grid in (q, chi_eff) with q in [0.3, 1.0] and chi_eff in [-0.5, 0.5]. Aligned
spins only (tilt_1, tilt_2 fixed to 0 or pi according to the sign of chi_eff,
|a_1| = |a_2| = |chi_eff|). Each injection is processed with both pipelines:

  - Baseline   : broad priors, production budget.
  - Phase 2    : Scout (cheap scan) + Refine (informed truncated-Gaussian
                 priors, reduced budget) + importance reweighting back to the
                 broad prior so the reported ln(BF) is honest.

The Delta_param fidelity metric is computed per parameter against the Baseline
per injection. The output JSON is the input to ``plot_b5_box.py`` which
produces the section 6.5 box-plot figure.

Approximant: ``IMRPhenomXAS`` (aligned-spin, ~3x cheaper than ``IMRPhenomXPHM``)
so each Baseline run lands near 4 wall-clock hours on the i7-10700KF, matching
the ~160 CPU-h budget in the fix plan.

This script is self-contained: it does NOT modify ``all_in_one_correct_phase2``;
it imports the supporting utilities (ASD discovery, informed priors, reweighting)
and defines a B5-specific PE entry point that swaps the waveform approximant.

Safe to interrupt: the master JSON is rewritten after every injection so a
crashed or killed run resumes from the next pending injection on restart.
"""

import json
import os
import sys
import time
from datetime import datetime

import bilby
import numpy as np
import pickle
from bilby.core.prior import Cosine, Sine, Uniform
from bilby.gw.detector import PowerSpectralDensity
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from scipy.stats import qmc

# Reuse the existing pipeline helpers. We only override the bits that need to
# change for the aligned-spin, multi-injection setting (waveform approximant
# and injection parameter generator).
from all_in_one_correct_phase2 import (
    OUTPUT_BASE,
    create_informed_priors,
    find_asd_scenarios,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_INJECTIONS = 20
DISTANCE_MPC = 150.0
Q_RANGE = (0.3, 1.0)
CHI_EFF_RANGE = (-0.5, 0.5)
WAVEFORM_APPROXIMANT = 'IMRPhenomXAS'  # aligned-spin, cheap
RANDOM_SEED = 20260620  # deterministic LHS so the grid is reproducible

# Parameters tracked in the Delta_param fidelity panel.
DELTA_PARAMS = (
    'chirp_mass',
    'mass_ratio',
    'luminosity_distance',
    'chi_1',
    'chi_2',
    'theta_jn',
)


# ---------------------------------------------------------------------------
# Injection-grid generation
# ---------------------------------------------------------------------------
def build_injection_grid(n=N_INJECTIONS, seed=RANDOM_SEED):
    """Latin-Hypercube grid of (q, chi_eff) injection points.

    Returns a list of dicts, each a full ``injection_parameters`` ready for
    bilby. Spins are aligned: |a_1| = |a_2| = |chi_eff|, tilts in {0, pi}.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    raw = sampler.random(n=n)
    qs = Q_RANGE[0] + raw[:, 0] * (Q_RANGE[1] - Q_RANGE[0])
    chi_effs = CHI_EFF_RANGE[0] + raw[:, 1] * (CHI_EFF_RANGE[1] - CHI_EFF_RANGE[0])

    injections = []
    for q, chi_eff in zip(qs, chi_effs):
        # Aligned-spin parameterisation: bilby's converter maps chi_i to
        # a_i = |chi_i|, tilt_i = 0 or pi, phi_12 = phi_jl = 0, which is what
        # IMRPhenomXAS requires (zero transverse spin components).
        injections.append({
            'chirp_mass': 30.0,
            'mass_ratio': float(q),
            'luminosity_distance': DISTANCE_MPC,
            'chi_1': float(chi_eff),
            'chi_2': float(chi_eff),
            'theta_jn': 0.8,
            'phase': 1.0,
            'ra': 1.5,
            'dec': -1.0,
            'psi': 2.5,
            'geocent_time': 1126259462.0,
            '_chi_eff_target': float(chi_eff),  # bookkeeping only; not used by bilby
        })
    return injections


def get_broad_priors(injection_params):
    """Broad priors for the aligned-spin B5 campaign.

    Uses ``chi_1`` and ``chi_2`` in place of the precessing spin parameters
    (``a_1``, ``a_2``, ``tilt_1``, ``tilt_2``, ``phi_12``, ``phi_jl``).
    Bilby's ``convert_to_lal_binary_black_hole_parameters`` maps each
    ``chi_i`` to ``a_i = |chi_i|``, ``tilt_i = 0`` (if chi >= 0) or ``pi``
    (if chi < 0), and pins ``phi_12 = phi_jl = 0``. This guarantees zero
    transverse spin components, which is required by ``IMRPhenomXAS``.
    """
    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    priors['chirp_mass'] = Uniform(25.0, 35.0, name='chirp_mass')
    priors['mass_ratio'] = Uniform(0.25, 1.0, name='mass_ratio')
    priors['luminosity_distance'] = Uniform(1.0, 10000.0, name='luminosity_distance')
    priors['geocent_time'] = Uniform(
        injection_params['geocent_time'] - 0.1,
        injection_params['geocent_time'] + 0.1,
        name='geocent_time',
    )
    priors['phase'] = Uniform(0, 2 * np.pi, name='phase', boundary='periodic')
    priors['theta_jn'] = Sine(name='theta_jn')
    priors['ra'] = Uniform(0, 2 * np.pi, name='ra', boundary='periodic')
    priors['dec'] = Cosine(name='dec')
    priors['psi'] = Uniform(0, np.pi, name='psi', boundary='periodic')
    # Flat aligned-spin priors (override AlignedSpin defaults for simplicity).
    priors['chi_1'] = Uniform(-0.99, 0.99, name='chi_1', latex_label=r'$\chi_1$')
    priors['chi_2'] = Uniform(-0.99, 0.99, name='chi_2', latex_label=r'$\chi_2$')
    return priors


# ---------------------------------------------------------------------------
# Aligned-spin PE runner
# ---------------------------------------------------------------------------
def run_pe_aligned(asd_files, label, outdir, injection_params, informed_priors=None):
    """Aligned-spin variant of ``run_pe``. Same I/O contract as the original.

    The only differences from the production ``run_pe`` are:
      - ``waveform_approximant`` is ``IMRPhenomXAS`` (aligned, ~3x cheaper).
      - ``injection_params`` is mandatory (this is a campaign, not a one-off).

    Budget/labelling rules follow the existing convention:
      - label contains ``scout`` -> Scout settings (cheap).
      - informed_priors is not None -> Refine settings.
      - otherwise -> Baseline settings (production budget).
    """
    print(f'\nRunning {label} PE...')

    # 1. Load the ASDs.
    asds = {}
    for det in ('H1', 'L1'):
        with open(asd_files[det], 'rb') as f:
            data = pickle.load(f)
        asds[det] = data['asd']

    # 2. Waveform generator (aligned-spin approximant).
    duration = 4
    sampling_frequency = 2048
    minimum_frequency = 20
    waveform_arguments = dict(
        waveform_approximant=WAVEFORM_APPROXIMANT,
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    # 3. Interferometers + PSD interpolation + injection.
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    n_freq = int(duration * sampling_frequency / 2) + 1
    frequencies = np.linspace(0, sampling_frequency / 2, n_freq)
    for ifo in ifos:
        asd = asds[ifo.name]
        interp = interp1d(
            asd.frequencies.value, asd.value,
            bounds_error=False, fill_value='extrapolate',
        )
        interpolated_asd = interp(frequencies)
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = sampling_frequency / 2
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration
        ifo.start_time = injection_params['geocent_time'] - duration + 0.5
        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=frequencies, psd_array=interpolated_asd ** 2,
        )
        ifo.strain_data.roll_off = 0.2
        ifo.strain_data.set_from_frequency_domain_strain(
            sampling_frequency=sampling_frequency,
            duration=duration,
            frequency_domain_strain=np.zeros(n_freq, dtype=complex),
        )
        # Strip bookkeeping keys before passing to bilby.
        clean_inj = {k: v for k, v in injection_params.items() if not k.startswith('_')}
        ifo.inject_signal(parameters=clean_inj, waveform_generator=waveform_generator)

    # 4. Priors. Same 15D structure as production so the rest of the pipeline
    #    (Re_Weight_Posterior, create_informed_priors, fidelity calc) is unchanged.
    priors = get_broad_priors(injection_params)

    if informed_priors is None:
        if 'scout' in label.lower():
            print('--- PHASE 1: SCOUT (Fast) ---')
            sampler_settings = {'npoints': 500, 'walks': 50}
            dlogz_val = 0.5
        else:
            print('--- BASELINE (Production budget) ---')
            sampler_settings = {'npoints': 2048, 'walks': 100}
            dlogz_val = 0.1
    else:
        priors.update(informed_priors)
        print('--- PHASE 2: REFINE (Informed priors, reduced budget) ---')
        sampler_settings = {'npoints': 1024, 'walks': 50}
        dlogz_val = 0.1

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator,
    )

    start = time.time()
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        outdir=outdir,
        label=label,
        injection_parameters={k: v for k, v in injection_params.items() if not k.startswith('_')},
        save=True,
        dlogz=dlogz_val,
        sample='rwalk',
        bound='multi',
        npool=1,
        check_point=False,
        print_progress=True,
        **sampler_settings,
    )
    runtime = time.time() - start
    print(f'{label} completed in {runtime / 3600:.2f} h')

    try:
        result.plot_corner()
    except Exception as exc:
        print(f'  (corner plot failed, non-critical: {exc})')

    return result, runtime


# ---------------------------------------------------------------------------
# Importance reweighting (local copy with explicit prior dict to avoid the
# wide-prior key-mismatch corner case that bit B3).
# ---------------------------------------------------------------------------
def reweight_to_broad(refined_result, broad_priors):
    """Importance-reweight refined posterior back onto the broad prior."""
    print('Reweighting Refine evidence to broad prior...')
    scout_priors = refined_result.priors
    posterior = refined_result.posterior
    valid_keys = list(broad_priors.keys())
    ln_w = np.zeros(len(posterior))
    for i in range(len(posterior)):
        raw = dict(posterior.iloc[i])
        clean = {k: raw[k] for k in valid_keys if k in raw}
        ln_w[i] = broad_priors.ln_prob(clean) - scout_priors.ln_prob(clean)
    shift = logsumexp(ln_w) - np.log(len(ln_w))
    return float(refined_result.log_bayes_factor + shift)


# ---------------------------------------------------------------------------
# Delta_param fidelity
# ---------------------------------------------------------------------------
def delta_params(refined_result, baseline_result, params=DELTA_PARAMS):
    """|median_refined - median_baseline| / std_baseline for each parameter."""
    out = {}
    for p in params:
        if p not in baseline_result.posterior or p not in refined_result.posterior:
            out[p] = None
            continue
        b = baseline_result.posterior[p].values
        r = refined_result.posterior[p].values
        b_std = float(np.std(b))
        if b_std == 0:
            out[p] = None
        else:
            out[p] = float(abs(np.median(r) - np.median(b)) / b_std)
    return out


# ---------------------------------------------------------------------------
# Resumable main loop
# ---------------------------------------------------------------------------
def main():
    print('=' * 78)
    print(f'B5 INJECTION CAMPAIGN: {N_INJECTIONS} injections at {DISTANCE_MPC} Mpc')
    print(f'q in {Q_RANGE} | chi_eff in {CHI_EFF_RANGE} | {WAVEFORM_APPROXIMANT}')
    print('=' * 78)

    full_files, half_files, _ = find_asd_scenarios()
    if not full_files:
        print('ERROR: no ASD scenarios found. Aborting.')
        sys.exit(1)

    # Output directory is deterministic on first call, then reused on resume.
    # The user can pass an explicit dir as argv[1] to resume a specific run.
    if len(sys.argv) > 1:
        outdir = sys.argv[1]
        if not os.path.isdir(outdir):
            print(f'ERROR: resume directory not found: {outdir}')
            sys.exit(1)
        print(f'Resuming into existing directory: {outdir}')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = os.path.join(OUTPUT_BASE, f'b5_injection_campaign_{timestamp}')
        os.makedirs(outdir, exist_ok=True)
        print(f'Output directory: {outdir}')

    json_path = os.path.join(outdir, 'b5_campaign_results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            master = json.load(f)
        print(f'Loaded existing checkpoint ({len(master.get("injections", {}))} entries).')
    else:
        master = {
            'config': {
                'n_injections': N_INJECTIONS,
                'distance_mpc': DISTANCE_MPC,
                'q_range': list(Q_RANGE),
                'chi_eff_range': list(CHI_EFF_RANGE),
                'waveform_approximant': WAVEFORM_APPROXIMANT,
                'random_seed': RANDOM_SEED,
                'delta_params': list(DELTA_PARAMS),
            },
            'injections': {},
        }

    injections = build_injection_grid()
    for idx, inj in enumerate(injections):
        key = f'inj_{idx:02d}'
        if key in master['injections'] and master['injections'][key].get('complete'):
            print(f'[{key}] already complete; skipping.')
            continue

        print('\n' + '#' * 78)
        print(f'[{key}] q={inj["mass_ratio"]:.3f} chi_eff={inj["_chi_eff_target"]:+.3f}')
        print('#' * 78)

        record = master['injections'].get(key, {
            'q_target': inj['mass_ratio'],
            'chi_eff_target': inj['_chi_eff_target'],
            'complete': False,
        })

        try:
            # --- Baseline --------------------------------------------------
            b_label = f'{key}_baseline'
            b_path = os.path.join(outdir, f'{b_label}_result.json')
            if os.path.exists(b_path):
                print(f'  Reusing existing Baseline result: {b_path}')
                b_res = bilby.result.read_in_result(b_path)
                b_time = record.get('baseline_time_s', float('nan'))
            else:
                b_res, b_time = run_pe_aligned(full_files, b_label, outdir, inj)
            record['baseline_log_bf'] = float(b_res.log_bayes_factor)
            record['baseline_time_s'] = float(b_time)

            # --- Scout -----------------------------------------------------
            s_label = f'{key}_scout'
            s_path = os.path.join(outdir, f'{s_label}_result.json')
            if os.path.exists(s_path):
                print(f'  Reusing existing Scout result: {s_path}')
                s_res = bilby.result.read_in_result(s_path)
                s_time = record.get('scout_time_s', float('nan'))
            else:
                s_res, s_time = run_pe_aligned(half_files, s_label, outdir, inj)
            record['scout_time_s'] = float(s_time)

            # --- Refine ----------------------------------------------------
            r_label = f'{key}_refine'
            r_path = os.path.join(outdir, f'{r_label}_result.json')
            informed = create_informed_priors(s_res)
            if os.path.exists(r_path):
                print(f'  Reusing existing Refine result: {r_path}')
                r_res = bilby.result.read_in_result(r_path)
                r_time = record.get('refine_time_s', float('nan'))
            else:
                r_res, r_time = run_pe_aligned(full_files, r_label, outdir, inj,
                                               informed_priors=informed)
            record['refine_time_s'] = float(r_time)

            # --- Math ------------------------------------------------------
            broad = get_broad_priors(inj)
            honest_bf = reweight_to_broad(r_res, broad)
            record['phase2_total_time_s'] = float(record['scout_time_s'] + record['refine_time_s'])
            record['phase2_honest_log_bf'] = honest_bf
            record['delta_log_bf'] = honest_bf - record['baseline_log_bf']
            record['delta_params'] = delta_params(r_res, b_res)
            record['time_savings_pct'] = float(
                (record['baseline_time_s'] - record['phase2_total_time_s'])
                / record['baseline_time_s'] * 100
            )
            record['complete'] = True

            print(f'  Baseline ln(BF)    = {record["baseline_log_bf"]:.2f}')
            print(f'  Phase 2 honest ln(BF) = {honest_bf:.2f} '
                  f'(delta = {record["delta_log_bf"]:+.2f})')
            print(f'  Time savings       = {record["time_savings_pct"]:.1f}%')
            print(f'  Delta_param        = {record["delta_params"]}')

        except Exception as exc:
            # Don't lose the whole campaign to one failed injection. Record
            # the error and move on; the JSON checkpoint preserves progress.
            record['complete'] = False
            record['error'] = repr(exc)
            print(f'  ERROR on {key}: {exc!r}')

        master['injections'][key] = record
        with open(json_path, 'w') as f:
            json.dump(master, f, indent=2)
        print(f'  [Checkpoint saved: {json_path}]')

    # ---- Final aggregate summary --------------------------------------
    completed = [r for r in master['injections'].values() if r.get('complete')]
    n_done = len(completed)
    print('\n' + '=' * 78)
    print(f'CAMPAIGN COMPLETE: {n_done}/{N_INJECTIONS} injections finished')
    print('=' * 78)
    if n_done == 0:
        print('No injections produced usable results. Inspect the JSON for errors.')
        return

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
    master['summary'] = summary

    time_savings = [r['time_savings_pct'] for r in completed if 'time_savings_pct' in r]
    if time_savings:
        master['summary']['time_savings_pct'] = {
            'median': float(np.median(time_savings)),
            'p10': float(np.percentile(time_savings, 10)),
            'p90': float(np.percentile(time_savings, 90)),
        }

    with open(json_path, 'w') as f:
        json.dump(master, f, indent=2)

    print('Delta_param summary (median | p90 | max):')
    for p, s in summary.items():
        if p == 'time_savings_pct':
            continue
        print(f'  {p:<22s} {s["median"]:.3f} | {s["p90"]:.3f} | {s["max"]:.3f}')
    if 'time_savings_pct' in master['summary']:
        ts = master['summary']['time_savings_pct']
        print(f'Time savings: median {ts["median"]:.1f}% (p10-p90: '
              f'{ts["p10"]:.1f}% to {ts["p90"]:.1f}%)')
    print(f'\nFull results: {json_path}')
    print('Run plot_b5_box.py with this directory to generate the section 6.5 figure.')


if __name__ == '__main__':
    main()
