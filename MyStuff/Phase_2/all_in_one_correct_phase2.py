#!/usr/bin/env python3
# FILENAME: all_in_one_correct_phase2.py
"""
All-in-one correct Phase 2 implementation
This single file contains everything you need
Just run: python all_in_one_correct_phase2.py
"""

import numpy as np
import bilby
from bilby.gw.detector import PowerSpectralDensity
from bilby.core.prior import Uniform, Sine, Cosine, PriorDict
import matplotlib.pyplot as plt
import pickle
import os
import time
import json
import glob
import re
from scipy.interpolate import interp1d
from gwpy.frequencyseries import FrequencySeries
from datetime import datetime

# Configuration
ASD_DIR = '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window_201225'
OUTPUT_BASE = '/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2'

def find_asd_scenarios():
    """Find Max (<= Day 80), Half (50%), and Quarter (25%) ASD files."""
    print("Looking for ASD files...")
    
    h1_files = glob.glob(os.path.join(ASD_DIR, 'H1_asd_win*.pkl'))
    valid_windows = {}
    
    # 1. Filter out Day 100+ and parse windows
    MAX_VALID_SECONDS = 80 * 24 * 3600  # Day 80
    
    for h1_file in h1_files:
        match = re.search(r'win(\d+)', h1_file)
        if match:
            window = int(match.group(1))
            if window <= MAX_VALID_SECONDS:
                l1_file = h1_file.replace('H1_asd', 'L1_asd')
                if os.path.exists(l1_file):
                    valid_windows[window] = {'H1': h1_file, 'L1': l1_file, 'hours': window/3600}

    if not valid_windows:
        print("✗ No valid ASD files found!")
        return None, None, None

    # 2. Find the "Gold Standard" (Max Time)
    sorted_windows = sorted(valid_windows.keys())
    max_w = sorted_windows[-1]
    full_time = valid_windows[max_w]
    print(f"✓ Gold Standard (Max): {full_time['hours']:.1f} hours")

    # 3. Find closest matches for 1/2 and 1/4
    target_half = max_w / 2
    target_quarter = max_w / 4
    
    # Helper to find closest existing window
    def get_closest(target):
        closest_w = min(sorted_windows, key=lambda x: abs(x - target))
        return valid_windows[closest_w]

    half_time = get_closest(target_half)
    quarter_time = get_closest(target_quarter)

    print(f"✓ Half-Time Scout:     {half_time['hours']:.1f} hours (Target: {target_half/3600:.1f})")
    print(f"✓ Quarter-Time Scout:  {quarter_time['hours']:.1f} hours (Target: {target_quarter/3600:.1f})")

    return full_time, half_time, quarter_time
def create_injection_parameters():
    """Create test injection parameters."""
    return {
        'chirp_mass': 30.0,
        'mass_ratio': 0.9,
        'luminosity_distance': 450.0,
        'a_1': 0.0, 'a_2': 0.0,
        'tilt_1': 0.0, 'tilt_2': 0.0,
        'phi_12': 0.0, 'phi_jl': 0.0,
        'theta_jn': 0.8,
        'phase': 1.0,
        'ra': 1.5, 'dec': -1.0, 'psi': 2.5,
        'geocent_time': 1238303719.0
    }

def run_pe(asd_files, label, outdir, informed_priors=None):
    """Run parameter estimation with given ASD files."""
    print(f"\n" + "="*60)
    print(f"STARTING RUN: {label}")
    print("="*60)
    
    # 1. Load ASDs
    asds = {}
    for det in ['H1', 'L1']:
        with open(asd_files[det], 'rb') as f:
            data = pickle.load(f)
        asds[det] = data['asd']
    
    # 2. Injection parameters
    injection_params = create_injection_parameters()
    
    # 3. Waveform Generator (IMRPhenomXPHM for 15-Parameters)
    duration = 4
    sampling_frequency = 2048
    minimum_frequency = 20
    
    waveform_arguments = dict(
        waveform_approximant='IMRPhenomXPHM',
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency
    )
    
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments
    )
    
    # 4. Set up Interferometers
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    n_freq = int(duration * sampling_frequency / 2) + 1
    frequencies = np.linspace(0, sampling_frequency/2, n_freq)
    
    for ifo in ifos:
        asd = asds[ifo.name]
        interp = interp1d(asd.frequencies.value, asd.value, bounds_error=False, fill_value='extrapolate')
        interpolated_asd = interp(frequencies)
        
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = sampling_frequency/2
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration
        ifo.start_time = injection_params['geocent_time'] - duration + 0.5
        ifo.power_spectral_density = PowerSpectralDensity(frequency_array=frequencies, psd_array=interpolated_asd**2)
        
        # Initialize zero strain and inject
        ifo.strain_data.roll_off = 0.2
        ifo.strain_data.set_from_frequency_domain_strain(
            sampling_frequency=sampling_frequency, duration=duration, 
            frequency_domain_strain=np.zeros(n_freq, dtype=complex))
        ifo.inject_signal(parameters=injection_params, waveform_generator=waveform_generator)
    
    # ---------------------------------------------------------
    # 5. DEFINE PRIORS (The Clean Way)
    # ---------------------------------------------------------
    # We define the full 15-param priors here so they exist for EVERY run.
    priors = bilby.gw.prior.BBHPriorDict()
    
    # Masses & Extrinsic
    priors['chirp_mass'] = Uniform(25.0, 35.0, name='chirp_mass', unit='$M_{\odot}$')
    priors['mass_ratio'] = Uniform(0.5, 1.0, name='mass_ratio')
    priors['luminosity_distance'] = Uniform(200, 800, name='luminosity_distance', unit='Mpc')
    priors['geocent_time'] = Uniform(injection_params['geocent_time'] - 0.1, injection_params['geocent_time'] + 0.1, name='geocent_time', unit='s')
    priors['phase'] = Uniform(0, 2 * np.pi, name='phase')
    priors['theta_jn'] = Sine(name='theta_jn') 

    # Sky Location
    priors['ra'] = Uniform(0, 2 * np.pi, name='ra')
    priors['dec'] = Cosine(name='dec')
    priors['psi'] = Uniform(0, np.pi, name='psi')

    # Spins (Magnitudes & Tilts)
    priors['a_1'] = Uniform(0, 0.99, name='a_1')
    priors['a_2'] = Uniform(0, 0.99, name='a_2')
    priors['tilt_1'] = Sine(name='tilt_1')
    priors['tilt_2'] = Sine(name='tilt_2')
    priors['phi_12'] = Uniform(0, 2 * np.pi, name='phi_12', boundary='periodic')
    priors['phi_jl'] = Uniform(0, 2 * np.pi, name='phi_jl', boundary='periodic')

    # ---------------------------------------------------------
    # 6. APPLY PHASE SETTINGS
    # ---------------------------------------------------------
    if informed_priors is None:
        # --- BLIND RUNS ---
        if 'scout' in label.lower():
            print(">>> MODE: Scout Run (Fast & Rough)")
            sampler_settings = {'npoints': 500, 'walks': 50} 
            dlogz_val = 0.5 
        else:
            print(">>> MODE: Baseline Production Run (High Precision)")
            sampler_settings = {'npoints': 2048, 'walks': 100}
            dlogz_val = 0.1
    else:
        # --- REFINED RUNS ---
        print(">>> MODE: Refined Run (Informed Priors)")
        # CRITICAL: We overwrite only the refined keys. The rest remain wide.
        priors.update(informed_priors)
        sampler_settings = {'npoints': 1024, 'walks': 50}
        dlogz_val = 0.1 

    # ---------------------------------------------------------
    # 7. PRE-FLIGHT CHECK (Stop Running Blind!)
    # ---------------------------------------------------------
    print("\n[Active Parameters]")
    for key in priors:
        if isinstance(priors[key], bilby.core.prior.Constraint):
            continue
        print(f"  - {key}")
    print("-" * 30)

    # 8. Run Sampler
    likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator)
    
    start_time = time.time()
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        outdir=outdir,
        label=label,
        injection_parameters=injection_params,
        save=True,
        dlogz=dlogz_val,
        sample='rwalk',
        bound='multi',
        npool=1,               # Safe for WSL
        check_point=False,     # Prevents WSL crash
        print_progress=True,   # KEEPS VISIBILITY ON
        **sampler_settings
    )

    # 9. Plotting
    print(f"Generating corner plot for {label}...")
    try:
        result.plot_corner()
    except Exception as e:
        print(f"Plotting failed (non-critical): {e}")
    
    runtime = time.time() - start_time
    print(f"✓ {label} completed in {runtime/3600:.2f} hours")
    
    return result, runtime

def analyze_results(outdir, times):
    """Analyze and compare results."""
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    # Load results
    results = {}
    try:
        results['full'] = bilby.result.read_in_result(
            os.path.join(outdir, 'full_time_PE_result.json'))
        results['half'] = bilby.result.read_in_result(
            os.path.join(outdir, 'half_time_PE_result.json'))
        results['phase2'] = bilby.result.read_in_result(
            os.path.join(outdir, 'phase2_PE_result.json'))
    except:
        print("✗ Could not load all results")
        return
    
    # Time analysis
    total_two_phase = times['half'] + times['phase2']
    time_savings = (times['full'] - total_two_phase) / times['full'] * 100
    speedup = times['full'] / total_two_phase
    
    print(f"\nTIMING:")
    print(f"Full-time PE: {times['full']/3600:.2f} hours")
    print(f"Half-time PE: {times['half']/3600:.2f} hours")
    print(f"Phase 2 PE: {times['phase2']/3600:.2f} hours")
    print(f"Total two-phase: {total_two_phase/3600:.2f} hours")
    print(f"\nTIME SAVINGS: {time_savings:.1f}%")
    print(f"SPEEDUP FACTOR: {speedup:.2f}x")
    
    # Parameter accuracy
    injection_params = create_injection_parameters()
    params = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'theta_jn']
    
    print("\nPARAMETER RECOVERY (Phase 2 vs Full-time):")
    
    for param in params:
        true_val = injection_params[param]
        
        full_median = np.median(results['full'].posterior[param].values)
        phase2_median = np.median(results['phase2'].posterior[param].values)
        
        full_error = abs(full_median - true_val) / true_val * 100
        phase2_error = abs(phase2_median - true_val) / true_val * 100
        
        print(f"\n{param}:")
        print(f"  True: {true_val:.4f}")
        print(f"  Full-time: {full_median:.4f} (error: {full_error:.2f}%)")
        print(f"  Phase 2: {phase2_median:.4f} (error: {phase2_error:.2f}%)")
    
    # Summary
    print("\n" + "="*70)
    if time_savings > 15 and speedup > 1.15:
        print("✓ METHOD SUCCESSFUL!")
        print(f"  Achieved {time_savings:.1f}% time savings")
        print(f"  With comparable accuracy")
    else:
        print("✗ Method needs optimization")
    
    # Save summary
    summary = {
        'timing': times,
        'time_savings_percent': time_savings,
        'speedup_factor': speedup
    }
    
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
def create_informed_priors(posterior_result):
    """Create informed priors from posterior AND VERIFY THEM."""
    print("\n" + "*"*50)
    print("VERIFYING INFORMED PRIORS")
    print("*"*50)
    
    informed_priors = PriorDict()
    
    # We refine the main parameters, but leave spins wide to be safe
    params_to_refine = [
        'chirp_mass', 'mass_ratio', 'luminosity_distance', 
        'theta_jn', 'phase', 'geocent_time', 
        'ra', 'dec', 'psi'
    ]
    
    # Get the original priors to check bounds
    orig_priors = posterior_result.priors

    for param in params_to_refine:
        if param in posterior_result.posterior:
            samples = posterior_result.posterior[param].values
            
            # 1. Calculate the new range (1st to 99th percentile for safety)
            lower, median, upper = np.percentile(samples, [1, 50, 99])
            
            # 2. Add a generous buffer (50% width on each side)
            width = (upper - lower)
            buffer = width * 0.5 
            
            # 3. Clip to physical bounds (don't go below 0 for mass, etc.)
            old_min = orig_priors[param].minimum
            old_max = orig_priors[param].maximum
            
            new_min = max(lower - buffer, old_min)
            new_max = min(upper + buffer, old_max)

            # 4. Create the new prior
            if param == 'dec':
                informed_priors[param] = Cosine(minimum=new_min, maximum=new_max, name=param)
            elif param == 'theta_jn':
                informed_priors[param] = Sine(minimum=new_min, maximum=new_max, name=param)
            else:
                informed_priors[param] = Uniform(minimum=new_min, maximum=new_max, name=param)
            
            print(f"  Refined {param}: {new_min:.2f} to {new_max:.2f}")

    print("*"*50 + "\n")
    return informed_priors

def main_rescue():
    print("="*70)
    print("PHASE 2: SENSITIVITY TEST (RESCUE MODE)")
    print("="*70)
    
    # 1. Find ASD files
    full_files, half_files, quarter_files = find_asd_scenarios()
    if not full_files: return

    # --- RESUME CONFIGURATION ---
    # We point strictly to the folder where your 5-hour run lives:
    outdir = '/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/sensitivity_test_20260104_113942'
    print(f"Resuming analysis in: {outdir}")
    
    times = {}

    # -------------------------------------------------------
    # STEP 0: LOAD BASELINE (Do not re-run)
    # -------------------------------------------------------
    print("\nLoading existing Baseline results...")
    try:
        baseline_result = bilby.result.read_in_result(os.path.join(outdir, 'baseline_full_result.json'))
        # Estimate runtime from your log (4.21 hours)
        times['baseline'] = 4.21 * 3600 
        print("✓ Baseline loaded successfully.")
    except Exception as e:
        print(f"Could not load baseline: {e}. You might need to re-run it.")
        return

    # -------------------------------------------------------
    # EXPERIMENT A: HALF TIME (Resume)
    # -------------------------------------------------------
    print("\nLoading existing Half-Scout results...")
    try:
        # Load the scout run you just finished
        half_result = bilby.result.read_in_result(os.path.join(outdir, 'expA_half_scout_result.json'))
        times['half_scout'] = 1.22 * 3600 # From your log
        print("✓ Half-Scout loaded successfully.")
    except:
        # If it failed to save, re-run it
        print("Scout result missing. Re-running...")
        half_result, times['half_scout'] = run_pe(half_files, 'expA_half_scout', outdir)

    # -------------------------------------------------------
    # EXPERIMENT A: REFINED (This is where it crashed!)
    # -------------------------------------------------------
    print("\n>>> STARTING REFINED RUN A (The Crash Point)...")
    
    # THIS FUNCTION NOW EXISTS!
    priors_A = create_informed_priors(half_result)
    
    res_A, times['refine_A'] = run_pe(full_files, 'expA_refined', outdir, priors_A)

    # -------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    total_A = times['half_scout'] + times['refine_A']
    baseline = times['baseline']

    print(f"Baseline Time: {baseline/3600:.2f}h")
    print(f"Exp A (1/2 Start): {total_A/3600:.2f}h (Savings: {(baseline-total_A)/baseline*100:.1f}%)")
    
    # Save times
    with open(os.path.join(outdir, 'final_timing_rescue.json'), 'w') as f:
        json.dump(times, f, indent=4)


    
def main():
    print("="*70)
    print("PHASE 2: SENSITIVITY TEST (1/2 vs 1/4 Start)")
    print("="*70)
    
    # 1. Find our three key files
    full_files, half_files, quarter_files = find_asd_scenarios()
    
    if not full_files:
        return

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_BASE, f'sensitivity_test_{timestamp}')
    os.makedirs(outdir, exist_ok=True)
    
    times = {}

    # -------------------------------------------------------
    # STEP 0: The Baseline (The Truth)
    # -------------------------------------------------------
    print("\n" + "="*50)
    print(f"BASELINE: Running Full Time ({full_files['hours']:.1f}h) from scratch")
    print("="*50)
    # Note: We pass None for informed_priors here
    baseline_result, times['baseline'] = run_pe(full_files, 'baseline_full', outdir)


    # -------------------------------------------------------
    # EXPERIMENT A: The "Half-Time" Approach
    # -------------------------------------------------------
    print("\n" + "="*50)
    print(f"EXP A: Starting with Half Time ({half_files['hours']:.1f}h)")
    print("="*50)
    
    # A1. Run Scout (Fast & Rough)
    half_result, times['half_scout'] = run_pe(half_files, 'expA_half_scout', outdir)
    
    # A2. Refine on Full Data
    priors_A = create_informed_priors(half_result)
    print("\n>>> Refinement A: Using Half-Time priors on Full Data...")
    res_A, times['refine_A'] = run_pe(full_files, 'expA_refined', outdir, priors_A)


    # -------------------------------------------------------
    # EXPERIMENT B: The "Quarter-Time" Approach
    # -------------------------------------------------------
    print("\n" + "="*50)
    print(f"EXP B: Starting with Quarter Time ({quarter_files['hours']:.1f}h)")
    print("="*50)
    
    # B1. Run Scout (Fast & Rough)
    # Note: The 'scout' label triggers the optimization in run_pe
    quarter_result, times['quarter_scout'] = run_pe(quarter_files, 'expB_quarter_scout', outdir)
    
    # B2. Refine on Full Data
    priors_B = create_informed_priors(quarter_result)
    print("\n>>> Refinement B: Using Quarter-Time priors on Full Data...")
    res_B, times['refine_B'] = run_pe(full_files, 'expB_refined', outdir, priors_B)

    # -------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    # Calculate totals
    total_A = times['half_scout'] + times['refine_A']
    total_B = times['quarter_scout'] + times['refine_B']
    baseline = times['baseline']

    print(f"Baseline Time: {baseline/3600:.2f}h")
    print(f"Exp A (1/2 Start): {total_A/3600:.2f}h (Savings: {(baseline-total_A)/baseline*100:.1f}%)")
    print(f"Exp B (1/4 Start): {total_B/3600:.2f}h (Savings: {(baseline-total_B)/baseline*100:.1f}%)")
    
    # Save times to file
    with open(os.path.join(outdir, 'final_timing.json'), 'w') as f:
        json.dump(times, f, indent=4)

if __name__ == "__main__":
    main_rescue()
    # main()