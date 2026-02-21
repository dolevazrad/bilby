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
from scipy.special import logsumexp

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
    print(f"\nRunning {label} PE...")
    
    # Load ASDs
    asds = {}
    for det in ['H1', 'L1']:
        with open(asd_files[det], 'rb') as f:
            data = pickle.load(f)
        asds[det] = data['asd']
    
    # Injection parameters (Includes tilts!)
    injection_params = create_injection_parameters()
    
    # Set up interferometers
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    
    # --- MAJOR CHANGE: WAVEFORM MODEL ---
    # We switch to IMRPhenomXPHM to support Precession (15 parameters)
    duration = 4
    sampling_frequency = 2048
    minimum_frequency = 20
    
    waveform_arguments = dict(
        waveform_approximant='IMRPhenomXPHM',  # <--- CHANGED FROM D TO XPHM
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
    
    # Set up interferometers
    n_freq = int(duration * sampling_frequency / 2) + 1
    frequencies = np.linspace(0, sampling_frequency/2, n_freq)
    
    for ifo in ifos:
        # Interpolate ASD
        asd = asds[ifo.name]
        interp = interp1d(asd.frequencies.value, asd.value, 
                         bounds_error=False, fill_value='extrapolate')
        interpolated_asd = interp(frequencies)
        
        # Set properties
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = sampling_frequency/2
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration
        ifo.start_time = injection_params['geocent_time'] - duration + 0.5
        
        # Set PSD
        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=frequencies,
            psd_array=interpolated_asd**2
        )
        
        # Initialize strain
        ifo.strain_data.roll_off = 0.2
        ifo.strain_data.set_from_frequency_domain_strain(
            sampling_frequency=sampling_frequency,
            duration=duration,
            frequency_domain_strain=np.zeros(n_freq, dtype=complex)
        )
        
        # Inject signal
        ifo.inject_signal(parameters=injection_params, waveform_generator=waveform_generator)
    
   # ---------------------------------------------------------
    # SET UP PRIORS (FULL 15 PARAMETERS)
    # ---------------------------------------------------------
    # DEFINE ALL 15 PRIORS FOR EVERY RUN FIRST
    print(">>> Configuring 15-Parameter Precessing Priors")
    priors = bilby.gw.prior.BBHPriorDict()
    
    # 1. Masses
    priors['chirp_mass'] = Uniform(25.0, 35.0, name='chirp_mass', unit='$M_{\odot}$')
    priors['mass_ratio'] = Uniform(0.5, 1.0, name='mass_ratio')
    
    # 2. Extrinsic
    priors['luminosity_distance'] = Uniform(200, 800, name='luminosity_distance', unit='Mpc')
    priors['geocent_time'] = Uniform(
        injection_params['geocent_time'] - 0.1,
        injection_params['geocent_time'] + 0.1,
        name='geocent_time', unit='s'
    )
    priors['phase'] = Uniform(0, 2 * np.pi, name='phase')
    priors['theta_jn'] = Sine(name='theta_jn') 

    # 3. Sky Location (Blind)
    priors['ra'] = Uniform(0, 2 * np.pi, name='ra')
    priors['dec'] = Cosine(name='dec')
    priors['psi'] = Uniform(0, np.pi, name='psi')

    # 4. SPIN MAGNITUDES
    priors['a_1'] = Uniform(0, 0.99, name='a_1')
    priors['a_2'] = Uniform(0, 0.99, name='a_2')

    # 5. SPIN TILTS & PHASES 
    priors['tilt_1'] = Sine(name='tilt_1')
    priors['tilt_2'] = Sine(name='tilt_2')
    priors['phi_12'] = Uniform(0, 2 * np.pi, name='phi_12', boundary='periodic')
    priors['phi_jl'] = Uniform(0, 2 * np.pi, name='phi_jl', boundary='periodic')

    # --- APPLY SAMPLER SETTINGS AND UPDATES ---
    if informed_priors is None:
        if 'scout' in label.lower():
            print("--- PHASE 1: SCOUT RUN (Fast) ---")
            sampler_settings = {'npoints': 500, 'walks': 50} 
            dlogz_val = 0.5 
        else:
            print("--- PHASE 0: BASELINE (PRODUCTION QUALITY) ---")
            sampler_settings = {'npoints': 2048, 'walks': 100}
            dlogz_val = 0.1
    else:
        # Phase 2: Refined
        print("--- PHASE 2: REFINED RUN (High Precision) ---")
        priors.update(informed_priors) # <--- THE FIX: Updates only the refined ones!
        sampler_settings = {'npoints': 1024, 'walks': 50}
        dlogz_val = 0.1
    
    # Set up likelihood
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator
    )
    
    # Run sampler (Safe Mode for WSL)
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
        npool=1,
        check_point=False,      
        print_progress=True,  
        **sampler_settings
    )

    # Generate plots
    print(f"Generating corner plot for {label}...")
    try:
        result.plot_corner()
    except Exception as e:
        print(f"Plotting failed (non-critical): {e}")
    
    runtime = time.time() - start_time
    print(f"{label} completed in {runtime/3600:.2f} hours")
    
    return result, runtime
def create_informed_priors(posterior_result):
    """Create informed priors from posterior AND VERIFY THEM."""
    print("\n" + "*"*50)
    print("VERIFYING INFORMED PRIORS")
    print("*"*50)
    
    informed_priors = PriorDict()
    
    # Define the original wide ranges (just for comparison)
    original_ranges = {
        'chirp_mass': 10.0,      # 35 - 25
        'mass_ratio': 0.5,       # 1.0 - 0.5
        'luminosity_distance': 600 # 800 - 200
    }
    
    params = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 
              'theta_jn', 'phase', 'geocent_time']
    
    for param in params:
        if param in posterior_result.posterior:
            samples = posterior_result.posterior[param].values
            
            # 1. Calculate the new range
            lower, median, upper = np.percentile(samples, [5, 50, 95])
            width = (upper - lower) * 1.5 / 2  # The buffer logic
            
            new_min = median - width
            new_max = median + width
            new_range = new_max - new_min

            # 2. Add to informed priors
            if param == 'theta_jn':
                informed_priors[param] = Sine(minimum=max(0, new_min), maximum=min(np.pi, new_max))
            elif param == 'phase':
                informed_priors[param] = Uniform(minimum=max(0, new_min), maximum=min(2*np.pi, new_max))
            else:
                informed_priors[param] = Uniform(minimum=new_min, maximum=new_max)

            # 3. PRINT THE COMPARISON (The Proof)
            if param in original_ranges:
                orig = original_ranges[param]
                improvement = (orig - new_range) / orig * 100
                print(f"PARAM: {param}")
                print(f"  Old Width: {orig:.2f}")
                print(f"  New Width: {new_range:.2f}")
                
                if improvement > 0:
                    print(f"  >>> SUCCESS: Range is {improvement:.1f}% tighter!")
                else:
                    print(f"  >>> WARNING: Range did not improve.")
            else:
                print(f"PARAM: {param} -> New range: {new_min:.2f} to {new_max:.2f}")

    print("*"*50 + "\n")
    return informed_priors

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

def Re_Weight_Posterior(refined_result, original_priors):
    """
    @brief Re-weights the posterior samples from a restricted prior run to reflect the original wide priors.
    @param refined_result Bilby result object from the Refined run.
    @param original_priors The original wide PriorDict.
    @return corrected_log_bayes_factor The mathematically honest Bayes Factor.
    """
    print("\n" + "="*60)
    print("PERFORMING IMPORTANCE RE-WEIGHTING (HONEST EVIDENCE)")
    print("="*60)
    
    scout_priors = refined_result.priors
    posterior = refined_result.posterior
    
    ln_weights = np.zeros(len(posterior))
    
    # Calculate the log prior probabilities for each sample
    for i in range(len(posterior)):
        # Extract the specific values for this sample
        sample = dict(posterior.iloc[i])
        
        # 1. Log probability under original wide prior
        ln_p_orig = original_priors.ln_prob(sample)
        
        # 2. Log probability under restricted scout prior
        ln_p_scout = scout_priors.ln_prob(sample)
        
        # 3. The weight in log space
        ln_weights[i] = ln_p_orig - ln_p_scout
        
    # Calculate the correction term for the Evidence: ln( (1/N) * sum(w_i) )
    # We use logsumexp to prevent computer underflow/overflow errors with tiny probabilities
    ln_weight_ratio = logsumexp(ln_weights) - np.log(len(ln_weights))
    
    corrected_log_evidence = refined_result.log_evidence + ln_weight_ratio
    corrected_log_bayes_factor = refined_result.log_bayes_factor + ln_weight_ratio
    
    print(f"Original (Inflated) ln(BF): {refined_result.log_bayes_factor:.2f}")
    print(f"Evidence Correction Shift:  {ln_weight_ratio:.2f}")
    print(f"Corrected (Honest) ln(BF):  {corrected_log_bayes_factor:.2f}")
    print("="*60 + "\n")
    
    return corrected_log_bayes_factor
def main():
    print("="*70)
    print("PHASE 2: SENSITIVITY TEST (Overnight Run with Re-weighting)")
    print("="*70)
    
    # 1. Find our files
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
    baseline_result, times['baseline'] = run_pe(full_files, 'baseline_full', outdir)

    # -------------------------------------------------------
    # EXPERIMENT A: STEP 1 - THE SCOUT RUN
    # -------------------------------------------------------
    print("\n" + "="*50)
    print(f"EXP A: Starting Scout Run with Half Time ({half_files['hours']:.1f}h)")
    print("="*50)
    half_result, times['half_scout'] = run_pe(half_files, 'expA_half_scout', outdir)

    # -------------------------------------------------------
    # EXPERIMENT A: STEP 2 - REFINED (With Importance Re-weighting)
    # -------------------------------------------------------
    print("\n>>> STARTING REFINED RUN A...")
    
    # 1. Create the tight priors for the run using the scout result
    priors_A = create_informed_priors(half_result)
    
    # 2. Run the PE (This saves the 17% time!)
    res_A, times['refine_A'] = run_pe(full_files, 'expA_refined', outdir, priors_A)

    # 3. Re-Weighting (The Mentor's Fix)
    # We grab the EXACT priors used in the Baseline run to ensure 100% mathematical accuracy.
    original_wide_priors = baseline_result.priors
    
    # Run the math
    honest_bf = Re_Weight_Posterior(res_A, original_wide_priors)

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
    print(f"Baseline ln(BF): {baseline_result.log_bayes_factor:.2f}")
    print(f"Refined Honest ln(BF): {honest_bf:.2f}")
    
    # Save times to file
    with open(os.path.join(outdir, 'final_timing.json'), 'w') as f:
        json.dump(times, f, indent=4)

if __name__ == "__main__":
    main()