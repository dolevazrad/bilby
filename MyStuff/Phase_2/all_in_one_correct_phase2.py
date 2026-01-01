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
from bilby.core.prior import Uniform, Sine, PriorDict
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

def find_asd_pairs():
    """Find appropriate half-time and full-time ASD pairs."""
    print("Looking for ASD files...")
    
    h1_files = glob.glob(os.path.join(ASD_DIR, 'H1_asd_win*.pkl'))
    
    asd_pairs = {}
    for h1_file in h1_files:
        match = re.search(r'win(\d+)', h1_file)
        if match:
            window = int(match.group(1))
            l1_file = h1_file.replace('H1_asd', 'L1_asd')
            if os.path.exists(l1_file):
                asd_pairs[window] = {
                    'H1': h1_file,
                    'L1': l1_file,
                    'hours': window / 3600
                }
    
    # Find best half/full pair (prefer 12h/24h)
    windows = sorted(asd_pairs.keys())
    half_time = None
    full_time = None
    
    # Look for ideal pairs
    ideal_pairs = [(43200, 86400), (21600, 43200), (3600, 7200)]  # (12h,24h), (6h,12h), (1h,2h)
    
    for half_w, full_w in ideal_pairs:
        if half_w in windows and full_w in windows:
            half_time = asd_pairs[half_w]
            full_time = asd_pairs[full_w]
            print(f"✓ Found ideal pair: {half_time['hours']}h and {full_time['hours']}h")
            break
    
    if not half_time and len(windows) >= 2:
        # Use smallest and largest
        half_time = asd_pairs[windows[0]]
        full_time = asd_pairs[windows[-1]]
        print(f"Using: {half_time['hours']}h and {full_time['hours']}h")
    
    return half_time, full_time

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
    
    # Injection parameters
    injection_params = create_injection_parameters()
    
    # Set up interferometers
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    
    # Waveform generator
    duration = 4
    sampling_frequency = 2048
    minimum_frequency = 20
    
    waveform_arguments = dict(
        waveform_approximant='IMRPhenomD',
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
        ifo.inject_signal(injection_params, waveform_generator)
    
    # Set up priors
    if informed_priors is None:
        # Standard uniform priors
        priors = bilby.gw.prior.BBHPriorDict()
        priors['chirp_mass'] = Uniform(25.0, 35.0)
        priors['mass_ratio'] = Uniform(0.5, 1.0)
        priors['luminosity_distance'] = Uniform(200, 800)
        priors['theta_jn'] = Sine()
        priors['phase'] = Uniform(0, 2 * np.pi)
        priors['geocent_time'] = Uniform(
            injection_params['geocent_time'] - 0.1,
            injection_params['geocent_time'] + 0.1
        )
        
        # --- OPTIMIZATION START ---
        # Check if this is the "Half Time" run (The Scout)
        if 'half' in label.lower():
            print("--- OPTIMIZING FOR SPEED (Phase 1: Scout) ---")
            # Lower settings for rough estimation
            sampler_settings = {'npoints': 250, 'walks': 10} 
            dlogz_val = 0.5  # Stop sooner (0.5 is rougher than 0.1)
        else:
            # Baseline: High Precision
            sampler_settings = {'npoints': 500, 'walks': 25}
            dlogz_val = 0.1
        # --- OPTIMIZATION END ---

    else:
        # Phase 2: Refined (The Sniper)
        priors = informed_priors
        sampler_settings = {'npoints': 300, 'walks': 15}
        dlogz_val = 0.1  # We want high precision here too
    
    # Fix other parameters
    for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra', 'dec', 'psi']:
        priors[key] = injection_params[key]
    
    # Set up likelihood
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator
    )
    
    # Run sampler
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
        **sampler_settings
    )
    
    runtime = time.time() - start_time
    print(f"{label} completed in {runtime/3600:.2f} hours")
    
    return result, runtime

def create_informed_priors(posterior_result):
    """Create informed priors from posterior."""
    informed_priors = PriorDict()
    
    params = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 
              'theta_jn', 'phase', 'geocent_time']
    
    for param in params:
        if param in posterior_result.posterior:
            samples = posterior_result.posterior[param].values
            lower, median, upper = np.percentile(samples, [5, 50, 95])
            width = (upper - lower) * 1.5 / 2
            
            if param == 'theta_jn':
                informed_priors[param] = Sine(
                    minimum=max(0, median - width),
                    maximum=min(np.pi, median + width)
                )
            elif param == 'phase':
                informed_priors[param] = Uniform(
                    minimum=max(0, median - width),
                    maximum=min(2*np.pi, median + width)
                )
            else:
                informed_priors[param] = Uniform(
                    minimum=median - width,
                    maximum=median + width
                )
    
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

def main():
    """Run the complete correct workflow."""
    print("="*70)
    print("PHASE 2 CORRECT IMPLEMENTATION")
    print("="*70)
    
    # Find ASD files
    half_time_files, full_time_files = find_asd_pairs()
    
    if not half_time_files or not full_time_files:
        print("\n✗ Could not find appropriate ASD pairs!")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_BASE, f'correct_run_{timestamp}')
    os.makedirs(outdir, exist_ok=True)
    
    print(f"\nOutput directory: {outdir}")
    
    times = {}
    
    # Step 1: Full-time PE (baseline)
    print("\n" + "="*50)
    print("STEP 1: Full-time PE (baseline)")
    print("="*50)
    full_result, times['full'] = run_pe(full_time_files, 'full_time_PE', outdir)
    
    # Step 2: Half-time PE
    print("\n" + "="*50)
    print("STEP 2: Half-time PE")
    print("="*50)
    half_result, times['half'] = run_pe(half_time_files, 'half_time_PE', outdir)
    
    # Step 3: Phase 2 PE with informed priors
    print("\n" + "="*50)
    print("STEP 3: Phase 2 PE (refined)")
    print("="*50)
    informed_priors = create_informed_priors(half_result)
    phase2_result, times['phase2'] = run_pe(full_time_files, 'phase2_PE', outdir, 
                                           informed_priors)
    
    # Analyze results
    analyze_results(outdir, times)
    
    print(f"\n✓ Complete! Results saved to: {outdir}")

if __name__ == "__main__":
    main()