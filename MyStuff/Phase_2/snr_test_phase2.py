#!/usr/bin/env python3
import os
import json
import bilby
import numpy as np
from datetime import datetime

from all_in_one_correct_phase2 import (
    find_asd_scenarios, 
    run_pe, 
    create_informed_priors, 
    Re_Weight_Posterior, 
    OUTPUT_BASE,
    create_injection_parameters
)

def Get_Original_Priors(injection_params):
    """Recreates the wide priors based on the specific injection time."""
    priors = bilby.gw.prior.BBHPriorDict()
    
    priors['chirp_mass'] = bilby.core.prior.Uniform(25.0, 35.0, name='chirp_mass')
    priors['mass_ratio'] = bilby.core.prior.Uniform(0.5, 1.0, name='mass_ratio')
    priors['luminosity_distance'] = bilby.core.prior.Uniform(200, 800, name='luminosity_distance')
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
    print("="*70)
    print("PHASE 2 SNR SCALING TEST (Varying Luminosity Distance)")
    print("="*70)
    
    full_files, half_files, _ = find_asd_scenarios()
    if not full_files:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_BASE, f'snr_test_{timestamp}')
    os.makedirs(outdir, exist_ok=True)
    
    # We will test a Loud, Medium, and Quiet signal
    distances_to_test = [300.0, 450.0, 600.0]
    results_summary = {}

    for dist in distances_to_test:
        print(f"\n" + "#"*70)
        print(f"TESTING DISTANCE: {dist} Mpc")
        print("#"*70)
        
        # 1. Create specific params and priors for this distance
        params = create_injection_parameters(distance=dist)
        orig_priors = Get_Original_Priors(params)
        
        # 2. BASELINE
        b_label = f'dist_{int(dist)}_baseline'
        b_res, b_time = run_pe(full_files, b_label, outdir, informed_priors=None, custom_injection_params=params)
        
        # 3. SCOUT
        s_label = f'dist_{int(dist)}_scout'
        s_res, s_time = run_pe(half_files, s_label, outdir, informed_priors=None, custom_injection_params=params)
        
        # 4. GAUSSIAN REFINE
        refined_priors = create_informed_priors(s_res)
        r_label = f'dist_{int(dist)}_refine'
        r_res, r_time = run_pe(full_files, r_label, outdir, informed_priors=refined_priors, custom_injection_params=params)
        
        # 5. RE-WEIGHT MATH
        honest_bf = Re_Weight_Posterior(r_res, orig_priors)
        total_p2_time = s_time + r_time
        time_savings = ((b_time - total_p2_time) / b_time) * 100
        
        # Store for summary
        results_summary[f"{dist}_Mpc"] = {
            'baseline_bf': b_res.log_bayes_factor,
            'honest_bf': honest_bf,
            'baseline_time_h': b_time / 3600,
            'scout_time_h': s_time / 3600,
            'refine_time_h': r_time / 3600,
            'total_phase2_time_h': total_p2_time / 3600,
            'time_savings_percent': time_savings
        }

    # PRINT FINAL SUMMARY TABLE
    print("\n" + "="*80)
    print("FINAL SNR SCALING SUMMARY")
    print("="*80)
    for dist, data in results_summary.items():
        print(f"\n--- DISTANCE: {dist} ---")
        print(f"Baseline ln(BF): {data['baseline_bf']:.2f}  |  Honest ln(BF): {data['honest_bf']:.2f}")
        print(f"Baseline Time:   {data['baseline_time_h']:.2f}h")
        print(f"Phase 2 Time:    {data['total_phase2_time_h']:.2f}h (Scout: {data['scout_time_h']:.2f}h + Refine: {data['refine_time_h']:.2f}h)")
        print(f"TIME SAVINGS:    {data['time_savings_percent']:.1f}%")
        
    with open(os.path.join(outdir, 'snr_scaling_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)

if __name__ == "__main__":
    main()