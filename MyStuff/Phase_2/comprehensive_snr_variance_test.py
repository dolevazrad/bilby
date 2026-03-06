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
    """Recreates the wide priors using the NEW 1-10000 Mpc bounds."""
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
    print("="*80)
    print("COMPREHENSIVE SYSTEMATIC SNR TEST (3 Iterations per Distance)")
    print("="*80)
    
    full_files, half_files, _ = find_asd_scenarios()
    if not full_files:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_BASE, f'comprehensive_snr_test_{timestamp}')
    os.makedirs(outdir, exist_ok=True)
    
    # The exact distances Ofek requested
    distances_to_test = [10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1300.0, 2500.0, 5000.0]
    
    # Master dictionary to hold all data
    master_results = {}
    json_path = os.path.join(outdir, 'systematic_snr_variance_results.json')

    for dist in distances_to_test:
        print(f"\n" + "#"*80)
        print(f"STARTING DISTANCE: {dist} Mpc")
        print("#"*80)
        
        params = create_injection_parameters(distance=dist)
        orig_priors = Get_Original_Priors(params)
        
        dist_data = {
            'baseline_bfs': [],
            'baseline_times': [],
            'honest_bfs': [],
            'phase2_times': []
        }

        # Run 3 Iterations for Variance
        for i in range(1, 4):
            print(f"\n--- DISTANCE {dist} Mpc | ITERATION {i}/3 ---")
            
            # BASELINE
            b_label = f'dist_{int(dist)}_iter_{i}_baseline'
            b_res, b_time = run_pe(full_files, b_label, outdir, informed_priors=None, custom_injection_params=params)
            
            # SCOUT
            s_label = f'dist_{int(dist)}_iter_{i}_scout'
            s_res, s_time = run_pe(half_files, s_label, outdir, informed_priors=None, custom_injection_params=params)
            
            # REFINE
            refined_priors = create_informed_priors(s_res)
            r_label = f'dist_{int(dist)}_iter_{i}_refine'
            r_res, r_time = run_pe(full_files, r_label, outdir, informed_priors=refined_priors, custom_injection_params=params)
            
            # MATH & TRACKING
            honest_bf = Re_Weight_Posterior(r_res, orig_priors)
            total_p2_time = s_time + r_time
            
            dist_data['baseline_bfs'].append(b_res.log_bayes_factor)
            dist_data['baseline_times'].append(b_time)
            dist_data['honest_bfs'].append(honest_bf)
            dist_data['phase2_times'].append(total_p2_time)
            
        # Calculate Stats for this Distance
        dist_data['stats'] = {
            'baseline_mean_bf': np.mean(dist_data['baseline_bfs']),
            'baseline_std_bf': np.std(dist_data['baseline_bfs']),
            'honest_mean_bf': np.mean(dist_data['honest_bfs']),
            'honest_std_bf': np.std(dist_data['honest_bfs']),
            'avg_baseline_time_h': np.mean(dist_data['baseline_times']) / 3600,
            'avg_phase2_time_h': np.mean(dist_data['phase2_times']) / 3600,
            'avg_time_savings_percent': ((np.mean(dist_data['baseline_times']) - np.mean(dist_data['phase2_times'])) / np.mean(dist_data['baseline_times'])) * 100
        }
        
        master_results[f"{dist}_Mpc"] = dist_data
        
        # SAVE PROGRESS TO DISK IMMEDIATELY (Checkpointing)
        with open(json_path, 'w') as f:
            json.dump(master_results, f, indent=4)
            
        print(f"\n[CHECKPOINT SAVED] Distance {dist} Mpc completed. JSON updated.")
        print(f"  -> Baseline Mean BF: {dist_data['stats']['baseline_mean_bf']:.2f} ± {dist_data['stats']['baseline_std_bf']:.2f}")
        print(f"  -> Phase 2 Mean BF:  {dist_data['stats']['honest_mean_bf']:.2f} ± {dist_data['stats']['honest_std_bf']:.2f}")
        print(f"  -> Avg Time Savings: {dist_data['stats']['avg_time_savings_percent']:.1f}%\n")

    print("\n" + "="*80)
    print("ALL SYSTEMATIC DISTANCES COMPLETED AND SAVED.")
    print("="*80)

if __name__ == "__main__":
    main()