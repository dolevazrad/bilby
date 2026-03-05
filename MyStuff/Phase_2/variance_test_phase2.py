#!/usr/bin/env python3
import os
import json
import numpy as np
import bilby
from datetime import datetime

# Import your working functions from the main script
from all_in_one_correct_phase2 import (
    find_asd_scenarios, 
    run_pe, 
    create_informed_priors, 
    Re_Weight_Posterior, 
    OUTPUT_BASE,
    create_injection_parameters
)

def Get_Original_Priors():
    """
    @brief Recreates the exact 15-parameter wide priors.
    @return bilby.gw.prior.BBHPriorDict original wide priors.
    """
    priors = bilby.gw.prior.BBHPriorDict()
    injection_params = create_injection_parameters()
    
    # 1. Masses
    priors['chirp_mass'] = bilby.core.prior.Uniform(25.0, 35.0, name='chirp_mass')
    priors['mass_ratio'] = bilby.core.prior.Uniform(0.5, 1.0, name='mass_ratio')
    
    # 2. Extrinsic
    priors['luminosity_distance'] = bilby.core.prior.Uniform(200, 800, name='luminosity_distance')
    priors['geocent_time'] = bilby.core.prior.Uniform(
        injection_params['geocent_time'] - 0.1,
        injection_params['geocent_time'] + 0.1,
        name='geocent_time'
    )
    priors['phase'] = bilby.core.prior.Uniform(0, 2 * np.pi, name='phase')
    priors['theta_jn'] = bilby.core.prior.Sine(name='theta_jn') 

    # 3. Sky Location (Blind)
    priors['ra'] = bilby.core.prior.Uniform(0, 2 * np.pi, name='ra')
    priors['dec'] = bilby.core.prior.Cosine(name='dec')
    priors['psi'] = bilby.core.prior.Uniform(0, np.pi, name='psi')

    # 4. SPIN MAGNITUDES
    priors['a_1'] = bilby.core.prior.Uniform(0, 0.99, name='a_1')
    priors['a_2'] = bilby.core.prior.Uniform(0, 0.99, name='a_2')

    # 5. SPIN TILTS & PHASES 
    priors['tilt_1'] = bilby.core.prior.Sine(name='tilt_1')
    priors['tilt_2'] = bilby.core.prior.Sine(name='tilt_2')
    priors['phi_12'] = bilby.core.prior.Uniform(0, 2 * np.pi, name='phi_12', boundary='periodic')
    priors['phi_jl'] = bilby.core.prior.Uniform(0, 2 * np.pi, name='phi_jl', boundary='periodic')
    
    return priors

def main():
    print("="*70)
    print("PHASE 2 VARIANCE TEST: Scout + Gaussian Refine (3 Iterations)")
    print("="*70)
    
    # 1. Grab the ASD files
    full_files, half_files, _ = find_asd_scenarios()
    if not full_files:
        print("Could not find ASD files.")
        return

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_BASE, f'variance_phase2_test_{timestamp}')
    os.makedirs(outdir, exist_ok=True)
    
    honest_bf_results = []
    total_runtimes = []
    
    # Recreate the original wide priors for the re-weighting math
    original_wide_priors = Get_Original_Priors()

    # 2. Loop the entire Phase 2 pipeline 3 times
    for i in range(1, 4):
        print(f"\n" + "-"*60)
        print(f"ITERATION {i} / 3: STARTING SCOUT")
        print("-"*60)
        
        # Step A: Scout Run
        half_label = f'expA_half_scout_run_{i}'
        half_result, scout_time = run_pe(half_files, half_label, outdir)
        
        # Step B: Gaussian Refinement Prior Creation
        priors_A = create_informed_priors(half_result)
        
        # Step C: Refined Run
        print(f"\n>>> ITERATION {i} / 3: STARTING REFINED PE...")
        refine_label = f'expA_refined_run_{i}'
        res_A, refine_time = run_pe(full_files, refine_label, outdir, priors_A)
        
        # Step D: Re-weighting
        honest_bf = Re_Weight_Posterior(res_A, original_wide_priors)
        
        total_time = scout_time + refine_time
        
        honest_bf_results.append(honest_bf)
        total_runtimes.append(total_time)
        
        print(f">>> Run {i} Honest ln(BF): {honest_bf:.2f}")
        print(f">>> Run {i} Total Time:    {total_time/3600:.2f} hours")

    # 3. Calculate the Statistics
    mean_bf = np.mean(honest_bf_results)
    std_bf = np.std(honest_bf_results)
    
    print("\n" + "="*70)
    print("PHASE 2 VARIANCE TEST RESULTS (Honest Evidence)")
    print("="*70)
    for i, bf in enumerate(honest_bf_results):
        print(f"Run {i+1} Honest ln(BF): {bf:.2f}  |  Total Time: {total_runtimes[i]/3600:.2f}h")
    
    print("-"*70)
    print(f"MEAN Honest ln(BF): {mean_bf:.2f}")
    print(f"STD DEV:            +/- {std_bf:.2f}")
    print("="*70)

    # Save to a file to show Ofek
    with open(os.path.join(outdir, 'variance_phase2_stats.json'), 'w') as f:
        json.dump({
            'honest_bf_results': honest_bf_results,
            'mean_bf': mean_bf,
            'std_bf': std_bf,
            'total_runtimes': total_runtimes
        }, f, indent=4)

if __name__ == "__main__":
    main()