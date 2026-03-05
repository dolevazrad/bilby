#!/usr/bin/env python3
import os
import json
import numpy as np
from datetime import datetime

# Import your working functions from your main script
from all_in_one_correct_phase2 import find_asd_scenarios, run_pe, OUTPUT_BASE

def main():
    print("="*70)
    print("BASELINE VARIANCE TEST (3 Iterations)")
    print("="*70)
    
    # 1. Grab the full ASD files (we only need the Baseline)
    full_files, _, _ = find_asd_scenarios()
    if not full_files:
        print("Could not find ASD files.")
        return

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_BASE, f'variance_test_{timestamp}')
    os.makedirs(outdir, exist_ok=True)
    
    bf_results = []
    runtimes = []

    # 2. Run the exact same Baseline 3 times
    for i in range(1, 4):
        label = f'baseline_variance_run_{i}'
        print(f"\n" + "-"*50)
        print(f"ITERATION {i} / 3")
        print("-"*50)
        
        # We pass informed_priors=None to force a standard Baseline run
        result, runtime = run_pe(full_files, label, outdir, informed_priors=None)
        
        bf_results.append(result.log_bayes_factor)
        runtimes.append(runtime)
        
        print(f">>> Run {i} ln(BF): {result.log_bayes_factor:.2f}")
        print(f">>> Run {i} Time:   {runtime/3600:.2f} hours")

    # 3. Calculate the Statistics
    mean_bf = np.mean(bf_results)
    std_bf = np.std(bf_results)
    
    print("\n" + "="*70)
    print("VARIANCE TEST RESULTS")
    print("="*70)
    for i, bf in enumerate(bf_results):
        print(f"Run {i+1} ln(BF): {bf:.2f}  |  Time: {runtimes[i]/3600:.2f}h")
    
    print("-"*70)
    print(f"MEAN ln(BF):  {mean_bf:.2f}")
    print(f"STD DEV:      +/- {std_bf:.2f}")
    print("="*70)

    # Save to a file for Ofek
    with open(os.path.join(outdir, 'variance_stats.json'), 'w') as f:
        json.dump({
            'bf_results': bf_results,
            'mean_bf': mean_bf,
            'std_bf': std_bf,
            'runtimes': runtimes
        }, f, indent=4)

if __name__ == "__main__":
    main()