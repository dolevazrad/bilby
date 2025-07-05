#!/usr/bin/env python3
"""
Simple metrics analysis - focuses on numerical results without complex plots
"""

import numpy as np
import bilby
import os
import json
import pandas as pd

def analyze_refinement_results(results_dir):
    """Analyze refinement results and print key metrics."""
    
    print("\n" + "="*70)
    print("PARAMETER ESTIMATION REFINEMENT - METRICS ANALYSIS")
    print("="*70)
    
    # Load results
    results = {}
    result_files = {
        'phase1': 'phase1_preliminary_result.json',
        'refined': 'phase2_refined_result.json',
        'baseline': 'baseline_from_scratch_result.json'
    }
    
    injection_parameters = None
    
    # Try to load summary for injection parameters
    summary_path = os.path.join(results_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            injection_parameters = summary.get('injection_parameters', {})
    
    # Load bilby results
    for label, filename in result_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                result = bilby.result.read_in_result(filepath)
                results[label] = result
                print(f"✓ Loaded {label} results")
                
                # Get injection parameters from result if not already loaded
                if injection_parameters is None and hasattr(result, 'injection_parameters'):
                    injection_parameters = result.injection_parameters
            except Exception as e:
                print(f"✗ Failed to load {label}: {e}")
    
    if not results:
        print("No results found!")
        return
    
    # Parameters to analyze
    params = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'theta_jn']
    
    # 1. PARAMETER RECOVERY ANALYSIS
    print("\n" + "-"*70)
    print("1. PARAMETER RECOVERY ACCURACY")
    print("-"*70)
    
    metrics_data = []
    
    for param in params:
        if injection_parameters and param in injection_parameters:
            true_value = injection_parameters[param]
            print(f"\n{param.upper().replace('_', ' ')}:")
            print(f"  True value: {true_value:.4f}")
            
            for method_name, result in results.items():
                if param in result.posterior:
                    samples = result.posterior[param].values
                    
                    # Calculate statistics
                    median = np.median(samples)
                    mean = np.mean(samples)
                    std = np.std(samples)
                    lower_68 = np.percentile(samples, 16)
                    upper_68 = np.percentile(samples, 84)
                    lower_90 = np.percentile(samples, 5)
                    upper_90 = np.percentile(samples, 95)
                    
                    # Calculate errors
                    rel_error = abs(median - true_value) / true_value * 100
                    bias = (median - true_value) / true_value * 100
                    
                    # Check if true value in CI
                    in_68_ci = lower_68 <= true_value <= upper_68
                    in_90_ci = lower_90 <= true_value <= upper_90
                    
                    print(f"\n  {method_name.upper()}:")
                    print(f"    Median: {median:.4f}")
                    print(f"    68% CI: [{lower_68:.4f}, {upper_68:.4f}]")
                    print(f"    90% CI: [{lower_90:.4f}, {upper_90:.4f}]")
                    print(f"    Relative error: {rel_error:.2f}%")
                    print(f"    Bias: {bias:+.2f}%")
                    print(f"    True in 68% CI: {'✓' if in_68_ci else '✗'}")
                    print(f"    True in 90% CI: {'✓' if in_90_ci else '✗'}")
                    
                    # Store for summary
                    metrics_data.append({
                        'Parameter': param,
                        'Method': method_name,
                        'Median': median,
                        'Relative_Error': rel_error,
                        'CI_90_Width': upper_90 - lower_90,
                        'In_90_CI': in_90_ci
                    })
    
    # 2. COMPUTATIONAL TIME ANALYSIS
    print("\n" + "-"*70)
    print("2. COMPUTATIONAL TIME ANALYSIS")
    print("-"*70)
    
    runtimes = {}
    for method_name, result in results.items():
        if hasattr(result, 'sampling_time'):
            runtime = result.sampling_time
            # Handle different formats
            if hasattr(runtime, 'total_seconds'):
                runtime_seconds = runtime.total_seconds()
            else:
                runtime_seconds = float(runtime)
            
            runtime_hours = runtime_seconds / 3600
            runtimes[method_name] = runtime_hours
            print(f"\n{method_name.upper()}: {runtime_hours:.2f} hours ({runtime_seconds:.0f} seconds)")
    
    # Calculate time savings
    if 'baseline' in runtimes and 'refined' in runtimes:
        baseline_time = runtimes['baseline']
        refined_time = runtimes['refined']
        phase1_time = runtimes.get('phase1', 0)
        
        total_refined_time = phase1_time + refined_time
        time_saved = baseline_time - refined_time
        percent_saved = (time_saved / baseline_time) * 100
        
        total_percent_saved = ((baseline_time - total_refined_time) / baseline_time) * 100
        
        print(f"\n" + "="*50)
        print("TIME SAVINGS SUMMARY:")
        print("="*50)
        print(f"Phase 1 time: {phase1_time:.2f} hours")
        print(f"Phase 2 (refined) time: {refined_time:.2f} hours")
        print(f"Total 2-phase time: {total_refined_time:.2f} hours")
        print(f"Baseline time: {baseline_time:.2f} hours")
        print(f"\nPhase 2 vs Baseline: {percent_saved:.1f}% faster")
        print(f"Total 2-phase vs Baseline: {total_percent_saved:.1f}% savings")
        print(f"Speed-up factor: {baseline_time/total_refined_time:.2f}x")
    
    # 3. PRECISION IMPROVEMENT
    print("\n" + "-"*70)
    print("3. PRECISION IMPROVEMENT ANALYSIS")
    print("-"*70)
    
    if 'phase1' in results and 'refined' in results:
        print("\nCI Width Reduction (Phase 1 → Refined):")
        
        for param in params:
            if param in results['phase1'].posterior and param in results['refined'].posterior:
                phase1_samples = results['phase1'].posterior[param].values
                refined_samples = results['refined'].posterior[param].values
                
                phase1_width = np.percentile(phase1_samples, 84) - np.percentile(phase1_samples, 16)
                refined_width = np.percentile(refined_samples, 84) - np.percentile(refined_samples, 16)
                
                improvement = (phase1_width - refined_width) / phase1_width * 100
                print(f"  {param}: {improvement:.1f}% narrower")
    
    # 4. SUMMARY STATISTICS
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        
        # Average errors by method
        avg_errors = df.groupby('Method')['Relative_Error'].mean()
        print("\nAverage Relative Errors:")
        for method, error in avg_errors.items():
            print(f"  {method}: {error:.2f}%")
        
        # Success rate (true value in 90% CI)
        success_rate = df.groupby('Method')['In_90_CI'].mean() * 100
        print("\nParameter Recovery Success Rate (90% CI):")
        for method, rate in success_rate.items():
            print(f"  {method}: {rate:.0f}%")
    
    # 5. KEY FINDINGS
    print("\n" + "="*70)
    print("KEY FINDINGS FOR YOUR THESIS")
    print("="*70)
    
    if 'baseline' in runtimes and 'refined' in runtimes:
        print(f"\n1. TIME EFFICIENCY:")
        print(f"   - Your 2-phase method saves {total_percent_saved:.1f}% computational time")
        print(f"   - Speed-up factor: {baseline_time/total_refined_time:.2f}x")
        
    if metrics_data:
        refined_avg_error = df[df['Method'] == 'refined']['Relative_Error'].mean()
        baseline_avg_error = df[df['Method'] == 'baseline']['Relative_Error'].mean()
        
        print(f"\n2. ACCURACY:")
        print(f"   - Refined method average error: {refined_avg_error:.2f}%")
        print(f"   - Baseline method average error: {baseline_avg_error:.2f}%")
        print(f"   - Accuracy maintained: {'✓' if refined_avg_error < baseline_avg_error * 1.5 else '✗'}")
        
    print(f"\n3. METHOD VALIDATION:")
    print(f"   - Your refinement method is {'SUCCESSFUL' if total_percent_saved > 30 and refined_avg_error < 5 else 'NEEDS TUNING'}")
    
    # Save summary to file
    summary_file = os.path.join(results_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("PE REFINEMENT ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Time Savings: {total_percent_saved:.1f}%\n")
        f.write(f"Speed-up Factor: {baseline_time/total_refined_time:.2f}x\n")
        f.write(f"Average Error (Refined): {refined_avg_error:.2f}%\n")
        f.write(f"Average Error (Baseline): {baseline_avg_error:.2f}%\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")


if __name__ == "__main__":
    # Find results directory
    results_dir = None
    possible_dirs = [
        'simple_refinement_test',
        'refinement_results',
        'minimal_test_results',
        'full_comparison_results'
    ]
    
    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            results_dir = dir_name
            break
    
    if results_dir:
        print(f"Analyzing results in: {results_dir}")
        analyze_refinement_results(results_dir)
    else:
        print("No results directory found!")
        print("Please specify the path to your results directory.")