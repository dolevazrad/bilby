import bilby
import numpy as np
import os
from scipy.stats import ks_2samp

# PATHS (Update these if you moved folders)
BASE_DIR = '/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/sensitivity_test_20260104_113942'
FILE_BASELINE = os.path.join(BASE_DIR, 'baseline_full_result.json')
FILE_REFINED = os.path.join(BASE_DIR, 'expA_refined_result.json')

def compare_results():
    print(f"Loading results from: {BASE_DIR}")
    try:
        res_base = bilby.result.read_in_result(FILE_BASELINE)
        
        
        
        
        res_ref = bilby.result.read_in_result(FILE_REFINED)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Parameters to compare (The 15D Physics)
    params = [
        'chirp_mass', 'mass_ratio', 'luminosity_distance', 
        'theta_jn', 'phase', 'geocent_time', 
        'ra', 'dec', 'psi', 'a_1', 'a_2', 'tilt_1', 'tilt_2'
    ]

    print(f"\n{'PARAMETER':<20} | {'KS STATISTIC':<12} | {'P-VALUE':<12} | {'CONCLUSION'}")
    print("-" * 65)

    ks_values = []
    
    for p in params:
        # Get samples
        samp_base = res_base.posterior[p].values
        samp_ref = res_ref.posterior[p].values
        
        # Calculate Kolmogorov-Smirnov Test
        # statistic: 0.0 (identical) to 1.0 (different)
        # pvalue: High (>0.05) means we cannot reject hypothesis that they are same
        ks_stat, p_val = ks_2samp(samp_base, samp_ref)
        ks_values.append(ks_stat)
        
        # Interpretation
        if ks_stat < 0.05:
            conclusion = "Excellent Match"
        elif ks_stat < 0.1:
            conclusion = "Good Match"
        else:
            conclusion = "Divergent"

        print(f"{p:<20} | {ks_stat:.4f}       | {p_val:.4f}       | {conclusion}")

    print("-" * 65)
    print(f"Average KS Statistic: {np.mean(ks_values):.4f} (Target: < 0.05)")
    
    # Bayes Factor Comparison
    bf_base = res_base.log_bayes_factor
    bf_ref = res_ref.log_bayes_factor
    print(f"\nLog Bayes Factors:")
    print(f"Baseline: {bf_base:.2f}")
    print(f"Refined:  {bf_ref:.2f}")
    print(f"Diff:     {bf_ref - bf_base:.2f}")

if __name__ == "__main__":
    compare_results()