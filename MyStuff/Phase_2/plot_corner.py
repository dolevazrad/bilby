import bilby
import os

# CONSTANTS
BASE_DIR = "/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/comprehensive_snr_test_20260306_113344"
BASELINE_FILE = os.path.join(BASE_DIR, "dist_40_iter_1_baseline_result.json")
PHASE2_FILE = os.path.join(BASE_DIR, "dist_40_iter_1_refine_result.json")
PARAMS_TO_PLOT = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'ra', 'dec']

def Generate_Corner_Plot():
    print("Loading results for corner plot...")
    baseline_result = bilby.result.read_in_result(filename=BASELINE_FILE)
    phase2_result = bilby.result.read_in_result(filename=PHASE2_FILE)
    
    # Give them labels for the legend
    baseline_result.label = 'Baseline (Blind)'
    phase2_result.label = 'Phase 2 (Scout + Refine)'
    
    output_filename = os.path.join(BASE_DIR, "Posterior_Overlap_40Mpc.png")
    print(f"Plotting multiple posteriors to {output_filename}...")
    
    bilby.core.result.plot_multiple(
        results=[baseline_result, phase2_result],
        filename=output_filename,
        parameters=PARAMS_TO_PLOT,
        colors=['blue', 'red'],
        titles=True
    )
    print("Done!")

if __name__ == "__main__":
    Generate_Corner_Plot()