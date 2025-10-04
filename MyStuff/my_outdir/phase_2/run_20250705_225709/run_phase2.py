
import sys
sys.path.insert(0, '.')

# Try to import the function with baseline, fall back to simple if not available
try:
    from phase2_complete import Run_Simple_Refinement_Test_With_Baseline
    run_function = Run_Simple_Refinement_Test_With_Baseline
    print("✓ Using refinement test with baseline comparison")
except ImportError:
    from phase2_complete import Run_Simple_Refinement_Test
    run_function = Run_Simple_Refinement_Test
    print("⚠ Using simple refinement test (no baseline comparison)")
    print("  To include baseline comparison, add Run_Simple_Refinement_Test_With_Baseline to phase2_complete.py")

# ASD files
base_asd_files = {
    'H1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/H1_asd_win86400.pkl',
    'L1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/L1_asd_win86400.pkl'
}

# Run the test
results = run_function(base_asd_files, outdir='/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/run_20250705_225709')

if results:
    print("\n✓ Phase 2 refinement test completed successfully!")
else:
    print("\n✗ Phase 2 refinement test failed!")
    sys.exit(1)
