
import sys
sys.path.insert(0, '.')
from phase2_complete import Run_Complete_Refinement_Test

# ASD files
base_asd_files = {
    'H1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/H1_asd_win86400.pkl',
    'L1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/L1_asd_win86400.pkl'
}

# Run the test
results = Run_Complete_Refinement_Test(base_asd_files, outdir='/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/run_20250705_115456')

if results:
    print("\n✓ Phase 2 refinement test completed successfully!")
else:
    print("\n✗ Phase 2 refinement test failed!")
    sys.exit(1)
