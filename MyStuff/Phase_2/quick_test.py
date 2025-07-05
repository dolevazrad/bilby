#!/usr/bin/env python3
"""
Quick test script to verify the Phase 2 setup before running the full workflow
This runs a minimal version with reduced sampling to check everything works
"""

import os
import sys
import numpy as np
import time

def quick_test():
    """Run a quick test with minimal sampling."""
    print("="*70)
    print("PHASE 2 QUICK TEST - Verify Setup")
    print("="*70)
    
    # Import the main module
    try:
        from phase2_complete import (
            ASD_Evolution_Model, 
            Create_Test_Injection_Parameters,
            Setup_Interferometers_With_ASD,
            Run_Phase1_PE
        )
        print("✓ Successfully imported Phase 2 modules")
    except Exception as e:
        print(f"✗ Failed to import modules: {e}")
        return False
    
    # Test ASD files
    base_asd_files = {
        'H1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/H1_asd_win86400.pkl',
        'L1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/L1_asd_win86400.pkl'
    }
    
    print("\nTesting ASD loading...")
    try:
        asd_models = {}
        for detector, asd_file in base_asd_files.items():
            if not os.path.exists(asd_file):
                print(f"✗ {detector} ASD file not found: {asd_file}")
                return False
            
            model = ASD_Evolution_Model(asd_file, evolution_stages=2)
            asd_models[detector] = model
            print(f"✓ Loaded {detector} ASD successfully")
            
            # Test ASD evolution
            preliminary_asd = model.Generate_Evolved_ASD(0)
            final_asd = model.Generate_Evolved_ASD(1)
            
            ratio = np.mean(preliminary_asd.value) / np.mean(final_asd.value)
            print(f"  - Preliminary/Final ASD ratio: {ratio:.3f}")
    except Exception as e:
        print(f"✗ ASD loading failed: {e}")
        return False
    
    print("\nTesting interferometer setup...")
    try:
        injection_params = Create_Test_Injection_Parameters()
        
        # Generate test ASDs
        test_asds = {}
        for detector, model in asd_models.items():
            test_asds[detector] = model.Generate_Evolved_ASD(0)
        
        # Set up interferometers
        ifos, wf_gen = Setup_Interferometers_With_ASD(test_asds, injection_params)
        print("✓ Interferometers set up successfully")
        
        # Check signal injection
        for ifo in ifos:
            max_strain = np.max(np.abs(ifo.strain_data.frequency_domain_strain))
            print(f"  - {ifo.name} max strain amplitude: {max_strain:.2e}")
    except Exception as e:
        print(f"✗ Interferometer setup failed: {e}")
        return False
    
    print("\nTesting minimal PE run (this may take 1-2 minutes)...")
    try:
        # Create test output directory
        test_dir = 'quick_test_output'
        os.makedirs(test_dir, exist_ok=True)
        
        # Set up minimal priors
        import bilby
        from bilby.core.prior import Uniform, Sine
        
        priors = bilby.gw.prior.BBHPriorDict()
        priors['chirp_mass'] = Uniform(29.0, 31.0, latex_label='$\\mathcal{M}$')
        priors['mass_ratio'] = Uniform(0.8, 1.0, latex_label='$q$')
        priors['luminosity_distance'] = Uniform(400, 500, latex_label='$d_L$')
        priors['theta_jn'] = Sine(latex_label='$\\theta_{JN}$')
        priors['phase'] = Uniform(0, 2 * np.pi, latex_label='$\\phi$')
        priors['geocent_time'] = Uniform(
            injection_params['geocent_time'] - 0.01,
            injection_params['geocent_time'] + 0.01,
            latex_label='$t_c$'
        )
        
        # Fix other parameters
        for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra', 'dec', 'psi']:
            priors[key] = injection_params[key]
        
        # Set up likelihood
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=wf_gen
        )
        
        # Run very minimal sampler
        print("  Running sampler with minimal settings...")
        start_time = time.time()
        
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty',
            npoints=50,  # Very low for quick test
            walks=5,     # Very low for quick test
            dlogz=1.0,   # Very loose for quick test
            sample='rwalk',
            outdir=test_dir,
            label='quick_test',
            injection_parameters=injection_params,
            save=False  # Don't save for quick test
        )
        
        runtime = time.time() - start_time
        print(f"✓ Quick PE test completed in {runtime:.1f} seconds")
        
        # Check results
        for param in ['chirp_mass', 'mass_ratio']:
            if param in result.posterior:
                true_val = injection_params[param]
                recovered = np.median(result.posterior[param].values)
                error = abs(recovered - true_val) / true_val * 100
                print(f"  - {param}: true={true_val:.3f}, recovered={recovered:.3f}, error={error:.1f}%")
        
        # Clean up
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
    except Exception as e:
        print(f"✗ PE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED! Ready to run full workflow.")
    print("="*70)
    print("\nTo run the complete workflow, use:")
    print("  python master_run_all.py")
    print("\nExpected runtime: 1-3 hours for full analysis")
    
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)