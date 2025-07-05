#!/usr/bin/env python3
"""
Script to add baseline comparison functionality to phase2_complete.py
Run this as: python add_baseline_to_phase2.py
"""

import os

def add_baseline_function():
    """Add the baseline comparison function to phase2_complete.py"""
    
    # The new function to add
    new_function = '''

# ============================================================================
# BASELINE COMPARISON VERSION
# ============================================================================

def Run_Simple_Refinement_Test_With_Baseline(base_asd_files: Dict[str, str], 
                              injection_parameters: Dict = None,
                              outdir: str = 'simple_refinement_test'):
    """
    Run a simplified refinement test with baseline comparison.
    
    Args:
        base_asd_files: Dictionary with paths to H1 and L1 ASD files
        injection_parameters: True injection parameters (if None, use defaults)
        outdir: Output directory
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(outdir, 'test.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting refinement test with baseline comparison")
    
    # Use default parameters if not provided
    if injection_parameters is None:
        injection_parameters = Create_Test_Injection_Parameters()
    
    # Initialize ASD models with just 2 stages (preliminary and final)
    asd_models = {}
    for detector, asd_file in base_asd_files.items():
        asd_models[detector] = ASD_Evolution_Model(asd_file, evolution_stages=2)
    
    # Store results
    results = {}
    
    try:
        # === PHASE 1: Initial PE with preliminary ASD ===
        logging.info("\\n=== PHASE 1: Initial PE with preliminary ASD ===")
        
        # Generate preliminary ASDs (stage 0)
        preliminary_asds = {}
        for detector, model in asd_models.items():
            preliminary_asds[detector] = model.Generate_Evolved_ASD(0)
        
        # Set up interferometers
        ifos_preliminary, waveform_gen = Setup_Interferometers_With_ASD(
            preliminary_asds, injection_parameters
        )
        
        # Run Phase 1 PE
        phase1_result = Run_Phase1_PE(
            ifos_preliminary, 
            waveform_gen,
            injection_parameters,
            outdir,
            'phase1_preliminary'
        )
        results['phase1'] = phase1_result
        
        # === PHASE 2: Refined PE with final ASD ===
        logging.info("\\n=== PHASE 2: Refined PE with final ASD ===")
        
        # Generate final ASDs (stage 1)
        final_asds = {}
        for detector, model in asd_models.items():
            final_asds[detector] = model.Generate_Evolved_ASD(1)
        
        # Set up interferometers with final ASD
        ifos_final, waveform_gen_final = Setup_Interferometers_With_ASD(
            final_asds, injection_parameters
        )
        
        # Create informed priors from Phase 1
        refinement_engine = PE_Refinement_Engine(outdir)
        informed_priors = refinement_engine.Create_Prior_From_Posterior(phase1_result)
        
        # Copy fixed parameters
        for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra', 'dec', 'psi']:
            informed_priors[key] = injection_parameters[key]
        
        # Run refined PE
        refined_result = refinement_engine.Run_Refined_PE(
            ifos_final,
            waveform_gen_final,
            informed_priors,
            injection_parameters,
            'phase2_refined'
        )
        results['refined'] = refined_result
        
        # === BASELINE: Traditional PE from scratch with final ASD ===
        logging.info("\\n=== BASELINE: Traditional PE from scratch with final ASD ===")
        
        baseline_result = Run_Phase1_PE(
            ifos_final,
            waveform_gen_final,
            injection_parameters,
            outdir,
            'baseline_from_scratch'
        )
        results['baseline'] = baseline_result
        
        # Create comparison plots
        Create_Comparison_Plots(results, injection_parameters, outdir)
        
        # Save summary
        summary = {
            'injection_parameters': injection_parameters,
            'phase1_runtime': phase1_result.sampling_time if hasattr(phase1_result, 'sampling_time') else None,
            'refined_runtime': refined_result.sampling_time if hasattr(refined_result, 'sampling_time') else None,
            'baseline_runtime': baseline_result.sampling_time if hasattr(baseline_result, 'sampling_time') else None,
        }
        
        with open(os.path.join(outdir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        logging.info("\\nTest completed successfully!")
        logging.info(f"Results saved to: {outdir}")
        
        return results
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
'''

    # Check if phase2_complete.py exists
    if not os.path.exists('phase2_complete.py'):
        print("✗ phase2_complete.py not found in current directory!")
        return False
    
    # Read the current file
    with open('phase2_complete.py', 'r') as f:
        content = f.read()
    
    # Check if function already exists
    if 'Run_Simple_Refinement_Test_With_Baseline' in content:
        print("✓ Function Run_Simple_Refinement_Test_With_Baseline already exists!")
        return True
    
    # Find where to insert (after the original Run_Simple_Refinement_Test)
    insert_pos = content.rfind('def Run_Simple_Refinement_Test')
    if insert_pos == -1:
        print("✗ Could not find Run_Simple_Refinement_Test function!")
        return False
    
    # Find the end of that function (look for the next function or end of file)
    # This is a simple approach - find the next 'def ' at the start of a line
    search_start = insert_pos + 1
    next_def = content.find('\ndef ', search_start)
    if next_def == -1:
        # No next function, insert at end
        insert_pos = len(content)
    else:
        insert_pos = next_def
    
    # Insert the new function
    new_content = content[:insert_pos] + new_function + content[insert_pos:]
    
    # Backup original file
    import shutil
    shutil.copy2('phase2_complete.py', 'phase2_complete_backup.py')
    print("✓ Created backup: phase2_complete_backup.py")
    
    # Write updated content
    with open('phase2_complete.py', 'w') as f:
        f.write(new_content)
    
    print("✓ Added Run_Simple_Refinement_Test_With_Baseline to phase2_complete.py")
    return True

if __name__ == "__main__":
    success = add_baseline_function()
    if success:
        print("\n✓ Successfully updated phase2_complete.py!")
        print("The baseline comparison function has been added.")
        print("You can now run: python master_run_all.py")
    else:
        print("\n✗ Failed to update phase2_complete.py")
        print("Please add the function manually from phase2_complete_update.py")