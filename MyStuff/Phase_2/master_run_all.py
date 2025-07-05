#!/usr/bin/env python3
"""
Master script to run the complete Phase 2 refinement workflow
This script runs all components in the correct order
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import shutil

# Configuration
BASE_OUTPUT_DIR = '/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2'
ASD_FILES = {
    'H1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/H1_asd_win86400.pkl',
    'L1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/L1_asd_win86400.pkl'
}

def check_prerequisites():
    """Check if all required files and modules are available."""
    print("Checking prerequisites...")
    
    # Check ASD files
    for detector, filepath in ASD_FILES.items():
        if os.path.exists(filepath):
            print(f"✓ {detector} ASD file found: {filepath}")
        else:
            print(f"✗ {detector} ASD file NOT found: {filepath}")
            return False
    
    # Check required modules
    required_modules = ['bilby', 'gwpy', 'numpy', 'matplotlib', 'corner', 'pandas']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ Module '{module}' is available")
        except ImportError:
            print(f"✗ Module '{module}' is NOT installed")
            return False
    
    # Check script files in current directory
    required_scripts = ['phase2_complete.py', 'enhanced_analysis.py', 'master_run_all.py']
    for script in required_scripts:
        if os.path.exists(script):
            print(f"✓ Script '{script}' found")
        else:
            print(f"✗ Script '{script}' NOT found in current directory")
            print(f"  Current directory: {os.getcwd()}")
            return False
    
    return True

def create_output_directory():
    """Create a timestamped output directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BASE_OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    return output_dir

def run_phase2_complete(output_dir):
    """Run the main Phase 2 complete refinement test."""
    print("\n" + "="*70)
    print("STEP 1: Running Phase 2 Complete Refinement Test")
    print("="*70)
    print("This will:")
    print("- Run Phase 1 PE with preliminary ASD")
    print("- Run Phase 2 refined PE with final ASD and informed priors")
    print("- Run baseline PE from scratch with final ASD")
    print("- Create comparison plots")
    print("\nThis may take 1-3 hours depending on your system...")
    
    # Create a temporary script that runs phase2_complete with the correct output directory
    temp_script = f"""
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
base_asd_files = {{
    'H1': '{ASD_FILES['H1']}',
    'L1': '{ASD_FILES['L1']}'
}}

# Run the test
results = run_function(base_asd_files, outdir='{output_dir}')

if results:
    print("\\n✓ Phase 2 refinement test completed successfully!")
else:
    print("\\n✗ Phase 2 refinement test failed!")
    sys.exit(1)
"""
    
    # Write temporary script
    temp_script_path = os.path.join(output_dir, 'run_phase2.py')
    with open(temp_script_path, 'w') as f:
        f.write(temp_script)
    
    # Run the script
    start_time = time.time()
    result = subprocess.run([sys.executable, temp_script_path], 
                          capture_output=False, text=True)
    runtime = time.time() - start_time
    
    print(f"\nPhase 2 test runtime: {runtime/3600:.2f} hours")
    
    if result.returncode != 0:
        print("✗ Phase 2 test failed!")
        return False
    
    return True

def run_analysis(output_dir):
    """Run the analysis script on the results."""
    print("\n" + "="*70)
    print("STEP 2: Running Analysis on Results")
    print("="*70)
    print("This will calculate:")
    print("- Parameter recovery accuracy")
    print("- Computational time savings")
    print("- Precision improvements")
    print("- Summary statistics")
    
    # Create analysis script
    analysis_script = f"""
import sys
sys.path.insert(0, '.')
from enhanced_analysis import analyze_refinement_results

# Run analysis
analyze_refinement_results('{output_dir}')
"""
    
    # Write and run analysis script
    temp_script_path = os.path.join(output_dir, 'run_analysis.py')
    with open(temp_script_path, 'w') as f:
        f.write(analysis_script)
    
    result = subprocess.run([sys.executable, temp_script_path], 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        print("✗ Analysis failed!")
        return False
    
    return True

def copy_scripts_to_output(output_dir):
    """Copy all scripts to output directory for reference."""
    scripts_to_copy = [
        'phase2_complete.py',
        'enhanced_analysis.py',
        'master_run_all.py',
        'quick_test.py',
        'test_fixed_analysis.py'
    ]
    
    scripts_dir = os.path.join(output_dir, 'scripts_used')
    os.makedirs(scripts_dir, exist_ok=True)
    
    for script in scripts_to_copy:
        if os.path.exists(script):
            shutil.copy2(script, scripts_dir)
            print(f"  ✓ Copied {script}")
    
    print(f"\nScripts copied to: {scripts_dir}")

def print_final_summary(output_dir):
    """Print final summary of the run."""
    print("\n" + "="*70)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nOutput files generated:")
    
    expected_files = [
        'phase1_preliminary_result.json',
        'phase2_refined_result.json', 
        'baseline_from_scratch_result.json',
        'parameter_comparison.png',
        'summary.json',
        'analysis_summary.txt',
        'test.log',
        'refinement.log'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  ✓ {filename} ({size:.1f} KB)")
        else:
            print(f"  ✗ {filename} (not found)")
    
    # Try to read and print key results from analysis summary
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    if os.path.exists(summary_file):
        print("\n" + "-"*50)
        print("KEY RESULTS FROM ANALYSIS:")
        print("-"*50)
        with open(summary_file, 'r') as f:
            print(f.read())

def main():
    """Main workflow execution."""
    print("="*70)
    print("PHASE 2 PARAMETER ESTIMATION REFINEMENT - COMPLETE WORKFLOW")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n✗ Prerequisites check failed! Please install missing components.")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Copy scripts for reference
    copy_scripts_to_output(output_dir)
    
    try:
        # Step 1: Run Phase 2 complete test
        if not run_phase2_complete(output_dir):
            print("\n✗ Phase 2 test failed! Check logs for details.")
            sys.exit(1)
        
        # Step 2: Run analysis
        if not run_analysis(output_dir):
            print("\n✗ Analysis failed! Check logs for details.")
            sys.exit(1)
        
        # Print final summary
        print_final_summary(output_dir)
        
    except KeyboardInterrupt:
        print("\n\n✗ Workflow interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ All steps completed successfully!")

if __name__ == "__main__":
    main()