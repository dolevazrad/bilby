#!/usr/bin/env python3
# FILENAME: diagnose_and_fix.py
"""
Diagnose issues with phase2_complete.py and provide solutions
"""

import os
import sys
import ast
import subprocess

def diagnose_phase2():
    """Diagnose what's in phase2_complete.py"""
    
    print("="*70)
    print("PHASE 2 DIAGNOSTIC TOOL")
    print("="*70)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"\nWorking directory: {os.getcwd()}")
    
    # Step 1: Check if phase2_complete.py exists
    if not os.path.exists('phase2_complete.py'):
        print("\n✗ ERROR: phase2_complete.py not found!")
        print("\nSOLUTION: You need to create or copy phase2_complete.py to this directory")
        return False
    
    print("\n✓ Found phase2_complete.py")
    
    # Step 2: Parse the file to find all functions
    print("\nAnalyzing phase2_complete.py...")
    
    try:
        with open('phase2_complete.py', 'r') as f:
            content = f.read()
        
        # Parse AST to find functions
        tree = ast.parse(content)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function name and first line of docstring
                func_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'args': [arg.arg for arg in node.args.args]
                }
                functions.append(func_info)
        
        print(f"\nFound {len(functions)} functions:")
        print("-"*50)
        
        run_functions = []
        for func in functions:
            print(f"Line {func['line']:4d}: {func['name']}({', '.join(func['args'])})")
            if 'run' in func['name'].lower() or 'Run' in func['name']:
                run_functions.append(func)
        
        print(f"\n\nFound {len(run_functions)} functions with 'run' in the name:")
        for func in run_functions:
            print(f"  - {func['name']}")
        
        # Step 3: Check what's missing
        print("\n" + "-"*50)
        print("CHECKING FOR EXPECTED FUNCTIONS:")
        
        expected = [
            'Run_Simple_Refinement_Test_With_Baseline',
            'Run_Simple_Refinement_Test',
            'Run_Complete_Refinement_Test'
        ]
        
        found_expected = False
        for exp_func in expected:
            if any(f['name'] == exp_func for f in functions):
                print(f"  ✓ {exp_func}")
                found_expected = True
            else:
                print(f"  ✗ {exp_func} (missing)")
        
        # Step 4: Provide solution
        print("\n" + "="*70)
        print("SOLUTION:")
        print("="*70)
        
        if found_expected:
            print("\n✓ Your phase2_complete.py has the expected functions!")
            print("\nTo run the workflow:")
            print("  python master_run_all.py")
        else:
            print("\n⚠ Your phase2_complete.py is missing expected functions.")
            print("\nYou have two options:")
            print("\n1. USE THE WRAPPER (Recommended):")
            print("   - Save run_phase2_wrapper.py in this directory")
            print("   - Run: python master_run_all.py")
            print("   - The wrapper will automatically use whatever function you have")
            
            print("\n2. ADD THE MISSING FUNCTION:")
            print("   - Run: python add_baseline_to_phase2.py")
            print("   - This will add the baseline comparison function")
            
            if run_functions:
                best_func = run_functions[0]['name']
                print(f"\n3. UPDATE master_run_all.py TO USE YOUR FUNCTION: {best_func}")
                print("   - Edit master_run_all.py")
                print(f"   - Change 'Run_Simple_Refinement_Test' to '{best_func}'")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error analyzing phase2_complete.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_simple_runner():
    """Create a simple runner that works with any phase2_complete.py"""
    
    runner_content = '''#!/usr/bin/env python3
# FILENAME: simple_runner.py
"""
Simple runner that works with any phase2_complete.py
"""

import os
import sys
import importlib

# Set paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

# Import phase2_complete
import phase2_complete

# Find the run function
run_func = None
for attr_name in dir(phase2_complete):
    if 'run' in attr_name.lower() and callable(getattr(phase2_complete, attr_name)):
        if not attr_name.startswith('_'):
            run_func = getattr(phase2_complete, attr_name)
            print(f"Using function: {attr_name}")
            break

if not run_func:
    print("No run function found!")
    sys.exit(1)

# ASD files
base_asd_files = {
    'H1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/H1_asd_win86400.pkl',
    'L1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/L1_asd_win86400.pkl'
}

# Output directory
outdir = '/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/test_run'

# Run it
print(f"\\nRunning {run_func.__name__}...")
try:
    result = run_func(base_asd_files, outdir=outdir)
    print("\\n✓ Completed!")
except TypeError:
    # Try without outdir
    print("Function doesn't accept outdir, trying without...")
    result = run_func(base_asd_files)
    print("\\n✓ Completed!")
'''
    
    with open('simple_runner.py', 'w') as f:
        f.write(runner_content)
    
    print("\n✓ Created simple_runner.py")
    print("  Run it with: python simple_runner.py")


if __name__ == "__main__":
    success = diagnose_phase2()
    
    if success:
        print("\n\nWould you like me to create a simple runner? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            create_simple_runner()
    
    print("\n" + "="*70)
    print("Diagnostic complete!")
    print("="*70)