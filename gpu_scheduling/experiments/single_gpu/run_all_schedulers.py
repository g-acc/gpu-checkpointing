#!/usr/bin/env python3
"""
Master script to run all scheduling experiments in the single_gpu directory.

This script runs all the different scheduling algorithms:
- Round Robin Equal Jobs
- Round Robin Big and Small Jobs
- Lottery Memory Proportional
- Lottery Memory Inverse Proportional
"""

import sys
import runpy
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
# Get the project root (3 levels up: single_gpu -> experiments -> gpu_scheduling -> project root)
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# Add project root to Python path so imports work when running directly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define all scheduling scripts to run
SCHEDULING_SCRIPTS = [
    {
        "name": "Round Robin Equal Jobs",
        "path": SCRIPT_DIR / "rr_equal" / "round_robin_equal_jobs.py",
        "module": "gpu_scheduling.experiments.single_gpu.rr_equal.round_robin_equal_jobs"
    },
    {
        "name": "Round Robin Big and Small Jobs",
        "path": SCRIPT_DIR / "rr_big_and_small" / "round_robin_big_and_small_jobs.py",
        "module": "gpu_scheduling.experiments.single_gpu.rr_big_and_small.round_robin_big_and_small_jobs"
    },
    {
        "name": "Lottery Memory Proportional",
        "path": SCRIPT_DIR / "lottery_memory_proportional" / "lottery_memory_proportional.py",
        "module": "gpu_scheduling.experiments.single_gpu.lottery_memory_proportional.lottery_memory_proportional"
    },
    {
        "name": "Lottery Memory Inverse Proportional",
        "path": SCRIPT_DIR / "lottery_memory_inv_proportional" / "lottery_memory_inv_proportional.py",
        "module": "gpu_scheduling.experiments.single_gpu.lottery_memory_inv_proportional.lottery_memory_inv_proportional"
    }
]


def run_script(script_info):
    """Run a single scheduling script."""
    name = script_info["name"]
    path = script_info["path"]
    module = script_info["module"]
    
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"Script: {path}")
    print(f"{'='*80}\n")
    
    # Check if script exists
    if not path.exists():
        print(f"⚠️  Warning: Script not found at {path}")
        return False
    
    # Check if script is empty
    if path.stat().st_size == 0:
        print(f"⚠️  Warning: Script is empty, skipping...")
        return False
    
    try:
        # Change to project root directory since scripts use relative paths
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        try:
            # Use runpy.run_path to execute the script file
            # This properly handles __name__ == "__main__" and imports
            runpy.run_path(str(path), run_name="__main__")
            
            print(f"\n✅ Successfully completed: {name}\n")
            return True
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)
    except Exception as e:
        print(f"\n❌ Error running {name}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all scheduling scripts."""
    print("="*80)
    print("GPU Scheduling Experiments - Running All Schedulers")
    print("="*80)
    
    results = []
    for script_info in SCHEDULING_SCRIPTS:
        success = run_script(script_info)
        results.append((script_info["name"], success))
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED/SKIPPED"
        print(f"{status}: {name}")
    
    # Exit with error code if any failed
    failed_count = sum(1 for _, success in results if not success)
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} script(s) failed or were skipped.")
        sys.exit(1)
    else:
        print("\n✅ All scripts completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

