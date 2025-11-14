#!/usr/bin/env python3
"""
Master script to run all scheduling experiments in the single_gpu directory.

This script runs all the different scheduling algorithms:
- Round Robin Equal Jobs
- Round Robin Big and Small Jobs
- Lottery Memory Proportional
- Lottery Memory Inverse Proportional
- Parallel GPT-2 Small Jobs
"""

import sys
import runpy
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
# Get the project root (3 levels up: single_gpu -> experiments -> gpu_scheduling -> project root)
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
# Results directory for organized output
RESULTS_DIR = PROJECT_ROOT / "results" / "single_gpu"

# Add project root to Python path so imports work when running directly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define all scheduling scripts to run with their output directory names
SCHEDULING_SCRIPTS = [
    {
        "name": "Round Robin Equal Jobs",
        "path": SCRIPT_DIR / "rr_equal" / "round_robin_equal_jobs.py",
        "module": "gpu_scheduling.experiments.single_gpu.rr_equal.round_robin_equal_jobs",
        "output_dir": "rr_equal"
    },
    {
        "name": "Round Robin Big and Small Jobs",
        "path": SCRIPT_DIR / "rr_big_and_small" / "round_robin_big_and_small_jobs.py",
        "module": "gpu_scheduling.experiments.single_gpu.rr_big_and_small.round_robin_big_and_small_jobs",
        "output_dir": "rr_big_and_small"
    },
    {
        "name": "Lottery Memory Proportional",
        "path": SCRIPT_DIR / "lottery_memory_proportional" / "lottery_memory_proportional.py",
        "module": "gpu_scheduling.experiments.single_gpu.lottery_memory_proportional.lottery_memory_proportional",
        "output_dir": "lottery_memory_proportional"
    },
    {
        "name": "Lottery Memory Inverse Proportional",
        "path": SCRIPT_DIR / "lottery_memory_inv_proportional" / "lottery_memory_inv_proportional.py",
        "module": "gpu_scheduling.experiments.single_gpu.lottery_memory_inv_proportional.lottery_memory_inv_proportional",
        "output_dir": "lottery_memory_inv_proportional"
    },
    {
        "name": "Parallel GPT-2 Small Jobs",
        "path": SCRIPT_DIR / "parallel_gpu2_small" / "parallel_gpu2_small.py",
        "module": "gpu_scheduling.experiments.single_gpu.parallel_gpu2_small.parallel_gpu2_small",
        "output_dir": "parallel_gpu2_small"
    }
]


def run_script(script_info):
    """Run a single scheduling script."""
    name = script_info["name"]
    path = script_info["path"]
    module = script_info["module"]
    output_dir_name = script_info["output_dir"]
    
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"Script: {path}")
    print(f"Output directory: results/single_gpu/{output_dir_name}/")
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
    
    # Create results directory structure
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Results will be saved to: {RESULTS_DIR}\n")
    
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

