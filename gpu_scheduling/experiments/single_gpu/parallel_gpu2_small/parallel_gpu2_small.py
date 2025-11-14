from gpu_scheduling import workqueue as wq
from pathlib import Path

# Organized output directory structure
OUTPUT_DIR = Path("results/single_gpu/parallel_gpu2_small")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CSV_DIR = OUTPUT_DIR / "csvs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

jobs = [
    wq.Job(
        name=str("gpt2-small, first parallel instance"),
        cmd=["python",
             "gpu_scheduling/model_training_scripts/train_gpt2.py",
             "--checkpoint_dir", str(CHECKPOINT_DIR / "proc1"),
             "--csv_file", str(CSV_DIR / "gpt2_small_first_parallel.csv")]
    ),
    wq.Job(
        name=str("gpt2-small, second parallel instance"),
        cmd=["python",
             "gpu_scheduling/model_training_scripts/train_gpt2.py",
             "--checkpoint_dir", str(CHECKPOINT_DIR / "proc2"),
             "--csv_file", str(CSV_DIR / "gpt2_small_second_parallel.csv")]
    )
]

def get_parallel_jobs(jobs_list):
    """Return indices of all available jobs to run in parallel"""
    return list(range(len(jobs_list)))

if __name__ == "__main__":
    parallel_scheduler = wq.Scheduler(get_next_job_fn=get_parallel_jobs, get_working_time_fn=lambda _: 60)
    exp = wq.WorkQueue(jobs, parallel_scheduler, str(OUTPUT_DIR))
    exp.manage_schedule()
