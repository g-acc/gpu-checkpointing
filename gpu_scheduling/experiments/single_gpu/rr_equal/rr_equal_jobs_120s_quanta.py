from gpu_scheduling import workqueue as wq
from pathlib import Path

# Organized output directory structure
OUTPUT_DIR = Path("results/single_gpu/rr_equal/120s_quanta")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CSV_DIR = OUTPUT_DIR / "csvs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

jobs = [
    wq.Job(
        name=str("gpt2-small, first instance"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "proc1"),
             "--csv_file", str(CSV_DIR / "gpt2_small_first_instance.csv")]
    ),
    wq.Job(
        name=str("gpt2-small, second instance"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py",  
             "--checkpoint_dir", str(CHECKPOINT_DIR / "proc2"),
             "--csv_file", str(CSV_DIR / "gpt2_small_second_instance.csv")]
    )
]

if __name__ == "__main__":
    round_robin_equal_time_scheduler = wq.Scheduler(get_next_job_fn=lambda _: 0, get_working_time_fn=lambda _: 120)
    exp = wq.WorkQueue(jobs, round_robin_equal_time_scheduler, str(OUTPUT_DIR))
    exp.manage_schedule()