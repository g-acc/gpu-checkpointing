from gpu_scheduling import workqueue as wq
import random
from pathlib import Path

"""
Priority scheduling based on memory usage. Equal quanta, lottery priority.
One big job, one small job.
"""

# Organized output directory structure
OUTPUT_DIR = Path("results/single_gpu/lottery_memory_proportional_big_and_small")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CSV_DIR = OUTPUT_DIR / "csvs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

jobs = [
    wq.Job(
        name=str("gpt2-small"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small"),
             "--csv_file", str(CSV_DIR / "gpt2_small.csv")]
    ),
wq.Job(
    name="gpt2-medium",
    cmd=[
        "python", "gpu_scheduling/model_training_scripts/train_gpt2.py",
        "--checkpoint_dir", str(CHECKPOINT_DIR / "medium"),
        "--csv_file", str(CSV_DIR / "gpt2_medium.csv"),
        "--model_name", "gpt2-medium",
        "--batch_size", "2",
        "--min_seq_len", "128",
        "--max_seq_len", "512",
        "--num_samples", "1500",
        "--save_every", "30"
    ]
)
]

def get_next_job_fn(jobs):
    """
    jobs: list[Job]
    Returns an integer index into the jobs list.
    """

    # Build lottery bucket
    ticket_bucket = []
    for idx, job in enumerate(jobs):
        # Convert bytes → ticket count
        tickets = max(int(job.memory_usage_bytes // 1_000_000_000), 1)
        ticket_bucket.extend([idx] * tickets)

    # Safety fallback
    if not ticket_bucket:
        return 0

    # Draw winner
    choice = random.choice(ticket_bucket)
    print("Lottery winner", choice)
    return choice

if __name__ == "__main__":
    round_robin_equal_time_scheduler = wq.Scheduler(get_next_job_fn=get_next_job_fn, get_working_time_fn=lambda _: 75)
    exp = wq.WorkQueue(jobs, round_robin_equal_time_scheduler, str(OUTPUT_DIR))
    exp.manage_schedule()