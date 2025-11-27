from gpu_scheduling import workqueue as wq
from pathlib import Path

# Organized output directory structure
OUTPUT_DIR = Path("results/single_gpu/rr_one_big_many_small/200s_quanta")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CSV_DIR = OUTPUT_DIR / "csvs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

jobs = [
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
),
    wq.Job(
        name=str("gpt2-small first"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small_one"),
             "--csv_file", str(CSV_DIR / "gpt2_small_one.csv")
        ]
    ),    wq.Job(
        name=str("gpt2-small second"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small_two"),
             "--csv_file", str(CSV_DIR / "gpt2_small_two.csv")
        ]
    ),    wq.Job(
        name=str("gpt2-small third"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small_three"),
             "--csv_file", str(CSV_DIR / "gpt2_small_three.csv")
        ]
    ),    wq.Job(
        name=str("gpt2-small fourth"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small_four"),
             "--csv_file", str(CSV_DIR / "gpt2_small_four.csv")
        ]
    )
]

if __name__ == "__main__":
    round_robin_equal_time_scheduler = wq.Scheduler(get_next_job_fn=lambda _: 0, get_working_time_fn=lambda _: 200)
    exp = wq.WorkQueue(jobs, round_robin_equal_time_scheduler, str(OUTPUT_DIR))
    exp.manage_schedule()