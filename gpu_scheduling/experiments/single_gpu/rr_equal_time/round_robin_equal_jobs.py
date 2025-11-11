from gpu_scheduling import workqueue as wq

jobs = [
            wq.Job(
                name=str("gpt2-small, first instance"),
                cmd=["python", 
                     "gpu_scheduling/model_training_scripts/train_gpt2.py", 
                     "--checkpoint_dir", "./proc1"]
            ),
            wq.Job(
                name=str("gpt2-small, second instance"),
                cmd=["python", 
                     "gpu_scheduling/model_training_scripts/train_gpt2.py",  
                     "--checkpoint_dir", "./proc2"]
            )
        ]

if __name__ == "__main__":
    round_robin_equal_time_scheduler = wq.Scheduler(get_next_job_fn=lambda _: 0, get_working_time_fn=lambda _: 60)
    exp = wq.WorkQueue(jobs, round_robin_equal_time_scheduler, "gpu_scheduling/experiments/single_gpu/rr_equal_time/")
    exp.manage_schedule()