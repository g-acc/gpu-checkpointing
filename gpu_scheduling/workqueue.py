import time
import subprocess
from pydantic import BaseModel
import sys
import torch
import psutil
import csv
import pynvml
import signal

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print("Device", DEVICE)
if DEVICE == "cuda":
    pynvml.nvmlInit()  # initialize NVML

class Job(BaseModel):
    name: str
    cmd: list[str]
    memory_usage_bytes: int = 0
    running_time: int = 0
    total_epochs: int = 0

class Scheduler:
    def __init__(self, get_next_job_fn, get_working_time_fn):
        self.get_next_job_fn = get_next_job_fn
        self.get_working_time_fn = get_working_time_fn

class WorkQueue():
    def __init__(self, jobs, scheduler, metrics_output_dir):
        self.jobs = jobs
        self.scheduler = scheduler
        self.metrics_output_dir = metrics_output_dir

    def get_gpu_stats(self):
        """Return memory used, total memory, and utilization for all GPUs"""
        stats = []
        if DEVICE == "cuda":
            num_gpus = pynvml.nvmlDeviceGetCount()
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats.append({
                    "mem_used": mem.used,
                    "mem_total": mem.total,
                    "util_gpu": util.gpu,
                    "util_mem": util.memory
                })
        return stats

    def manage_schedule(self):
        # Prepare CSV fieldnames
        fieldnames = [
            "timestamp", "job_names",
            "working_time", "total_running_times"
        ]
        if DEVICE == "cuda":
            num_gpus = pynvml.nvmlDeviceGetCount()
            for i in range(num_gpus):
                fieldnames += [
                    f"gpu{i}_mem_used", f"gpu{i}_mem_total",
                    f"gpu{i}_util_gpu", f"gpu{i}_util_mem"
                ]
        elif DEVICE == "mps":
            fieldnames += ["mem_used", "mem_total"]

        with open(self.metrics_output_dir + "/timeseries.csv", "w", newline='') as timeseries:
            writer = csv.DictWriter(timeseries, fieldnames=fieldnames)
            writer.writeheader()

            while self.jobs:
                print("Num jobs", len(self.jobs))
                # Get jobs to run (can be single or multiple)
                job_indices = self.scheduler.get_next_job_fn(self.jobs)
                if isinstance(job_indices, int):
                    job_indices = [job_indices]  # Handle single job case

                running_jobs = []
                running_procs = []

                # Sort indices in descending order to avoid index shifting when popping
                job_indices = sorted(set(job_indices), reverse=True)
                for idx in job_indices:
                    if idx < len(self.jobs):
                        job = self.jobs.pop(idx)
                        running_jobs.append(job)

                if not running_jobs:
                    continue  # No valid jobs to run

                working_time = self.scheduler.get_working_time_fn(self.jobs)

                print("_______________________________________________")
                print("Running", len(running_jobs), "jobs for", working_time, "secs:")
                for job in running_jobs:
                    job.running_time += working_time
                    print(" -", job.name)

                # Start all processes
                for job in running_jobs:
                    proc = subprocess.Popen(job.cmd, stdout=sys.stdout, stderr=sys.stderr)
                    running_procs.append(proc)

                time.sleep(working_time)

                # Collect metrics
                job_names = [job.name for job in running_jobs]
                total_running_times = [job.running_time for job in running_jobs]

                row = {
                    "timestamp": int(time.time()),
                    "job_names": ";".join(job_names),
                    "working_time": working_time,
                    "total_running_times": ";".join(str(t) for t in total_running_times)
                }

                if DEVICE == "mps":
                    mem_used = psutil.virtual_memory().used
                    mem_total = psutil.virtual_memory().total
                    row.update({"mem_used": mem_used, "mem_total": mem_total})
                elif DEVICE == "cuda":
                    gpu_stats = self.get_gpu_stats()
                    for i, g in enumerate(gpu_stats):
                        row.update({
                            f"gpu{i}_mem_used": g["mem_used"],
                            f"gpu{i}_mem_total": g["mem_total"],
                            f"gpu{i}_util_gpu": g["util_gpu"],
                            f"gpu{i}_util_mem": g["util_mem"]
                        })

                print(row)
                writer.writerow(row)
                timeseries.flush()

                # Terminate all processes and check completion
                for job, proc in zip(running_jobs, running_procs):
                    poll = proc.poll()
                    if poll is None:
                        proc.send_signal(signal.SIGTERM)  # send SIGTERM
                        try:
                            proc.wait(timeout=10)
                            print("Job exited gracefully:", job.name)
                        except subprocess.TimeoutExpired:
                            print("Job did not exit in time, killing:", job.name)
                            proc.kill()  # force kill
                            proc.wait()
                        finally:
                            self.jobs.append(job)  # Put back unfinished job
                    else:
                        print("Job finished:", job.name, "exit code:", poll)