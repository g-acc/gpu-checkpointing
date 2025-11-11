import time
import subprocess
from pydantic import BaseModel
import sys
import torch
import psutil
import csv


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print("Device", DEVICE)

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


    def manage_schedule(self):
        with open(self.metrics_output_dir + "/timeseries.csv", "w", newline='') as timeseries:
            writer = csv.DictWriter(timeseries, fieldnames=
                                    ["timestamp","job_name","memory_usage","total_memory",
                                     "working_time","total_running_time"])
            writer.writeheader()
            while self.jobs:
                print("Num jobs", len(self.jobs))

                job = self.jobs.pop(self.scheduler.get_next_job_fn(self.jobs))
                working_time = self.scheduler.get_working_time_fn(self.jobs)
                job.running_time += working_time

                print("_______________________________________________")
                print("Running", job.name, "for", working_time, "secs")

                proc = subprocess.Popen(job.cmd, stdout=sys.stdout, stderr=sys.stderr)
                time.sleep(working_time)
                total_memory = 0
                if DEVICE == "mps":
                    job.memory_usage_bytes = psutil.virtual_memory().used
                    total_memory = psutil.virtual_memory().total
                elif DEVICE == "cuda":
                    job.memory_usage_bytes = torch.cuda.memory_allocated(0)
                    total_memory = torch.cuda.get_device_properties(0).total_memory

                print("Running time", job.running_time, "seconds")
                print("Mem usage", job.memory_usage_bytes / 1000000000, "GB",)
                writer.writerow({
                    "timestamp" : int(time.time()),
                    "job_name": job.name,
                    "memory_usage": job.memory_usage_bytes,
                    "total_memory": total_memory,
                    "working_time": working_time,
                    "total_running_time": job.running_time
                })
                timeseries.flush()
                poll = proc.poll()
                if poll is None:
                    proc.terminate()
                    print("Terminated job", job.name, "appending to queue")
                    self.jobs.append(job)
                else:
                    print("Job finished: ", job.name, poll)
                    proc.stdout
