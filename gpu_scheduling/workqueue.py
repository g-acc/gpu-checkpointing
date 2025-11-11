import time
import subprocess
from pydantic import BaseModel
import sys
import torch
import psutil


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
    def __init__(self, jobs, scheduler):
        self.jobs = jobs
        self.scheduler = scheduler

    def manage_schedule(self):
        while self.jobs:
            # TODO: report execution chain, memory usage in a timeseries log
            print("Num jobs", len(self.jobs))
            job = self.jobs.pop(self.scheduler.get_next_job_fn(self.jobs))
            working_time = self.scheduler.get_working_time_fn(self.jobs)
            print("_______________________________________________")
            print("Running", job.name, "for", working_time, "secs")
            job.running_time += working_time
            proc = subprocess.Popen(job.cmd, stdout=sys.stdout, stderr=sys.stderr)
            time.sleep(working_time)
            if DEVICE == "mps":
                job.memory_usage_bytes = psutil.virtual_memory().used
            elif DEVICE == "cuda":
                job.memory_usage_bytes = torch.cuda.memory_allocated(0)
            print("Running time", job.running_time, "seconds")
            print("Mem usage", job.memory_usage_bytes / 1000000000, "GB",)
            poll = proc.poll()
            if poll is None:
                proc.terminate()
                print("Terminated job", job.name, "appending to queue")
                self.jobs.append(job)
            else:
                print("Job finished: ", job.name, poll)
                proc.stdout
