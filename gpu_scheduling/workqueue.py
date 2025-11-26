import time
import subprocess
from pydantic import BaseModel
import sys
import torch
import psutil
import csv
import pynvml
import signal
import threading
import queue

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

    def _output_reader(self, proc, job_name, stream_name, output_queue):
        """Thread function to read output from a subprocess stream and prefix it."""
        try:
            for line in iter(proc.stdout.readline if stream_name == 'stdout' else proc.stderr.readline, b''):
                if line:  # Only process non-empty lines
                    decoded_line = line.decode('utf-8', errors='replace').rstrip()
                    if decoded_line:  # Skip empty lines after stripping
                        prefixed_line = f"[{job_name}] {decoded_line}"
                        output_queue.put(prefixed_line)
        except Exception as e:
            output_queue.put(f"[{job_name}] Error reading {stream_name}: {e}")

    def _display_output(self, output_queue):
        """Thread function to display queued output."""
        while True:
            try:
                line = output_queue.get(timeout=0.1)
                print(line)
                output_queue.task_done()
            except queue.Empty:
                continue

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
            "working_time", "total_running_times", "scheduler_overhead", "total_scheduler_overhead"
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

            total_scheduler_overhead = 0
            while self.jobs:
                scheduler_start = time.time()
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

                # Start all processes with output capture
                output_queue = queue.Queue()
                threads = []
                
                scheduler_end = time.time()
                total_scheduler_overhead += scheduler_end - scheduler_start
                for job in running_jobs:
                    # Start process with captured output
                    proc = subprocess.Popen(
                        job.cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=1,  # Line buffered
                        universal_newlines=False  # We'll decode manually
                    )
                    running_procs.append(proc)

                    # Start output reader threads for stdout and stderr
                    stdout_thread = threading.Thread(
                        target=self._output_reader,
                        args=(proc, job.name, 'stdout', output_queue)
                    )
                    stderr_thread = threading.Thread(
                        target=self._output_reader,
                        args=(proc, job.name, 'stderr', output_queue)
                    )
                    stdout_thread.daemon = True
                    stderr_thread.daemon = True
                    stdout_thread.start()
                    stderr_thread.start()
                    threads.extend([stdout_thread, stderr_thread])

                # Start display thread
                display_thread = threading.Thread(target=self._display_output, args=(output_queue,))
                display_thread.daemon = True
                display_thread.start()

                time.sleep(working_time)

                # Give a moment for any remaining output to be processed
                time.sleep(0.1)
                output_queue.join()  # Wait for all queued output to be displayed

                # Collect metrics
                job_names = [job.name for job in running_jobs]
                total_running_times = [job.running_time for job in running_jobs]

                row = {
                    "timestamp": int(time.time()),
                    "job_names": ";".join(job_names),
                    "working_time": working_time,
                    "total_running_times": ";".join(str(t) for t in total_running_times),
                    "scheduler_overhead": scheduler_end - scheduler_start,
                    "total_scheduler_overhead": total_scheduler_overhead 
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
                            proc.wait(timeout=20)
                            print("Job exited gracefully:", job.name)
                        except subprocess.TimeoutExpired:
                            print("Job did not exit in time, killing:", job.name)
                            proc.kill()  # force kill
                            proc.wait()
                        finally:
                            self.jobs.append(job)  # Put back unfinished job
                    else:
                        print("Job finished:", job.name, "exit code:", poll)