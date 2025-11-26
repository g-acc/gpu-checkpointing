import os
import pandas as pd
import numpy as np

# ================================================================
# Utility: Jain's Fairness Index
# ================================================================
def jain_fairness(values):
    arr = np.array(values, dtype=float)
    if np.sum(arr) == 0:
        return 1.0
    return (np.sum(arr) ** 2) / (len(arr) * np.sum(arr ** 2))


# ================================================================
# Slowdown computation
# ================================================================
def compute_slowdown(df, jobs):
    slowdown = {}
    details = {}

    for job in jobs:
        df_j = df[df["job_names"] == job]
        if df_j.empty:
            continue

        arrival = df_j["timestamp"].min()
        finish = df_j["timestamp"].max()

        compute_time = df_j["working_time"].sum()
        wait_time = (finish - arrival) - compute_time

        if compute_time > 0:
            sd = (compute_time + wait_time) / compute_time
        else:
            sd = 1.0

        slowdown[job] = sd
        details[job] = {
            "arrival_time": arrival,
            "finish_time": finish,
            "compute_time": compute_time,
            "wait_time": wait_time,
            "slowdown": sd,
        }

    return slowdown, details


# ================================================================
# Core analysis for one CSV file
# ================================================================
def analyze_csv(path, combined_file):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR reading {path}: {e}")
        return

    if df.empty:
        print(f"SKIP (empty CSV): {path}")
        return

    required_cols = {
        "timestamp", "job_names", "working_time",
        "total_running_times", "mem_used", "mem_total"
    }
    if not required_cols.issubset(df.columns):
        print(f"SKIP (missing required columns): {path}")
        return

    jobs = df["job_names"].unique()

    # ------------------------------------------------------------
    # Total running time per job
    # ------------------------------------------------------------
    total_runtime = (
        df.groupby("job_names")["working_time"].sum().to_dict()
    )

    # ------------------------------------------------------------
    # Average memory usage
    # ------------------------------------------------------------
    avg_mem = (
        df.groupby("job_names")["mem_used"].mean().to_dict()
    )

    # ------------------------------------------------------------
    # Time before first scheduled
    # ------------------------------------------------------------
    t0 = df["timestamp"].iloc[0]
    first_time = df.groupby("job_names")["timestamp"].min().to_dict()
    time_before_scheduled = {job: first_time[job] - t0 for job in jobs}

    # ------------------------------------------------------------
    # Slowdown
    # ------------------------------------------------------------
    slowdown, slowdown_details = compute_slowdown(df, jobs)

    # ------------------------------------------------------------
    # Fairness
    # ------------------------------------------------------------
    fairness = jain_fairness(list(total_runtime.values()))

    # ------------------------------------------------------------
    # Scheduler overhead
    # ------------------------------------------------------------
    if "scheduler_overhead" in df.columns:
        total_overhead = df["scheduler_overhead"].sum()
    else:
        total_overhead = 0.0

    # ============================================================
    # Append all results to the unified results file
    # ============================================================
    with open(combined_file, "a") as f:
        f.write(f"\n\n==== RESULTS FOR {path} ====\n\n")

        f.write("Total Running Time:\n")
        for j, v in total_runtime.items():
            f.write(f"  {j}: {v:.2f}\n")

        f.write("\nAvg Memory Usage:\n")
        for j, v in avg_mem.items():
            f.write(f"  {j}: {v:.2f}\n")

        f.write("\nTime Before First Scheduled:\n")
        for j, v in time_before_scheduled.items():
            f.write(f"  {j}: {v}\n")

        f.write("\nSlowdown (with actual timing):\n")
        for j, vals in slowdown_details.items():
            f.write(f"  {j}:\n")
            f.write(f"    compute_time:  {vals['compute_time']:.2f} seconds\n")
            f.write(f"    wait_time:           {vals['wait_time']:.2f} seconds\n")
            f.write(f"    slowdown factor:            {vals['slowdown']:.4f}\n")

        f.write(f"\nFairness: {fairness:.4f}\n")
        f.write(f"Total Scheduler Overhead: {total_overhead:.4f} seconds\n")

    print(f"Processed {path}")


# ================================================================
# Directory traversal (ONLY one results file)
# ================================================================
def traverse_and_process(root_dir):
    combined_file = os.path.join(root_dir, "combined_timeseries_results.txt")

    with open(combined_file, "w") as f:
        f.write("### Combined GPU Scheduler Analysis ###\n")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "timeseries.csv":
                full_path = os.path.join(dirpath, filename)
                analyze_csv(full_path, combined_file)

    print(f"\nAll results written to: {combined_file}\n")


# ================================================================
# Main entry point
# ================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python evaluate_timeseries_csv.py <root_results_directory>")
        exit(1)

    root_dir = sys.argv[1]
    traverse_and_process(root_dir)
