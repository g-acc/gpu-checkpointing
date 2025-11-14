# gpu-checkpointing
Applications of GPU checkpointing to scheduling

## Installation

Install dependencies using uv (this will automatically create a virtual environment if needed):

```bash
# Install the package and dependencies (creates .venv automatically)
uv sync
```

Or manually create the venv first:

```bash
# Create virtual environment
uv venv

# Install the package and dependencies
uv sync
```

## Running Experiments

After installation, you can run experiments individually or run all scheduling algorithms at once:

### Run All Experiments

To run all scheduling experiments automatically:

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run all scheduling experiments
python gpu_scheduling/experiments/single_gpu/run_all_schedulers.py
```

Or use uv directly:

```bash
uv run python gpu_scheduling/experiments/single_gpu/run_all_schedulers.py
```

This will execute all available scheduling algorithms in sequence:
- Round Robin Equal Jobs
- Round Robin Big and Small Jobs
- Lottery Memory Proportional
- Lottery Memory Inverse Proportional
- Parallel GPT-2 Small Jobs

Results will be saved to the `results/single_gpu/` directory.

### Run Individual Experiments

To run a specific scheduling experiment:

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run experiments
python -m gpu_scheduling.experiments.single_gpu.rr_equal.round_robin_equal_jobs
```

Or use uv directly:

```bash
uv run python -m gpu_scheduling.experiments.single_gpu.rr_equal.round_robin_equal_jobs
```