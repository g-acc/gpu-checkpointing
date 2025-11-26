import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import random
import argparse
import os
import signal
import sys
import time
import csv
import psutil

# -----------------------------
# 1. Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="GPT-2 training with variable-length sequences, checkpointing, and GPU/MPS stats logging")
parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--min_seq_len", type=int, default=128)
parser.add_argument("--max_seq_len", type=int, default=1024)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
parser.add_argument("--csv_file", type=str, default="./gpu_stats.csv")
parser.add_argument("--save_every", type=int, default=30)
args = parser.parse_args()

# -----------------------------
# 2. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    import pynvml
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()
else:
    num_gpus = 0

# -----------------------------
# 3. Checkpoint setup
# -----------------------------
os.makedirs(args.checkpoint_dir, exist_ok=True)
ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")

def save_checkpoint(step):
    tmp_path = ckpt_path + ".tmp"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, tmp_path)
    os.replace(tmp_path, ckpt_path)
    print(f"[Checkpoint] Saved at step {step}")

current_step = 0

# -----------------------------
# 4. Load tokenizer and model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# -----------------------------
# 5. Load dataset
# -----------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{args.num_samples}]")

def tokenize_variable(batch):
    seq_lens = [random.randint(args.min_seq_len, args.max_seq_len) for _ in batch["text"]]
    tokenized = [tokenizer(text, truncation=True, max_length=l, padding="max_length") for text, l in zip(batch["text"], seq_lens)]
    return {k: [d[k] for d in tokenized] for k in tokenized[0].keys()}

dataset = dataset.map(tokenize_variable, batched=True, remove_columns=["text"])

# -----------------------------
# 6. Data collator & loader
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)

# -----------------------------
# 7. Optimizer
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# -----------------------------
# 8. Resume from checkpoint if exists
# -----------------------------
start_step = 0
if os.path.exists(ckpt_path):
    print(f"Resuming from checkpoint {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_step = checkpoint["step"] + 1
    current_step = start_step

# -----------------------------
# 9. CSV setup
# -----------------------------

fieldnames = ["timestamp", "step", "loss"]
if device.type == "cuda":
    for i in range(num_gpus):
        fieldnames += [f"gpu{i}_mem_used", f"gpu{i}_mem_total", f"gpu{i}_util_gpu", f"gpu{i}_util_mem"]
elif device.type == "mps":
    fieldnames += ["mps_mem_used", "mps_mem_total"]

# Check if file exists and is non-empty
file_exists = os.path.isfile(args.csv_file)
file_empty = not file_exists or os.path.getsize(args.csv_file) == 0

# Open in append mode, line-buffered
csv_file = open(args.csv_file, "a", newline='', buffering=1)
csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

# Only write header if file is empty
if file_empty:
    csv_writer.writeheader()
    csv_file.flush()


# -----------------------------
# 10. Signal handler for checkpoint
# -----------------------------
def handle_exit(signum, frame):
    print(f"\n[Signal {signum}] Exiting. Saving checkpoint at step {current_step}...")
    save_checkpoint(current_step)
    csv_file.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

# -----------------------------
# 11. Training loop
# -----------------------------
model.train()
for step, batch in enumerate(loader):
    if step < start_step:
        continue
    current_step = step

    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Collect device stats
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
        row = {"timestamp": int(time.time()), "step": step, "loss": loss.item()}
        if device.type == "cuda":
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                row.update({
                    f"gpu{i}_mem_used": mem.used,
                    f"gpu{i}_mem_total": mem.total,
                    f"gpu{i}_util_gpu": util.gpu,
                    f"gpu{i}_util_mem": util.memory
                })
        elif device.type == "mps":
            # MPS stats via PyTorch
            mem_used = psutil.virtual_memory().used
            mem_total = psutil.virtual_memory().total
            row.update({"mps_mem_used": mem_used, "mps_mem_total": mem_total})

        csv_writer.writerow(row)
        csv_file.flush()

    if args.save_every != 0 and step > 0 and step % args.save_every == 0:
        save_checkpoint(step)

# -----------------------------
# 12. End of training
# -----------------------------
print("Training complete.")
save_checkpoint(current_step)
csv_file.close()
