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

# -----------------------------
# 1. Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="GPT-2 training with variable-length sequences and checkpointing")
parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--min_seq_len", type=int, default=128)
parser.add_argument("--max_seq_len", type=int, default=1024)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
parser.add_argument("--save_every", type=int, default=20)
args = parser.parse_args()

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(args.checkpoint_dir, exist_ok=True)
ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")

# -----------------------------
# 2. Load tokenizer and model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# -----------------------------
# 3. Load dataset
# -----------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{args.num_samples}]")

def tokenize_variable(batch):
    seq_lens = [random.randint(args.min_seq_len, args.max_seq_len) for _ in batch["text"]]
    tokenized = []
    for text, l in zip(batch["text"], seq_lens):
        tokenized.append(tokenizer(text, truncation=True, max_length=l, padding="max_length"))
    out = {k: [d[k] for d in tokenized] for k in tokenized[0].keys()}
    return out

dataset = dataset.map(tokenize_variable, batched=True, remove_columns=["text"])

# -----------------------------
# 4. Data collator & loader
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)

# -----------------------------
# 5. Optimizer
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# -----------------------------
# 6. Resume from checkpoint if exists
# -----------------------------
start_step = 0
if os.path.exists(ckpt_path):
    print(f"Resuming from checkpoint {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_step = checkpoint["step"] + 1

# -----------------------------
# 7. Training loop
# -----------------------------
model.train()
for step, batch in enumerate(loader):
    if step < start_step:
        continue

    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")


    # Save checkpoint periodically
    if step > 0 and step % args.save_every == 0:
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)
        print(f"Checkpoint saved at step {step}")

print("Training complete.")
