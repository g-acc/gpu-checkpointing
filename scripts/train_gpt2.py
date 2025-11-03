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

# -----------------------------
# 1. Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="GPT-2 training with variable-length sequences")
parser.add_argument("--model_name", type=str, default="gpt2-medium")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--min_seq_len", type=int, default=128)
parser.add_argument("--max_seq_len", type=int, default=1024)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--lr", type=float, default=5e-5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Randomly truncate to variable lengths to stress memory
def tokenize_variable(batch):
    seq_lens = [random.randint(args.min_seq_len, args.max_seq_len) for _ in batch["text"]]
    tokenized = []
    for text, l in zip(batch["text"], seq_lens):
        tokenized.append(tokenizer(text, truncation=True, max_length=l, padding="max_length"))
    # Convert to dict of lists
    out = {k: [d[k] for d in tokenized] for k in tokenized[0].keys()}
    return out

dataset = dataset.map(tokenize_variable, batched=True, remove_columns=["text"])

# -----------------------------
# 4. Data collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=data_collator
)

# -----------------------------
# 5. Optimizer
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# -----------------------------
# 6. Training loop
# -----------------------------
model.train()
for step, batch in enumerate(loader):
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

