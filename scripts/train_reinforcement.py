import torch
import torch.nn.functional as F
import gymnasium as gym
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. Environment
# -----------------------------
env_name = "CartPole-v1"
env = gym.make(env_name)
num_actions = env.action_space.n

# -----------------------------
# 3. Policy model
# -----------------------------
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
policy = AutoModelForCausalLM.from_pretrained(model_name).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    policy.config.pad_token_id = policy.config.eos_token_id

policy.train()

# -----------------------------
# 4. Hyperparameters
# -----------------------------
max_episodes = 20
max_steps_base = 50
max_steps_extra = 50
seq_len_base = 16
seq_len_extra = 32
long_episode_prob = 0.3  # 30% of episodes are "long"
gamma = 0.99

replay_buffer = []
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

# -----------------------------
# 5. Training loop with spikes
# -----------------------------
for ep in range(max_episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0

    # Base step count
    max_steps = max_steps_base + random.randint(0, max_steps_extra)

    # Occasionally trigger long-horizon episode
    if random.random() < long_episode_prob:
        # Double sequence length and step count to spike memory
        seq_len = seq_len_base + 2 * seq_len_extra
        max_steps += max_steps_extra
    else:
        seq_len = seq_len_base + int(seq_len_extra * (ep / max_episodes))

    past_obs = []

    for step in range(max_steps):
        # Stack past observations to simulate memory-hungry sequences
        past_obs.append(str(obs))
        if len(past_obs) > 4:  # keep last 4 observations
            past_obs.pop(0)
        obs_str = " ".join(past_obs * (seq_len // len(past_obs)))

        inputs = tokenizer(obs_str, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=seq_len).to(device)

        outputs = policy(**inputs)
        logits = outputs.logits
        last_logits_step = logits[0, -1, :num_actions]
        probs = F.softmax(last_logits_step, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        replay_buffer.append((obs, action, reward, next_obs, done))

        # Policy gradient update
        loss = -torch.log(probs[action]) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # VRAM logging
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(
            f"Episode {ep} | Step {step} | SeqLen {seq_len} | Allocated: {allocated:.1f}MB | Reserved: {reserved:.1f}MB | ReplayBuf: {len(replay_buffer)}"
        )

        if done:
            break
        obs = next_obs

print("Training complete.")

