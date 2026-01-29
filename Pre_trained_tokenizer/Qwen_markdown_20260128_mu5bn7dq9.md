# DreamerV4 Stage 2–3 Reproduction Plan  
*Testing PMPO Imagination RL with Pretrained Tokenizer on Consumer Hardware*

## 🎯 Goal
Reproduce **Stage 2 (task-conditioned BC + reward modeling)** and **Stage 3 (PMPO imagination RL)** of DreamerV4 using a **pretrained tokenizer** (e.g., NVIDIA Cosmos) to bypass Stage 1. Target hardware: **RTX 3060/4090 (12–24 GB VRAM)**.

---

## ✅ Methodology Overview

### Step 1: Use Pretrained Tokenizer (Skip Stage 1)
- **Tokenizer**: NVIDIA Cosmos (`cosmos_dv8x8x8` for discrete tokens)
- **Action**:
  - Run tokenizer **once offline** on dataset (e.g., MineRL)
  - Save latents as `.npz` files
  - **Freeze tokenizer permanently** (no gradients)

> 💡 *Why? Avoids 8–12 days of pretraining; leverages NVIDIA’s large-scale vision model.*

---

### Step 2: Prepare Task-Conditioned Data
- **Dataset Requirements**:
  - Latents (from Cosmos)
  - Actions (keyboard/mouse or joint velocities)
  - Task labels (e.g., `"obtain_iron"`)
- **Sequence Format**:  
  `[TASK_TOKEN, a₀, z₀, a₁, z₁, ..., aₜ, zₜ]`

---

### Step 3: Train Stage 2 — Multi-Task BC + Reward Model
- **Model**: Scaled-down transformer (12 layers, 256 tokens/frame)
- **Heads**:
  - **Policy head**: Action prediction (BC loss)
  - **Reward head**: Symexp two-hot classification
- **Training**:
  - Freeze backbone; train only MLP heads initially
  - Optional: Unfreeze last 2–4 transformer layers later
- **Hardware**: 1× RTX 4090 or 2× RTX 3060 (with ZeRO)
- **Time**: 12–36 hours

---

### Step 4: Implement Stage 3 — PMPO Imagination RL
- **World Model**: Fully frozen (tokenizer + dynamics)
- **Rollouts**:
  - Generate trajectories **inside world model** (no env interaction)
  - Length: 32–64 steps (adjust for VRAM)
- **PMPO Loss**:
  ```python
  advantages = compute_gae(rewards, values)
  signed_adv = torch.sign(advantages).detach()  # Critical: use only sign!
  loss = -(signed_adv * log_prob(actions)).mean()