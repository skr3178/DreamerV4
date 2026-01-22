The best way to test the entire pipeline in your repo (https://github.com/skr3178/DreamerV4), especially when using **MineRL** instead of the full VPT dataset, is to follow a staged, incremental validation approach. This helps catch issues early (data loading, tokenization, model forward passes, losses decreasing, imagination rollouts coherent) before wasting GPU time on long runs.

Your repo already has good scaffolding for this — scripts like `test_pipeline.py`, `analyze_dataset.py`, `inspect_minerl_dataset.py`, `extract_minerl_frames.py`, and phase-specific training files (`train_phase*.py`). Leverage those heavily.

### Recommended Testing Strategy (from easiest/fastest to full end-to-end)

1. **Dataset & Preprocessing Sanity (fastest, CPU-only, minutes)**
   - Run the inspection / analysis scripts first:
     ```
     python inspect_minerl_dataset.py   # checks structure, action keys, frame shapes, etc.
     python analyze_dataset.py          # stats: how many episodes, avg length, action distribution, missing data?
     ```
   - Then extract a **tiny subset** (e.g. 5–20 episodes or ~1–5 hours total gameplay):
     - Modify extraction script or config to limit number of trajectories/files.
     - Run:
       ```
       python extract_minerl_frames.py --subset-size 10 --output-dir data/minerl_subset/
       ```
   - Visually verify:
     ```
     python create_video_from_frames.py --input-dir data/minerl_subset/frames/ --output video_sample.mp4
     python view_mcap.py   # if using MCAP format
     ```
   - Goal: confirm frames look like Minecraft gameplay, actions are parsed sensibly (23 binary keys + 121-class mouse), no crashes on loading.

2. **Phase 1 – World Model Pretraining (most critical to get right)**
   - Use your tiny subset (~few thousand frames total).
   - Run tokenizer training alone first (disable dynamics if possible via config):
     ```
     python train_phase1.py --config configs/phase1_tokenizer_only.yaml --max-steps 2000 --batch-size 4 --subset data/minerl_subset/
     ```
     → Watch for: reconstruction quality (MSE + LPIPS going down), latents staying in reasonable range after tanh.
   - Then full Phase 1 (tokenizer + dynamics with shortcut forcing):
     ```
     python train_phase1.py --config configs/phase1_full.yaml --max-steps 5000 --context-length 32 --batch-size 4
     ```
     → Monitor:
       - Loss components (flow-matching / shortcut loss, ramp weight effect)
       - Generate short rollouts every 1k–2k steps → run a generation script (if exists) or add one that samples K=4 steps from a starting frame + dummy actions → check if video stays somewhat coherent (no instant collapse to noise/artifacts).
   - Quick qualitative test: feed real past frames + zero/no-op actions → does the model predict reasonable future frames?

3. **Phase 2 – Behavior Cloning + Reward/Value Heads (agent finetuning)**
   - Use the same small subset.
   - Only after Phase 1 checkpoint looks stable:
     ```
     python train_phase2.py --world-model-checkpoint logs/phase1/best_model.pt --max-steps 3000 --batch-size 4
     ```
     → Check:
       - Action prediction accuracy (cross-entropy on keyboard/mouse)
       - Reward prediction MSE (if sparse rewards exist in MineRL data)
       - Whether heads overfit quickly on tiny data (expected → good sign model can learn)

4. **Phase 3 – Imagination RL with PMPO (the hardest / most fragile part)**
   - Start with frozen world model + tiny imagination budget.
     ```
     python train_phase3.py --phase2-checkpoint logs/phase2/final.pt --imagination-steps 8 --num-imagined-trajectories 64 --max-epochs 20
     ```
     → Key things to watch:
       - Imagined trajectories: log some generated latents → decode to frames periodically (add logging if missing) → do they look like plausible Minecraft sequences?
       - Advantage binning working (D+ and D- populations exist and are balanced-ish)
       - Policy loss terms: negative advantages get pushed down, positive up, KL not exploding
       - Value loss converging (TD error decreasing)
   - If PMPO feels unstable → try higher β (KL reg), smaller learning rate, or fall back to simpler BC/REINFORCE baseline temporarily.

5. **Full Pipeline Smoke Test (end-to-end on small data)**
   - Chain them in one go with a tiny config:
     - 1–2k steps Phase 1 → save checkpoint
     - Load into Phase 2 → 1–2k steps
     - Load into Phase 3 → 10–30 short imagination epochs
   - Use your `test_pipeline.py` (or extend it) to automate this chain and report success/failure metrics:
     - Reconstruction PSNR/SSIM on hold-out frames
     - Action BCE / categorical accuracy on hold-out
     - Imagined rollout length before divergence (visual inspection or simple metric like average latent norm explosion)

### Practical Tips for Fast Iteration
- **Subset aggressively**: Start with 1 environment / 5–10 episodes → scale up only when everything is stable.
- **Reduce context length** initially (32–64 instead of 192) and small batch size (2–8).
- **Lower model size** if possible (smaller hidden dim, fewer layers) for debug runs.
- **Logging & Checkpoints**: Log every 100–500 steps: losses, sample reconstructions, short imagined videos.
- **Hardware**: Even on single RTX 4090 / A100 you can debug Phase 1+2 with tiny data in <1 hour per run.
- **Common failure modes to watch**:
  - Latents exploding → gradient clipping, better init, QK-norm issues
  - Sampling collapses → ramp weight, bootstrap loss, or x-pred vs velocity mismatch
  - PMPO unstable → advantage normalization, bin sizes, α/β tuning

Once the pipeline runs end-to-end on a small MineRL subset without crashing **and** produces vaguely sensible imagined videos + policy improving on imagined returns, you have good confidence the code structure is correct. Only then scale to larger portions of MineRL (or full if you manage to process it).

If `test_pipeline.py` doesn't yet chain everything + report metrics — consider adding that as your "integration test". Good luck — this is an ambitious reproduction, so catching bugs early with small data is key!