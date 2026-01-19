1. soar dataset:

https://rail.eecs.berkeley.edu/datasets/soar_release/test/

2. MinAtar 

https://github.com/kenjyoung/MinAtar

3. NH Dataset
Downloaded

Trajectories downloaded in form of: Expert, mixed-small, mixed-large

https://huggingface.co/datasets/nicklashansen/dreamer4 

4. Original VPT dataset


Open X-Embodiment “OXE”  (robotics)
URL:  https://robotics-transformer-x.github.io
Stats: ~1 M real-robot episodes, 20 Hz, 720p wrist + base cameras, 6-DoF delta EE + gripper.
Why useful: Largest open real-world robot video–action corpus; already tokenised in RT-X papers, so the action space is discrete/continuous labels you can directly feed to DreamerV4.
RoboNet v2  (robotic manipulation)
URL:  https://robotics-transformer-x.github.io/robonet.html
Stats: 2 k h in 131 kitchens, 640×480 @ 15 FPS, 6-DoF arm + gripper.
Why useful: Static camera + wrist camera, diverse objects, long-horizon stacking/pouring.
BridgeData v2  (widowX arm, 7-DoF)
URL:  https://rail.eecs.berkeley.edu/datasets/bridge_release
Stats: 60 k trajectories, 256×256 @ 10 Hz, language-labelled.
Why useful: Language can be treated as an extra modality (task embedding) exactly like DreamerV4’s “agent token”.
DeepMind Control + Action Labels  (simulated continuous control)
URL:  https://github.com/deepmind/dm_control + https://github.com/google-research/d4rl
Stats: 1 k Hz physics, 84×84 @ 30 FPS videos bundled with the “pixels” branch of D4RL.
Why useful: Cheap, infinite data for debugging; same action space (torques) used in Dreamer-3 papers.
Atari 100k/200M + Sticky Actions  (discrete controller)
URL:  https://github.com/MG2033/Arcade-Learning-Environment + https://github.com/openai/atari-replay-dataset
Stats: 500 M frames, 160×210 @ 60 FPS, 18-button joystick.
Why useful: Standard RL benchmark; DreamerV4’s categorical policy head drops straight in.
CARLA Autonomous-Driving Dataset  (urban driving)
URL:  https://github.com/carla-simulator/autonomous_driving_dataset
Stats: 2 k interactive episodes, 800×600 @ 20 FPS, 2 × forward-facing cameras, continuous steer/throttle/brake.
Why useful: Long-horizon, complex dynamics, good test of shortcut-forcing on high-speed sequences.
GTA-V + Obstacle-Traversal  (simulated urban, ego-centric)
Paper:  “GAIA-1/2”  (Russell et al., 2023/25)
URL:  https://github.com/wayve-ai/gaia-2  (download scripts)
Stats: 2 k h 1080p video, 30 Hz, steering wheel + keyboard actions.
Why useful: Realistic graphics, complex physics, already shown compatible with diffusion-world-models.
MineDojo  (another Minecraft corpus)
URL:  https://github.com/MineDojo/MineDojo
Stats: 1 M YouTube Minecraft clips with automatically extracted ASR, 720p @ 30 FPS; 10 k labelled contractor trajectories with mouse/keyboard.
Why useful: Same action space as VPT but more recent, includes creative-mode and red-stone tasks.
Procgen + Video-Action Dump  (2-D procedural games)
URL:  https://github.com/openai/procgen + https://github.com/leonardoduher/procgen_offline_dataset
Stats: 16 simple arcade games, 256×256 @ 15 FPS, 15-button discrete actions.
Why useful: Fast sanity-check of imagination-training loop; 100 k trajectories per game.
Habitat-Web  (ego-centric indoor navigation)
URL:  https://github.com/facebookresearch/habitat-web
Stats: 120 k human tele-op episodes in 1.5 k Gibson apartments, 640×480 @ 30 FPS, discrete (go fwd, rotate, stop) + continuous velocity.
Why useful: Tests long-horizon memory (navigation graphs) and 3-D continuity.


✅ 1. MinAtar (Already on your list — but worth highlighting)
Domain: Simplified arcade games (Breakout, Seaquest, etc.)
Observations: 10×10 symbolic or rendered RGB
Actions: Discrete (4–10 actions)
Why it’s great:
Fast simulation (~10k FPS)
Designed for world models & RL research
Used in DreamerV3/V4 ablations
Limitation: Too simple vs. Minecraft
🔗 https://github.com/kenjyoung/MinAtar

✅ 2. MineDojo + YouTube Dataset
Domain: Minecraft (same as VPT, but open alternative)
Observations: 640×480 RGB from YouTube videos
Actions: ❌ Not directly provided — but you can infer them:
Use OpenAI’s VPT action tokenizer (publicly released) to label videos.
Or use MineCLIP + behavioral priors to estimate actions.
Size: 730K+ videos, 30K+ annotated clips
Tasks: 200+ natural language tasks
Perfect if you want Minecraft without VPT licensing
🔗 https://minedojo.org/
📄 MineDojo Paper (NeurIPS 2022)

✅ 3. Crafter
Domain: 64×64 procedurally generated 2D world with crafting, mining, building
Observations: RGB (64×64) or symbolic
Actions: Discrete (17 actions: move, sleep, craft, etc.)
Why it’s great:
Minecraft-like mechanics (hierarchical goals, sparse rewards)
Open-source, lightweight, fast
Designed for long-horizon RL
Used in: DreamerV3, EfficientZero, etc.
🔗 https://github.com/danijar/crafter
🎮 Example task: “Obtain diamond” (takes ~10k steps)

✅ 4. Procgen (Especially "Maze", "Heist", "Chaser")
Domain: 16 procedurally generated 2D games
Observations: 64×64 RGB
Actions: Discrete (15 actions max)
Why consider:
Generalization benchmark
Fast, stable, widely used
Limitation: Less "open-world" than Minecraft
🔗 https://github.com/openai/procgen

✅ 5. TextWorld / Jericho (for text-based, but with actions)
Not visual, but included for completeness:
Observations: text descriptions
Actions: text commands (“take sword”, “go north”)
Not suitable for DreamerV4 (which needs pixels), but good for hybrid models.
✅ 6. Malmo / Project Malmo (Minecraft AI Platform)
Domain: Official Microsoft Minecraft AI platform
Observations: RGB, depth, symbolic
Actions: Full Minecraft controls (discrete keyboard + mouse)
Why it’s ideal:
Real Minecraft engine
You can record your own trajectories with human or scripted agents
Generate custom offline datasets with aligned actions
Catch: Requires running Minecraft instances → not pre-recorded
🔗 https://github.com/microsoft/malmo
💡 Use with MineRL data collection tools to record datasets.

✅ 7. MineRL (Competition Datasets)
Domain: Minecraft (treechop, navigate, obtain diamond)
Observations: 64×64 RGB
Actions: Mouse + keyboard (discretized)
Size: 600K+ human demos
License: Research-only, but publicly downloadable
Perfect for DreamerV4-style training
🔗 https://minerl.io/dataset/
📦 Hugging Face: minerl

⚠️ Note: Lower resolution than VPT (64×64 vs 360×640), but actions are real and aligned.

🏆 Best Recommendation for You
Goal
Best Dataset
Minecraft-like, open-source, lightweight
Crafter
Real Minecraft + real actions + public
MineRL
Minecraft at scale (like VPT)
MineDojo + VPT action tokenizer
Custom Minecraft data recording
Malmo + your own demos
Final Tip
If you want Minecraft with actual actions and don’t need VPT-scale data, start with:

MineRL → ready-to-use, action-aligned, public.
Crafter → if you want fast iteration and Minecraft-like logic.