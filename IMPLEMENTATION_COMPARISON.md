# DreamerV4 Implementation Comparison

Comparison of reference implementations with our MineRL implementation.
Analysis performed: January 2026

## Reference Implementations Reviewed

| Implementation | Source | Target Domain | GPU Requirements |
|----------------|--------|---------------|------------------|
| **Dreamer4** | V100 impl | Minecraft/Atari | 8× 16GB V100 |
| **dreamer4-experiments** | lucidrains | Generic | Not specified |
| **dreamer4** | Nicklas Hansen | DMControl (30 tasks) | 8× 24GB RTX 3090 |

---

## Dataset Comparison

| Dataset | Frames | Resolution | Storage | Trajectories |
|---------|--------|------------|---------|--------------|
| **Our MineRL (full)** | 4,350,893 | 64×64 | 52GB | 759 |
| **Our MineRL (subset)** | 71,279 | 64×64 | 864MB | 10 |
| **Hansen DMControl** | 3,600,000 | 128×128 | 350GB (processed) | 7,200 |
| **OpenAI Contractor** | Millions | 360×640 | Large | Thousands |

---

## Architecture Comparison

### Tokenizer

| Parameter | **Ours** | Dreamer4 (V100) | Hansen | lucidrains |
|-----------|----------|-----------------|--------|------------|
| embed_dim | 256 | 256 | 256 | 512 |
| depth | 6 | 8 | 8 | 4+4 (enc+dec) |
| num_heads | 8 | 8 | 4 | 8 |
| latent_dim | 32 | 256 | 32 | 32 |
| num_latents | 16 | 64 | 16 | 4 |
| num_registers | 4 | - | - | - |
| patch_size | 8 | 8 | 4 | 32 |
| **Parameters** | **5.66M** | ~15-20M | ~8-12M | ~30-50M |

### Dynamics Model

| Parameter | **Ours** | Dreamer4 (V100) | Hansen | lucidrains |
|-----------|----------|-----------------|--------|------------|
| embed_dim | 256 | 512 | 512 | 512 |
| depth | 6 | 12 | 8 | 4 |
| num_heads | 8 | 8 | 4 | 8 |
| latent_dim | 32 | 256 | 32 | 32 |
| num_registers | 4 | 8 | 4 | 8 |
| max_shortcut_steps | 6 | - | 8 | - |
| **Parameters** | **5.93M** | ~50-80M | ~25-40M | ~40-60M |

### Total Model Size

| Implementation | Tokenizer | Dynamics | Heads | **Total** |
|----------------|-----------|----------|-------|-----------|
| **Ours** | 5.66M | 5.93M | 1.58M | **13.17M** |
| Dreamer4 (V100) | ~15-20M | ~50-80M | ~5M | ~70-100M |
| Hansen | ~8-12M | ~25-40M | ~5M | ~35-55M |
| lucidrains | ~30-50M | ~40-60M | ~10M | ~70-120M |

---

## GPU Memory Analysis

### Why Reference Implementations Need 192GB+ (8×24GB)

Model parameters are only a small fraction of GPU memory. The real memory consumption comes from **activations** stored for backpropagation.

#### Memory Components

| Component | Formula | Our Setup | Hansen (per GPU) |
|-----------|---------|-----------|------------------|
| Parameters | params × 4B | 52MB | 200MB |
| Optimizer (AdamW) | params × 8B | 104MB | 400MB |
| Gradients | params × 4B | 52MB | 200MB |
| **Subtotal** | | **~210MB** | **~800MB** |

#### Attention Memory (Dominant Factor)

```
attention_memory = batch × heads × seq_len² × 4 bytes
```

| Setup | Sequence Length | Attention Memory/Layer |
|-------|-----------------|------------------------|
| **Ours** | 1,344 tokens | ~920MB |
| **Hansen** | 33,408 tokens | ~430GB (requires distribution) |

The **sequence length squared** term dominates:
- Our 64×64 with patch=8: 64 patches/frame × 16 frames = 1,024 + overhead = ~1,344 tokens
- Hansen 128×128 with patch=4: 1,024 patches/frame × 32 frames = ~33,000 tokens

Memory ratio: (33,408/1,344)² × batch_ratio × embed_ratio ≈ **15,000×**

### Our Efficiency Wins

| Design Choice | Memory Savings |
|---------------|----------------|
| 64×64 resolution (native MineRL) | 4× fewer pixels than 128×128 |
| 64 patches vs 1024 | 16× shorter sequence |
| 16 latent tokens vs 64 | 4× fewer latents |
| 6 layers vs 8-12 | ~1.5× fewer activations |
| Batch 16 vs 192 | 12× smaller batch |

---

## Training Time Estimates

### Hardware: RTX 3060 12GB

#### Full MineRL Dataset (4.35M frames)

| Phase | Epochs | Time (Typical) | Time (Optimistic) |
|-------|--------|----------------|-------------------|
| Phase 1: Tokenizer | 100 | 7.9 days | 4.7 days |
| Phase 2: Dynamics | 50 | 3.9 days | 2.4 days |
| Phase 3: Agent (PMPO) | 100 | 7.9 days | 4.7 days |
| **TOTAL** | 250 | **~20 days** | **~12 days** |

#### Current Subset (71K frames)

| Phase | Epochs | Time |
|-------|--------|------|
| Full training | 250 | **~8 hours** |

#### Training Parameters

- Batch size: 16
- Sequence length: 16 frames
- Frames per step: 256
- Steps per epoch (full): 16,995
- Steps per epoch (subset): 278
- Estimated time per step: 0.25-0.60 seconds

### Comparison with Reference

| Setup | Hardware | Dataset | Training Time |
|-------|----------|---------|---------------|
| **Ours (full)** | 1× RTX 3060 | 4.35M frames | ~20 days |
| **Hansen** | 8× RTX 3090 | 3.6M frames | ~3 days |
| **Speedup factor** | 8× GPUs + 12× batch | Similar data | ~7× faster |

---

## Key Architectural Differences

### Our Implementation Features

| Feature | Status | Notes |
|---------|--------|-------|
| Block-causal attention | ✅ | With asymmetric masks for tokenizer |
| RoPE position encoding | ✅ | Plus learned frame embeddings |
| SwiGLU activation | ✅ | Same as references |
| RMSNorm | ✅ | Same as references |
| QK-Norm | ✅ | For training stability |
| Attention soft cap (50.0) | ✅ | Gemma-2 style, not in references |
| Separate CNN decoder | ✅ | For decode-only path (dynamics predictions) |
| GQA support | ✅ | Configurable, not used by default |
| Context corruption (Sec 4.2) | ✅ | τ_ctx for context frames |
| Multi-discrete actions | ✅ | Proper MineRL action space |

### Potential Gaps vs References

1. **Model capacity**: ~5-8× smaller than smallest reference
2. **Latent compression**: 16 tokens × 32 dim = 512 dims vs 16,384 dims (V100)
3. **Dynamics embed_dim**: 256 vs 512 in references
4. **No LPIPS in dynamics eval**: References use for generation quality

---

## Recommendations

### Model Scaling (If Needed)

Within 12GB constraints:
1. Increase depth to 8 (matches Hansen tokenizer)
2. Increase dynamics embed_dim to 384
3. Consider num_latents=32 for more spatial detail
4. Keep batch_size at 16 with gradient accumulation

### Training Strategy Options

#### Option A: Full Dataset, Patient Training
- Time: ~20 days
- Pros: Best possible results
- Cons: Long training time

#### Option B: 20% Subset (~870K frames)
- Time: ~4 days
- Pros: Faster iteration
- Cons: May underfit slightly

#### Option C: Progressive Training
1. Train on subset (8 hours) → validate approach
2. Train on 20% (4 days) → check scaling
3. Train on full (20 days) → final model

### Optimization Options

| Optimization | Potential Speedup |
|--------------|-------------------|
| `torch.compile()` | 1.3-1.5× |
| Increase batch to 24-32 | 1.2× |
| Gradient accumulation | Better convergence |
| Early stopping | 20-40% time savings |
| Reduce depth to 4 | 1.5× (capacity tradeoff) |

---

## Conclusion

Our 13M parameter model is appropriately sized for:
- Single-GPU training on RTX 3060 12GB
- MineRL at native 64×64 resolution
- 4.35M frame dataset

The reference implementations solve a harder problem (multi-task, higher-res, more diverse data) and are scaled accordingly. Our setup represents a reasonable single-domain, single-GPU configuration.

**Key insight**: The bottleneck is data quantity (4.35M frames), not model capacity (13M params). The data/param ratio of ~330 samples/param is healthy and should allow learning without severe overfitting.
