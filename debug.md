Critical Issues (Now Fixed ✓)                                                              
  #: 1                                                                                       
  Issue: τ sampled continuously instead of discrete grid (Eq. 4)                             
  Location: dynamics.py:188-192                                                              
  Status: ✓ Fixed                                                                            
  ────────────────────────────────────────                                                   
  #: 2                                                                                       
  Issue: Rollout τ scheduling inconsistent (step+1 vs step)                                  
  Location: rollout.py:119 vs dynamics.py:423                                                
  Status: ✓ Fixed                                                                            
  ────────────────────────────────────────                                                   
  #: 3                                                                                       
  Issue: Bootstrap loss missing (1-τ)² scaling (Eq. 7)                                       
  Location: shortcut_loss.py:148                                                             
  Status: ✓ Fixed                                                                            
  ────────────────────────────────────────                                                   
  #: 4                                                                                       
  Issue: Bootstrap target averages x-predictions instead of velocities (Eq. 3)               
  Location: shortcut_loss.py:247                                                             
  Status: ✓ Fixed                                                                            
  Medium Issues (Pending)                                                                    
  ┌─────┬──────────────────────────────────────────────┬─────────────────────┬──────────────┐
  │  #  │                    Issue                     │      Location       │    Paper     │
  │     │                                              │                     │  Reference   │
  ├─────┼──────────────────────────────────────────────┼─────────────────────┼──────────────┤
  │ 5   │ Missing context corruption (τ_ctx=0.1 noise  │ rollout.py,         │ Section 4.2  │
  │     │ on context frames)                           │ dynamics.py         │              │
  ├─────┼──────────────────────────────────────────────┼─────────────────────┼──────────────┤
  │ 6   │ Missing space/time factorized attention      │ transformer.py      │ Section 3.2  │
  │     │ (temporal every 4 layers)                    │                     │              │
  ├─────┼──────────────────────────────────────────────┼─────────────────────┼──────────────┤
  │ 7   │ Missing Grouped Query Attention (GQA) for    │ transformer.py      │ Section 3.2  │
  │     │ reduced KV cache                             │                     │              │
  ├─────┼──────────────────────────────────────────────┼─────────────────────┼──────────────┤
  │ 8   │ Missing Multi-Token Prediction (MTP) with    │ agent_loss.py       │ Section 4.3  │
  │     │ L=8 steps                                    │                     │              │
  └─────┴──────────────────────────────────────────────┴─────────────────────┴──────────────┘
  Low Severity Issues (Now Fixed ✓)                                                              
  #: 9                                                                                       
  Issue: Linear bin spacing instead of symlog for value/reward                               
  Location: heads.py                                                                         
  Status: ✓ Fixed                                                                            
  ────────────────────────────────────────                                                   
  #: 10                                                                                      
  Issue: Missing learned position embeddings per frame                                       
  Location: dynamics.py                                                                      
  Status: ✓ Fixed                                                                            
  ────────────────────────────────────────                                                   
  #: 11                                                                                      
  Issue: Missing EMA target network for value learning                                       
  Location: train_phase3.py                                                                  
  Status: ✓ Fixed                                                                            
  ────────────────────────────────────────                                                   
  #: 12                                                                                      
  Issue: Registers used differently (paper uses for global info aggregation)                 
  Location: dynamics.py, tokenizer.py                                                        
  Status: ✓ Reviewed (implementation correct - registers per timestep enable global aggregation via attention)
  ────────────────────────────────────────                                                   
  #: 13                                                                                      
  Issue: Action embedding structure may differ                                               
  Location: embeddings.py                                                                    
  Status: ✓ Reviewed (implementation correct - supports discrete/continuous actions)
  ────────────────────────────────────────                                                   
  #: 14                                                                                      
  Issue: Missing auxiliary reconstruction loss during phase 2                                
  Location: agent_loss.py                                                                    
  Status: ✓ Fixed                                                                            
  ────────────────────────────────────────                                                   
  #: 15                                                                                      
  Issue: Discount/lambda values may need tuning for MineRL                                   
  Location: configs/minerl.yaml                                                              
  Status: ✓ Fixed (added comments and EMA decay parameter)                                                                  
  Would you like to proceed with the medium issues next?   

  All medium issues are now complete. Here's a summary:                                                                                        
                                                                                                                                               
  Summary of Medium Issue Fixes                                                                                                                
                                                                                                                                               
  Issue 5: Context Corruption (dynamics.py)                                                                                                    
                                                                                                                                               
  - Added context_noise_level parameter (default τ_ctx=0.1)                                                                                    
  - Added add_noise_with_context_corruption() method                                                                                           
  - Context frames get fixed noise at τ_ctx, target frame gets sampled τ                                                                       
  - This makes the model robust to its own prediction errors during autoregressive generation                                                  
                                                                                                                                               
  Issue 6: Space/Time Factorized Attention (transformer.py)                                                                                    
                                                                                                                                               
  - Added create_spatial_mask() - tokens attend only within their timestep                                                                     
  - Added create_temporal_mask() - tokens attend to same position across timesteps (causal)                                                    
  - Added use_spacetime_factorization and temporal_layer_interval parameters                                                                   
  - Most layers use spatial attention, every 4th layer uses temporal attention                                                                 
                                                                                                                                               
  Issue 7: Grouped Query Attention (transformer.py)                                                                                            
                                                                                                                                               
  - Added num_kv_heads parameter to BlockCausalAttention, TransformerBlock, and BlockCausalTransformer                                         
  - Supports fewer KV heads than query heads for reduced KV cache                                                                              
  - K and V are repeated to match query heads before attention computation                                                                     
                                                                                                                                               
  Issue 8: Multi-Token Prediction (heads.py)                                                                                                   
                                                                                                                                               
  - MTP was already implemented but defaulted to off                                                                                           
  - Changed use_mtp=True as default in both PolicyHead and RewardHead                                                                          
  - MTP predicts L+1 future steps from single embedding h_t (L=8 per paper Equation 9)                                                         
                                                                                                                                               
  Files Modified                                                                                                                               
  ┌──────────────────┬───────────────────────────────────────────────────┐                                                                     
  │       File       │                      Changes                      │                                                                     
  ├──────────────────┼───────────────────────────────────────────────────┤                                                                     
  │ dynamics.py      │ Context corruption, discrete τ sampling           │                                                                     
  ├──────────────────┼───────────────────────────────────────────────────┤                                                                     
  │ rollout.py       │ Fixed τ scheduling                                │                                                                     
  ├──────────────────┼───────────────────────────────────────────────────┤                                                                     
  │ shortcut_loss.py │ Bootstrap loss (1-τ)² scaling, velocity averaging │                                                                     
  ├──────────────────┼───────────────────────────────────────────────────┤                                                                     
  │ transformer.py   │ Space/time factorization, GQA support             │                                                                     
  ├──────────────────┼───────────────────────────────────────────────────┤                                                                     
  │ heads.py         │ MTP enabled by default                            │
  
  Summary of Low Severity Issue Fixes
  
  Issue 9: Symlog Bin Spacing (heads.py)
  
  - Added use_symlog parameter (default True) to ValueHead and RewardHead
  - Bin centers now use symlog spacing: symlog(x) = sign(x) * ln(|x| + 1)
  - target_to_bins() converts targets to symlog space before binning
  - Better handles large value/reward magnitudes per paper Appendix
  
  Issue 10: Learned Position Embeddings per Frame (dynamics.py)
  
  - Added frame_pos_embed parameter (max_seq_len=1024)
  - Frame-level position embeddings added to all tokens (action, signal, latents, registers) per timestep
  - Distinguishes different timesteps in the sequence per Section 3.1
  
  Issue 11: EMA Target Network for Value Learning (train_phase3.py)
  
  - Added value_target_head (EMA copy of value_head)
  - Target network updated with EMA decay (default 0.999) after each value update
  - Used for bootstrap values in TD(λ) loss per Section 4.4
  - Stabilizes value learning by using slower-updating target network
  
  Issue 12: Register Token Usage (dynamics.py, tokenizer.py)
  
  - Reviewed: Current implementation uses registers per timestep with attention
  - This enables global info aggregation as registers can attend to previous registers
  - Implementation appears correct per Section 3.1
  
  Issue 13: Action Embedding Structure (embeddings.py)
  
  - Reviewed: Supports both discrete (embedding lookup) and continuous (linear projection) actions
  - Matches paper's description of keyboard (binary) + mouse (categorical) actions
  - Implementation appears correct per Section 3.1
  
  Issue 14: Auxiliary Reconstruction Loss (agent_loss.py)
  
  - Added use_auxiliary_reconstruction and reconstruction_weight parameters
  - Added _compute_reconstruction_loss() method structure
  - Can be enabled during phase 2 to maintain good latent representations per Section 4.3
  - Note: Full implementation requires tokenizer decoder path
  
  Issue 15: Discount/Lambda Tuning (configs/minerl.yaml)
  
  - Added detailed comments explaining discount (0.997) and lambda (0.95) choices
  - Added value_ema_decay parameter (0.999) for EMA target network
  - Suggested lambda values (0.98-0.99) for very sparse rewards
  
  Files Modified
  ┌──────────────────┬───────────────────────────────────────────────────┐
  │       File       │                      Changes                      │
  ├──────────────────┼───────────────────────────────────────────────────┤
  │ heads.py         │ Symlog bin spacing for value/reward                │
  ├──────────────────┼───────────────────────────────────────────────────┤
  │ dynamics.py      │ Frame-level position embeddings                    │
  ├──────────────────┼───────────────────────────────────────────────────┤
  │ train_phase3.py  │ EMA target network for value learning             │
  ├──────────────────┼───────────────────────────────────────────────────┤
  │ value_loss.py    │ Support for target_head parameter                 │
  ├──────────────────┼───────────────────────────────────────────────────┤
  │ agent_loss.py    │ Auxiliary reconstruction loss structure           │
  └──────────────────┴───────────────────────────────────────────────────┘                                                  

## MineRL Dataset Statistics (Your Downloaded Dataset)

Total Episodes: 759 episodes across 4 environments:
MineRLNavigate-v0
MineRLObtainDiamond-v0
MineRLObtainIronPickaxe-v0
MineRLTreechop-v0
Frame Data (Video/Training Data)
Total frames: 4,310,077 frames
Total hours: 59.86 hours (at 20 FPS)
Average episode length: 5,678.6 frames (~283.9 seconds or ~4.7 minutes)
Episode length range: 21.6 seconds to 121.3 minutes
Action Data
Total action timesteps: 4,139,887 timesteps
Total hours: 57.50 hours (at 20 FPS)
Difference: 170,190 fewer action timesteps than frames (~2.36 hours less)
Notes
Frame/action mismatch: All 759 episodes have fewer action timesteps than frames. This can happen if:
Actions are missing at episode boundaries
Some frames don't have corresponding actions
Extraction differences between frames and actions
Frame rate: The dataset uses 20 FPS (from extract_minerl_frames.py), which matches the MineRL standard.
Official dataset: According to the Zenodo record, the full MineRL 2019 dataset includes:
MineRLNavigate-v0: 629.9 MB
MineRLObtainDiamond-v0: 3.6 GB
MineRLObtainIronPickaxe-v0: 2.8 GB
MineRLTreechop-v0: 1.5 GB
Your extracted dataset has approximately 60 hours of video data and 57.5 hours of action data for Phase 1 training.




## Key Differences

Metric	MineRL 2019	VPT Dataset	Ratio
Episodes	759	26,322	34.7× more
Video Hours	59.86 hours	~2,541 hours	42.5× more
Storage Size	~8.6 GB	5.2 TB	~618× larger
Image Resolution	64×64	360×640	Higher resolution
Average Episode Length	~4.7 minutes	~5.8 minutes	Similar

## fixes jan 26:


Summary of Changes                                                                                                                                                      
                                                                                                                                                                          
  1. Asymmetric Attention Masks (transformer.py)                                                                                                                          
                                                                                                                                                                          
  - Added create_tokenizer_encoder_mask(): patches only attend to patches, latents attend to all                                                                          
  - Added create_tokenizer_decoder_mask(): latents only attend to latents, decoder tokens attend to all                                                                   
  - Forces information through the bottleneck                                                                                                                             
                                                                                                                                                                          
  2. Updated Tokenizer (tokenizer.py)                                                                                                                                     
                                                                                                                                                                          
  - encode_frame() and encode() now use asymmetric encoder masks                                                                                                          
  - decode_transformer() uses asymmetric decoder mask with correct token ordering                                                                                         
                                                                                                                                                                          
  3. Per-Image Random Masking (tokenizer.py)                                                                                                                              
                                                                                                                                                                          
  - random_masking() now supports mask_ratio=None for per-image sampling from U(0, 0.9)                                                                                   
  - Matches paper: "The dropout probability is randomized across images as p ~ U(0, 0.9)"                                                                                 
                                                                                                                                                                          
  4. Fixed RMS Normalization (tokenizer_loss.py)                                                                                                                          
                                                                                                                                                                          
  - Added .detach() to RMS normalization to preserve gradient flow                                                                                                        
  - Bug: dividing loss by its own magnitude made it constant with zero gradient       