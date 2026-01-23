import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
import numpy as np
import os
import glob
from tqdm import tqdm

# =============================================================================
# CONFIGURATION: Edit these paths for your setup
# =============================================================================

# Path to your HY-Motion model directory (contains latest.ckpt or .safetensors)
# Automatically resolved relative to ComfyUI models directory if possible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "models", "hymotion", "HY-Motion-1.0")

# Path to your converted NPZ files (output of fbx_to_smplh_converter.py)
DATA_DIR = "/path/to/your/converted/npz"  # <-- CHANGE THIS!

# Output directory for checkpoints
CHECKPOINT_DIR = "./pose_adapter_checkpoints"


# =============================================================================
# MODEL LOADER: Loads the pre-trained HY-Motion DiT
# =============================================================================

def load_pretrained_hymotion(model_dir: str = MODEL_DIR, device: str = "cuda"):
    """
    Load the pre-trained HunyuanMotionMMDiT model.
    
    This replicates the loading logic from nodes_modular.py but returns
    just the raw network for training.
    
    Args:
        model_dir: Path to the HY-Motion-1.0 folder
        device: "cuda" or "cpu"
    
    Returns:
        network: The loaded HunyuanMotionMMDiT model
        config: The model configuration dict
    """
    from .hymotion.utils.loaders import load_object
    
    # Default config for the full 1B model
    config = {
        "network_module": "hymotion/network/hymotion_mmdit.HunyuanMotionMMDiT",
        "network_module_args": {
            "apply_rope_to_single_branch": False,
            "ctxt_input_dim": 4096,
            "dropout": 0.0,
            "feat_dim": 1280,
            "input_dim": 201,
            "mask_mode": "narrowband",
            "mlp_ratio": 4.0,
            "num_heads": 20,
            "num_layers": 27,
            "time_factor": 1000.0,
            "vtxt_input_dim": 768
        }
    }
    
    print(f"[PoseAdapter] Loading base model from: {model_dir}")
    
    # Create the network
    network = load_object(config["network_module"], config["network_module_args"])
    
    # Find checkpoint file
    ckpt_path = os.path.join(model_dir, "latest.ckpt")
    if not os.path.exists(ckpt_path):
        # Try other file types
        for ext in [".ckpt", ".pth", ".safetensors"]:
            candidates = glob.glob(os.path.join(model_dir, f"*{ext}"))
            if candidates:
                ckpt_path = candidates[0]
                break
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    print(f"[PoseAdapter] Loading weights from: {os.path.basename(ckpt_path)}")
    
    # Load checkpoint
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    
    # Strip prefixes from keys
    prefixes_to_strip = ["network.", "motion_transformer.", "module."]
    network_state = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in prefixes_to_strip:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        
        # Filter out non-network keys
        if not any(new_k.startswith(p) for p in ["text_encoder.", "vae.", "total_params", "mean", "std", "null_"]):
            network_state[new_k] = v
    
    # Load weights
    missing, unexpected = network.load_state_dict(network_state, strict=False)
    print(f"[PoseAdapter] Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    network = network.to(device)
    network.eval()  # Base model stays in eval mode
    
    print(f"[PoseAdapter] [OK] Base model loaded successfully!")
    return network, config



# =============================================================================
# PART 1: THE POSE ENCODER (THE "Conditioning Brain")
# =============================================================================

class PoseEncoder(nn.Module):
    """
    Takes a Start Frame and End Frame (201-dims each) and turns them into
    a conditioning embedding for the MMDiT model.
    
    Architecture: 3-Layer MLP with Zero-Init final layer for safe training start.
    """
    def __init__(self, input_dim: int = 201, feat_dim: int = 1024):
        super().__init__()
        self.feat_dim = feat_dim
        
        # We use a simple but effective 3-layer MLP with LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        # CRITICAL: Zero-initialize the final layer.
        # This keeps the AI "safe" at the start of training (0 impact).
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, first_frame: torch.Tensor, last_frame: torch.Tensor):
        """
        Args:
            first_frame: [B, 201] or [B, 1, 201]
            last_frame:  [B, 201] or [B, 1, 201]
        Returns:
            pose_embedding: [B, 1, feat_dim]
        """
        x = torch.cat([first_frame.flatten(1), last_frame.flatten(1)], dim=-1)
        return self.encoder(x).unsqueeze(1)  # [B, 1, 1024]


# =============================================================================
# PART 2: CUSTOM MOTION DATASET (For converted FBX -> NPZ files)
# =============================================================================

class ConvertedMotionDataset(Dataset):
    """
    Loads motion clips from NPZ files converted by fbx_to_smplh_converter.py.
    
    The NPZ files contain:
        - poses: [num_frames, 52, 3] axis-angle rotations
        - trans: [num_frames, 3] root translations
        - fps: frame rate
    
    Usage:
        1. First convert your FBX files:
           python fbx_to_smplh_converter.py --input /path/to/fbx --output /path/to/npz
        2. Then use this dataset:
           dataset = ConvertedMotionDataset("/path/to/npz")
    """
    def __init__(self, data_root: str, seq_len: int = 120, stride: int = 60, target_fps: float = 30.0):
        self.seq_len = seq_len
        self.target_fps = target_fps
        self.clips = []
        
        print(f"[MotionDataset] Scanning {data_root} for converted NPZ files...")
        npz_files = glob.glob(os.path.join(data_root, "**", "*.npz"), recursive=True)
        print(f"[MotionDataset] Found {len(npz_files)} files. Processing...")
        
        for npz_path in tqdm(npz_files, desc="Loading Motion Data"):
            try:
                data = np.load(npz_path, allow_pickle=True)
                
                # Check for required fields
                if 'poses' not in data or 'trans' not in data:
                    continue
                
                poses = data['poses']  # [L, 52, 3] axis-angle
                trans = data['trans']  # [L, 3]
                
                # Handle different pose formats
                if poses.ndim == 2:
                    if poses.shape[1] == 156:
                        poses = poses.reshape(-1, 52, 3)
                    elif poses.shape[1] == 66:
                        # 22 joints only - pad to 52
                        poses = poses.reshape(-1, 22, 3)
                        full_poses = np.zeros((len(poses), 52, 3), dtype=np.float32)
                        full_poses[:, :22, :] = poses
                        poses = full_poses
                
                L = min(len(poses), len(trans))
                if L < seq_len:
                    print(f"  Skip (too short): {os.path.basename(npz_path)} ({L} frames)")
                    continue
                
                # Window the sequence into clips
                for start in range(0, L - seq_len, stride):
                    clip_poses = poses[start:start+seq_len]  # [seq_len, 52, 3]
                    clip_trans = trans[start:start+seq_len]  # [seq_len, 3]
                    
                    # Convert to 201-dim format:
                    # [0:3]     = translation (3)
                    # [3:135]   = 22 joints * 6 (rot6d) = 132
                    # [135:201] = 22 joints * 3 (keypoint velocities, placeholder) = 66
                    #
                    # For simplicity in Phase 1, we use axis-angle directly:
                    # [0:3]   = translation (3)
                    # [3:69]  = 22 joints * 3 (axis-angle) = 66
                    # [69:201] = padding (zeros)
                    
                    clip = np.zeros((seq_len, 201), dtype=np.float32)
                    clip[:, 0:3] = clip_trans
                    clip[:, 3:3+22*3] = clip_poses[:, :22, :].reshape(seq_len, -1)
                    
                    self.clips.append(clip)
                    
            except Exception as e:
                print(f"  Error loading {npz_path}: {e}")
                continue
        
        print(f"[MotionDataset] Loaded {len(self.clips)} training clips from {len(npz_files)} files.")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = torch.from_numpy(self.clips[idx])
        return {"motion": clip}


# Alias for backward compatibility
AMASSMotionDataset = ConvertedMotionDataset


# =============================================================================
# PART 3: FLOW-MATCHING NOISE SCHEDULER
# =============================================================================

class FlowMatchingScheduler:
    """
    Implements the Flow-Matching (Rectified Flow) noise schedule.
    This is the native scheduler used by HY-Motion.
    """
    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps
        # Linear schedule from t=0 (clean) to t=1 (noise)
        self.timesteps = torch.linspace(0, 1, num_steps)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise according to flow-matching: x_t = (1-t)*x0 + t*noise
        
        Args:
            x0: Clean motion [B, L, D]
            t: Timestep values [B] in range [0, 1]
        Returns:
            x_t: Noisy motion
            noise: The noise that was added
        """
        noise = torch.randn_like(x0)
        t = t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting
        x_t = (1 - t) * x0 + t * noise
        return x_t, noise

    def get_velocity_target(self, x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        The target for flow-matching is the "velocity": v = noise - x0
        """
        return noise - x0


# =============================================================================
# PART 4: PHYSICS LOSS (FOOT-SLIDE PENALTY)
# =============================================================================

def compute_foot_slide_loss(motion: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """
    Penalizes foot sliding: If a foot is on the ground (Y < threshold),
    its XZ velocity should be zero.
    
    Args:
        motion: [B, L, 201] - assumes trans is in [:, :, 0:3]
    Returns:
        loss: Scalar penalty
    """
    # Extract translation (root position)
    trans = motion[:, :, :3]  # [B, L, 3]
    
    # Compute velocity
    velocity = trans[:, 1:, :] - trans[:, :-1, :]  # [B, L-1, 3]
    
    # Check if root is near ground (Y < threshold)
    y_pos = trans[:, :-1, 1]  # [B, L-1]
    on_ground = (y_pos < threshold).float()
    
    # XZ velocity should be zero when on ground
    xz_velocity = velocity[:, :, [0, 2]]  # [B, L-1, 2]
    xz_speed = torch.norm(xz_velocity, dim=-1)  # [B, L-1]
    
    # Penalize XZ movement when on ground
    slide_penalty = (on_ground * xz_speed).mean()
    
    return slide_penalty


# =============================================================================
# PART 5: THE MODIFIED MMDiT WRAPPER
# =============================================================================

class GuidedHunyuanMMDiT(nn.Module):
    """
    A wrapper that injects our Pose Embedding into the native MMDiT flow.
    The base model stays frozen; only the PoseEncoder is trained.
    """
    def __init__(self, base_model: nn.Module, pose_encoder: PoseEncoder, injection_scale: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.pose_encoder = pose_encoder
        self.injection_scale = injection_scale
        
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Keep pose encoder trainable
        for param in self.pose_encoder.parameters():
            param.requires_grad = True

    def forward(self, x, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal,
                first_frame: torch.Tensor, last_frame: torch.Tensor):
        """
        Forward pass with pose conditioning injection.
        """
        # Get pose embedding from our adapter
        pose_cond = self.pose_encoder(first_frame, last_frame)  # [B, 1, 1024]
        
        # The base model computes: adapter = timestep_feat + vtxt_feat
        # We need to inject: adapter = timestep_feat + vtxt_feat + pose_cond
        # This requires either patching or modifying the base forward.
        
        # For Phase 1, we simulate by directly calling base and adding to its adapter
        # (In production, we'd fork hymotion_mmdit.py to accept pose_cond)
        
        return self.base_model(
            x=x,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
            pose_cond=pose_cond * self.injection_scale  # Scaled injection
        )


# =============================================================================
# PART 6: THE FULL L40S TRAINING LOOP
# =============================================================================

class PoseAdapterTrainer:
    """
    Complete training pipeline for the Pose-to-Pose Adapter.
    Optimized for L40S (48GB VRAM).
    """
    def __init__(
        self,
        model: GuidedHunyuanMMDiT,
        dataset: AMASSMotionDataset,
        lr: float = 1e-4,
        batch_size: int = 64,  # L40S can handle large batches
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        val_every: int = 500,
    ):
        self.model = model.to(device)
        self.device = device
        self.scheduler = FlowMatchingScheduler(num_steps=50)
        self.checkpoint_dir = checkpoint_dir
        self.val_every = val_every
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer (only PoseEncoder params)
        self.optimizer = torch.optim.AdamW(
            self.model.pose_encoder.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        # LR Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.dataloader) * 100  # 100 epochs
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        motion = batch["motion"].to(self.device)  # [B, 120, 201]
        B, L, D = motion.shape
        
        # Extract endpoints
        first_frame = motion[:, 0, :]   # [B, 201]
        last_frame = motion[:, -1, :]   # [B, 201]
        
        # Sample random timesteps
        t = torch.rand(B, device=self.device)
        
        # Add noise
        noisy_motion, noise = self.scheduler.add_noise(motion, t)
        
        # Get velocity target
        velocity_target = self.scheduler.get_velocity_target(motion, noise)
        
        # Create dummy masks (full sequence valid)
        x_mask = torch.ones(B, L, dtype=torch.bool, device=self.device)
        ctxt_mask = torch.ones(B, 1, dtype=torch.bool, device=self.device)
        
        # Dummy text features (we're training pose-only for now)
        vtxt = torch.zeros(B, 1, 768, device=self.device)
        ctxt = torch.zeros(B, 128, 4096, device=self.device)
        
        # Convert t to integer timesteps for the model
        timesteps = (t * 999).long()
        
        # Forward pass
        pred_velocity = self.model(
            x=noisy_motion,
            ctxt_input=ctxt,
            vtxt_input=vtxt,
            timesteps=timesteps,
            x_mask_temporal=x_mask,
            ctxt_mask_temporal=ctxt_mask,
            first_frame=first_frame,
            last_frame=last_frame
        )
        
        # Compute losses
        loss_flow = F.mse_loss(pred_velocity, velocity_target)
        loss_physics = compute_foot_slide_loss(motion)
        
        total_loss = loss_flow + 0.1 * loss_physics
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.pose_encoder.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        
        return {
            "loss_total": total_loss.item(),
            "loss_flow": loss_flow.item(),
            "loss_physics": loss_physics.item(),
        }

    def validate(self, step: int):
        """Generate a sample transition and save it for visual inspection."""
        self.model.eval()
        with torch.no_grad():
            # Create a simple test: identity pose at start, slightly rotated at end
            first_frame = torch.zeros(1, 201, device=self.device)
            last_frame = torch.zeros(1, 201, device=self.device)
            last_frame[:, 2] = 2.0  # Move forward 2 meters
            
            # Generate motion (simplified - in practice, run full ODE)
            # For now, just check that the encoder produces valid output
            pose_cond = self.model.pose_encoder(first_frame, last_frame)
            print(f"[Val @ Step {step}] Pose embedding stats: "
                  f"mean={pose_cond.mean().item():.4f}, "
                  f"std={pose_cond.std().item():.4f}, "
                  f"max={pose_cond.max().item():.4f}")
        
        self.model.train()

    def save_checkpoint(self, step: int):
        """Save the PoseEncoder weights."""
        path = os.path.join(self.checkpoint_dir, f"pose_encoder_step{step}.pt")
        torch.save(self.model.pose_encoder.state_dict(), path)
        print(f"[Checkpoint] Saved: {path}")

    # =========================================================================
    # SANITY CHECK SUITE (Run this first to avoid wasting GPU money!)
    # =========================================================================
    
    def run_sanity_checks(self) -> bool:
        """
        Run early verification tests before starting real training.
        Returns True if all checks pass, False if something is wrong.
        
        ESTIMATED TIME: 5-10 minutes on L40S
        """
        print("\n" + "=" * 60)
        print("SANITY CHECK SUITE - Verifying before training...")
        print("=" * 60)
        
        all_passed = True
        
        # =====================================================================
        # CHECK 1: Gradient Flow Verification
        # =====================================================================
        print("\n[Check 1/4] Gradient Flow Test...")
        try:
            # Get a single batch
            batch = next(iter(self.dataloader))
            losses = self.train_step(batch)
            
            # Check if gradients are flowing
            total_grad_norm = 0.0
            num_params = 0
            for param in self.model.pose_encoder.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
                    num_params += 1
            
            if num_params == 0 or total_grad_norm == 0:
                print("  ❌ FAIL: No gradients detected! The encoder is not learning.")
                all_passed = False
            else:
                print(f"  ✅ PASS: Gradients flowing (norm={total_grad_norm:.6f} across {num_params} params)")
        except Exception as e:
            print(f"  ❌ FAIL: Error during gradient check: {e}")
            all_passed = False
        
        # =====================================================================
        # CHECK 2: Single-Batch Overfit Test
        # =====================================================================
        print("\n[Check 2/4] Single-Batch Overfit Test (50 steps)...")
        try:
            # Get a single batch and try to overfit it
            batch = next(iter(self.dataloader))
            initial_loss = None
            final_loss = None
            
            for i in range(50):
                losses = self.train_step(batch)
                if i == 0:
                    initial_loss = losses['loss_flow']
                if i == 49:
                    final_loss = losses['loss_flow']
                if i % 10 == 0:
                    print(f"    Step {i}: loss={losses['loss_flow']:.6f}")
            
            # Check if loss decreased
            if final_loss >= initial_loss * 0.95:
                print(f"  ❌ FAIL: Loss didn't decrease! ({initial_loss:.6f} -> {final_loss:.6f})")
                print("           The model is not learning from the data.")
                all_passed = False
            else:
                reduction = (1 - final_loss/initial_loss) * 100
                print(f"  ✅ PASS: Loss decreased by {reduction:.1f}% ({initial_loss:.6f} -> {final_loss:.6f})")
        except Exception as e:
            print(f"  ❌ FAIL: Error during overfit test: {e}")
            all_passed = False
        
        # =====================================================================
        # CHECK 3: Pose Encoder Output Verification
        # =====================================================================
        print("\n[Check 3/4] Pose Encoder Output Test...")
        try:
            self.model.eval()
            with torch.no_grad():
                # Test with two different inputs
                frame_a = torch.zeros(1, 201, device=self.device)
                frame_b = torch.zeros(1, 201, device=self.device)
                frame_b[:, 2] = 2.0  # Different translation
                
                # Same input should give same output
                out1 = self.model.pose_encoder(frame_a, frame_a)
                out2 = self.model.pose_encoder(frame_a, frame_a)
                
                # Different input should give different output
                out3 = self.model.pose_encoder(frame_a, frame_b)
                
                same_diff = (out1 - out2).abs().max().item()
                diff_diff = (out1 - out3).abs().max().item()
                
                if same_diff > 1e-5:
                    print(f"  ❌ FAIL: Encoder is non-deterministic (diff={same_diff:.6f})")
                    all_passed = False
                elif diff_diff < 1e-5:
                    print(f"  ❌ FAIL: Encoder ignores input differences (diff={diff_diff:.6f})")
                    all_passed = False
                else:
                    print(f"  ✅ PASS: Encoder is deterministic and input-sensitive (diff={diff_diff:.4f})")
            self.model.train()
        except Exception as e:
            print(f"  ❌ FAIL: Error during encoder test: {e}")
            all_passed = False
        
        # =====================================================================
        # CHECK 4: Memory Usage Test
        # =====================================================================
        print("\n[Check 4/4] GPU Memory Usage Test...")
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                print(f"    Allocated: {allocated:.2f} GB")
                print(f"    Reserved:  {reserved:.2f} GB")
                print(f"    Total:     {total:.2f} GB")
                
                if reserved > total * 0.95:
                    print("  [WARNING] Memory is almost full! Consider reducing batch size.")
                else:
                    headroom = total - reserved
                    print(f"  [OK] {headroom:.1f} GB headroom available")
            else:
                print("  [WARNING] Not running on CUDA")
        except Exception as e:
            print(f"  [WARNING] Could not check memory: {e}")
        
        # =====================================================================
        # FINAL VERDICT
        # =====================================================================
        print("\n" + "=" * 60)
        if all_passed:
            print("[OK] ALL SANITY CHECKS PASSED - Safe to start training!")
            print("=" * 60)
            return True
        else:
            print("[ERROR] SANITY CHECKS FAILED - DO NOT START TRAINING!")
            print("   Fix the issues above before renting GPU time.")
            print("=" * 60)
            return False

    def train(self, num_epochs: int = 100, skip_sanity_checks: bool = False):
        """
        Full training loop with optional sanity checks.
        
        Args:
            num_epochs: Number of epochs to train
            skip_sanity_checks: Set True to skip initial verification (not recommended)
        """
        # Run sanity checks first (unless skipped)
        if not skip_sanity_checks:
            if not self.run_sanity_checks():
                print("\n[STOP] Training aborted due to failed sanity checks.")
                print("   Use train(skip_sanity_checks=True) to force start (not recommended)")
                return
        
        print("\n" + "=" * 60)
        print("PHASE 1: POSE-TO-POSE ADAPTER TRAINING")
        print(f"Device: {self.device}")
        print(f"Batch Size: {self.dataloader.batch_size}")
        print(f"Dataset Size: {len(self.dataloader.dataset)} clips")
        print("=" * 60)
        
        global_step = 0
        loss_history = []
        
        for epoch in range(num_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                losses = self.train_step(batch)
                global_step += 1
                loss_history.append(losses['loss_flow'])
                
                pbar.set_postfix({
                    "loss": f"{losses['loss_total']:.4f}",
                    "flow": f"{losses['loss_flow']:.4f}",
                    "phys": f"{losses['loss_physics']:.4f}",
                })
                
                # EARLY ABORT: If loss explodes, stop immediately
                if losses['loss_flow'] > 100 or np.isnan(losses['loss_flow']):
                    print(f"\n[EMERGENCY STOP] Loss exploded ({losses['loss_flow']})")
                    print("   This usually means learning rate is too high.")
                    self.save_checkpoint(global_step)
                    return
                
                # EARLY ABORT: If loss hasn't decreased after 1000 steps
                if global_step == 1000:
                    first_100 = np.mean(loss_history[:100])
                    last_100 = np.mean(loss_history[-100:])
                    if last_100 >= first_100 * 0.98:
                        print(f"\n[WARNING] Loss stagnant after 1000 steps")
                        print(f"   First 100 avg: {first_100:.6f}")
                        print(f"   Last 100 avg:  {last_100:.6f}")
                        print("   Consider stopping and checking the setup.")
                
                # Validation preview
                if global_step % self.val_every == 0:
                    self.validate(global_step)
                    self.save_checkpoint(global_step)
        
        print("\n[OK] Training Complete!")
        self.save_checkpoint(global_step)


# =============================================================================
# PART 7: COMPLETE TRAINING SCRIPT
# =============================================================================

def main():
    """
    Complete Phase 1 Training Script for Pose-to-Pose Adapter.
    
    Before running:
    1. Convert your FBX animations:
       python fbx_to_smplh_converter.py --input /your/fbx/folder --output /your/npz/folder
    
    2. Edit the paths at the top of this file:
       - MODEL_DIR: Path to HY-Motion-1.0 folder
       - DATA_DIR: Path to converted NPZ files
       - CHECKPOINT_DIR: Where to save trained adapters
    
    3. Run this script:
       python hymotion_pose_adapter.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Pose-to-Pose Adapter for HY-Motion")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR, help="Path to HY-Motion model")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Path to converted NPZ data")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (reduce if OOM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip sanity checks (not recommended)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HY-Motion Pose-to-Pose Adapter - Phase 1 Training")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Verify paths
    # =========================================================================
    print(f"\n[INFO] Model directory: {args.model_dir}")
    print(f"[INFO] Data directory:  {args.data_dir}")
    print(f"[INFO] Output directory: {args.checkpoint_dir}")
    
    if not os.path.exists(args.model_dir):
        print(f"\n[ERROR] Model directory not found: {args.model_dir}")
        print("   Please download HY-Motion-1.0 and set the correct path.")
        return
    
    if args.data_dir == "/path/to/your/converted/npz":
        print(f"\n[ERROR] You need to set DATA_DIR to your converted NPZ folder!")
        print("   1. First convert your FBX files:")
        print("      python fbx_to_smplh_converter.py --input /your/fbx --output /your/npz")
        print("   2. Then edit DATA_DIR at the top of this file")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"\n[ERROR] Data directory not found: {args.data_dir}")
        print("   Run the FBX converter first:")
        print("   python fbx_to_smplh_converter.py --input /your/fbx --output /your/npz")
        return
    
    # =========================================================================
    # Step 2: Load dataset
    # =========================================================================
    print(f"\n[INFO] Loading dataset from {args.data_dir}...")
    dataset = ConvertedMotionDataset(args.data_dir, seq_len=120, stride=60)
    
    if len(dataset) == 0:
        print("[ERROR] No training clips found in dataset!")
        print("   Make sure your NPZ files contain 'poses' and 'trans' arrays.")
        return
    
    print(f"[OK] Loaded {len(dataset)} training clips")
    
    # =========================================================================
    # Step 3: Load base model
    # =========================================================================
    print(f"\n[INFO] Loading base HY-Motion model...")
    try:
        base_model, config = load_pretrained_hymotion(args.model_dir, args.device)
    except Exception as e:
        print(f"[ERROR] Failed to load base model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get the adapter feat_dim from base model config
    feat_dim = config["network_module_args"]["feat_dim"]  # 1280 for full, 1024 for lite
    print(f"[OK] Base model loaded. Adapter dimension: {feat_dim}")
    
    # =========================================================================
    # Step 4: Create Pose Encoder and Guided Model
    # =========================================================================
    print(f"\n[INFO] Creating Pose Encoder...")
    pose_encoder = PoseEncoder(input_dim=201, feat_dim=feat_dim)
    guided_model = GuidedHunyuanMMDiT(base_model, pose_encoder, injection_scale=1.0)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in pose_encoder.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in base_model.parameters())
    print(f"[OK] Trainable parameters: {trainable_params:,} (Pose Encoder)")
    print(f"   Frozen parameters:   {frozen_params:,} (Base Model)")
    
    # =========================================================================
    # Step 5: Create trainer and start training
    # =========================================================================
    print(f"\n[INFO] Initializing trainer...")
    trainer = PoseAdapterTrainer(
        model=guided_model,
        dataset=dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        val_every=500,
    )
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"   Epochs:      {args.epochs}")
    print(f"   Batch Size:  {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Device:      {args.device}")
    print(f"   Checkpoints: {args.checkpoint_dir}")
    print(f"{'='*60}")
    
    # Start training!
    trainer.train(num_epochs=args.epochs, skip_sanity_checks=args.skip_sanity)
    
    print("\n" + "=" * 60)
    print("Training session complete!")
    print(f"   Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
