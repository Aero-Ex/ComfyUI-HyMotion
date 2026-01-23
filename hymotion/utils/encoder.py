"""
Motion Encoder for HY-Motion 1.0

Encodes motion data into the 201-dimensional latent space used by the DiT model.

Latent Vector Layout (201 Dimensions):
    0:3     - Root Translation (X, Y, Z)
    3:135   - 22 Joints in 6D Column-Major Rotation (22 × 6 = 132)
    135:201 - 22 Joint 3D Positions relative to root (22 × 3 = 66)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Any


class MotionEncoder:
    """
    Encodes and decodes motion data to/from the 201-dimensional latent space.
    
    The latent format matches HY-Motion 1.0's training data:
    - Positions 0-2: Root translation (X, Y, Z)
    - Positions 3-134: 22 joints × 6D rotation (column-major)
    - Positions 135-200: 22 joints × 3D position (relative to root)
    """
    
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        """
        Initialize the encoder.
        
        Args:
            mean: Optional normalization mean tensor
            std: Optional normalization std tensor
        """
        self.mean = mean
        self.std = std
        # J0 rest-pose offset from HY-Motion's wooden mesh skeleton
        # This ensures world-to-local consistency
        self.j0_offset = torch.tensor([-0.00179506, -0.190827, 0.02821912])
    
    def encode(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        positions: torch.Tensor,
        dit_model: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Encode raw motion tensors into the 201-dimensional normalized latent.
        
        Args:
            rotations: (B, T, 22, 6) - 6D Column-Major rotations per joint
            translations: (B, T, 3) - Root translation (World or Local)
            positions: (B, T, 22, 3) - Joint positions relative to root
            dit_model: Optional DiT model to pull normalization stats from
            
        Returns:
            latent: (B, T, 201) - Normalized latent tensor
        """
        device = rotations.device
        B, T, J, _ = rotations.shape
        
        # Validate input dimensions
        assert J >= 22, f"Expected at least 22 joints, got {J}"
        assert rotations.shape[-1] == 6, f"Expected 6D rotations, got {rotations.shape[-1]}"
        assert translations.shape[-1] == 3, f"Expected 3D translations, got {translations.shape[-1]}"
        assert positions.shape[-1] == 3, f"Expected 3D positions, got {positions.shape[-1]}"
        
        # 1. Flatten rotations (132 dims) - only first 22 joints
        rot_flat = rotations[:, :, :22, :].reshape(B, T, -1)  # (B, T, 132)
        
        # 2. Extract relative positions (66 dims) - only first 22 joints
        # CRITICAL: Official spec requires positions to EXCLUDE root translation (Issue #33)
        pos_relative = positions[:, :, :22, :] - translations.unsqueeze(2)
        pos_flat = pos_relative.reshape(B, T, -1)  # (B, T, 66)
        
        # 3. Translation (3 dims)
        transl = translations  # (B, T, 3)
        
        # Assemble Raw Latent (201 dims)
        # Sequence: Transl(3) + Rot(132) + Pos(66)
        latent = torch.cat([transl, rot_flat, pos_flat], dim=-1)  # (B, T, 201)
        
        # 4. Normalization
        if dit_model is not None:
            latent = self._normalize_with_model(latent, dit_model, device)
        elif self.mean is not None and self.std is not None:
            m = self.mean.to(device)
            s = self.std.to(device)
            latent = (latent - m) / (s + 1e-8)
            
        return latent
    
    def _normalize_with_model(
        self,
        latent: torch.Tensor,
        dit_model: Any,
        device: torch.device
    ) -> torch.Tensor:
        """
        Pull normalization stats directly from the DiT model buffers.
        
        The model may have stats for 52 joints (315 dims), so we slice
        to match our 22-joint 201-dim format:
        - Stats indices 0:3 -> Latent 0:3 (translation)
        - Stats indices 3:135 -> Latent 3:135 (22-joint rotations)
        - Stats indices 159:225 -> Latent 135:201 (22-joint positions)
        
        Note: The gap (135:159) in model stats corresponds to joints 23-26
        which we don't use in the 22-joint format.
        """
        m = dit_model.mean.to(device)
        s = dit_model.std.to(device)
        
        # Handle stats that are sized for 52 joints (315 dims) or larger
        if m.shape[-1] > 201:
            # Slice the stats to match 201-dim layout
            # Translation: 0:3
            # Rotations (22 joints): 3:135
            # Positions (22 joints): 159:225 in 52-joint format -> 135:201 in 22-joint
            m = torch.cat([m[..., :3], m[..., 3:135], m[..., 159:225]], dim=-1)
            s = torch.cat([s[..., :3], s[..., 3:135], s[..., 159:225]], dim=-1)
        
        # Apply normalization with epsilon for numerical stability
        latent = (latent - m) / (s + 1e-8)
        
        return latent
    
    def decode_to_components(
        self,
        latent: torch.Tensor,
        dit_model: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode a latent tensor back to motion components.
        
        Args:
            latent: (B, T, 201) - Normalized latent tensor
            dit_model: Optional DiT model for denormalization stats
            
        Returns:
            transl: (B, T, 3) - Root translation
            rot6d: (B, T, 22, 6) - 6D rotations per joint
            pos3d: (B, T, 22, 3) - 3D positions relative to root
        """
        device = latent.device
        B = latent.shape[0]
        T = latent.shape[1]
        
        # Denormalize if model stats available
        if dit_model is not None:
            latent = self._denormalize_with_model(latent, dit_model, device)
        elif self.mean is not None and self.std is not None:
            m = self.mean.to(device)
            s = self.std.to(device)
            latent = latent * s + m
        
        # Extract components
        transl = latent[..., :3]  # (B, T, 3)
        rot6d = latent[..., 3:135].view(B, T, 22, 6)  # (B, T, 22, 6)
        pos_rel = latent[..., 135:201].view(B, T, 22, 3)  # (B, T, 22, 3)
        
        # Re-apply root translation to positions for world-space output
        pos3d = pos_rel + transl.unsqueeze(2)
        
        return transl, rot6d, pos3d
    
    def _denormalize_with_model(
        self,
        latent: torch.Tensor,
        dit_model: Any,
        device: torch.device
    ) -> torch.Tensor:
        """
        Denormalize using model stats. Handles 52-joint stats format.
        """
        m = dit_model.mean.to(device)
        s = dit_model.std.to(device)
        
        if m.shape[-1] > 201:
            # Slice stats for 22-joint format
            m = torch.cat([m[..., :3], m[..., 3:135], m[..., 159:225]], dim=-1)
            s = torch.cat([s[..., :3], s[..., 3:135], s[..., 159:225]], dim=-1)
        
        # Denormalize
        latent = latent * s + m
        
        return latent
    
    def encode_single_frame(
        self,
        rot6d: torch.Tensor,
        transl: torch.Tensor,
        keypoints3d: Optional[torch.Tensor] = None,
        dit_model: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Convenience method to encode a single frame.
        
        Args:
            rot6d: (B, 22, 6) or (B, 1, 22, 6) - 6D rotations
            transl: (B, 3) or (B, 1, 3) - Translation
            keypoints3d: Optional (B, 22, 3) or (B, 1, 22, 3) - Joint positions
            dit_model: Optional DiT model for normalization
            
        Returns:
            latent: (B, 1, 201) - Encoded latent for single frame
        """
        # Ensure temporal dimension
        if rot6d.dim() == 3:
            rot6d = rot6d.unsqueeze(1)  # (B, 1, 22, 6)
        if transl.dim() == 2:
            transl = transl.unsqueeze(1)  # (B, 1, 3)
        if keypoints3d is not None:
            if keypoints3d.dim() == 3:
                keypoints3d = keypoints3d.unsqueeze(1)  # (B, 1, 22, 3)
        else:
            # Create zero positions if not provided
            keypoints3d = torch.zeros(
                rot6d.shape[0], 1, 22, 3,
                device=rot6d.device, dtype=rot6d.dtype
            )
        
        return self.encode(rot6d, transl, keypoints3d, dit_model)
