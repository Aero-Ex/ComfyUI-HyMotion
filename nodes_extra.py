"""
Extra utility nodes for HY-Motion 1.0

Provides nodes for:
- Encoding raw motion data into latent space
- Slicing motion data (tail-encoding for chaining)
"""

import torch
from typing import Optional, Dict, Any

from .hymotion.utils.data_types import HYMotionData
from .hymotion.utils.encoder import MotionEncoder








class HYMotionDecodeLatent:
    """
    Decode a latent tensor back to motion components.
    
    Useful for:
    - Inspecting what the latent encodes
    - Converting AI-generated latents to usable motion data
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Latent tensor of shape (B, T, 201)"
                }),
            },
            "optional": {
                "dit_model": ("HYMOTION_DIT", {
                    "tooltip": "DiT model for denormalization. Required if latent is normalized."
                }),
                "text": ("STRING", {
                    "default": "decoded motion",
                    "tooltip": "Description text for the decoded motion"
                }),
            }
        }

    RETURN_TYPES = ("HYMOTION_DATA", "ROTATIONS", "TRANSLATIONS", "POSITIONS")
    RETURN_NAMES = ("motion_data", "rotations", "translations", "positions")
    FUNCTION = "decode"
    CATEGORY = "HY-Motion/utils"

    def decode(
        self,
        latent: torch.Tensor,
        dit_model: Optional[Any] = None,
        text: str = "decoded motion"
    ):
        """
        Decode latent to motion components.
        """
        encoder = MotionEncoder()
        
        transl, rot6d, pos3d = encoder.decode_to_components(latent, dit_model)
        
        T = rot6d.shape[1]
        output_dict = {
            "rot6d": rot6d,
            "transl": transl,
            "keypoints3d": pos3d
        }
        
        motion_data = HYMotionData(
            output_dict=output_dict,
            text=text,
            duration=T / 30.0,
            seeds=[0],
            device_info=str(latent.device)
        )
        
        print(f"[HY-Motion] Decoded latent: shape {latent.shape} -> {T} frames")
        
        return (motion_data, rot6d, transl, pos3d)


class HYMotionDecomposeData:
    """
    Decompose HYMotionData into its raw tensor components.
    
    Useful for:
    - Extracting raw rotations/translations for custom processing
    - Inspecting the data structure
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA", {
                    "tooltip": "Input motion data to decompose"
                }),
            }
        }

    RETURN_TYPES = ("ROTATIONS", "TRANSLATIONS", "POSITIONS", "STRING")
    RETURN_NAMES = ("rotations", "translations", "positions", "text")
    FUNCTION = "decompose"
    CATEGORY = "HY-Motion/utils"

    def decompose(self, motion_data: HYMotionData):
        """
        Extract components from motion data.
        """
        d = motion_data.output_dict
        
        rot6d = d["rot6d"].clone()
        transl = d["transl"].clone()
        
        if "keypoints3d" in d and d["keypoints3d"] is not None:
            keypoints3d = d["keypoints3d"].clone()
        else:
            keypoints3d = torch.zeros_like(rot6d[..., :3]) # Fallback
            
        print(f"[HY-Motion] Decomposed data: {motion_data.text} ({rot6d.shape[1]} frames)")
        
        return (rot6d, transl, keypoints3d, motion_data.text)



# Node mappings for ComfyUI registration
class HYMotionSliceAndEncode:
    """
    Combines Slicing and Encoding into a single node for easier chaining workflow.
    
    Inputs:
    - Full Motion Data
    - Slice Parameters (frames, mode)
    - Encoding Options (DiT Model)
    
    Outputs:
    - Latent (Velocity/Momentum Hint)
    - Motion Data (The Sliced Subset)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA", {
                    "tooltip": "Input full motion data to slice and encode"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Starting frame index (only used in 'from_start' mode)"
                }),
                "num_frames": ("INT", {
                    "default": 5, # Standard for momentum injection
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Number of frames to extract (5-10 recommended for momentum)"
                }),
                "mode": (["from_start", "from_end"], {
                    "default": "from_end",
                    "tooltip": "'from_end' extracts the last N frames (for chaining). 'from_start' extracts from start_frame."
                }),
            },
            "optional": {
                "dit_model": ("HYMOTION_DIT", {
                    "tooltip": "DiT model to use for normalization stats. Highly recommended for accurate latent encoding."
                }),
                "text": ("STRING", {
                    "default": "sliced encoded motion",
                    "tooltip": "Description text for the encoded motion"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "HYMOTION_DATA")
    RETURN_NAMES = ("latent", "sliced_motion_data")
    FUNCTION = "slice_and_encode"
    CATEGORY = "HY-Motion/utils"

    def slice_and_encode(
        self,
        motion_data: HYMotionData,
        start_frame: int,
        num_frames: int,
        mode: str,
        dit_model: Optional[Any] = None,
        text: str = "sliced encoded motion"
    ):
        # === 1. Internal Slicing Logic ===
        d = motion_data.output_dict
        total_frames = d["rot6d"].shape[1]
        
        # Calculate slice indices
        if mode == "from_end":
            actual_start = max(0, total_frames - num_frames)
            actual_end = total_frames
        else:  # from_start
            actual_start = min(start_frame, total_frames - 1)
            actual_end = min(total_frames, start_frame + num_frames)
        
        # Validate we have frames to slice
        if actual_end <= actual_start:
            raise ValueError(f"Invalid slice: start={actual_start}, end={actual_end}")
        
        # Slice all components
        sliced_rot6d = d["rot6d"][:, actual_start:actual_end].clone()
        sliced_transl = d["transl"][:, actual_start:actual_end].clone()
        
        # Handle optional keypoints3d
        if "keypoints3d" in d and d["keypoints3d"] is not None:
            sliced_keypoints = d["keypoints3d"][:, actual_start:actual_end].clone()
        else:
            # Create zero keypoints if not available
            B = sliced_rot6d.shape[0]
            L = sliced_rot6d.shape[1]
            J = sliced_rot6d.shape[2]
            sliced_keypoints = torch.zeros((B, L, J, 3), device=sliced_rot6d.device)
        
        sliced_dict = {
            "rot6d": sliced_rot6d,
            "transl": sliced_transl,
            "keypoints3d": sliced_keypoints
        }
        
        # Preserve root rotations if available
        if "root_rotations_mat" in d and d["root_rotations_mat"] is not None:
            sliced_dict["root_rotations_mat"] = d["root_rotations_mat"][:, actual_start:actual_end].clone()
        
        # Calculate new duration
        slice_frames = actual_end - actual_start
        new_duration = slice_frames / 30.0
        
        # Create new motion data wrapper
        sliced_data = HYMotionData(
            output_dict=sliced_dict,
            text=f"Slice ({mode}: {slice_frames} frames) of {motion_data.text}",
            duration=new_duration,
            seeds=motion_data.seeds,
            device_info=motion_data.device_info
        )
        
        # === 2. Internal Encoding Logic ===
        encoder = MotionEncoder()
        
        # Ensure tensors are float32
        rotations = sliced_rot6d.float()
        translations = sliced_transl.float()
        positions = sliced_keypoints.float()
        
        # Encode to latent
        latent = encoder.encode(rotations, translations, positions, dit_model)
        
        print(f"[HY-Motion] SliceAndEncode: Processed {num_frames} frames ({mode}). ready for chaining.")
        
        return (latent, sliced_data)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS_EXTRA = {
    "HYMotionDecodeLatent": HYMotionDecodeLatent,
    "HYMotionDecomposeData": HYMotionDecomposeData,
    "HYMotionSliceAndEncode": HYMotionSliceAndEncode,
}

NODE_DISPLAY_NAME_MAPPINGS_EXTRA = {
    "HYMotionDecodeLatent": "HY-Motion Decode Latent",
    "HYMotionDecomposeData": "HY-Motion Decompose Data",
    "HYMotionSliceAndEncode": "HY-Motion Slice & Encode (Chain)",
}
