import torch
from typing import Dict, List, Any, Optional

class HYMotionData:
    """Class to hold motion data results"""
    def __init__(self, output_dict: Dict[str, Any], text: str, duration: float, seeds: List[int], device_info: str = "unknown"):
        self.output_dict = output_dict
        self.text = text
        self.duration = duration
        self.seeds = seeds
        self.device_info = device_info
        self.batch_size = output_dict["keypoints3d"].shape[0] if "keypoints3d" in output_dict else 1

class HYMotionTextEmbeds:
    """Wrapper for encoded text embeddings"""
    def __init__(self, vtxt, ctxt, ctxt_length, text=""):
        self.vtxt = vtxt  # [batch, 1, 768] - CLIP embeddings
        self.ctxt = ctxt  # [batch, max_len, 4096] - Qwen3 embeddings
        self.ctxt_length = ctxt_length  # [batch] - actual lengths
        self.text = text

class HYMotionFrame:
    """Wrapper for a single frame of motion data"""
    def __init__(self, rot6d: torch.Tensor, transl: torch.Tensor):
        self.rot6d = rot6d  # [1, 22, 6]
        self.transl = transl  # [1, 3]
