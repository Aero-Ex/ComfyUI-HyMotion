
import os
import sys
from unittest.mock import MagicMock

# Mock ComfyUI modules
sys.modules["folder_paths"] = MagicMock()
comfy = MagicMock()
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Mock COMFY_OUTPUT_DIR which is imported from nodes_modular
import builtins
builtins.COMFY_OUTPUT_DIR = "/tmp"

import torch
import numpy as np

from nodes_modular import HYMotionData, HYMotionFrame, HYMotionExtractFrame, construct_smpl_data_dict
from hymotion.utils.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_angle_axis

def test_extraction():
    print("Testing HYMotionExtractFrame...")
    
    # 1. Create mock motion data (100 frames, 22 joints)
    B, L, J = 1, 100, 22
    # Create some non-zero rotations (e.g., identity + small offset)
    rot6d = torch.zeros((B, L, J, 6))
    rot6d[..., 0] = 1.0
    rot6d[..., 4] = 1.0
    # Add a specific pose at the last frame
    rot6d[0, -1, 1, 0] = 0.5 # Bend the second joint
    
    transl = torch.zeros((B, L, 3))
    transl[0, -1, 1] = 1.5 # Move up at the last frame
    
    output_dict = {
        "rot6d": rot6d,
        "transl": transl,
        "keypoints3d": torch.zeros((B, L, J, 3))
    }
    
    motion_data = HYMotionData(
        output_dict=output_dict,
        text="test motion",
        duration=L/30.0,
        seeds=[42]
    )
    
    extractor = HYMotionExtractFrame()
    
    # 2. Extract last frame
    print(f"Extracting frame -1 (index {L-1})...")
    frame_wrapper, frame_motion_data = extractor.extract(motion_data, -1)
    
    # 3. Verify extracted data
    ext_rot6d = frame_motion_data.output_dict["rot6d"]
    ext_transl = frame_motion_data.output_dict["transl"]
    
    print(f"Extracted rot6d shape: {ext_rot6d.shape}")
    print(f"Extracted transl shape: {ext_transl.shape}")
    
    # Check if values match the last frame
    match_rot = torch.allclose(ext_rot6d[0, 0], rot6d[0, -1])
    match_trans = torch.allclose(ext_transl[0, 0], transl[0, -1])
    
    print(f"Rotations match: {match_rot}")
    print(f"Translations match: {match_trans}")
    
    if not match_rot or not match_trans:
        print("ERROR: Extracted values do not match source!")
        return
    
    # 4. Test construct_smpl_data_dict
    print("Testing construct_smpl_data_dict with extracted frame...")
    smpl_data = construct_smpl_data_dict(ext_rot6d[0], ext_transl[0])
    
    poses = smpl_data["poses"]
    trans = smpl_data["trans"]
    
    print(f"SMPL-H poses shape: {poses.shape}")
    print(f"SMPL-H trans shape: {trans.shape}")
    
    # Check if poses are non-zero (not a T-pose)
    # Identity in angle-axis is [0, 0, 0]
    is_t_pose = np.allclose(poses, 0.0)
    print(f"Is T-pose (all zeros): {is_t_pose}")
    
    if is_t_pose:
        print("ERROR: Resulting SMPL-H data is a T-pose!")
    else:
        print("SUCCESS: Extracted frame has non-zero poses.")

if __name__ == "__main__":
    test_extraction()
