"""
FBX to SMPL-H Converter
=======================
Converts FBX animation files to SMPL-H format (.npz) for training the Pose Adapter.

Usage:
    python fbx_to_smplh_converter.py --input /path/to/fbx/folder --output /path/to/output

Requirements:
    - FBX SDK (already installed for ComfyUI-HyMotion)
    - NumPy
    - SciPy
"""

import os
import glob
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R

try:
    import fbx
except ImportError:
    print("ERROR: FBX SDK not found. Please install it first.")
    fbx = None


# =============================================================================
# SKELETON MAPPINGS (Add your rig's mapping here)
# =============================================================================

# SMPL-H Joint Order (52 joints)
SMPLH_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    # Fingers (22-51)
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]

# Mixamo to SMPL-H Mapping
MIXAMO_TO_SMPLH = {
    "mixamorig:Hips": "Pelvis",
    "mixamorig:LeftUpLeg": "L_Hip",
    "mixamorig:RightUpLeg": "R_Hip",
    "mixamorig:Spine": "Spine1",
    "mixamorig:LeftLeg": "L_Knee",
    "mixamorig:RightLeg": "R_Knee",
    "mixamorig:Spine1": "Spine2",
    "mixamorig:LeftFoot": "L_Ankle",
    "mixamorig:RightFoot": "R_Ankle",
    "mixamorig:Spine2": "Spine3",
    "mixamorig:LeftToeBase": "L_Foot",
    "mixamorig:RightToeBase": "R_Foot",
    "mixamorig:Neck": "Neck",
    "mixamorig:LeftShoulder": "L_Collar",
    "mixamorig:RightShoulder": "R_Collar",
    "mixamorig:Head": "Head",
    "mixamorig:LeftArm": "L_Shoulder",
    "mixamorig:RightArm": "R_Shoulder",
    "mixamorig:LeftForeArm": "L_Elbow",
    "mixamorig:RightForeArm": "R_Elbow",
    "mixamorig:LeftHand": "L_Wrist",
    "mixamorig:RightHand": "R_Wrist",
    # Fingers (if available)
    "mixamorig:LeftHandIndex1": "L_Index1",
    "mixamorig:LeftHandIndex2": "L_Index2",
    "mixamorig:LeftHandIndex3": "L_Index3",
    "mixamorig:LeftHandMiddle1": "L_Middle1",
    "mixamorig:LeftHandMiddle2": "L_Middle2",
    "mixamorig:LeftHandMiddle3": "L_Middle3",
    "mixamorig:LeftHandPinky1": "L_Pinky1",
    "mixamorig:LeftHandPinky2": "L_Pinky2",
    "mixamorig:LeftHandPinky3": "L_Pinky3",
    "mixamorig:LeftHandRing1": "L_Ring1",
    "mixamorig:LeftHandRing2": "L_Ring2",
    "mixamorig:LeftHandRing3": "L_Ring3",
    "mixamorig:LeftHandThumb1": "L_Thumb1",
    "mixamorig:LeftHandThumb2": "L_Thumb2",
    "mixamorig:LeftHandThumb3": "L_Thumb3",
    "mixamorig:RightHandIndex1": "R_Index1",
    "mixamorig:RightHandIndex2": "R_Index2",
    "mixamorig:RightHandIndex3": "R_Index3",
    "mixamorig:RightHandMiddle1": "R_Middle1",
    "mixamorig:RightHandMiddle2": "R_Middle2",
    "mixamorig:RightHandMiddle3": "R_Middle3",
    "mixamorig:RightHandPinky1": "R_Pinky1",
    "mixamorig:RightHandPinky2": "R_Pinky2",
    "mixamorig:RightHandPinky3": "R_Pinky3",
    "mixamorig:RightHandRing1": "R_Ring1",
    "mixamorig:RightHandRing2": "R_Ring2",
    "mixamorig:RightHandRing3": "R_Ring3",
    "mixamorig:RightHandThumb1": "R_Thumb1",
    "mixamorig:RightHandThumb2": "R_Thumb2",
    "mixamorig:RightHandThumb3": "R_Thumb3",
}

# Generic Humanoid Mapping (for rigs without prefix)
GENERIC_TO_SMPLH = {
    "Hips": "Pelvis",
    "pelvis": "Pelvis",
    "LeftUpLeg": "L_Hip",
    "left_hip": "L_Hip",
    "RightUpLeg": "R_Hip",
    "right_hip": "R_Hip",
    "Spine": "Spine1",
    "spine1": "Spine1",
    "LeftLeg": "L_Knee",
    "left_knee": "L_Knee",
    "RightLeg": "R_Knee",
    "right_knee": "R_Knee",
    "Spine1": "Spine2",
    "spine2": "Spine2",
    "LeftFoot": "L_Ankle",
    "left_ankle": "L_Ankle",
    "RightFoot": "R_Ankle",
    "right_ankle": "R_Ankle",
    "Spine2": "Spine3",
    "spine3": "Spine3",
    "LeftToeBase": "L_Foot",
    "left_foot": "L_Foot",
    "RightToeBase": "R_Foot",
    "right_foot": "R_Foot",
    "Neck": "Neck",
    "neck": "Neck",
    "LeftShoulder": "L_Collar",
    "left_collar": "L_Collar",
    "RightShoulder": "R_Collar",
    "right_collar": "R_Collar",
    "Head": "Head",
    "head": "Head",
    "LeftArm": "L_Shoulder",
    "left_shoulder": "L_Shoulder",
    "RightArm": "R_Shoulder",
    "right_shoulder": "R_Shoulder",
    "LeftForeArm": "L_Elbow",
    "left_elbow": "L_Elbow",
    "RightForeArm": "R_Elbow",
    "right_elbow": "R_Elbow",
    "LeftHand": "L_Wrist",
    "left_wrist": "L_Wrist",
    "RightHand": "R_Wrist",
    "right_wrist": "R_Wrist",
}


# =============================================================================
# FBX LOADING UTILITIES
# =============================================================================

def load_fbx_scene(filepath: str):
    """Load an FBX file and return the scene."""
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(filepath, -1, manager.GetIOSettings()):
        raise Exception(f"Failed to load FBX: {filepath}")
    
    scene = fbx.FbxScene.Create(manager, "")
    importer.Import(scene)
    importer.Destroy()
    
    return manager, scene


def collect_skeleton_nodes(node, skeleton_dict=None):
    """Recursively collect all skeleton nodes."""
    if skeleton_dict is None:
        skeleton_dict = {}
    
    attr = node.GetNodeAttribute()
    if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
        skeleton_dict[node.GetName()] = node
    
    for i in range(node.GetChildCount()):
        collect_skeleton_nodes(node.GetChild(i), skeleton_dict)
    
    return skeleton_dict


def get_animation_length(scene) -> Tuple[int, float]:
    """Get the number of frames and FPS from the scene."""
    anim_stack = scene.GetCurrentAnimationStack()
    if not anim_stack:
        # Try to get the first animation stack
        if scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)) > 0:
            anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    
    if not anim_stack:
        return 0, 30.0
    
    # Get time span
    time_span = anim_stack.GetLocalTimeSpan()
    start_time = time_span.GetStart()
    end_time = time_span.GetStop()
    
    # Get FPS from global settings
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    fps = fbx.FbxTime().GetFrameRate(time_mode)
    if fps <= 0:
        fps = 30.0
    
    # Calculate frame count
    duration_seconds = end_time.GetSecondDouble() - start_time.GetSecondDouble()
    num_frames = int(duration_seconds * fps) + 1
    
    return num_frames, fps


def extract_joint_rotations(node, num_frames: int, fps: float) -> np.ndarray:
    """Extract rotation keyframes for a joint as Euler angles."""
    rotations = np.zeros((num_frames, 3), dtype=np.float32)
    
    time = fbx.FbxTime()
    for frame in range(num_frames):
        time.SetSecondDouble(frame / fps)
        
        # Get local rotation at this time
        rot = node.EvaluateLocalRotation(time)
        rotations[frame] = [rot[0], rot[1], rot[2]]  # XYZ Euler in degrees
    
    return rotations


def extract_joint_translations(node, num_frames: int, fps: float) -> np.ndarray:
    """Extract translation keyframes for a joint (usually root only)."""
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    
    time = fbx.FbxTime()
    for frame in range(num_frames):
        time.SetSecondDouble(frame / fps)
        
        # Get local translation at this time
        trans = node.EvaluateLocalTranslation(time)
        translations[frame] = [trans[0], trans[1], trans[2]]
    
    return translations


def euler_to_axis_angle(euler_degrees: np.ndarray, order: str = "xyz") -> np.ndarray:
    """Convert Euler angles (degrees) to axis-angle representation."""
    # Shape: [N, 3] -> [N, 3]
    euler_radians = np.deg2rad(euler_degrees)
    r = R.from_euler(order, euler_radians)
    return r.as_rotvec().astype(np.float32)


# =============================================================================
# MAIN CONVERSION LOGIC
# =============================================================================

def auto_detect_mapping(skeleton_nodes: Dict[str, any]) -> Dict[str, str]:
    """Auto-detect which mapping to use based on bone names."""
    node_names = set(skeleton_nodes.keys())
    
    # Check for Mixamo
    if any("mixamorig:" in name for name in node_names):
        print("[Converter] Detected Mixamo rig")
        return MIXAMO_TO_SMPLH
    
    # Check for direct SMPL-H
    if "Pelvis" in node_names and "L_Hip" in node_names:
        print("[Converter] Detected SMPL-H rig (no conversion needed)")
        return {j: j for j in SMPLH_JOINTS}
    
    # Fall back to generic
    print("[Converter] Using generic humanoid mapping")
    return GENERIC_TO_SMPLH


def convert_fbx_to_smplh(fbx_path: str, output_dir: str) -> Optional[str]:
    """
    Convert a single FBX file to SMPL-H format (.npz).
    
    Returns the output path on success, None on failure.
    """
    print(f"\n[Converting] {os.path.basename(fbx_path)}")
    
    try:
        manager, scene = load_fbx_scene(fbx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    
    try:
        # Collect skeleton
        root_node = scene.GetRootNode()
        skeleton_nodes = collect_skeleton_nodes(root_node)
        
        if not skeleton_nodes:
            print("  ERROR: No skeleton found in FBX")
            return None
        
        print(f"  Found {len(skeleton_nodes)} bones")
        
        # Get animation info
        num_frames, fps = get_animation_length(scene)
        if num_frames < 30:
            print(f"  SKIP: Animation too short ({num_frames} frames)")
            return None
        
        print(f"  Animation: {num_frames} frames @ {fps:.1f} FPS")
        
        # Auto-detect mapping
        mapping = auto_detect_mapping(skeleton_nodes)
        
        # Create output arrays
        poses = np.zeros((num_frames, 52, 3), dtype=np.float32)  # Axis-angle
        trans = np.zeros((num_frames, 3), dtype=np.float32)
        
        # Extract data for each SMPL-H joint
        mapped_count = 0
        for smplh_idx, smplh_name in enumerate(SMPLH_JOINTS):
            # Find the source bone
            source_bone = None
            for src_name, dst_name in mapping.items():
                if dst_name == smplh_name and src_name in skeleton_nodes:
                    source_bone = skeleton_nodes[src_name]
                    break
            
            # Also try direct match
            if source_bone is None and smplh_name in skeleton_nodes:
                source_bone = skeleton_nodes[smplh_name]
            
            if source_bone is None:
                continue  # Leave as identity (zeros)
            
            # Extract rotations
            euler_rots = extract_joint_rotations(source_bone, num_frames, fps)
            axis_angle = euler_to_axis_angle(euler_rots)
            poses[:, smplh_idx, :] = axis_angle
            mapped_count += 1
            
            # Extract translation for root (Pelvis)
            if smplh_idx == 0:
                trans = extract_joint_translations(source_bone, num_frames, fps)
                # Convert cm to meters if needed (common in FBX)
                if np.abs(trans).max() > 10:
                    trans = trans / 100.0
        
        print(f"  Mapped {mapped_count}/52 joints")
        
        # Save as NPZ
        base_name = os.path.splitext(os.path.basename(fbx_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.npz")
        
        np.savez(
            output_path,
            poses=poses,           # [num_frames, 52, 3]
            trans=trans,           # [num_frames, 3]
            fps=np.array([fps]),
            source_file=fbx_path
        )
        
        print(f"  SAVED: {output_path}")
        return output_path
        
    finally:
        manager.Destroy()


def batch_convert(input_dir: str, output_dir: str):
    """Convert all FBX files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    fbx_files = glob.glob(os.path.join(input_dir, "**", "*.fbx"), recursive=True)
    fbx_files += glob.glob(os.path.join(input_dir, "**", "*.FBX"), recursive=True)
    
    print(f"Found {len(fbx_files)} FBX files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for fbx_path in fbx_files:
        result = convert_fbx_to_smplh(fbx_path, output_dir)
        if result:
            success += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"CONVERSION COMPLETE")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Output:  {output_dir}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FBX animations to SMPL-H format")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing FBX files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for NPZ files")
    
    args = parser.parse_args()
    
    if fbx is None:
        print("FBX SDK not available. Cannot convert.")
        exit(1)
    
    batch_convert(args.input, args.output)
