"""
Retargeting utilities for converting HY-Motion data to custom FBX skeletons.
Extracted from retargetfbxnpzfull.py for ComfyUI integration.
"""
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import fbx
    from fbx import *
    HAS_FBX_SDK = True
except ImportError:
    HAS_FBX_SDK = False

# SMPL-H Mean Hand Pose Constants
LEFT_HAND_MEAN_AA = np.array([
    0.1117,  0.0429, -0.4164,  0.1088, -0.0660, -0.7562, -0.0964, -0.0909,
    -0.1885, -0.1181,  0.0509, -0.5296, -0.1437,  0.0552, -0.7049, -0.0192,
    -0.0923, -0.3379, -0.4570, -0.1963, -0.6255, -0.2147, -0.0660, -0.5069,
    -0.3697, -0.0603, -0.0795, -0.1419, -0.0859, -0.6355, -0.3033, -0.0579,
    -0.6314, -0.1761, -0.1321, -0.3734,  0.8510,  0.2769, -0.0915, -0.4998,
    0.0266,  0.0529,  0.5356,  0.0460, -0.2774]
)
RIGHT_HAND_MEAN_AA = np.array([
    0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
    0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
    0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
    -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
    0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
    -0.0266, -0.0529,  0.5356, -0.0460,  0.2774]
)

# Base bone mapping for Mixamo rigs
BASE_BONE_MAPPING = {
    "hips": "mixamorig:hips", "pelvis": "mixamorig:hips",
    "spine": "mixamorig:spine", "spine1": "mixamorig:spine",
    "spine2": "mixamorig:spine1", "spine3": "mixamorig:spine2",
    "chest": "mixamorig:spine2", "neck": "mixamorig:neck", "head": "mixamorig:head",
    "leftupleg": "mixamorig:leftupleg", "rightupleg": "mixamorig:rightupleg",
    "leftleg": "mixamorig:leftleg", "rightleg": "mixamorig:rightleg",
    "leftfoot": "mixamorig:leftfoot", "rightfoot": "mixamorig:rightfoot",
    "leftshoulder": "mixamorig:leftshoulder", "rightshoulder": "mixamorig:rightshoulder",
    "leftarm": "mixamorig:leftarm", "rightarm": "mixamorig:rightarm",
    "leftforearm": "mixamorig:leftforearm", "rightforearm": "mixamorig:rightforearm",
    "lefthand": "mixamorig:lefthand", "righthand": "mixamorig:righthand",
    
    # SMPL-H style
    "l_collar": "mixamorig:leftshoulder", "r_collar": "mixamorig:rightshoulder",
    "l_shoulder": "mixamorig:leftarm", "r_shoulder": "mixamorig:rightarm",
    "l_elbow": "mixamorig:leftforearm", "r_elbow": "mixamorig:rightforearm",
    "l_wrist": "mixamorig:lefthand", "r_wrist": "mixamorig:righthand",
    
    # Fingers
    "l_index1": "mixamorig:lefthandindex1", "l_index2": "mixamorig:lefthandindex2", "l_index3": "mixamorig:lefthandindex3",
    "r_index1": "mixamorig:righthandindex1", "r_index2": "mixamorig:righthandindex2", "r_index3": "mixamorig:righthandindex3",
    "l_middle1": "mixamorig:lefthandmiddle1", "l_middle2": "mixamorig:lefthandmiddle2", "l_middle3": "mixamorig:lefthandmiddle3",
    "r_middle1": "mixamorig:righthandmiddle1", "r_middle2": "mixamorig:righthandmiddle2", "r_middle3": "mixamorig:righthandmiddle3",
    "l_ring1": "mixamorig:lefthandring1", "l_ring2": "mixamorig:lefthandring2", "l_ring3": "mixamorig:lefthandring3",
    "r_ring1": "mixamorig:righthandring1", "r_ring2": "mixamorig:righthandring2", "r_ring3": "mixamorig:righthandring3",
    "l_pinky1": "mixamorig:lefthandpinky1", "l_pinky2": "mixamorig:lefthandpinky2", "l_pinky3": "mixamorig:lefthandpinky3",
    "r_pinky1": "mixamorig:righthandpinky1", "r_pinky2": "mixamorig:righthandpinky2", "r_pinky3": "mixamorig:righthandpinky3",
    "l_thumb1": "mixamorig:lefthandthumb1", "l_thumb2": "mixamorig:lefthandthumb2", "l_thumb3": "mixamorig:lefthandthumb3",
    "r_thumb1": "mixamorig:righthandthumb1", "r_thumb2": "mixamorig:righthandthumb2", "r_thumb3": "mixamorig:righthandthumb3",
}

FUZZY_ALIASES = {
    'hips': ['pelvis', 'root_joint', 'spine_01'],
    'spine': ['spine', 'chest'],
    'neck': ['neck'],
    'head': ['head', 'head_top'],
    'upleg': ['thigh', 'hip'],
    'leg': ['knee', 'leg'],
    'foot': ['ankle', 'foot'],
    'toe': ['toe', 'ball'],
    'shoulder': ['collar', 'clavicle', 'shoulder_01'],
    'arm': ['shoulder', 'upperarm', 'arm'],
    'forearm': ['elbow', 'forearm'],
    'hand': ['wrist', 'hand'],
    'thumb': ['thumb'],
    'index': ['index'],
    'middle': ['middle'],
    'ring': ['ring'],
    'pinky': ['pinky', 'little'],
}

def load_bone_mapping(filepath: str) -> dict:
    """Load bone mapping from JSON file, merging with base mapping."""
    mapping = BASE_BONE_MAPPING.copy()
    if not os.path.exists(filepath):
        return mapping
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    bones = data.get("bones", {})
    for key, values in bones.items():
        if isinstance(values, list):
            if len(values) >= 2:
                src = values[0].lower()
                tgt = values[-1].lower()
                mapping[src] = tgt
            elif len(values) == 1:
                mapping[key.lower()] = values[0].lower()
        elif isinstance(values, str):
            mapping[key.lower()] = values.lower()
    
    return mapping


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.sum(q**2)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
