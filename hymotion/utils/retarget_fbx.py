from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# SMPL-H Mean Hand Pose Constants (from ComfyUI-HyMotion/body_model.py)
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

# Attempt to import FBX SDK
try:
    import fbx
    from fbx import *
    HAS_FBX_SDK = True
except ImportError:
    HAS_FBX_SDK = False

# =============================================================================
# Math Utilities
# =============================================================================

def fbx_matrix_to_numpy(fbx_mat) -> np.ndarray:
    """Convert FBX matrix (FbxMatrix or FbxAMatrix) to numpy 4x4."""
    mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            mat[i, j] = fbx_mat.Get(i, j)
    return mat

def matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Convert 4x4 or 3x3 matrix to [w, x, y, z] quaternion with normalization."""
    m33 = mat[:3, :3].copy()
    # Orthonormalize basis vectors (rows) to strip scaling/shearing
    for i in range(3):
        norm = np.linalg.norm(m33[i])
        if norm > 1e-9: m33[i] /= norm
        
    # FBX basis vectors are in rows (Row-Major convention V' = V * M)
    # SciPy expects basis vectors in columns (Column-Major y = M * x)
    m33_col = m33.T 
    rot = R.from_matrix(m33_col)
    q = rot.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])

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

def solve_rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns a 3x3 rotation matrix that aligns v1 with v2 via shortest path."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-9 or v2_norm < 1e-9: return np.eye(3)
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    dot = np.dot(v1, v2)
    if dot > 0.999999: return np.eye(3)
    if dot < -0.999999:
        # 180 deg: Pick a perpendicular axis
        axis = np.array([0, 1, 0]) if abs(v1[0]) > 0.9 else np.array([1, 0, 0])
        axis = np.cross(v1, axis)
        axis /= (np.linalg.norm(axis) + 1e-9)
        return R.from_rotvec(axis * np.pi).as_matrix()
    
    axis = np.cross(v1, v2)
    axis_len = np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis / axis_len * angle).as_matrix()

def look_at_matrix(fwd: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Creates a 3x3 rotation matrix where X is fwd and Z is up_hint."""
    f = fwd / (np.linalg.norm(fwd) + 1e-9)
    s = np.cross(up_hint, f)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(f, s)
    return np.stack((f, s, u), axis=-1)

def decompose_swing_twist(q: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose quaternion q [w, x, y, z] into swing and twist relative to axis.
    Twist is the rotation around the axis. Swing is the remaining rotation.
    """
    # q_xyz = [x, y, z]
    q_xyz = q[1:]
    # Project q_xyz onto axis
    projection = np.dot(q_xyz, axis) * axis
    # Twist = [w, projection]
    twist = np.array([q[0], projection[0], projection[1], projection[2]])
    mag = np.linalg.norm(twist)
    if mag > 1e-9:
        twist /= mag
    else:
        twist = np.array([1.0, 0, 0, 0])
    # Swing = q * twist_inv
    swing = quaternion_multiply(q, quaternion_inverse(twist))
    return swing, twist

# =============================================================================
# Core Data Structures
# =============================================================================

# =============================================================================
# Core Data Structures
# =============================================================================

class BoneData:
    def __init__(self, name: str):
        self.name = name
        self.parent_name = None  # Immediate FBX node parent
        self.bone_parent_name = None # Parent in bone hierarchy
        self.local_matrix = np.eye(4)
        self.world_matrix = np.eye(4)
        self.head: np.ndarray = np.zeros(3)
        self.has_skeleton_attr: bool = False
        self.rest_rotation = np.array([1, 0, 0, 0])
        self.animation: dict[int, np.ndarray] = {}  # frame -> local [w, x, y, z]
        self.world_animation: dict[int, np.ndarray] = {} # frame -> world [w, x, y, z]
        self.location_animation: dict[int, np.ndarray] = {} # frame -> local pos
        self.world_location_animation: dict[int, np.ndarray] = {} # frame -> world pos

class Skeleton:
    def __init__(self, name: str = "Skeleton"):
        self.name = name
        self.bones: dict[str, BoneData] = {}
        self.all_nodes: dict[str, str] = {} # node_name -> node_name (for hierarchy)
        self.node_rest_rotations: dict[str, np.ndarray] = {}  # node -> world_rest_q [w,x,y,z]
        self.node_world_matrices: dict[str, np.ndarray] = {} # node -> world_rest_mat (4x4)
        self.fps = 30.0
        self.frame_start = 0
        self.frame_end = 0

    def add_bone(self, bone: BoneData):
        self.bones[bone.name.lower()] = bone
    
    def get_bone_case_insensitive(self, name: str) -> BoneData:
        if not name: return None
        lower_name = name.lower()
        if lower_name in self.bones: return self.bones[lower_name]
        
        # 1. Try stripping namespaces (e.g. Unirig:Hips -> Hips)
        simplified = lower_name.split(":")[-1]
        if simplified in self.bones: return self.bones[simplified]
        
        # 2. Try strict exact match on simplified names
        for bname, bone in self.bones.items():
            if bname.split(":")[-1] == simplified:
                return bone
        
        # 3. Strip standard prefixes and characters
        def clean(s):
            return s.lower().split(":")[-1].replace("bip01 ", "").replace(".", "").replace("_", "").replace("-", "").replace(" ", "")
            
        clean_query = clean(simplified)
        for bname, bone in self.bones.items():
            if clean(bname) == clean_query:
                return bone
                
        return None

# =============================================================================
# NPZ Support (SMPL-H)
# =============================================================================

def rot6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """Numpy version of 6D rotation to 3x3 matrix."""
    shape = d6.shape[:-1]
    x = d6.reshape(-1, 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-9)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2, axis=1)
    return np.stack((b1, b2, b3), axis=-1).reshape(*shape, 3, 3)

def load_npz(filepath: str) -> Skeleton:
    """Load motion data from NPZ file (HyMotion/SMPL-H format)."""
    data = np.load(filepath)
    # Typical HyMotion NPZ structure:
    # keypoints3d (T, 52, 3), rot6d (T, 22, 6), transl (T, 3), root_rotations_mat (T, 3, 3)
    kps = data['keypoints3d']
    transl = data['transl']
    rot6d = data.get('rot6d')
    root_mat = data.get('root_rotations_mat')
    poses = data.get('poses')
    
    T = kps.shape[0]
    # Global coordinates = Local keypoints + Translation
    global_kps = kps + transl[:, np.newaxis, :]
    
    names = [
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3", 
        "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", 
        "L_Wrist", "R_Wrist", "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3", "L_Pinky1", "L_Pinky2", 
        "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Index1", "R_Index2", "R_Index3", 
        "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", 
        "R_Thumb2", "R_Thumb3"
    ]
    
    parents = [
        -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19,
        20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
        21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50
    ]
    
    skel = Skeleton(os.path.basename(filepath))
    skel.frame_start = 0
    skel.frame_end = T - 1
    skel.fps = 30.0
    
    # Reconstruct World Rotations via Forward Kinematics
    world_rots = np.zeros((T, 52, 3, 3))
    
    # 1. Compute Canonical Rest Pose (All Identity Local)
    # This keeps the character stable and prevents 'morphing'
    rest_world_rots = np.zeros((52, 3, 3))
    rest_world_rots[0] = np.eye(3) # Root
    for i in range(1, 52):
        p = parents[i]
        rest_world_rots[i] = rest_world_rots[p] # Local Identity
            
    # Child lookup for vector-based solving (fingers)
    children = {}
    for idx, p in enumerate(parents):
        if p != -1 and p not in children: # Find first child
            children[p] = idx

    # 1. Determine Local Rotations
    local_mats = np.zeros((T, 52, 3, 3))
    for f in range(T):
        for i in range(52):
            local_mats[f, i] = np.eye(3)

    if poses is not None:
        print(f"Using full 52-joint 'poses' from NPZ...")
        aa = poses.reshape(T, 52, 3)
        for f in range(T):
            for i in range(52):
                local_mats[f, i] = R.from_rotvec(aa[f, i]).as_matrix()
    elif rot6d is not None:
        print(f"Using 22-joint 'rot6d' (with mean hand injection) from NPZ...")
        rot_mats_22 = rot6d_to_matrix_np(rot6d)
        l_hand_mats = R.from_rotvec(LEFT_HAND_MEAN_AA.reshape(15, 3)).as_matrix()
        r_hand_mats = R.from_rotvec(RIGHT_HAND_MEAN_AA.reshape(15, 3)).as_matrix()
        for f in range(T):
            lim = min(22, rot_mats_22.shape[1])
            for i in range(lim):
                local_mats[f, i] = rot_mats_22[f, i]
            for i in range(15):
                if 22+i < 52: local_mats[f, 22+i] = l_hand_mats[i]
                if 37+i < 52: local_mats[f, 37+i] = r_hand_mats[i]

    # 2. Forward Kinematics to get World Rotations
    for f in range(T):
        # Root (Pelvis)
        if root_mat is not None:
            world_rots[f, 0] = root_mat[f]
        else:
            world_rots[f, 0] = local_mats[f, 0]
            
        for i in range(1, 52):
            p = parents[i]
            world_rots[f, i] = world_rots[f, p] @ local_mats[f, i]

    # 3. Create a Standing Rest Pose for displacement reference
    # We calculate the height of the character in a standing pose
    standing_kps = np.zeros((52, 3))
    standing_kps[0] = [0, 0, 0] # Root at origin
    for i in range(1, 52):
        p = parents[i]
        # In SMPL-H rest pose (Identity), bones are oriented such that 
        # the vector to the child is the rest-pose offset.
        # We'll just use the first frame's bone lengths but oriented vertically.
        dist = np.linalg.norm(kps[0, i] - kps[0, p])
        # Rough vertical stack for spine/legs
        if i in [1, 2, 4, 5, 7, 8, 10, 11]: # Legs
            standing_kps[i] = standing_kps[p] + [0, -dist, 0]
        else: # Torso/Arms
            standing_kps[i] = standing_kps[p] + [0, dist, 0]

    # Shift standing kps so the lowest point is at Y=0
    y_min = np.min(standing_kps[:, 1])
    standing_kps[:, 1] -= y_min

    for i, name in enumerate(names):
        bone = BoneData(name)
        p_idx = parents[i]
        if p_idx != -1: bone.parent_name = names[p_idx]
        
        for f in range(T):
            bone.world_animation[f] = matrix_to_quaternion(world_rots[f, i].T)
            bone.world_location_animation[f] = global_kps[f, i]
            
        bone.rest_rotation = matrix_to_quaternion(rest_world_rots[i].T)
        
        # Use standing pose for the rest matrix reference
        bone.head = standing_kps[i]
        bone.world_matrix = np.eye(4)
        bone.world_matrix[:3, :3] = rest_world_rots[i]
        bone.world_matrix[3, :3] = standing_kps[i]
        
        skel.add_bone(bone)
        skel.all_nodes[name] = name
        skel.node_rest_rotations[name] = bone.rest_rotation
        skel.node_world_matrices[name] = bone.world_matrix
        
    return skel

# =============================================================================
BASE_BONE_MAPPING = {
    "hips": "mixamorig:hips",
    "pelvis": "mixamorig:hips",
    "spine": "mixamorig:spine",
    "spine1": "mixamorig:spine",
    "spine2": "mixamorig:spine1",
    "spine3": "mixamorig:spine2",
    "chest": "mixamorig:spine2",

    "neck": "mixamorig:neck",
    "neck1": "mixamorig:neck",
    "neck2": "mixamorig:neck",
    "head": "mixamorig:head",
    "head1": "mixamorig:head",
    "leftupleg": "mixamorig:leftupleg",
    "rightupleg": "mixamorig:rightupleg",
    "leftleg": "mixamorig:leftleg",
    "rightleg": "mixamorig:rightleg",
    "leftfoot": "mixamorig:leftfoot",
    "rightfoot": "mixamorig:rightfoot",
    "leftshoulder": "mixamorig:leftshoulder",
    "rightshoulder": "mixamorig:rightshoulder",
    "leftarm": "mixamorig:leftarm",
    "rightarm": "mixamorig:rightarm",
    "leftforearm": "mixamorig:leftforearm",
    "rightforearm": "mixamorig:rightforearm",
    "lefthand": "mixamorig:lefthand",
    "righthand": "mixamorig:righthand",

    # SMPL-H Naming Style (Explicit)
    "l_collar": "mixamorig:leftshoulder", "r_collar": "mixamorig:rightshoulder",
    "l_shoulder": "mixamorig:leftarm", "r_shoulder": "mixamorig:rightarm",
    "l_elbow": "mixamorig:leftforearm", "r_elbow": "mixamorig:rightforearm",
    "l_wrist": "mixamorig:lefthand", "r_wrist": "mixamorig:righthand",
    
    # SMPL-H Uppercase variants (CRITICAL for proper matching)
    "L_Wrist": "mixamorig:lefthand", "R_Wrist": "mixamorig:righthand",
    "L_Elbow": "mixamorig:leftforearm", "R_Elbow": "mixamorig:rightforearm",
    "L_Shoulder": "mixamorig:leftarm", "R_Shoulder": "mixamorig:rightarm",
    "L_Collar": "mixamorig:leftshoulder", "R_Collar": "mixamorig:rightshoulder",

    # Explicit SMPL-H and Common Hand variations
    "left_hand": "mixamorig:lefthand", "right_hand": "mixamorig:righthand",
    "left_wrist": "mixamorig:lefthand", "right_wrist": "mixamorig:righthand",
    "lhand": "mixamorig:lefthand", "rhand": "mixamorig:righthand",
    
    # Fingers - Left Hand
    "leftthumb1": "mixamorig:lefthandthumb1", 
    "leftthumbmedial": "mixamorig:lefthandthumb1",
    "leftthumb2": "mixamorig:lefthandthumb2", 
    "leftthumbdistal": "mixamorig:lefthandthumb2",
    "leftthumb3": "mixamorig:lefthandthumb3",
    "l_thumb1": "mixamorig:lefthandthumb1", 
    "l_thumb2": "mixamorig:lefthandthumb2", 
    "l_thumb3": "mixamorig:lefthandthumb3",
    
    "leftindex1": "mixamorig:lefthandindex1", 
    "leftindexmedial": "mixamorig:lefthandindex1",
    "leftindex2": "mixamorig:lefthandindex2", 
    "leftindexdistal": "mixamorig:lefthandindex2",
    "leftindex3": "mixamorig:lefthandindex3",
    "l_index1": "mixamorig:lefthandindex1", 
    "l_index2": "mixamorig:lefthandindex2", 
    "l_index3": "mixamorig:lefthandindex3",
    
    "leftmiddle1": "mixamorig:lefthandmiddle1",
    "leftmiddle2": "mixamorig:lefthandmiddle2",
    "leftmiddle3": "mixamorig:lefthandmiddle3",
    "l_middle1": "mixamorig:lefthandmiddle1", 
    "l_middle2": "mixamorig:lefthandmiddle2", 
    "l_middle3": "mixamorig:lefthandmiddle3",
    
    "leftring1": "mixamorig:lefthandring1", 
    "leftringmedial": "mixamorig:lefthandring1",
    "leftring2": "mixamorig:lefthandring2", 
    "leftringdistal": "mixamorig:lefthandring2",
    "leftring3": "mixamorig:lefthandring3",
    "l_ring1": "mixamorig:lefthandring1", 
    "l_ring2": "mixamorig:lefthandring2", 
    "l_ring3": "mixamorig:lefthandring3",
    
    "leftpinky1": "mixamorig:lefthandpinky1", 
    "leftlittlemedial": "mixamorig:lefthandpinky1",
    "leftpinky2": "mixamorig:lefthandpinky2", 
    "leftlittledistal": "mixamorig:lefthandpinky2",
    "leftpinky3": "mixamorig:lefthandpinky3",
    "l_pinky1": "mixamorig:lefthandpinky1", 
    "l_pinky2": "mixamorig:lefthandpinky2", 
    "l_pinky3": "mixamorig:lefthandpinky3",
    
    # Fingers - Right Hand
    "rightthumb1": "mixamorig:righthandthumb1",
    "rightthumbmedial": "mixamorig:righthandthumb1",
    "rightthumb2": "mixamorig:righthandthumb2",
    "rightthumbdistal": "mixamorig:righthandthumb2",
    "rightthumb3": "mixamorig:righthandthumb3",
    "r_thumb1": "mixamorig:righthandthumb1", 
    "r_thumb2": "mixamorig:righthandthumb2", 
    "r_thumb3": "mixamorig:righthandthumb3",
    
    "rightindex1": "mixamorig:righthandindex1",
    "rightindexmedial": "mixamorig:righthandindex1",
    "rightindex2": "mixamorig:righthandindex2",
    "rightindexdistal": "mixamorig:righthandindex2",
    "rightindex3": "mixamorig:righthandindex3",
    "r_index1": "mixamorig:righthandindex1", 
    "r_index2": "mixamorig:righthandindex2", 
    "r_index3": "mixamorig:righthandindex3",
    
    "rightmiddle1": "mixamorig:righthandmiddle1",
    "rightmiddle2": "mixamorig:righthandmiddle2",
    "rightmiddle3": "mixamorig:righthandmiddle3",
    "r_middle1": "mixamorig:righthandmiddle1", 
    "r_middle2": "mixamorig:righthandmiddle2", 
    "r_middle3": "mixamorig:righthandmiddle3",
    
    "rightring1": "mixamorig:righthandring1",
    "rightringmedial": "mixamorig:righthandring1",
    "rightring2": "mixamorig:righthandring2",
    "rightringdistal": "mixamorig:righthandring2",
    "rightring3": "mixamorig:righthandring3",
    "r_ring1": "mixamorig:righthandring1", 
    "r_ring2": "mixamorig:righthandring2", 
    "r_ring3": "mixamorig:righthandring3",
    
    "rightpinky1": "mixamorig:righthandpinky1",
    "rightlittlemedial": "mixamorig:righthandpinky1",
    "rightpinky2": "mixamorig:righthandpinky2",
    "rightlittledistal": "mixamorig:righthandpinky2",
    "rightpinky3": "mixamorig:righthandpinky3",
    "r_pinky1": "mixamorig:righthandpinky1", 
    "r_pinky2": "mixamorig:righthandpinky2", 
    "r_pinky3": "mixamorig:righthandpinky3",
    
    # Feet/Toes
    "l_foot": "mixamorig:lefttoebase",
    "r_foot": "mixamorig:righttoebase",
    
    # SMPL-H Uppercase leg/foot variants (CRITICAL for proper matching)
    "L_Hip": "mixamorig:leftupleg", "R_Hip": "mixamorig:rightupleg",
    "L_Knee": "mixamorig:leftleg", "R_Knee": "mixamorig:rightleg",
    "L_Ankle": "mixamorig:leftfoot", "R_Ankle": "mixamorig:rightfoot",
    "L_Foot": "mixamorig:lefttoebase", "R_Foot": "mixamorig:righttoebase",

    "bone_0": "mixamorig:hips", # Unirig Hip
    "bone_8": "mixamorig:leftforearm", # Unirig L_Elbow
    "bone_26": "mixamorig:rightarm", # Unirig R_Shoulder
    "bone_27": "mixamorig:rightforearm", # Unirig R_Elbow
}

# Comprehensive bone name aliases for fuzzy matching
FUZZY_ALIASES = {
    'hips': ['pelvis', 'root_joint', 'spine_01', 'hip', 'root', 'cog', 'center', 'base'],
    'spine': ['spine', 'chest', 'back', 'torso', 'spine1', 'spine2', 'spine3'],
    'neck': ['neck', 'neck_01', 'neckbase', 'neck1', 'neck2', 'cervical', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'],
    'head': ['head', 'head_top', 'skull', 'cranium', 'head1', 'head2'],
    'upleg': ['thigh', 'hip', 'upperleg', 'leg_upper', 'femur'],
    'leg': ['knee', 'leg', 'lowerleg', 'leg_lower', 'shin', 'calf'],
    'foot': ['ankle', 'foot', 'ankle_01'],
    'toe': ['toe', 'ball', 'toebase', 'foot_end'],
    'shoulder': ['collar', 'clavicle', 'shoulder_01', 'scapula'],
    'arm': ['shoulder', 'upperarm', 'arm', 'arm_upper', 'humerus'],
    'forearm': ['elbow', 'forearm', 'arm_lower', 'lowerarm', 'ulna'],
    'hand': ['wrist', 'hand', 'palm'],
    'thumb': ['thumb', 'pollex'],
    'index': ['index', 'pointer'],
    'middle': ['middle', 'long'],
    'ring': ['ring', 'third'],
    'pinky': ['pinky', 'little', 'small'],
}

BONE_KEYWORDS = {
    'root': ['hips', 'pelvis', 'root', 'cog', 'center'],
    'spine': ['spine', 'chest', 'back', 'torso'],
    'neck': ['neck'],
    'head': ['head', 'skull'],
    'leg': ['upleg', 'thigh', 'knee', 'leg', 'ankle', 'foot', 'toe'],
    'arm': ['shoulder', 'collar', 'clavicle', 'arm', 'elbow', 'forearm', 'hand', 'wrist'],
    'finger': ['thumb', 'index', 'middle', 'ring', 'pinky', 'digit'],
}

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings for typo detection"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def normalize_bone_name(name: str) -> str:
    """Normalize bone name by removing prefixes and common variations"""
    name_lower = name.lower()
    
    # Remove namespaces and standard prefixes
    if ":" in name_lower:
        name_lower = name_lower.split(":")[-1]
        
    prefixes = ['bip01_', 'bip001_', 'joint_', 'bone_', 'def_', 'rig_', 'valvebiped_', 'mixamorig_']
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix):]
    
    # Remove underscores and dots for comparison
    name_lower = name_lower.replace('_', '').replace('.', '').replace('-', '')
    
    return name_lower

def detect_side(name: str) -> tuple[bool, bool]:
    """
    Detect if bone is left or right side with improved logic.
    Returns: (is_left, is_right)
    """
    name_lower = name.lower()
    
    # Explicit left markers
    is_left = any(x in name_lower for x in [
        'left', '.l', '_l', 'l_', 'lhand', 'lfoot', 'larm', 'lleg'
    ]) or (name_lower.startswith('l') and len(name_lower) > 1 and name_lower[1] in ['_', '.'])
    
    # Explicit right markers
    is_right = any(x in name_lower for x in [
        'right', '.r', '_r', 'r_', 'rhand', 'rfoot', 'rarm', 'rleg'
    ]) or (name_lower.startswith('r') and len(name_lower) > 1 and name_lower[1] in ['_', '.'])
    
    return is_left, is_right

def classify_bone(name: str) -> list[str]:
    """Classify bone into categories (root, spine, leg, arm, finger, etc.)"""
    name_norm = normalize_bone_name(name)
    categories = []
    
    for category, keywords in BONE_KEYWORDS.items():
        if any(kw in name_norm for kw in keywords):
            categories.append(category)
    
    return categories

def calculate_bone_similarity(source_name: str, target_name: str, use_aliases: bool = True) -> float:
    """
    Calculate similarity score between two bone names (0.0 = no match, 1.0 = perfect match)
    Uses multiple matching strategies.
    """
    s_norm = normalize_bone_name(source_name)
    t_norm = normalize_bone_name(target_name)
    
    # Strategy 1: Exact match (perfect score)
    if s_norm == t_norm:
        return 1.0
    
    # Strict category separation
    s_categories = classify_bone(source_name)
    t_categories = classify_bone(target_name)
    
    # Prevent 'hand' bones from matching 'finger' bones (common failure case)
    if ('hand' in s_categories or 'arm' in s_categories) and 'finger' in t_categories:
        # A generic hand bone should never match a specific finger bone
        return 0.0
    if 'finger' in s_categories and ('hand' in t_categories or 'arm' in t_categories):
        return 0.0

    # Strategy 2: One contains the other (substring match)
    if s_norm in t_norm or t_norm in s_norm:
        overlap = min(len(s_norm), len(t_norm))
        total = max(len(s_norm), len(t_norm))
        score = 0.85 * (overlap / total)
        
        # Strict finger separation for substring matches
        if 'finger' in s_categories or 'finger' in t_categories:
            # If both are fingers, must share same type and segment
            s_type = next((c for c in ['thumb', 'index', 'middle', 'ring', 'pinky'] if c in s_categories), None)
            t_type = next((c for c in ['thumb', 'index', 'middle', 'ring', 'pinky'] if c in t_categories), None)
            if s_type != t_type: return 0.0
            
            # Segment check
            s_seg = 1 if '1' in s_norm else (2 if '2' in s_norm else (3 if '3' in s_norm else 0))
            t_seg = 0
            if 'proximal' in t_norm: t_seg = 1
            elif 'distal' in t_norm or 'dist' in t_norm: t_seg = 3
            elif 'medial' in t_norm or 'inter' in t_norm: t_seg = 2
            elif '1' in t_norm: t_seg = 1
            elif '2' in t_norm: t_seg = 2
            elif '3' in t_norm: t_seg = 3
            
            if s_seg != 0 and t_seg != 0 and s_seg != t_seg: return 0.0
        
        # Apply category penalty if they don't share enough categories
        if s_categories and t_categories:
            intersect = set(s_categories) & set(t_categories)
            if not intersect:
                score *= 0.1 # Very heavy penalty for body part mismatch
        return score
    
    # Strategy 3: Alias matching
    if use_aliases:
        for target_keyword, source_aliases in FUZZY_ALIASES.items():
            if target_keyword in t_norm:
                for alias in source_aliases:
                    if alias in s_norm:
                        return 0.75  # Good match via alias
    
    # Strategy 4: Levenshtein distance (typo tolerance)
    edit_distance = levenshtein_distance(s_norm, t_norm)
    max_len = max(len(s_norm), len(t_norm))
    if max_len > 0:
        similarity = 1.0 - (edit_distance / max_len)
        if similarity > 0.6:  # Only accept if reasonably similar
            score = similarity * 0.7
            if s_categories and t_categories:
                if not (set(s_categories) & set(t_categories)):
                    score *= 0.3 # Heavy penalty for category mismatch
            return score
    
    # Strategy 5: Category matching (same type of bone)
    if s_categories and t_categories:
        # Check finger type equality FIRST
        s_finger_types = set(['thumb', 'index', 'middle', 'ring', 'pinky']) & set(s_categories)
        t_finger_types = set(['thumb', 'index', 'middle', 'ring', 'pinky']) & set(t_categories)
        if s_finger_types or t_finger_types:
            if s_finger_types != t_finger_types: return 0.0

        overlap = len(set(s_categories) & set(t_categories))
        if overlap > 0:
            score = 0.5 * (overlap / max(len(s_categories), len(t_categories)))
            
            # Segment specific adjustment for fingers
            if 'finger' in s_categories:
                s_seg = 1 if '1' in s_norm else (2 if '2' in s_norm else (3 if '3' in s_norm else 0))
                t_seg = 0
                if 'proximal' in t_norm: t_seg = 1
                elif 'distal' in t_norm: t_seg = 3
                elif 'medial' in t_norm: t_seg = 2
                elif '1' in t_norm: t_seg = 1
                elif '2' in t_norm: t_seg = 2
                elif '3' in t_norm: t_seg = 3
                
                if s_seg != 0 and t_seg != 0:
                    if s_seg == t_seg: score += 0.4
                    else: return 0.0
            
            # if 'index' in s_norm or 'index' in t_norm or 'middle' in s_norm or 'middle' in t_norm:
            #    print(f"  [DEBUG SCORE] {s_bone_name} vs {t_bone_name}: {score} (cats: {s_categories} & {t_categories})")
            return score
    
    return 0.0  # No match

def find_best_bone_match(
    target_bone_name: str,
    source_skeleton: 'Skeleton',
    already_mapped_sources: set[str],
    require_side_match: bool = True
) -> tuple[str, float]:
    """
    Find the best matching source bone for a target bone.
    Returns: (source_bone_name, confidence_score)
    """
    t_left, t_right = detect_side(target_bone_name)
    best_match = None
    best_score = 0.0
    
    for s_bone_name in source_skeleton.bones.keys():
        # Skip already mapped bones
        if s_bone_name in already_mapped_sources:
            continue
        
        # Side matching check
        if require_side_match:
            s_left, s_right = detect_side(s_bone_name)
            # If one is sided and the other isn't, or they're opposite sides, skip
            if (t_left != s_left) or (t_right != s_right):
                continue
        
        # Calculate similarity
        score = calculate_bone_similarity(s_bone_name, target_bone_name)
        
        if score > best_score:
            best_score = score
            best_match = s_bone_name
    
    return best_match, best_score

def get_skeleton_height(skeleton: Skeleton, mapping_dict: dict = None) -> float:
    """
    Calculate skeleton height using bone lengths (Pelvis -> Spine -> Neck -> Head).
    This is pose-invariant and works even if the character is lying down.
    """
    # 1. Try Bone Chain Method (Robust)
    # Standard SMPL-H / Mixamo chain
    pelvis_names = ['pelvis', 'hips', 'bone_0', 'mixamorig:hips']
    spine_names = ['spine', 'spine1', 'spine2', 'spine3', 'chest', 'mixamorig:spine', 'mixamorig:spine1', 'mixamorig:spine2']
    neck_names = ['neck', 'mixamorig:neck']
    head_names = ['head', 'mixamorig:head']

    def find_bone(names):
        for n in names:
            b = skeleton.get_bone_case_insensitive(n)
            if b: return b
        return None

    pelvis = find_bone(pelvis_names)
    head = find_bone(head_names)

    if pelvis and head:
        # Calculate distance between joints in rest pose
        # We sum the lengths of the segments to get the "true" height
        total_len = 0.0
        curr = head
        visited = set()
        while curr and curr.name.lower() not in [p.lower() for p in pelvis_names] and curr.name not in visited:
            visited.add(curr.name)
            pname = curr.parent_name
            if not pname: break
            parent = skeleton.get_bone_case_insensitive(pname)
            if not parent: break
            
            # Length is distance between parent head and child head in rest pose
            dist = np.linalg.norm(curr.head - parent.head)
            total_len += dist
            curr = parent
        
        # Add leg length (approximate from Pelvis to Foot)
        # We look for a leg chain
        leg_names = ['l_hip', 'l_knee', 'l_ankle', 'leftupleg', 'leftleg', 'leftfoot', 'mixamorig:leftupleg', 'mixamorig:leftleg', 'mixamorig:leftfoot']
        l_hip = find_bone(['l_hip', 'leftupleg', 'mixamorig:leftupleg'])
        l_foot = find_bone(['l_ankle', 'leftfoot', 'mixamorig:leftfoot'])
        
        if l_hip and l_foot:
            leg_len = 0.0
            curr = l_foot
            while curr and curr != l_hip and curr.name not in visited:
                pname = curr.parent_name
                if not pname: break
                parent = skeleton.get_bone_case_insensitive(pname)
                if not parent: break
                leg_len += np.linalg.norm(curr.head - parent.head)
                curr = parent
            total_len += leg_len
        else:
            # Fallback: if no leg found, just double the torso (rough)
            total_len *= 2.0
            
        if total_len > 0.1:
            print(f"[DEBUG] Height via Bone Chain: {total_len:.4f}")
            return total_len

    # 2. Fallback to Y-Range Method (Legacy)
    y_min, y_max = 999999.0, -999999.0
    found_any = False
    for _, bone in skeleton.bones.items():
        h_val = bone.head[1]
        if abs(h_val) < 1e-6: continue
        y_min = min(y_min, h_val)
        y_max = max(y_max, h_val)
        found_any = True
            
    if not found_any or y_max <= y_min: return 1.0
    print(f"[DEBUG] Height via Y-Range: {y_max - y_min:.4f}")
    return y_max - y_min

def load_bone_mapping(filepath: str, src_skel: Skeleton, tgt_skel: Skeleton) -> dict[str, str]:
    mapping = BASE_BONE_MAPPING.copy()
    if not filepath or not os.path.exists(filepath):
        print(f"Using hardcoded bone mappings (no JSON file needed)")
        return mapping
        
    print(f"Loading Mapping File: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    json_bones = data.get("bones", {})
    
    match_count = 0
    mapped_sources = set()
    mapped_targets = set()
    
    # Strategy: Try to find a source match for the JSON label
    # We use a custom order: first strictly named joints, then limbs, then fingers
    def priority_sort(item):
        k = item[0].lower()
        if any(x in k for x in ['hip', 'pelvis', 'root']): return 0
        if any(x in k for x in ['spine', 'neck', 'head']): return 1
        if any(x in k for x in ['collar', 'shoulder']): return 2
        if any(x in k for x in ['arm', 'leg', 'foot', 'hand', 'wrist']): return 3
        return 4
        
    print(f"[DEBUG] Target bones ({len(tgt_skel.bones)}): {list(tgt_skel.bones.keys())[:20]}...")
    sorted_json = sorted(json_bones.items(), key=priority_sort)

    for gen_name, aliases in sorted_json:
        if not isinstance(aliases, list): aliases = [aliases]
        search_list = [gen_name] + aliases
        
        best_tgt = None
        for a in search_list:
            found = tgt_skel.get_bone_case_insensitive(a)
            if found:
                if found.name not in mapped_targets:
                    best_tgt = found.name
                break
        
        if not best_tgt: continue

        best_src = None
        # 1. Direct Search
        for a in search_list:
            found = src_skel.get_bone_case_insensitive(a)
            if found:
                if found.name not in mapped_sources:
                    best_src = found.name
                break
        
        # 2. Fuzzy Search specifically targeting the GENERIC name (e.g. "leftUpLeg")
        if not best_src:
            for rep_name in [gen_name] + aliases:
                # Ignore raw "bone_X" for source searching
                if rep_name.lower().startswith("bone_") and len(rep_name) < 8: continue
                best_f, confidence = find_best_bone_match(rep_name, src_skel, mapped_sources, require_side_match=True)
                if best_f and confidence >= 0.3:
                    best_src = best_f
                    break

        if best_src and best_tgt:
            mapping[best_src.lower()] = best_tgt.lower()
            match_count += 1
            mapped_sources.add(best_src.lower())
            mapped_targets.add(best_tgt.lower())
            # print(f"  [JSON MATCH] {best_src} -> {best_tgt} (via {gen_name})")
            
    print(f"[Retarget] load_bone_mapping: Found {match_count} matches in JSON.")
    return mapping

def get_fbx_rotation_order_str(node: fbx.FbxNode) -> str:
    order = node.RotationOrder.Get()
    mapping = {0:'xyz', 1:'xzy', 2:'yzx', 3:'yxz', 4:'zxy', 5:'zyx'}
    return mapping.get(order, 'xyz')

# =============================================================================
# FBX Logic
# =============================================================================

def collect_skeleton_nodes(node: fbx.FbxNode, skeleton: Skeleton, immediate_parent: str = None, last_bone_parent: str = None, depth: int = 0, sampling_time: fbx.FbxTime = None, bind_pose_map: dict = None):
    node_name = node.GetName()
    attr = node.GetNodeAttribute()
    is_bone = False
    
    # 1. Coordinate Sampling (Bind Pose preferred)
    t_eval = sampling_time if sampling_time is not None else fbx.FbxTime(0)
    node_world_mat = fbx_matrix_to_numpy(node.EvaluateGlobalTransform(t_eval))
    node_local_mat = fbx_matrix_to_numpy(node.EvaluateLocalTransform(t_eval))
    
    if bind_pose_map and node_name in bind_pose_map:
        node_world_mat = bind_pose_map[node_name]
        # Recalculate local matching the bind world
        if immediate_parent:
            p_world_mat = skeleton.node_world_matrices.get(immediate_parent)
            if p_world_mat is not None:
                node_local_mat = node_world_mat @ np.linalg.inv(p_world_mat)
    
    skeleton.node_world_matrices[node_name] = node_world_mat
    skeleton.node_rest_rotations[node_name] = matrix_to_quaternion(node_world_mat)
    
    # 2. Bone Identification
    if attr:
        at = attr.GetAttributeType()
        # 3 = eSkeleton, 4 = eLimbNode, 2 = eNull
        if at in [3, 4]: is_bone = True
        elif at == 2 and (node.GetChildCount() > 0 or immediate_parent): is_bone = True
    
    name_lower = node_name.lower()
    keywords = ['hips', 'hip', 'spine', 'neck', 'head', 'arm', 'leg', 'foot', 'ankle', 'knee', 'shoulder', 'elbow', 'pelvis', 'joint', 'mixamo', 'thigh', 'upper', 'forearm', 'hand', 'finger', 'clavicle', 'collar', 'toe', 'thumb', 'index', 'middle', 'ring', 'pinky', 'upleg', 'downleg', 'wrist', 'chest', 'belly', 'bone_', 'character']
    if any(k in name_lower for k in keywords): is_bone = True
    
    if is_bone:
        existing = skeleton.get_bone_case_insensitive(node_name)
        if existing and (attr and attr.GetAttributeType() in [3, 4]):
            skeleton.bones.pop(existing.name.lower(), None)
        elif existing:
            is_bone = False

    if is_bone:
        bone = BoneData(node_name)
        bone.has_skeleton_attr = (attr and attr.GetAttributeType() in [3, 4])
        bone.parent_name = immediate_parent
        bone.bone_parent_name = last_bone_parent
        bone.local_matrix = node_local_mat
        bone.world_matrix = node_world_mat
        bone.head = node_world_mat[3, :3]
        bone.rest_rotation = skeleton.node_rest_rotations[node_name]
        skeleton.add_bone(bone)
        last_bone_parent = node_name
    
    skeleton.all_nodes[node_name] = node_name
    imm_p = node_name
    for i in range(node.GetChildCount()):
        collect_skeleton_nodes(node.GetChild(i), skeleton, imm_p, last_bone_parent, depth + 1, sampling_time, bind_pose_map)

def extract_animation(scene: FbxScene, skeleton: Skeleton):
    stack = scene.GetCurrentAnimationStack()
    if not stack: return
    time_span = stack.GetLocalTimeSpan()
    start = time_span.GetStart()
    stop = time_span.GetStop()
    mode = scene.GetGlobalSettings().GetTimeMode()
    skeleton.frame_start = int(start.GetFrameCount(mode))
    skeleton.frame_end = int(stop.GetFrameCount(mode))
    skeleton.fps = FbxTime.GetFrameRate(mode)
    
    def sample(node):
        bone = skeleton.get_bone_case_insensitive(node.GetName())
        if bone:
            for f in range(skeleton.frame_start, skeleton.frame_end + 1):
                t = FbxTime()
                t.SetFrame(f, mode)
                lmat = fbx_matrix_to_numpy(node.EvaluateLocalTransform(t))
                wmat = fbx_matrix_to_numpy(node.EvaluateGlobalTransform(t))
                bone.animation[f] = matrix_to_quaternion(lmat)
                bone.world_animation[f] = matrix_to_quaternion(wmat)
                # Save both Local and World Translation for root analysis
                bone.location_animation[f] = lmat[3, :3]
                bone.world_location_animation[f] = wmat[3, :3] # NEW: world-space pos
        for i in range(node.GetChildCount()): sample(node.GetChild(i))
    sample(scene.GetRootNode())

def load_fbx(filepath: str, sample_rest_frame: int = None):
    print(f"[Retarget] Loading FBX: {filepath}")
    if not os.path.exists(filepath):
        print(f"[Retarget] ERROR: File not found: {filepath}")
        return None, None, None

    manager = fbx.FbxManager.Create()
    if not manager:
        print("[Retarget] ERROR: Failed to create FbxManager")
        return None, None, None

    # Create IO settings - CRITICAL for stability
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(filepath, -1, ios):
        print(f"[Retarget] ERROR: Importer initialization failed: {importer.GetStatus().GetErrorString()}")
        manager.Destroy()
        return None, None, None
        
    scene = fbx.FbxScene.Create(manager, "")
    if not importer.Import(scene):
        print(f"[Retarget] ERROR: Scene import failed: {importer.GetStatus().GetErrorString()}")
        importer.Destroy()
        manager.Destroy()
        return None, None, None
    
    importer.Destroy()
    
    # Pre-extract Bind Pose into a global map for total consistency
    bind_pose_map = {}
    for i in range(scene.GetPoseCount()):
        pose = scene.GetPose(i)
        if pose and pose.IsBindPose():
            for j in range(pose.GetCount()):
                pnode = pose.GetNode(j)
                if pnode:
                    bind_pose_map[pnode.GetName()] = fbx_matrix_to_numpy(pose.GetMatrix(j))
            if bind_pose_map: break # Preference: use the first bind pose found
    
    skel = Skeleton(os.path.basename(filepath))
    collect_skeleton_nodes(scene.GetRootNode(), skel, sampling_time=FbxTime(0) if sample_rest_frame is not None else None, bind_pose_map=bind_pose_map)
    
    # Refresh frame range if needed
    return manager, scene, skel

def apply_retargeted_animation(scene, skeleton, ret_rots, ret_locs, fstart, fend, source_time_mode=None):
    if source_time_mode:
        scene.GetGlobalSettings().SetTimeMode(source_time_mode)
    else:
        # Default to 30 FPS (HyMotion standard) if no source time mode provided (e.g. NPZ)
        scene.GetGlobalSettings().SetTimeMode(fbx.FbxTime.EMode.eFrames30)
        
    tmode = scene.GetGlobalSettings().GetTimeMode()
    unit = scene.GetGlobalSettings().GetSystemUnit()
    print(f"[DEBUG] FBX TimeMode: {tmode}, SystemUnit ScaleFactor: {unit.GetScaleFactor()}")
    
    # Clear old stacks
    for i in range(scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(FbxAnimStack.ClassId)) - 1, -1, -1):
        s = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(FbxAnimStack.ClassId), i)
        scene.DisconnectSrcObject(s)
        s.Destroy()
        
    stack = FbxAnimStack.Create(scene, "Take 001")
    layer = FbxAnimLayer.Create(scene, "BaseLayer")
    stack.AddMember(layer)
    scene.SetCurrentAnimationStack(stack)
    
    def apply_node(node):
        name = node.GetName()
        if name in ret_rots:
            node.LclRotation.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
            ord_str = get_fbx_rotation_order_str(node)
            # PreRotation and PostRotation handling
            # The original code had a duplicated `cx` line and incorrect pre/post rotation application.
            # This section is updated to correctly apply pre/post rotations using scipy.
            
            # Retrieve PreRotation/PostRotation and convert to Quat (FBX stores these as Eulers)
            pq_v = node.PreRotation.Get()
            poq_v = node.PostRotation.Get()
            
            # Using scipy for robust rotation math
            # FBX Pre/Post rotations are typically baked as XYZ eulers
            pre_q = R.from_euler('xyz', [pq_v[0], pq_v[1], pq_v[2]], degrees=True)
            post_q = R.from_euler('xyz', [poq_v[0], poq_v[1], poq_v[2]], degrees=True)
            
            ord_lower = get_fbx_rotation_order_str(node).lower()
            
            cx = node.LclRotation.GetCurve(layer, "X", True)
            cy = node.LclRotation.GetCurve(layer, "Y", True)
            cz = node.LclRotation.GetCurve(layer, "Z", True)
            cx.KeyModifyBegin(); cy.KeyModifyBegin(); cz.KeyModifyBegin()
            
            for f, q_local in ret_rots[name].items():
                t = FbxTime()
                t.SetFrame(f, tmode)
                
                # Logic: R_combined = R_pre * R_local * R_post
                # Thus R_local = R_pre_inv * R_combined * R_post_inv
                ql = R.from_quat([q_local[1], q_local[2], q_local[3], q_local[0]])
                q_final = pre_q.inv() * ql * post_q.inv()
                
                # Convert back to Euler matching the node's custom rotation order
                e = q_final.as_euler(ord_lower, degrees=True)
                
                curve_map = {'x': cx, 'y': cy, 'z': cz}
                for i, axis in enumerate(ord_lower):
                    c = curve_map[axis]
                    idx = c.KeyAdd(t)[0]
                    c.KeySetValue(idx, float(e[i]))
                    c.KeySetInterpolation(idx, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
            cx.KeyModifyEnd(); cy.KeyModifyEnd(); cz.KeyModifyEnd()
            
        if name in ret_locs:
            node.LclTranslation.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
            tx = node.LclTranslation.GetCurve(layer, "X", True)
            ty = node.LclTranslation.GetCurve(layer, "Y", True)
            tz = node.LclTranslation.GetCurve(layer, "Z", True)
            tx.KeyModifyBegin(); ty.KeyModifyBegin(); tz.KeyModifyBegin()
            for f, loc in ret_locs[name].items():
                t = FbxTime()
                t.SetFrame(f, tmode)
                for c, val in zip([tx, ty, tz], loc):
                    idx = c.KeyAdd(t)[0]
                    c.KeySetValue(idx, float(val))
                    c.KeySetInterpolation(idx, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
            tx.KeyModifyEnd(); ty.KeyModifyEnd(); tz.KeyModifyEnd()
            
        for i in range(node.GetChildCount()): apply_node(node.GetChild(i))
    apply_node(scene.GetRootNode())

def retarget_animation(src_skel: Skeleton, tgt_skel: Skeleton, mapping: dict[str, str], force_scale: float = 0.0, yaw_offset: float = 0.0, neutral_fingers: bool = True, in_place: bool = False):
    print("Retargeting Animation...")
    ret_rots = {}
    ret_locs = {}
    
    yaw_q_raw = R.from_euler('y', yaw_offset, degrees=True).as_quat()
    yaw_q = np.array([yaw_q_raw[3], yaw_q_raw[0], yaw_q_raw[1], yaw_q_raw[2]])
    
    # 1. Base Mapping
    active = []
    mapped_targets = set()
    mapped_sources = set()

    for s_key, t_key in mapping.items():
        s_bone = src_skel.get_bone_case_insensitive(s_key)
        t_bone = tgt_skel.get_bone_case_insensitive(t_key)
        
        if s_bone and t_bone:
            # Skip if either bone is already part of a mapping
            if t_bone.name.lower() in mapped_targets or s_bone.name.lower() in mapped_sources:
                continue

            s_rest = s_bone.rest_rotation
            
            # NEUTRAL FINGERS: If the source is in a curled pose, it 'cancels out' the motion.
            if neutral_fingers:
                is_finger = any(f in s_bone.name.lower() for f in ['index', 'middle', 'ring', 'pinky', 'thumb', 'toe'])
                if is_finger:
                    pname = s_bone.parent_name
                    p_bone = src_skel.get_bone_case_insensitive(pname)
                    if p_bone:
                        s_rest = p_bone.rest_rotation
            
            off = quaternion_multiply(quaternion_inverse(s_rest), t_bone.rest_rotation)
            active.append((s_bone, t_bone, off))
            mapped_targets.add(t_bone.name.lower())
            mapped_sources.add(s_bone.name.lower())
            
    # 2. Smart fuzzy matching for unmapped bones using advanced algorithm
    # Sort target bones: Hips/Root first, then limbs, then fingers
    def sort_key(b):
        n = b.name.lower()
        if 'hips' in n or 'root' in n or 'pelvis' in n: return 0
        if 'spine' in n or 'chest' in n: return 1
        if 'neck' in n or 'head' in n: return 2
        if 'leg' in n or 'arm' in n: return 3
        return 10  # Fingers and other bones last
    
    tgt_list = sorted(tgt_skel.bones.values(), key=sort_key)
    
    # Track matches with confidence scores for reporting
    fuzzy_matches = []
    
    for t_bone in tgt_list:
        if t_bone.name.lower() in mapped_targets:
            continue
        
        # Use the advanced bone matching system
        best_source_name, confidence = find_best_bone_match(
            t_bone.name,
            src_skel,
            mapped_sources,
            require_side_match=True
        )
        
        # Accept match if confidence is above threshold
        if best_source_name and confidence >= 0.5:  # 50% confidence minimum
            s_bone = src_skel.bones[best_source_name]
            
            #Ensure this source bone hasn't been mapped yet
            if s_bone.name.lower() in mapped_sources:
                continue
            
            s_rest = s_bone.rest_rotation
            if neutral_fingers:
                is_finger = any(f in s_bone.name.lower() for f in ['index', 'middle', 'ring', 'pinky', 'thumb', 'toe'])
                if is_finger:
                    pname = s_bone.parent_name
                    p_bone = src_skel.get_bone_case_insensitive(pname)
                    if p_bone:
                        s_rest = p_bone.rest_rotation

            off = quaternion_multiply(quaternion_inverse(s_rest), t_bone.rest_rotation)
            active.append((s_bone, t_bone, off))
            mapped_sources.add(s_bone.name.lower())
            mapped_targets.add(t_bone.name.lower())
            fuzzy_matches.append((s_bone.name, t_bone.name, confidence))
    
    # Print mapping results with confidence scores
    print(f"\n[Retarget] Bone Mapping Results:")
    print(f"  Total: {len(active)} bones mapped")
    if fuzzy_matches:
        print(f"  Fuzzy matched: {len(fuzzy_matches)} bones")
        for s_name, t_name, conf in fuzzy_matches:
            print(f"    {s_name} -> {t_name} (confidence: {conf:.2f})")
            
    print(f"DEBUG: Final Mapping Count: {len(active)} bones")
    # Sort for cleaner debug output
    final_mappings = sorted(active, key=lambda x: x[1].name)
    for s, t, _ in final_mappings:
        print(f"  - {s.name} -> {t.name}")
            
    src_h = get_skeleton_height(src_skel, mapping)
    tgt_h = get_skeleton_height(tgt_skel, mapping)
    print(f"[DEBUG] Skeleton Heights - Source: {src_h:.4f}, Target: {tgt_h:.4f}")
    scale = force_scale if force_scale > 1e-4 else (tgt_h / src_h if src_h > 0.01 else 1.0)
    print(f"Scale: {scale:.4f}")
    
    tgt_world_anims = {}
    frames = sorted(list(src_skel.bones.values())[0].world_animation.keys()) if src_skel.bones else []
    if not frames: frames = range(src_skel.frame_start, src_skel.frame_end + 1)
    
    # 1. World Rotations and Root Displacement Reference
    root_mapped = False
    # Detect Target Up Axis (Blender is Z-Up)
    t_up_axis = np.array([0, 1, 0]) # Default Y-Up
    t_head = tgt_skel.get_bone_case_insensitive('head')
    if t_head:
        if abs(t_head.head[2]) > abs(t_head.head[1]):
            t_up_axis = np.array([0, 0, 1])
            print("[DEBUG] Detected Target Up Axis: Z-Up")

    # Coordinate Transform: Map Source Up ([0,1,0]) to Target Up
    if t_up_axis[2] == 1: # Z-Up (Blender)
        # Source Y -> Target Z, Source Z -> Target -Y, Source X -> Target X
        coord_q = R.from_euler('x', 90, degrees=True)
    else: # Y-Up
        coord_q = R.identity()

    # Realignment Logic (Calculated in Target Space)
    realignment_tgt_q = R.identity()
    t_up = t_up_axis # Default Up Axis
    
    s_lhip = src_skel.get_bone_case_insensitive('l_hip')
    s_rhip = src_skel.get_bone_case_insensitive('r_hip')
    t_lhip = tgt_skel.get_bone_case_insensitive('l_hip')
    t_rhip = tgt_skel.get_bone_case_insensitive('r_hip')
    
    # We need a root bone for displacement reference
    root_bone_name = None
    s_root_bone = None
    for s_bone, t_bone, _ in active:
        if any(x in t_bone.name.lower() or x in s_bone.name.lower() for x in ['hips', 'pelvis', 'root', 'center', 'cog']):
            root_bone_name = t_bone.name
            s_root_bone = s_bone
            break

    if s_lhip and s_rhip and t_lhip and t_rhip and s_root_bone:
        
        # Construct Source Frame (Rest Pose)
        s_spine = src_skel.get_bone_case_insensitive('spine') or src_skel.get_bone_case_insensitive('spine1') or src_skel.get_bone_case_insensitive('Spine1')
        s_up = (s_spine.world_matrix[3, :3] - s_root_bone.world_matrix[3, :3]) if s_spine else np.array([0, 1, 0])

        s_side = s_rhip.world_matrix[3, :3] - s_lhip.world_matrix[3, :3]
        s_fwd = np.cross(s_side, s_up)
        s_up = np.cross(s_fwd, s_side) # Orthogonalize
        
        s_fwd /= (np.linalg.norm(s_fwd) + 1e-9)
        s_up /= (np.linalg.norm(s_up) + 1e-9)
        s_side /= (np.linalg.norm(s_side) + 1e-9)
        s_mat = np.stack((s_side, s_up, s_fwd), axis=-1)
        
        # Construct Target Frame (Rest Pose)
        # Use a more stable Up vector (Pelvis to Spine/Head)
        t_spine = tgt_skel.get_bone_case_insensitive('spine') or tgt_skel.get_bone_case_insensitive('spine1')
        t_root_bone = tgt_skel.get_bone_case_insensitive(root_bone_name)
        t_up = (t_spine.world_matrix[3, :3] - t_root_bone.world_matrix[3, :3]) if t_spine else t_up_axis
        t_side = t_rhip.world_matrix[3, :3] - t_lhip.world_matrix[3, :3]
        t_fwd = np.cross(t_side, t_up)
        t_up = np.cross(t_fwd, t_side) # Orthogonalize
        
        t_fwd /= (np.linalg.norm(t_fwd) + 1e-9)
        t_up /= (np.linalg.norm(t_up) + 1e-9)
        t_side /= (np.linalg.norm(t_side) + 1e-9)
        t_mat = np.stack((t_side, t_up, t_fwd), axis=-1)
        
        # Map Source Frame to Target Space via coord_q
        s_mat_tgt = coord_q.as_matrix() @ s_mat
        
        # Find rotation that aligns Source Frame (in Target Space) to Target Frame
        # R_s_tgt * R_align = R_t  => R_align = R_s_tgt_inv * R_t
        s_rot_tgt = R.from_matrix(s_mat_tgt)
        t_rot = R.from_matrix(t_mat)
        realignment_tgt_q = s_rot_tgt.inv() * t_rot
        
        print(f"[DEBUG] Pose-Invariant Realignment Applied")

    # Find Target Spine and Hips for dynamic leveling
    t_spine_bone = tgt_skel.get_bone_case_insensitive('spine') or tgt_skel.get_bone_case_insensitive('spine1') or tgt_skel.get_bone_case_insensitive('chest')
    t_spine_name = t_spine_bone.name if t_spine_bone else None
    t_hips_bone = tgt_skel.get_bone_case_insensitive('hips') or tgt_skel.get_bone_case_insensitive('pelvis')
    
    t_spine_local_up = np.array([0, 1, 0])
    if t_spine_bone and t_hips_bone:
        # Calculate Local Up vector for spine in rest pose
        # This is the vector from Hips to Spine, transformed into Spine's local space
        v_up_world = t_spine_bone.head - t_hips_bone.head
        r_spine_rest = R.from_quat([t_spine_bone.rest_rotation[1], t_spine_bone.rest_rotation[2], t_spine_bone.rest_rotation[3], t_spine_bone.rest_rotation[0]])
        t_spine_local_up = r_spine_rest.inv().apply(v_up_world)
        t_spine_local_up /= (np.linalg.norm(t_spine_local_up) + 1e-9)
        print(f"[DEBUG] Spine Local Up: {t_spine_local_up}")

    # Unified Global Transformation
    # Order: Coord Map -> Realignment -> User Yaw
    yaw_q = R.from_euler('y' if t_up_axis[1] == 1 else 'z', yaw_offset, degrees=True)
    global_transform_q = yaw_q * realignment_tgt_q * coord_q
    
    gt_q = global_transform_q.as_quat() # [x, y, z, w]
    gt_q_np = np.array([gt_q[3], gt_q[0], gt_q[1], gt_q[2]]) # [w, x, y, z]

    # 3. Pass 1: Standard Retargeting for all bones
    for s_bone, t_bone, _ in active:
        tgt_world_anims[t_bone.name] = {}
        s_rest_q = s_bone.rest_rotation
        s_rest_tgt_q = quaternion_multiply(gt_q_np, s_rest_q)
        t_rest_q = t_bone.rest_rotation
        off_q = quaternion_multiply(quaternion_inverse(s_rest_tgt_q), t_rest_q)
        
        for f in frames:
            s_rot_f = s_bone.world_animation.get(f, s_rest_q)
            t_rot_f = quaternion_multiply(gt_q_np, s_rot_f)
            t_rot_f = quaternion_multiply(t_rot_f, off_q)
            tgt_world_anims[t_bone.name][f] = t_rot_f


    # NOTE: Head leveling removed - the head should follow the body naturally

    #Inherit parent rotation for uanimated head/neck bones
    # If the source head has identity rotation while the body is rotated,
    # the head animation was not properly baked. We fix this by inheriting
    # the parent's rotation.
    for s_bone, t_bone, _ in active:
        is_head_neck = any(x in t_bone.name.lower() for x in ['head', 'neck'])
        if is_head_neck:
            # Check if the source head is uanimated (stays at rest for all frames)
            s_rest_q = s_bone.rest_rotation
            is_uanimated = True
            for f in frames[:min(10, len(frames))]:  # Check first 10 frames
                s_rot_f = s_bone.world_animation.get(f, s_rest_q)
                if np.linalg.norm(s_rot_f - s_rest_q) > 0.01:
                    is_uanimated = False
                    break
            
            if is_uanimated:
                print(f"[FIX] {s_bone.name} detected as uanimated - inheriting parent rotation")
                # Find parent bone
                pname = s_bone.parent_name
                if pname:
                    p_bone = src_skel.get_bone_case_insensitive(pname)
                    if p_bone:
                        for f in frames:
                            # Get parent's animated world rotation
                            p_rot_f = p_bone.world_animation.get(f, p_bone.rest_rotation)
                            # Transform to target space
                            p_rot_tgt = quaternion_multiply(gt_q_np, p_rot_f)
                            # Apply the same offset as we would for the head
                            off_q = quaternion_multiply(quaternion_inverse(quaternion_multiply(gt_q_np, s_rest_q)), t_bone.rest_rotation)
                            t_rot_f = quaternion_multiply(p_rot_tgt, off_q)
                            tgt_world_anims[t_bone.name][f] = t_rot_f


    # 5. Displacement and Local Rotations
    for s_bone, t_bone, _ in active:
        if t_bone.name == root_bone_name:
            ret_locs[t_bone.name] = {}
            s_rest_pos = s_bone.world_matrix[3, :3]
            
            # Calculate Horizontal Offset at Frame 0 to center at origin
            s_pos_0 = s_bone.world_location_animation.get(frames[0], s_rest_pos)
            t_pos_0 = global_transform_q.apply(s_pos_0 * scale)
            
            # horizontal_offset = -t_pos_0 (masking out the vertical axis)
            h_offset = -t_pos_0.copy()
            if t_up_axis[1] == 1: h_offset[1] = 0 # Y-Up: Keep Y (vertical)
            else: h_offset[2] = 0 # Z-Up: Keep Z (vertical)
            
            for f in frames:
                s_pos_f = s_bone.world_location_animation.get(f, s_rest_pos)
                t_pos_f = global_transform_q.apply(s_pos_f * scale) + h_offset
                
                # IN-PLACE: Remove horizontal movement relative to start position
                if in_place:
                    if t_up_axis[1] == 1: # Y-Up
                        t_pos_f[0] = t_pos_0[0] + h_offset[0] # Lock X
                        t_pos_f[2] = t_pos_0[2] + h_offset[2] # Lock Z
                    else: # Z-Up
                        t_pos_f[0] = t_pos_0[0] + h_offset[0] # Lock X
                        t_pos_f[1] = t_pos_0[1] + h_offset[1] # Lock Y
                
                # Transform to Parent Local Space
                pname = t_bone.parent_name
                p_world_mat = tgt_skel.node_world_matrices.get(pname, np.eye(4))
                prot_q = tgt_world_anims.get(pname, {}).get(f)
                if prot_q is not None:
                    p_mat = np.eye(4)
                    p_mat[:3, :3] = R.from_quat([prot_q[1], prot_q[2], prot_q[3], prot_q[0]]).as_matrix()
                    p_mat[3, :3] = p_world_mat[3, :3]
                    p_world_mat = p_mat
                
                p_mat_inv = np.linalg.inv(p_world_mat)
                v_homog = np.append(t_pos_f, 1.0)
                local_pos_f = (v_homog @ p_mat_inv)[:3]
                ret_locs[t_bone.name][f] = local_pos_f

    # 3. Local Rotations
    for s_bone, t_bone, _ in active:
        ret_rots[t_bone.name] = {}
        pname = t_bone.parent_name
        
        is_head = 'head' in t_bone.name.lower() and 'neck' not in t_bone.name.lower()
        
        for f in frames:
            prot = tgt_world_anims.get(pname, {}).get(f)
            if prot is None:
                # Fallback: Check if target skeleton has this node's rest orientation
                prot = tgt_skel.node_rest_rotations.get(pname, np.array([1, 0, 0, 0]))
                # Apply global rotation to parent if it's not animated
                prot = quaternion_multiply(gt_q_np, prot)
                
            # Target world rotation to local space: Local = ParentWorld.inv @ World
            l_rot = quaternion_multiply(quaternion_inverse(prot), tgt_world_anims[t_bone.name][f])
            ret_rots[t_bone.name][f] = l_rot
            
    return ret_rots, ret_locs, active

def copy_textures_for_scene(scene, output_fbx_path):
    """Copy all texture files referenced by the scene to the output FBX location"""
    import shutil
    
    output_dir = os.path.dirname(os.path.abspath(output_fbx_path))
    fbx_filename = os.path.basename(output_fbx_path)
    fbx_base = os.path.splitext(fbx_filename)[0]
    
    # Create texture folder named after the FBX file
    texture_dir = os.path.join(output_dir, f"{fbx_base}_textures")
    
    copied_count = 0
    
    # Iterate through all materials in the scene
    for i in range(scene.GetMaterialCount()):
        material = scene.GetMaterial(i)
        
        # Check common material properties for textures
        texture_props = [
            FbxSurfaceMaterial.sDiffuse,
            FbxSurfaceMaterial.sNormalMap,
            FbxSurfaceMaterial.sSpecular,
            FbxSurfaceMaterial.sEmissive,
            FbxSurfaceMaterial.sBump,
            "DiffuseColor",
            "NormalMap",
            "SpecularColor",
        ]
        
        for prop_name in texture_props:
            prop = material.FindProperty(prop_name)
            if prop.IsValid():
                # Get texture count for this property  
                tex_count = prop.GetSrcObjectCount()
                for j in range(tex_count):
                    texture = prop.GetSrcObject(j)
                    # Check if it's a file texture (has GetFileName method)
                    if texture and hasattr(texture, 'GetFileName'):
                        original_path = texture.GetFileName()
                        if original_path and os.path.exists(original_path):
                            # Create texture directory if needed
                            os.makedirs(texture_dir, exist_ok=True)
                            
                            # Copy texture file
                            filename = os.path.basename(original_path)
                            dest_path = os.path.join(texture_dir, filename)
                            
                            if not os.path.exists(dest_path):
                                shutil.copy2(original_path, dest_path)
                                copied_count += 1
                                print(f"  Copied: {filename}")
                            
                            # Update texture path to be in the same directory as FBX
                            # This ensures the web browser can access it
                            relative_path = os.path.join(f"{fbx_base}_textures", filename)
                            texture.SetFileName(relative_path)
                            texture.SetRelativeFileName(relative_path)
    
    if copied_count > 0:
        print(f"[Retarget] Copied {copied_count} texture file(s) to {os.path.basename(texture_dir)}")
    
    return copied_count

def save_fbx(manager, scene, path):
    """Save FBX with materials and textures preserved (matching HY-Motion implementation)"""
    
    exporter = FbxExporter.Create(manager, "")
    
    # Get or create IO settings
    ios = manager.GetIOSettings()
    if not ios:
        ios = FbxIOSettings.Create(manager, "IOSRoot")
        manager.SetIOSettings(ios)
    
    # CRITICAL: Use the CORRECT FBX SDK constants (from HY-Motion's working code)
    # These will embed the materials and textures directly into the FBX file
    try:
        ios.SetBoolProp(EXP_FBX_EMBEDDED, True)  # Embed media
        ios.SetBoolProp(EXP_FBX_MATERIAL, True)   # Include materials
        ios.SetBoolProp(EXP_FBX_TEXTURE, True)    # Include textures
        print("[Retarget] Configured FBX export with embedded materials")
    except Exception as e:
        print(f"[Retarget] Warning: Could not set embedded export properties: {e}")
        print("[Retarget]  Trying fallback properties...")
        # Fallback to string-based properties (some FBX SDK versions use these)
        try:
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Material", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Texture", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Model", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Animation", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Shape", True)
            ios.SetBoolProp("Export|AdvOptGrp|Fbx|Skin", True)
            print("[Retarget]  Fallback properties set")
        except Exception as fallback_error:
            print(f"[Retarget]  Fallback also failed: {fallback_error}")
    
    # Initialize exporter
    file_format = manager.GetIOPluginRegistry().GetNativeWriterFormat()
    if not exporter.Initialize(path, file_format, ios):
        raise RuntimeError(f"Failed to initialize FBX exporter: {exporter.GetStatus().GetErrorString()}")
    
    # Export the scene
    if not exporter.Export(scene):
        raise RuntimeError(f"Failed to export FBX: {exporter.GetStatus().GetErrorString()}")
    
    exporter.Destroy()
    print(f"[Retarget] Saved FBX to: {path}")

def main():
    print("[Retarget] Starting retargeting script...")
    if not HAS_FBX_SDK:
        print("[Retarget] ERROR: FBX SDK not found. Please install 'fbx' python package.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True)
    parser.add_argument('--target', '-t', required=True)
    parser.add_argument('--mapping', '-m', default='', help='Optional bone mapping file (uses hardcoded mappings if not provided)')
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--yaw', '-y', type=float, default=0.0)
    parser.add_argument('--scale', '-sc', type=float, default=0.0)
    parser.add_argument('--no-neutral', dest='neutral', action='store_false', help="Disable neutral finger rest-pose")
    parser.add_argument('--in-place', action='store_true', help="Convert animation to in-place (removes horizontal movement)")
    parser.set_defaults(neutral=True, in_place=False)
    args = parser.parse_args()
    
    if args.source.lower().endswith('.npz'):
        print(f"[Retarget] Loading NPZ Source: {args.source}")
        src_man, src_scene = None, None
        src_skel = load_npz(args.source)
    else:
        print(f"[Retarget] Loading FBX Source: {args.source}")
        src_man, src_scene, src_skel = load_fbx(args.source, sample_rest_frame=None)
        if src_skel is None:
            print(f"[Retarget] ERROR: Failed to load source FBX: {args.source}")
            sys.exit(1)
            
        src_h = get_skeleton_height(src_skel, [])
        # FALLBACK: If Bind Pose is collapsed (height ~0), try frame 0
        if src_h < 0.1:
            print("[Retarget] DEBUG: Bind Pose collapsed, falling back to frame 0 for rest pose.")
            src_man, src_scene, _ = load_fbx(args.source, sample_rest_frame=0)
            # Refresh skeleton with sampled data
            src_skel = Skeleton(os.path.basename(args.source))
            collect_skeleton_nodes(src_scene.GetRootNode(), src_skel, sampling_time=FbxTime())
        
        extract_animation(src_scene, src_skel)
    
    print(f"[Retarget] Loading Target FBX: {args.target}")
    tgt_man, tgt_scene, tgt_skel = load_fbx(args.target)
    if tgt_skel is None:
        print(f"[Retarget] ERROR: Failed to load target FBX: {args.target}")
        sys.exit(1)
    
    print("[Retarget] Loading bone mapping...")
    mapping = load_bone_mapping(args.mapping, src_skel, tgt_skel)
    
    print("[Retarget] Calculating retargeting...")
    rots, locs, active = retarget_animation(src_skel, tgt_skel, mapping, args.scale, args.yaw, args.neutral, args.in_place)
    
    print(f"\n[Retarget] Bone Mapping Results:\n  Total: {len(active)} bones mapped")
    if len(active) < 15:
        print(f"[Retarget] [WARNING] Very few bones mapped ({len(active)}). Retargeting might be poor.")
        
    src_time_mode = src_scene.GetGlobalSettings().GetTimeMode() if src_scene else tgt_scene.GetGlobalSettings().GetTimeMode()
    print("[Retarget] Applying retargeted animation to scene...")
    apply_retargeted_animation(tgt_scene, tgt_skel, rots, locs, src_skel.frame_start, src_skel.frame_end, src_time_mode)
    
    print(f"[Retarget] Saving result to: {args.output}")
    save_fbx(tgt_man, tgt_scene, args.output)
    
    # Explicit cleanup to prevent SIGSEGV on exit
    # We use os._exit(0) to bypass the Python interpreter's cleanup sequence
    # which often conflicts with the FBX SDK's own cleanup.
    print("[Retarget] Done!")
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
