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
        self.bone_children: dict[str, list[str]] = {} # bone_name -> list of child bone names
        self.unit_scale = 1.0 # Multiplier to convert internal units to Centimeters (1.0 for CM, 100.0 for M)

    def add_bone(self, bone: BoneData):
        self.bones[bone.name.lower()] = bone
        if bone.bone_parent_name:
            pname = bone.bone_parent_name.lower()
            if pname not in self.bone_children:
                self.bone_children[pname] = []
            if bone.name not in self.bone_children[pname]:
                self.bone_children[pname].append(bone.name)

    def get_children(self, bone_name: str) -> list[BoneData]:
        names = self.bone_children.get(bone_name.lower(), [])
        return [self.bones[n.lower()] for n in names if n.lower() in self.bones]
    
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

    def get_geometric_features(self) -> dict[str, dict]:
        """
        Calculate geometric features for all bones in the skeleton.
        Returns a dict mapping bone name to its features:
        { 'pos': normalized_pos, 'depth': hierarchy_depth, 'side': -1|0|1 }
        """
        features = {}
        
        # 1. Find Pelvis/Root for normalization
        pelvis = self.get_bone_case_insensitive("Pelvis")
        if not pelvis:
            # Fallback to first bone if no pelvis
            pelvis = next(iter(self.bones.values())) if self.bones else None
        
        if not pelvis: return {}

        # 2. Get height for scaling
        height = get_skeleton_height(self)
        if height < 0.001: height = 1.0
        
        # 3. Calculate features for each bone
        for name, bone in self.bones.items():
            # Relative position to pelvis, normalized by height
            rel_pos = (bone.head - pelvis.head) / height
            
            # Depth in hierarchy
            depth = 0
            curr = bone
            visited = set()
            while curr and curr.bone_parent_name and curr.name not in visited:
                visited.add(curr.name)
                curr = self.get_bone_case_insensitive(curr.bone_parent_name)
                depth += 1
            
            # Side detection: -1 (Left), 0 (Center), 1 (Right)
            # Based on X-coordinate in local space (assuming Y-up or Z-up standard)
            side = 0
            if rel_pos[0] > 0.05: side = 1
            elif rel_pos[0] < -0.05: side = -1
            
            # Refine side based on name if possible
            lower_name = bone.name.lower()
            if "left" in lower_name or "l_" in lower_name or ".l" in lower_name: side = -1
            elif "right" in lower_name or "r_" in lower_name or ".r" in lower_name: side = 1

            features[bone.name] = {
                'pos': rel_pos,
                'depth': depth,
                'side': side
            }
            
        return features

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

def load_npz(data_or_path: str | dict) -> Skeleton:
    """Load motion data from NPZ file or dictionary (HyMotion/SMPL-H format)."""
    if isinstance(data_or_path, str):
        data = np.load(data_or_path)
        name = os.path.basename(data_or_path)
    else:
        data = data_or_path
        name = "In-Memory Data"
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
    
    skel = Skeleton(name)
    skel.frame_start = 0
    skel.frame_end = T - 1
    skel.fps = 30.0
    skel.unit_scale = 100.0 # HyMotion/SMPL-H is ALWAYS in Meters
    
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

    # 3. Create a Canonical T-Pose for rest pose reference
    # This ensures stable retargeting offsets (off_q) and accurate geometric matching.
    t_pose_kps = np.zeros((52, 3))
    t_pose_kps[0] = [0, 0, 0] # Pelvis at origin
    
    # Define T-pose directions for SMPL-H
    # Y is Up, X is Side, Z is Forward
    for i in range(1, 52):
        p = parents[i]
        dist = np.linalg.norm(kps[0, i] - kps[0, p])
        name = names[i].lower()
        
        # Default direction: Up
        dir_vec = np.array([0.0, 1.0, 0.0])
        
        if any(x in name for x in ['hip', 'knee', 'ankle', 'foot']):
            dir_vec = np.array([0.0, -1.0, 0.0]) # Legs down
            if 'hip' in name:
                # Add a small X offset to separate hips for side detection
                if name.startswith('l_'): dir_vec[0] = 0.5
                else: dir_vec[0] = -0.5
                dir_vec /= np.linalg.norm(dir_vec)
        elif 'collar' in name or 'shoulder' in name or 'elbow' in name or 'wrist' in name or 'index' in name or 'middle' in name or 'pinky' in name or 'ring' in name or 'thumb' in name:
            if name.startswith('l_'):
                dir_vec = np.array([1.0, 0.0, 0.0]) # Left arm out (+X)
            else:
                dir_vec = np.array([-1.0, 0.0, 0.0]) # Right arm out (-X)
        
        t_pose_kps[i] = t_pose_kps[p] + dir_vec * dist

    for i, name in enumerate(names):
        bone = BoneData(name)
        p_idx = parents[i]
        if p_idx != -1: 
            bone.parent_name = names[p_idx]
            bone.bone_parent_name = names[p_idx]
        
        # Rest Pose = Canonical T-Pose
        bone.rest_rotation = np.array([1, 0, 0, 0]) # Identity
        bone.head = t_pose_kps[i]
        bone.world_matrix = np.eye(4)
        bone.world_matrix[:3, :3] = np.eye(3) # Identity world rotation
        bone.world_matrix[3, :3] = t_pose_kps[i]
        
        # Animation
        for f in range(T):
            bone.world_animation[f] = matrix_to_quaternion(world_rots[f, i].T)
            bone.world_location_animation[f] = global_kps[f, i]
            
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

# ArticulationXL (UniRig) Bone Mapping (Strictly from unirig.json)
ARTICULATION_MAPPING = {
    "Pelvis": ["bone_0"],
    "L_Hip": ["bone_44"], "L_Knee": ["bone_45"], "L_Ankle": ["bone_46"], "L_Foot": ["bone_47"],
    "R_Hip": ["bone_48"], "R_Knee": ["bone_49"], "R_Ankle": ["bone_50"], "R_Foot": ["bone_51"],
    "Spine1": ["bone_1"], "Spine2": ["bone_2"], "Spine3": ["bone_3"],
    "Neck": ["bone_4"], "Head": ["bone_5"],
    "L_Collar": ["bone_6"], "L_Shoulder": ["bone_7"], "L_Elbow": ["bone_8"], "L_Wrist": ["bone_9"],
    "R_Collar": ["bone_25"], "R_Shoulder": ["bone_26"], "R_Elbow": ["bone_27"], "R_Wrist": ["bone_28"],
    "L_Index1": ["bone_13"], "L_Index2": ["bone_14"], "L_Index3": ["bone_15"],
    "L_Middle1": ["bone_16"], "L_Middle2": ["bone_17"], "L_Middle3": ["bone_18"],
    "L_Ring1": ["bone_19"], "L_Ring2": ["bone_20"], "L_Ring3": ["bone_21"],
    "L_Pinky1": ["bone_22"], "L_Pinky2": ["bone_23"], "L_Pinky3": ["bone_24"],
    "L_Thumb1": ["bone_10"], "L_Thumb2": ["bone_11"], "L_Thumb3": ["bone_12"],
    "R_Index1": ["bone_32"], "R_Index2": ["bone_33"], "R_Index3": ["bone_34"],
    "R_Middle1": ["bone_35"], "R_Middle2": ["bone_36"], "R_Middle3": ["bone_37"],
    "R_Ring1": ["bone_38"], "R_Ring2": ["bone_39"], "R_Ring3": ["bone_40"],
    "R_Pinky1": ["bone_41"], "R_Pinky2": ["bone_42"], "R_Pinky3": ["bone_43"],
    "R_Thumb1": ["bone_29"], "R_Thumb2": ["bone_30"], "R_Thumb3": ["bone_31"],
}

    # UE5 Bone Mapping (Strictly from user request)
UE5_MAPPING = {
    "Pelvis": ["pelvis"], 
    "Spine1": ["spine_01", "spine_02"], 
    "Spine2": ["spine_03"], 
    "Spine3": ["spine_04", "spine_05"],
    "Neck": ["neck_01", "neck_02", "neck_03"], 
    "Head": ["head"],
    "L_Collar": ["clavicle_l"], "R_Collar": ["clavicle_r"],
    "L_Shoulder": ["upperarm_l"], "R_Shoulder": ["upperarm_r"],
    "L_Elbow": ["lowerarm_l"], "R_Elbow": ["lowerarm_r"],
    "L_Wrist": ["hand_l"], "R_Wrist": ["hand_r"],
    "L_Hip": ["thigh_l"], "L_Knee": ["calf_l"], "L_Ankle": ["foot_l"],
    "R_Hip": ["thigh_r"], "R_Knee": ["calf_r"], "R_Ankle": ["foot_r"],
    "L_Foot": ["ball_l"], "R_Foot": ["ball_r"],
    "L_Thumb1": ["thumb_01_l"], "L_Thumb2": ["thumb_02_l"], "L_Thumb3": ["thumb_03_l"],
    "L_Index1": ["index_01_l"], "L_Index2": ["index_02_l"], "L_Index3": ["index_03_l"],
    "L_Middle1": ["middle_01_l"], "L_Middle2": ["middle_02_l"], "L_Middle3": ["middle_03_l"],
    "L_Ring1": ["ring_01_l"], "L_Ring2": ["ring_02_l"], "L_Ring3": ["ring_03_l"],
    "L_Pinky1": ["pinky_01_l"], "L_Pinky2": ["pinky_02_l"], "L_Pinky3": ["pinky_03_l"],
    "R_Thumb1": ["thumb_01_r"], "R_Thumb2": ["thumb_02_r"], "R_Thumb3": ["thumb_03_r"],
    "R_Index1": ["index_01_r"], "R_Index2": ["index_02_r"], "R_Index3": ["index_03_r"],
    "R_Middle1": ["middle_01_r"], "R_Middle2": ["middle_02_r"], "R_Middle3": ["middle_03_r"],
    "R_Ring1": ["ring_01_r"], "R_Ring2": ["ring_02_r"], "R_Ring3": ["ring_03_r"],
    "R_Pinky1": ["pinky_01_r"], "R_Pinky2": ["pinky_02_r"], "R_Pinky3": ["pinky_03_r"],
}

# Add fuzzy aliases FOR THE SOURCE only - prevents mapping to incorrect target bones like thigh_twist
UE5_SOURCE_ALIASES = {
    "Spine2": ["Spine2", "Chest"],
    "Spine3": ["Spine3", "Chest"],
    "L_Knee": ["L_Knee", "LowerLeg_L"],
    "R_Knee": ["R_Knee", "LowerLeg_R"],
    "L_Ankle": ["L_Ankle", "L_Foot", "Ankle_L"],
    "R_Ankle": ["R_Ankle", "R_Foot", "Ankle_R"],
    "L_Foot": ["L_Toe", "Ball_L"],
    "R_Foot": ["R_Toe", "Ball_R"],
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

def get_skeleton_height(skeleton: Skeleton) -> float:
    """
    Calculate skeleton height using bone lengths (Pelvis -> Spine -> Neck -> Head).
    This is pose-invariant and works even if the character is lying down.
    """
    # 1. Try Bone Chain Method (Robust)
    pelvis_names = ['pelvis', 'hips', 'bone_0', 'mixamorig:hips']
    head_names = ['head', 'mixamorig:head']

    def find_bone(names):
        for n in names:
            b = skeleton.get_bone_case_insensitive(n)
            if b: return b
        return None

    pelvis = find_bone(pelvis_names)
    head = find_bone(head_names)

    if pelvis and head:
        total_len = 0.0
        curr = head
        visited = set()
        while curr and curr.name.lower() not in [p.lower() for p in pelvis_names] and curr.name not in visited:
            visited.add(curr.name)
            pname = curr.parent_name
            if not pname: break
            parent = skeleton.get_bone_case_insensitive(pname)
            if not parent: break
            
            dist = np.linalg.norm(curr.head - parent.head)
            total_len += dist
            curr = parent
        
        # Add leg length (approximate from Pelvis to Foot)
        l_hip = find_bone(['l_hip', 'leftupleg', 'mixamorig:leftupleg', 'L_Hip'])
        # Search for foot first, then ankle
        l_foot = find_bone(['lefttoe', 'l_foot', 'leftfoot', 'L_Foot', 'l_ankle', 'L_Ankle'])
        
        if l_hip and l_foot:
            leg_len = 0.0
            curr = l_foot
            # Traverse from foot up to PELVIS, not just hip, to catch the hip-to-pelvis distance
            # This ensures we get the full vertical height even if the structure is non-standard
            while curr and curr.name.lower() not in [p.lower() for p in pelvis_names] and curr.name not in visited:
                visited.add(curr.name)
                pname = curr.parent_name
                if not pname: break
                parent = skeleton.get_bone_case_insensitive(pname)
                if not parent: break
                leg_len += np.linalg.norm(curr.head - parent.head)
                curr = parent
            
            total_len += leg_len
        else:
            # approximated fallback
            total_len *= 2.0
            
        if total_len > 0.1:
            return total_len

    # 2. Fallback to Y-Range
    y_min, y_max = 999999.0, -999999.0
    found_any = False
    for _, bone in skeleton.bones.items():
        h_val = bone.head[1]
        if abs(h_val) < 1e-6: continue
        y_min = min(y_min, h_val)
        y_max = max(y_max, h_val)
        found_any = True
            
    if not found_any or y_max <= y_min: return 1.0
    return y_max - y_min

def load_bone_mapping(filepath: str, src_skel: Skeleton, tgt_skel: Skeleton) -> dict[str, str]:
    is_ue5 = filepath and filepath.lower() == "ue5"
    if not is_ue5:
        # Auto-detect UE5 based on specific bone names
        if tgt_skel.get_bone_case_insensitive("clavicle_l") and \
           tgt_skel.get_bone_case_insensitive("upperarm_l") and \
           tgt_skel.get_bone_case_insensitive("spine_01"):
            is_ue5 = True
            print("[Retarget] Auto-detected UE5 skeleton structure.")

    mapping = {} if is_ue5 else BASE_BONE_MAPPING.copy()
    if is_ue5:
        # Pass the flag through the mapping for the animation retargeter
        mapping["__preset__"] = "ue5"
    
    # Check if target is UniRig (ArticulationXL) - look for any bone_X pattern
    is_unirig = False
    if not is_ue5:
        is_unirig = tgt_skel.get_bone_case_insensitive("bone_0") is not None
        if not is_unirig:
            # Check for any bone_X naming pattern
            is_unirig = any("bone_" in name.lower() for name in tgt_skel.bones.keys())
    
    json_bones = {}
    if is_ue5:
        print(f"Detected UE5 mapping preset, using hardcoded UE5 mapping...")
        json_bones = UE5_MAPPING
    elif filepath and os.path.exists(filepath):
        print(f"Loading Mapping File: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        json_bones = data.get("bones", {})
    elif is_unirig:
        # Try to find unirig.json in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        unirig_json_path = os.path.join(script_dir, "unirig.json")
        if os.path.exists(unirig_json_path):
            print(f"Detected UniRig target, loading mapping from: {unirig_json_path}")
            with open(unirig_json_path, 'r') as f:
                data = json.load(f)
            json_bones = data.get("bones", {})
        else:
            print(f"Detected UniRig target, using hardcoded articulation mapping...")
            json_bones = ARTICULATION_MAPPING
    else:
        if not filepath:
            print(f"Using hardcoded bone mappings (no JSON file needed)")
        # Don't return early - still try structural BFS!

    match_count = 0
    mapped_targets = set()
    mapped_sources = set()
    
    # Strategy: 
    # 1. Name-based matching for stable semantic names.
    # 2. Geometric matching for everything else (especially UniRig bone_X).
    
    # Pre-calculate heights for normalization
    src_h = get_skeleton_height(src_skel)
    tgt_h = get_skeleton_height(tgt_skel)
    if src_h < 0.01: src_h = 1.0
    if tgt_h < 0.01: tgt_h = 1.0
    
    sorted_json = sorted(json_bones.items())

    # Phase 1: Semantic Name Matching (High Confidence)
    for gen_name, aliases in sorted_json:
        if not isinstance(aliases, list): aliases = [aliases]
        
        # Find Source Bone (Robust Search)
        source_bone = src_skel.get_bone_case_insensitive(gen_name)
        if not source_bone:
            # Try specific source aliases for UE5 if active
            if is_ue5 and gen_name in UE5_SOURCE_ALIASES:
                for sa in UE5_SOURCE_ALIASES[gen_name]:
                    found = src_skel.get_bone_case_insensitive(sa)
                    if found:
                        source_bone = found
                        break
            
            # Standard fuzzy aliases (only if not found via UE5 specific ones)
            if not source_bone:
                lower_gen = gen_name.lower()
                search_group = []
                if lower_gen in FUZZY_ALIASES:
                    search_group = [lower_gen] + FUZZY_ALIASES[lower_gen]
                else:
                    for k, v in FUZZY_ALIASES.items():
                        if lower_gen in v:
                            search_group = [k] + v
                            break
                for sa in search_group:
                    found = src_skel.get_bone_case_insensitive(sa)
                    if found:
                        source_bone = found
                        break
        
        if not source_bone: continue

        # Find Target Bone (Semantic Match)
        # For UE5/Presets, we allow mapping ONE source bone to MULTIPLE target bones (chains)
        found_names = []
        for a in aliases:
            if is_unirig and a.startswith("bone_") and a != "bone_0":
                continue # Skip unstable UniRig indices for name matching
            
            found = tgt_skel.get_bone_case_insensitive(a)
            if found:
                found_names.append(found.name)
                if not is_ue5: break # Only UE5/Chains need multiple targets
        
        if found_names and source_bone.name.lower() not in mapped_sources:
            mapping[source_bone.name.lower()] = found_names
            match_count += len(found_names)
            for fn in found_names:
                mapped_targets.add(fn)
            mapped_sources.add(source_bone.name.lower())

    print(f"[Retarget] load_bone_mapping: Found {match_count} matches via semantic names.")
    
    if is_ue5:
        print(f"[Retarget] UE5 preset active: Skipping structural matching and returning explicit mapping ({len(mapping)} bones).")
        return mapping

    # Phase 2: Structural BFS Matching (Hierarchy-Aware)
    # We start from the root and traverse down
    # Helper to check if a bone is a finger
    def is_finger(name):
        n = name.lower()
        return any(k in n for k in ['thumb', 'index', 'middle', 'ring', 'pinky', 'finger'])

    # Helper to find root/pelvis bone in target skeleton
    def find_target_root(skel):
        """Find the root/pelvis bone using multiple fallback strategies."""
        
        # Strategy 1: Look for bone_0 (UniRig standard)
        root = skel.get_bone_case_insensitive("bone_0")
        if root:
            print(f"[Retarget] Found target root via 'bone_0': {root.name}")
            return root
        
        # Strategy 2: Name-based detection
        root_names = ['pelvis', 'hips', 'root', 'cog', 'center', 'base', 'spine_01', 'hip']
        for rn in root_names:
            root = skel.get_bone_case_insensitive(rn)
            if root:
                print(f"[Retarget] Found target root via name '{rn}': {root.name}")
                return root
        
        # Strategy 3: Topological detection - find bone with most children/descendants
        best_root = None
        best_score = -1
        for name, bone in skel.bones.items():
            children = skel.get_children(name)
            if not children: continue
            
            # Count descendants recursively
            def count_descendants(bone_name, visited=None):
                if visited is None: visited = set()
                if bone_name in visited: return 0
                visited.add(bone_name)
                children = skel.get_children(bone_name)
                count = len(children)
                for c in children:
                    count += count_descendants(c.name, visited)
                return count
            
            descendant_count = count_descendants(name)
            child_count = len(children)
            
            # Score: prefer bones with 3+ children (spine + 2 legs) and many descendants
            score = child_count * 10 + descendant_count
            if child_count >= 3: score += 100  # Bonus for typical root structure
            
            if score > best_score:
                best_score = score
                best_root = bone
        
        if best_root:
            print(f"[Retarget] Found target root via topology (score {best_score}): {best_root.name}")
            return best_root
        
        # Strategy 4: Geometric detection - find bone closest to center with children on both sides
        center_y = 0
        for bone in skel.bones.values():
            center_y = max(center_y, bone.head[1])
        center_y /= 2  # Approximate vertical center
        
        for name, bone in skel.bones.items():
            children = skel.get_children(bone)
            if len(children) < 2: continue
            
            # Check if children spread both left and right
            has_left = any(c.head[0] < bone.head[0] - 0.05 for c in children)
            has_right = any(c.head[0] > bone.head[0] + 0.05 for c in children)
            
            if has_left and has_right:
                print(f"[Retarget] Found target root via geometry (bilateral children): {bone.name}")
                return bone
        
        print(f"[Retarget] WARNING: Could not find target root bone!")
        return None

    src_root = src_skel.get_bone_case_insensitive("Pelvis")
    tgt_root = find_target_root(tgt_skel)
    
    if src_root and tgt_root:
        print(f"[Retarget] Starting Structural BFS Matching from {src_root.name} -> {tgt_root.name}...")
        
        # Queue of (src_bone, tgt_bone) pairs to explore children of
        queue = [(src_root, tgt_root)]
        
        while queue:
            s_parent, t_parent = queue.pop(0)
            
            s_children = src_skel.get_children(s_parent.name)
            t_children = tgt_skel.get_children(t_parent.name)
            
            if not s_children or not t_children: continue
            
            # Filter out fingers if requested
            s_children = [c for c in s_children if not is_finger(c.name)]
            t_children = [c for c in t_children if not is_finger(c.name)]
            
            # Side matching check
            def get_side(v, name):
                n = name.lower()
                if "left" in n or "l_" in n or ".l" in n: return -1
                if "right" in n or "r_" in n or ".r" in n: return 1
                if abs(v[0]) < 0.05 * (tgt_h if v is t_rel else src_h): return 0 # Height-relative threshold
                return 1 if v[0] > 0 else -1

            # Match children of s_parent to children of t_parent
            for tc in t_children:
                if tc.name in mapped_targets: continue
                
                best_sc = None
                best_score = 999999.0
                
                # 1. Try Name-based Hint (Fuzzy)
                # Check if any source child matches tc's name or its semantic meaning
                for sc in s_children:
                    if sc.name.lower() in mapped_sources: continue
                    
                    # Check if sc name matches tc name (fuzzy)
                    # Or if sc name matches the gen_name associated with tc in unirig.json
                    is_match = False
                    if sc.name.lower() == tc.name.lower(): is_match = True
                    else:
                        for gen_name, aliases in json_bones.items():
                            if sc.name.lower() == gen_name.lower() or sc.name.lower() in [a.lower() for a in (aliases if isinstance(aliases, list) else [aliases])]:
                                if tc.name in (aliases if isinstance(aliases, list) else [aliases]):
                                    is_match = True
                                    break
                    
                    if is_match:
                        best_sc = sc
                        best_score = 0.0
                        break
                
                # 2. Geometric Fallback
                if not best_sc:
                    t_rel = tc.head - t_parent.head
                    t_side = get_side(t_rel, tc.name)
                    
                    for sc in s_children:
                        if sc.name.lower() in mapped_sources: continue
                        
                        s_rel = sc.head - s_parent.head
                        s_side = get_side(s_rel, sc.name)
                        t_rel_norm = t_rel / tgt_h
                        s_rel_norm = s_rel / src_h
                        
                        dist = np.linalg.norm(t_rel_norm - s_rel_norm)
                        
                        # Relaxed side matching for limb chains
                        if t_side != s_side and dist > 0.2: continue

                        if dist < best_score:
                            best_score = dist
                            best_sc = sc
                
                # 3. Single-child Fallback: If target has exactly 1 unmatched child and source has 1 unmatched child, force match
                if not best_sc and len(t_children) == 1 and len([c for c in s_children if c.name.lower() not in mapped_sources]) == 1:
                    remaining_s = [c for c in s_children if c.name.lower() not in mapped_sources][0]
                    best_sc = remaining_s
                    best_score = 0.99  # Mark as forced match
                
                # Increased threshold from 0.5 to 1.0 to allow more matches in limb chains
                if best_sc and best_score < 1.0:
                    mapping[best_sc.name.lower()] = tc.name
                    mapped_targets.add(tc.name)
                    mapped_sources.add(best_sc.name.lower())
                    match_count += 1
                    queue.append((best_sc, tc))

        print(f"[Retarget] Structural BFS Matching complete. Total matches: {match_count}")

    # Phase 3: Hand and Finger Mapping (Phase 2 of the plan)
    # Identify mapped wrists and explore their children
    wrists = [('L_Wrist', ['left', 'l_']), ('R_Wrist', ['right', 'r_'])]
    for wrist_gen_name, side_keys in wrists:
        # Find the mapped target wrist
        target_wrist_name = None
        source_wrist_name = None
        
        # Look for the wrist in the current mapping
        # We need to find a match where both bones actually exist in the skeletons
        for s_name, t_name in mapping.items():
            s_lower = s_name.lower()
            if any(k in s_lower for k in ['wrist', 'hand']):
                if any(sk in s_lower for sk in side_keys):
                    # Check if bones exist
                    s_bone = src_skel.get_bone_case_insensitive(s_name)
                    t_bone = tgt_skel.get_bone_case_insensitive(t_name)
                    if s_bone and t_bone:
                        source_wrist_name = s_name
                        target_wrist_name = t_name
                        break
        
        if not target_wrist_name:
            continue
        
        s_wrist = src_skel.get_bone_case_insensitive(source_wrist_name)
        t_wrist = tgt_skel.get_bone_case_insensitive(target_wrist_name)
        
        if not s_wrist or not t_wrist: continue
        
        s_fingers = [c for c in src_skel.get_children(s_wrist.name) if is_finger(c.name)]
        t_fingers = [c for c in tgt_skel.get_children(t_wrist.name)] # All children of target wrist are likely fingers
        
        if not t_fingers:
            continue
        
        # Case A: Single-Finger Rig (e.g. SK_tira)
        if len(t_fingers) == 1:
            # Map source Middle finger to this single target finger
            # Find source middle finger
            src_middle = None
            for sf in s_fingers:
                if 'middle' in sf.name.lower():
                    src_middle = sf
                    break
            if not src_middle and s_fingers: src_middle = s_fingers[len(s_fingers)//2] # Fallback to middle-ish
            
            if src_middle:
                mapping[src_middle.name.lower()] = t_fingers[0].name
                mapped_targets.add(t_fingers[0].name)
                mapped_sources.add(src_middle.name.lower())
                match_count += 1
                
                # Recursively map down the chain
                s_curr, t_curr = src_middle, t_fingers[0]
                while True:
                    s_next = [c for c in src_skel.get_children(s_curr.name) if is_finger(c.name)]
                    t_next = tgt_skel.get_children(t_curr.name)
                    if not s_next or not t_next: break
                    
                    mapping[s_next[0].name.lower()] = t_next[0].name
                    mapped_targets.add(t_next[0].name)
                    mapped_sources.add(s_next[0].name.lower())
                    match_count += 1
                    s_curr, t_curr = s_next[0], t_next[0]

        # Case B: Multi-Finger Rig
        else:
            # Use geometric matching relative to wrist
            for tf in t_fingers:
                if tf.name in mapped_targets: continue
                
                best_sf = None
                best_score = 999999.0
                t_rel = tf.head - t_wrist.head
                
                for sf in s_fingers:
                    if sf.name.lower() in mapped_sources: continue
                    t_rel_norm = t_rel / tgt_h
                    s_rel_norm = s_rel / src_h
                    dist = np.linalg.norm(t_rel_norm - s_rel_norm)
                    if dist < best_score:
                        best_score = dist
                        best_sf = sf
                
                if best_sf and best_score < 0.3:
                    mapping[best_sf.name.lower()] = tf.name
                    mapped_targets.add(tf.name)
                    mapped_sources.add(best_sf.name.lower())
                    match_count += 1
                    
                    # Recursively map down the chain
                    s_curr, t_curr = best_sf, tf
                    while True:
                        s_next = [c for c in src_skel.get_children(s_curr.name) if is_finger(c.name)]
                        t_next = tgt_skel.get_children(t_curr.name)
                        if not s_next or not t_next: break
                        
                        mapping[s_next[0].name.lower()] = t_next[0].name
                        mapped_targets.add(t_next[0].name)
                        mapped_sources.add(s_next[0].name.lower())
                        match_count += 1
                        s_curr, t_curr = s_next[0], t_next[0]

    # Phase 4: Arm Fallback Detection
    # If arms were not mapped via structural BFS, try to find them using alternative strategies
    arm_bones = ['collar', 'shoulder', 'elbow', 'wrist']
    for side, side_keys in [('L', ['left', 'l_']), ('R', ['right', 'r_'])]:
        # Check if this side's arm is already mapped
        side_arm_mapped = any(
            any(sk in s.lower() for sk in side_keys) and any(ab in s.lower() for ab in arm_bones)
            for s in mapped_sources
        )
        
        if side_arm_mapped:
            continue  # Already mapped via BFS
        
        print(f"[Retarget] Arm Fallback: Searching for {side} arm...")
        
        # Find source arm chain
        src_collar = src_skel.get_bone_case_insensitive(f"{side}_Collar")
        if not src_collar:
            continue
        
        # Strategy 1: Find target bones by name hints
        for ab in arm_bones:
            src_bone = src_skel.get_bone_case_insensitive(f"{side}_{ab.capitalize()}")
            if not src_bone or src_bone.name.lower() in mapped_sources:
                continue
            
            # Find unmapped target bone with similar name or position
            best_tgt = None
            best_score = 999999.0
            
            for t_name, t_bone in tgt_skel.bones.items():
                if t_name in mapped_targets:
                    continue
                
                # Skip non-skeleton nodes (root containers, meshes, etc.)
                t_lower = t_name.lower()
                excluded_names = ['armature', 'character', 'root', 'scene', 'skeleton', 'rig']
                if any(ex == t_lower or t_lower.startswith(ex + '_') for ex in excluded_names):
                    continue
                t_lower = t_name.lower()
                if ab in t_lower:
                    # Check side
                    if any(sk in t_lower for sk in side_keys):
                        best_tgt = t_bone
                        best_score = 0.0
                        break
                
                # Geometric fallback: find bone with similar relative position
                # Use position relative to a common reference (pelvis)
                src_pelvis = src_skel.get_bone_case_insensitive("Pelvis")
                tgt_pelvis_bone = None
                for mapped_s, mapped_t in mapping.items():
                    if 'pelvis' in mapped_s.lower():
                        if isinstance(mapped_t, list):
                            for t_name in mapped_t:
                                tgt_pelvis_bone = tgt_skel.get_bone_case_insensitive(t_name)
                                if tgt_pelvis_bone:
                                    break
                        else:
                            tgt_pelvis_bone = tgt_skel.get_bone_case_insensitive(mapped_t)
                        if tgt_pelvis_bone:
                            break
                
                if src_pelvis and tgt_pelvis_bone:
                    src_rel = (src_bone.head - src_pelvis.head) / src_h
                    tgt_rel = (t_bone.head - tgt_pelvis_bone.head) / tgt_h
                    
                    # Check side consistency
                    src_side = 1 if src_rel[0] > 0.05 else (-1 if src_rel[0] < -0.05 else 0)
                    tgt_side = 1 if tgt_rel[0] > 0.05 else (-1 if tgt_rel[0] < -0.05 else 0)
                    
                    if src_side != tgt_side:
                        continue
                    
                    dist = np.linalg.norm(src_rel - tgt_rel)
                    if dist < best_score:
                        best_score = dist
                        best_tgt = t_bone
            
            if best_tgt and best_score < 1.0:
                if src_bone.name.lower() not in mapped_sources and best_tgt.name not in mapped_targets:
                    mapping[src_bone.name.lower()] = best_tgt.name
                    mapped_targets.add(best_tgt.name)
                    mapped_sources.add(src_bone.name.lower())
                    match_count += 1

    print(f"[Retarget] Final load_bone_mapping: Found {match_count} total matches.")
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
    keywords = ['hips', 'hip', 'spine', 'neck', 'head', 'arm', 'leg', 'foot', 'ankle', 'knee', 'shoulder', 'elbow', 'pelvis', 'joint', 'mixamo', 'thigh', 'upper', 'forearm', 'hand', 'finger', 'clavicle', 'collar', 'toe', 'thumb', 'index', 'middle', 'ring', 'pinky', 'upleg', 'downleg', 'wrist', 'chest', 'belly', 'bone_', 'character', 'calf', 'ball']
    if any(k in name_lower for k in keywords): 
        is_bone = True
    
    # Explicitly ignore IK bones for UE5 and other skeletons
    if 'ik_' in name_lower or '_ik' in name_lower:
        is_bone = False
    
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
    print(f"[HY-Motion] Loading FBX for retargeting: {filepath}")
    manager = fbx.FbxManager.Create()
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(filepath, -1, manager.GetIOSettings()):
        error_msg = f"Failed to initialize FBX importer for: {filepath}. Error: {importer.GetStatus().GetErrorString()}"
        print(f"[HY-Motion] ERROR: {error_msg}")
        raise RuntimeError(error_msg)
        
    scene = fbx.FbxScene.Create(manager, "")
    importer.Import(scene)
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
    
    unit = scene.GetGlobalSettings().GetSystemUnit()
    skel.unit_scale = unit.GetScaleFactor()
    
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

def retarget_animation(src_skel: Skeleton, tgt_skel: Skeleton, mapping: dict[str, str], force_scale: float = 0.0, yaw_offset: float = 0.0, neutral_fingers: bool = True, in_place: bool = False, in_place_x: bool = False, in_place_y: bool = False, in_place_z: bool = False, preserve_position: bool = False):
    print("Retargeting Animation...")

    is_ue5 = mapping.get("__preset__") == "ue5"
    # Remove internal flag from mapping
    mapping = {k: v for k, v in mapping.items() if k != "__preset__"}
    
    ret_rots = {}
    ret_locs = {}
    
    yaw_q_raw = R.from_euler('y', yaw_offset, degrees=True).as_quat()
    yaw_q = np.array([yaw_q_raw[3], yaw_q_raw[0], yaw_q_raw[1], yaw_q_raw[2]])
    
    # 1. Base Mapping
    active = []
    mapped_targets = set()
    mapped_sources = set()

    for s_key, t_val in mapping.items():
        s_bone = src_skel.get_bone_case_insensitive(s_key)
        if not s_bone: continue
        
        # Support both single target string and list of multiple targets (for chains)
        t_keys = [t_val] if isinstance(t_val, str) else t_val
        
        for t_key in t_keys:
            t_bone = tgt_skel.get_bone_case_insensitive(t_key)
            if t_bone:
                # Skip if target bone is already part of a mapping
                if t_bone.name.lower() in mapped_targets:
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
    
    # 2. Smart fuzzy matching for unmapped bones using advanced algorithm
    # DISABLED for UE5 to ensure ONLY the requested 52 bones are mapped.
    if not is_ue5:
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
            
    # Sort for cleaner debug output
    final_mappings = sorted(active, key=lambda x: x[1].name)
    for s, t, _ in final_mappings:
        print(f"  - {s.name} -> {t.name}")
            
    src_h = get_skeleton_height(src_skel)
    tgt_h = get_skeleton_height(tgt_skel)
    
    # Anatomical Scale: Normalized to Centimeters for unit-agnostic comparison
    src_h_cm = src_h * src_skel.unit_scale
    tgt_h_cm = tgt_h * tgt_skel.unit_scale
    
    # The anatomical scale is the actual size ratio between the two characters
    anatomical_scale = tgt_h_cm / src_h_cm if src_h_cm > 0.01 else 1.0
    
    # Final numeric scale to apply to raw source coordinates to get target units
    # Logic: TargetUnits = (SourceUnits * SourceScaleToCM) * AnatomicalScale / TargetScaleFromCM
    # DRIFT FIX: When Auto-Scale (force_scale=0) is used, we enforce Absolute Motion Matching.
    # We IGNORE anatomical scale for translation to ensure 1:1 physics (if source moves 1m, target moves 1m).
    # This prevents "drifting" where a smaller character covers less distance than the source over time.
    if force_scale > 1e-4:
        scale = force_scale
        print(f"[INFO] Using Forced Scale: {scale:.4f}")
    else:
        # Absolute Scaling: Just convert units (M -> CM usually)
        # s_skel.unit_scale is usually 100 (M to CM)
        # t_skel.unit_scale is usually 1.0 (CM to CM) or 100 (M to CM)
        # Result: 100/1 = 100.0 (Scale meters to cm). 100/100 = 1.0 (Meters to Meters).
        scale = src_skel.unit_scale / tgt_skel.unit_scale
        print(f"[INFO] Auto-Scale (Absolute): {scale:.4f} (Anatomical Ratio {anatomical_scale:.4f} ignored for physics)")
    
    print(f"[INFO] Height Check - Source: {src_h_cm:.2f}cm, Target: {tgt_h_cm:.2f}cm")
    print(f"[INFO] Final Retargeting Scale: {scale:.4f}")
    
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

    # Coordinate Transform: Map Source Up ([0,1,0]) to Target Up
    if t_up_axis[2] == 1: # Z-Up (Blender)
        # Source Y -> Target Z, Source Z -> Target -Y, Source X -> Target X
        coord_q = R.from_euler('x', 90, degrees=True)
    else: # Y-Up
        coord_q = R.identity()

    # Realignment Logic (Calculated in Target Space)
    realignment_tgt_q = R.identity()
    t_up = t_up_axis 
    
    # NEW: Unified robust search for alignment anchors and root bone
    t_lhip, t_rhip, t_spine = None, None, None
    s_lhip, s_rhip, s_spine = None, None, None
    root_bone_name = None
    s_root_bone = None
    
    for s_bone, t_bone_val, _ in active:
        t_bone = t_bone_val[0] if isinstance(t_bone_val, list) else t_bone_val
        if not t_bone: continue
        
        sb_name = s_bone.name.lower()
        tb_name = t_bone.name.lower()
        
        if sb_name == 'l_hip': 
            s_lhip, t_lhip = s_bone, t_bone
        elif sb_name == 'r_hip': 
            s_rhip, t_rhip = s_bone, t_bone
        elif sb_name in ['spine3', 'spine2', 'spine']:
            if not s_spine or sb_name == 'spine3' or (sb_name == 'spine2' and s_spine.name.lower() == 'spine'):
                s_spine, t_spine = s_bone, t_bone
        
        if not root_bone_name:
            if any(x in tb_name or x in sb_name for x in ['pelvis', 'hips', 'root', 'center', 'cog']):
                root_bone_name = t_bone.name
                s_root_bone = s_bone

    if s_lhip and s_rhip and t_lhip and t_rhip:
        # Construct Source Frame
        s_mid_hip = (s_lhip.world_matrix[3, :3] + s_rhip.world_matrix[3, :3]) * 0.5
        s_up = (s_spine.world_matrix[3, :3] - s_mid_hip) if s_spine else np.array([0, 1, 0])
        s_side = s_rhip.world_matrix[3, :3] - s_lhip.world_matrix[3, :3]
        s_fwd = np.cross(s_side, s_up)
        s_up = np.cross(s_fwd, s_side) 
        
        s_fwd /= (np.linalg.norm(s_fwd) + 1e-9)
        s_up /= (np.linalg.norm(s_up) + 1e-9)
        s_side /= (np.linalg.norm(s_side) + 1e-9)
        s_mat = np.stack((s_side, s_up, s_fwd), axis=-1)
        
        # Construct Target Frame
        t_mid_hip = (t_lhip.world_matrix[3, :3] + t_rhip.world_matrix[3, :3]) * 0.5
        t_up_vec = (t_spine.world_matrix[3, :3] - t_mid_hip) if t_spine else t_up_axis
        t_side = t_rhip.world_matrix[3, :3] - t_lhip.world_matrix[3, :3]
        t_fwd = np.cross(t_side, t_up_vec)
        t_up = np.cross(t_fwd, t_side) 
        
        t_fwd /= (np.linalg.norm(t_fwd) + 1e-9)
        t_up /= (np.linalg.norm(t_up) + 1e-9)
        t_side /= (np.linalg.norm(t_side) + 1e-9)
        t_mat = np.stack((t_side, t_up, t_fwd), axis=-1)
        
        real_mat = t_mat @ s_mat.T @ coord_q.as_matrix().T
        realignment_tgt_q = R.from_matrix(real_mat)
        print(f"[Retarget] Computed Alignment Matrix (Euler XYZ): {realignment_tgt_q.as_euler('xyz', degrees=True)}")
    else:
        print(f"[Retarget] WARNING: Unable to compute alignment matrix - missing hip/thigh bones")

    # Final Global Transformation
    yaw_q = R.from_euler('y' if t_up_axis[1] == 1 else 'z', yaw_offset, degrees=True)
    global_transform_q = yaw_q * realignment_tgt_q * coord_q
    
    gt_q = global_transform_q.as_quat()
    gt_q_np = np.array([gt_q[3], gt_q[0], gt_q[1], gt_q[2]]) 
    # 3. Pass 1: Standard Retargeting for all bones
    for s_bone, t_bone_val, _ in active:
        # Filter None bones out of the list (e.g. if neck_03 doesn't exist)
        if isinstance(t_bone_val, list):
            t_bone_list = [b for b in t_bone_val if b is not None]
        else:
            t_bone_list = [t_bone_val] if t_bone_val is not None else []
            
        if not t_bone_list: continue
        
        num_targets = len(t_bone_list)
        s_rest_q = s_bone.rest_rotation
        
        for i, tb in enumerate(t_bone_list):
            if tb.name not in tgt_world_anims:
                tgt_world_anims[tb.name] = {}
            
            # Calculate rest pose offset in WORLD space
            # This captures the T-pose to A-pose difference automatically
            s_rest_world_q = quaternion_multiply(gt_q_np, s_rest_q)
            t_rest_world_q = tb.rest_rotation
            
            # The offset transforms from source rest to target rest
            off_q = quaternion_multiply(quaternion_inverse(s_rest_world_q), t_rest_world_q)
            
            for f in frames:
                s_rot_f = s_bone.world_animation.get(f, s_rest_q)
                
                # If chain, distribute the relative rotation
                if num_targets > 1:
                    # Convert source rotation to euler to divide it
                    rel_q = quaternion_multiply(quaternion_inverse(s_rest_q), s_rot_f)
                    rel_r = R.from_quat([rel_q[1], rel_q[2], rel_q[3], rel_q[0]])
                    
                    # Instead of euler, use SLERP for safer distribution
                    # We want to find rel_q ^ ((i+1)/num_targets)
                    # For stability we just use euler for now as previously done 
                    # but ensure we handle potential flip issues
                    euler = rel_r.as_euler('xyz')
                    dist_euler = euler * ((i + 1) / num_targets)
                    step_rel_r = R.from_euler('xyz', dist_euler)
                    
                    step_rel_q = step_rel_r.as_quat()
                    step_rel_q = np.array([step_rel_q[3], step_rel_q[0], step_rel_q[1], step_rel_q[2]])
                    s_rot_f_distributed = quaternion_multiply(s_rest_q, step_rel_q)
                else:
                    s_rot_f_distributed = s_rot_f

                # Apply the rotation: GlobalTransform * SourceAnim * Offset
                # This maps source animation through coordinate system and applies rest-pose correction
                s_rot_f_world = quaternion_multiply(gt_q_np, s_rot_f_distributed)
                t_rot_f = quaternion_multiply(s_rot_f_world, off_q)
                
                tgt_world_anims[tb.name][f] = t_rot_f


    # NOTE: Head leveling removed - the head should follow the body naturally

    #Inherit parent rotation for uanimated head/neck bones
    # If the source head has identity rotation while the body is rotated,
    # the head animation was not properly baked. We fix this by inheriting
    # the parent's rotation.
    for s_bone, t_bone_val, _ in active:
        t_bone_list = t_bone_val if isinstance(t_bone_val, list) else [t_bone_val]
        # Check if any bone in this mapping is head or neck
        is_head_neck = any(any(x in tb.name.lower() for x in ['head', 'neck']) for tb in t_bone_list)
        
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
                            
                            for tb in t_bone_list:
                                # Apply the same offset as we would for the head
                                off_q = quaternion_multiply(quaternion_inverse(quaternion_multiply(gt_q_np, s_rest_q)), tb.rest_rotation)
                                t_rot_f = quaternion_multiply(p_rot_tgt, off_q)
                                tgt_world_anims[tb.name][f] = t_rot_f


    # 5. Displacement and Local Rotations
    for s_bone, t_bone_val, _ in active:
        # Handle list or single bone
        t_bone_list = t_bone_val if isinstance(t_bone_val, list) else [t_bone_val]
        t_bone_main = t_bone_list[0]
        
        if t_bone_main.name == root_bone_name:
            ret_locs[t_bone_main.name] = {}
            
            # GROUNDING FIX: Calculate floor levels for proper vertical alignment
            # Source floor: Y=0 (SMPL-H/HyMotion convention)
            s_floor = 0.0
            
            # Target floor: Minimum foot Y coordinate (handles rigs centered at hips)
            t_floor = 0.0
            t_lfoot = tgt_skel.get_bone_case_insensitive('leftfoot') or tgt_skel.get_bone_case_insensitive('l_foot')
            t_rfoot = tgt_skel.get_bone_case_insensitive('rightfoot') or tgt_skel.get_bone_case_insensitive('r_foot')
            if t_lfoot or t_rfoot:
                foot_ys = []
                if t_lfoot: foot_ys.append(t_lfoot.head[1])
                if t_rfoot: foot_ys.append(t_rfoot.head[1])
                t_floor = min(foot_ys) if foot_ys else 0.0
                print(f"[Retarget] Ground Level Detection: Target floor at Y={t_floor:.2f}")
            
            # Where the hips START in the source animation (Scaled and Transformed)
            s_rest_pos = s_bone.world_matrix[3, :3]
            s_pos_0 = s_bone.world_location_animation.get(frames[0], s_rest_pos)
            t_pos_0 = global_transform_q.apply(s_pos_0 * scale)
            
            # AXIS-AWARE GROUNDING:
            h_offset = np.zeros(3)
            ground_offset = np.zeros(3)
            
            if t_up_axis[1] == 1: # Y-Up (Unity, Unreal)
                # Ground is X/Z plane, Up is Y
                ground_offset = np.array([0.0, t_floor - (s_floor * scale), 0.0])
                # Center X and Z (horizontal)
                h_offset = np.array([-t_pos_0[0], 0.0, -t_pos_0[2]])
                print(f"[Retarget] Axis System: Y-Up. Applying grounding to Y, centering X/Z.")
            
            else: # Z-Up (Blender, 3ds Max)
                # Ground is X/Y plane, Up is Z
                # Note: t_floor detection above might need adjustment for Z-Up if not handled
                s_floor_z = 0.0 # Hypothetical Z-floor for source
                
                # Re-detect floor on Z axis for target if needed
                if t_lfoot or t_rfoot:
                     foot_zs = []
                     if t_lfoot: foot_zs.append(t_lfoot.head[2])
                     if t_rfoot: foot_zs.append(t_rfoot.head[2])
                     t_floor = min(foot_zs) if foot_zs else 0.0
                
                ground_offset = np.array([0.0, 0.0, t_floor - (s_floor_z * scale)])
                # Center X and Y (horizontal in Z-up)
                h_offset = np.array([-t_pos_0[0], -t_pos_0[1], 0.0])
                print(f"[Retarget] Axis System: Z-Up. Applying grounding to Z, centering X/Y.")
            
            anchor_offset = ground_offset + h_offset
            
            if preserve_position:
                anchor_offset[:] = 0.0
                print(f"[Retarget] Preserving absolute world position (using sampler coordinates).")
            else:
                print(f"[Retarget] Grounded root: Floor offset={ground_offset[1]:.2f}, Horizontal centering applied.")
            
            for f in frames:
                s_pos_f = s_bone.world_location_animation.get(f, s_rest_pos)
                t_pos_f = global_transform_q.apply(s_pos_f * scale) + anchor_offset
                
                # Lock movement based on granular toggles or the legacy in_place flag
                # Note: These values are in Target Space (t_pos_f)
                
                # Determine what needs to be locked
                lock_x = in_place_x or in_place
                lock_z = in_place_z or in_place
                lock_y = in_place_y
                
                # STITCHING & COORD DEBUG
                f_idx = frames.index(f)
                is_key_frame = (f_idx < 3) or (f_idx > len(frames) - 3)
                # Check for possible stitch junction (every 90-120 frames usually)
                if not is_key_frame and (f_idx % 30 == 0):
                    is_key_frame = True
                
                if is_key_frame:
                    lock_str = f"[{'X' if lock_x else '.'}{'Y' if lock_y else '.'}{'Z' if lock_z else '.'}]"

                if t_up_axis[1] == 1: # Y-Up (e.g. Unity, Unreal default)
                    if lock_x: t_pos_f[0] = t_rest_pos[0]
                    if lock_z: t_pos_f[2] = t_rest_pos[2]
                    if lock_y: t_pos_f[1] = t_rest_pos[1]
                else: # Z-Up (e.g. Blender)
                    if lock_z: t_pos_f[2] = t_rest_pos[2] # Vertical in Z-Up
                    if lock_x: t_pos_f[0] = t_rest_pos[0]
                    if lock_y: t_pos_f[1] = t_rest_pos[1] # Horizontal in Z-Up
                
                # Transform to Parent Local Space
                pname = t_bone_main.parent_name
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
                ret_locs[t_bone_main.name][f] = local_pos_f

    # 3. Local Rotations
    for s_bone, t_bone_val, _ in active:
        # Handle list or single bone
        t_bone_list = t_bone_val if isinstance(t_bone_val, list) else [t_bone_val]
        
        for tb in t_bone_list:
            ret_rots[tb.name] = {}
            pname = tb.parent_name
            
            is_head = 'head' in tb.name.lower() and 'neck' not in tb.name.lower()
            
            for f in frames:
                prot = tgt_world_anims.get(pname, {}).get(f)
                if prot is None:
                    # Fallback: Check if target skeleton has this node's rest orientation
                    prot = tgt_skel.node_rest_rotations.get(pname, np.array([1, 0, 0, 0]))
                    # Apply global rotation to parent if it's not animated
                    prot = quaternion_multiply(gt_q_np, prot)
                    
                # Target world rotation to local space: Local = ParentWorld.inv @ World
                l_rot = quaternion_multiply(quaternion_inverse(prot), tgt_world_anims[tb.name][f])
                ret_rots[tb.name][f] = l_rot
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True)
    parser.add_argument('--target', '-t', required=True)
    parser.add_argument('--mapping', '-m', default='', help='Optional bone mapping file (uses hardcoded mappings if not provided)')
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--yaw', '-y', type=float, default=0.0)
    parser.add_argument('--scale', '-sc', type=float, default=0.0)
    parser.add_argument('--no-neutral', dest='neutral', action='store_false', help="Disable neutral finger rest-pose")
    parser.add_argument('--in-place', action='store_true', help="Lock horizontal movement")
    parser.set_defaults(neutral=True)
    args = parser.parse_args()
    
    if args.source.lower().endswith('.npz'):
        print(f"Loading NPZ Source: {args.source}")
        src_man, src_scene = None, None
        src_skel = load_npz(args.source)
    else:
        # Sampling characters often use frame 0 as bind, but real FBX files have a Bind Pose 
        # reachable by setting current animation stack to None.
        src_man, src_scene, src_skel = load_fbx(args.source, sample_rest_frame=None)
        src_h = get_skeleton_height(src_skel)
        # FALLBACK: If Bind Pose is collapsed (height ~0), try frame 0
        if src_h < 0.1:
            src_man, src_scene, _ = load_fbx(args.source, sample_rest_frame=0)
            # Refresh skeleton with sampled data
            src_skel = Skeleton(os.path.basename(args.source))
            collect_skeleton_nodes(src_scene.GetRootNode(), src_skel, sampling_time=FbxTime())
        
        extract_animation(src_scene, src_skel)
    
    tgt_man, tgt_scene, tgt_skel = load_fbx(args.target)
    
    mapping = load_bone_mapping(args.mapping, src_skel, tgt_skel)
    
    rots, locs, active = retarget_animation(src_skel, tgt_skel, mapping, args.scale, args.yaw, args.neutral, args.in_place)
    
    print(f"\n[Retarget] Bone Mapping Results:\n  Total: {len(active)} bones mapped")
    if len(active) < 15:
        print(f"  [WARNING] Very few bones mapped ({len(active)}). Retargeting might be poor for high-fidelity animations.")
        
    src_time_mode = src_scene.GetGlobalSettings().GetTimeMode() if src_scene else tgt_scene.GetGlobalSettings().GetTimeMode()
    apply_retargeted_animation(tgt_scene, tgt_skel, rots, locs, src_skel.frame_start, src_skel.frame_end, src_time_mode)
    
    save_fbx(tgt_man, tgt_scene, args.output)
    print("Done!")

if __name__ == "__main__":
    main()
