from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# =============================================================================
# Constants & Hand Poses
# =============================================================================

# Imports from presets.py
from .presets import (
    LEFT_HAND_MEAN_AA,
    RIGHT_HAND_MEAN_AA,
    BASE_BONE_MAPPING,
    ARTICULATION_MAPPING,
    UE5_MAPPING,
    G9_MAPPING,
    G3_MAPPING,
    UE5_SOURCE_ALIASES,
    FUZZY_ALIASES,
    BONE_KEYWORDS,
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
        if norm > 1e-9:
            m33[i] /= norm

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
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def solve_rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns a 3x3 rotation matrix that aligns v1 with v2 via shortest path."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-9 or v2_norm < 1e-9:
        return np.eye(3)
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    dot = np.dot(v1, v2)
    if dot > 0.999999:
        return np.eye(3)
    if dot < -0.999999:
        # 180 deg: Pick a perpendicular axis
        axis = np.array([0, 1, 0]) if abs(v1[0]) > 0.9 else np.array([1, 0, 0])
        axis = np.cross(v1, axis)
        axis /= np.linalg.norm(axis) + 1e-9
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


def decompose_swing_twist(
    q: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


class BoneData:
    def __init__(self, name: str):
        self.name = name
        self.parent_name = None  # Immediate FBX node parent
        self.bone_parent_name = None  # Parent in bone hierarchy
        self.local_matrix = np.eye(4)
        self.world_matrix = np.eye(4)
        self.head: np.ndarray = np.zeros(3)
        self.has_skeleton_attr: bool = False
        self.rest_rotation = np.array([1, 0, 0, 0])
        self.animation: dict[int, np.ndarray] = {}  # frame -> local [w, x, y, z]
        self.world_animation: dict[int, np.ndarray] = {}  # frame -> world [w, x, y, z]
        self.location_animation: dict[int, np.ndarray] = {}  # frame -> local pos
        self.world_location_animation: dict[int, np.ndarray] = {}  # frame -> world pos


class Skeleton:
    def __init__(self, name: str = "Skeleton"):
        self.name = name
        self.bones: dict[str, BoneData] = {}
        self.all_node_parents: dict[str, str] = {}  # node_name -> parent_name
        self.node_rest_rotations: dict[
            str, np.ndarray
        ] = {}  # node -> world_rest_q [w,x,y,z]
        self.node_world_matrices: dict[
            str, np.ndarray
        ] = {}  # node -> world_rest_mat (4x4)
        self.node_local_matrices: dict[
            str, np.ndarray
        ] = {}  # node -> local_rest_mat (4x4)
        self.fps = 30.0
        self.frame_start = 0
        self.frame_end = 0
        self.bone_children: dict[
            str, list[str]
        ] = {}  # bone_name -> list of child bone names
        self.unit_scale = 1.0  # Multiplier to convert internal units to Centimeters (1.0 for CM, 100.0 for M)

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
        if not name:
            return None
        lower_name = name.lower()
        if lower_name in self.bones:
            return self.bones[lower_name]

        # 1. Try stripping namespaces (e.g. Unirig:Hips -> Hips)
        simplified = lower_name.split(":")[-1]
        if simplified in self.bones:
            return self.bones[simplified]

        # 2. Try strict exact match on simplified names
        for bname, bone in self.bones.items():
            if bname.split(":")[-1] == simplified:
                return bone

        # 3. Strip standard prefixes and characters
        def clean(s):
            return (
                s.lower()
                .split(":")[-1]
                .replace("bip01 ", "")
                .replace(".", "")
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
            )

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

        if not pelvis:
            return {}

        # 2. Get height for scaling
        height = get_skeleton_height(self)
        if height < 0.001:
            height = 1.0

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
            if rel_pos[0] > 0.05:
                side = 1
            elif rel_pos[0] < -0.05:
                side = -1

            # Refine side based on name if possible
            lower_name = bone.name.lower()
            if "left" in lower_name or "l_" in lower_name or ".l" in lower_name:
                side = -1
            elif "right" in lower_name or "r_" in lower_name or ".r" in lower_name:
                side = 1

            features[bone.name] = {"pos": rel_pos, "depth": depth, "side": side}

        return features


def get_skeleton_height(skeleton: Skeleton) -> float:
    """
    Calculate skeleton height using bone lengths (Pelvis -> Spine -> Neck -> Head).
    This is pose-invariant and works even if the character is lying down.
    """
    # 1. Try Bone Chain Method (Robust)
    pelvis_names = ["pelvis", "hips", "hip", "bone_0", "mixamorig:hips"]
    head_names = ["head", "mixamorig:head"]

    def find_bone(names):
        for n in names:
            b = skeleton.get_bone_case_insensitive(n)
            if b:
                return b
        return None

    pelvis = find_bone(pelvis_names)
    head = find_bone(head_names)

    if pelvis and head:
        total_len = 0.0
        curr = head
        visited = set()
        while (
            curr
            and curr.name.lower() not in [p.lower() for p in pelvis_names]
            and curr.name not in visited
        ):
            visited.add(curr.name)
            pname = curr.parent_name
            if not pname:
                break
            parent = skeleton.get_bone_case_insensitive(pname)
            if not parent:
                break

            dist = np.linalg.norm(curr.head - parent.head)
            total_len += dist
            curr = parent

        # Add leg length (approximate from Pelvis to Foot)
        l_hip = find_bone(
            [
                "hip",
                "pelvis",
                "l_hip",
                "leftupleg",
                "mixamorig:leftupleg",
                "L_Hip",
                "lThighBend",
            ]
        )
        # Search for foot first, then ankle
        l_foot = find_bone(
            [
                "lefttoe",
                "l_foot",
                "leftfoot",
                "L_Foot",
                "l_ankle",
                "L_Ankle",
                "ltoe",
                "lfoot",
                "lFoot",
            ]
        )

        if l_hip and l_foot:
            leg_len = 0.0
            curr = l_foot
            # Traverse from foot up to PELVIS, not just hip, to catch the hip-to-pelvis distance
            # This ensures we get the full vertical height even if the structure is non-standard
            while (
                curr
                and curr.name.lower() not in [p.lower() for p in pelvis_names]
                and curr.name not in visited
            ):
                visited.add(curr.name)
                pname = curr.parent_name
                if not pname:
                    break
                parent = skeleton.get_bone_case_insensitive(pname)
                if not parent:
                    break
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
        if abs(h_val) < 1e-6:
            continue
        y_min = min(y_min, h_val)
        y_max = max(y_max, h_val)
        found_any = True

    if not found_any or (y_max - y_min) < 0.1:
        return 1.0
    return y_max - y_min


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
    kps = data["keypoints3d"]
    transl = data["transl"]
    rot6d = data.get("rot6d")
    root_mat = data.get("root_rotations_mat")
    poses = data.get("poses")

    T = kps.shape[0]

    # HY-MOTION SYNC FIX: Detect if keypoints are already in World Space
    # In some NPZ variants (like the user's), keypoints3d already includes translation.
    # In others (standard SMPL), they are relative to Pelvis at origin.
    kps_travel = np.linalg.norm(kps[-1, 0] - kps[0, 0])
    if kps_travel > 0.1:
        print(
            f"[Retarget] NPZ Detect: Keypoints are already animated (Travel={kps_travel:.2f}m). Skipping Transl addition."
        )
        global_kps = kps.copy()
    else:
        print(
            f"[Retarget] NPZ Detect: Keypoints are relative to root. Adding Transl vector."
        )
        global_kps = kps + transl[:, np.newaxis, :]

    names = [
        "Pelvis",
        "L_Hip",
        "R_Hip",
        "Spine1",
        "L_Knee",
        "R_Knee",
        "Spine2",
        "L_Ankle",
        "R_Ankle",
        "Spine3",
        "L_Foot",
        "R_Foot",
        "Neck",
        "L_Collar",
        "R_Collar",
        "Head",
        "L_Shoulder",
        "R_Shoulder",
        "L_Elbow",
        "R_Elbow",
        "L_Wrist",
        "R_Wrist",
        "L_Index1",
        "L_Index2",
        "L_Index3",
        "L_Middle1",
        "L_Middle2",
        "L_Middle3",
        "L_Pinky1",
        "L_Pinky2",
        "L_Pinky3",
        "L_Ring1",
        "L_Ring2",
        "L_Ring3",
        "L_Thumb1",
        "L_Thumb2",
        "L_Thumb3",
        "R_Index1",
        "R_Index2",
        "R_Index3",
        "R_Middle1",
        "R_Middle2",
        "R_Middle3",
        "R_Pinky1",
        "R_Pinky2",
        "R_Pinky3",
        "R_Ring1",
        "R_Ring2",
        "R_Ring3",
        "R_Thumb1",
        "R_Thumb2",
        "R_Thumb3",
    ]

    parents = [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        20,
        25,
        26,
        20,
        28,
        29,
        20,
        31,
        32,
        20,
        34,
        35,
        21,
        37,
        38,
        21,
        40,
        41,
        21,
        43,
        44,
        21,
        46,
        47,
        21,
        49,
        50,
    ]

    skel = Skeleton(name)
    skel.frame_start = 0
    skel.frame_end = T - 1
    skel.fps = 30.0
    skel.unit_scale = 100.0  # HyMotion/SMPL-H is ALWAYS in Meters

    # Reconstruct World Rotations via Forward Kinematics
    world_rots = np.zeros((T, 52, 3, 3))

    # 1. Compute Canonical Rest Pose (All Identity Local)
    # This keeps the character stable and prevents 'morphing'
    rest_world_rots = np.zeros((52, 3, 3))
    rest_world_rots[0] = np.eye(3)  # Root
    for i in range(1, 52):
        p = parents[i]
        rest_world_rots[i] = rest_world_rots[p]  # Local Identity

    # Child lookup for vector-based solving (fingers)
    children = {}
    for idx, p in enumerate(parents):
        if p != -1 and p not in children:  # Find first child
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
                if 22 + i < 52:
                    local_mats[f, 22 + i] = l_hand_mats[i]
                if 37 + i < 52:
                    local_mats[f, 37 + i] = r_hand_mats[i]

    # 2. Forward Kinematics to get World Rotations
    for f in range(T):
        # Root (Pelvis)
        if root_mat is not None:
            # Use the actual root rotation matrix from the source
            # (Allows the character to lean/bow naturally)
            world_rots[f, 0] = root_mat[f]
        else:
            world_rots[f, 0] = local_mats[f, 0]

        for i in range(1, 52):
            p = parents[i]
            # Multiplication order for world rotations: Parent @ Local (Stable convention)
            world_rots[f, i] = world_rots[f, p] @ local_mats[f, i]

    # 3. Create a Canonical T-Pose for rest pose reference
    # This ensures stable retargeting offsets (off_q) and accurate geometric matching.
    t_pose_kps = np.zeros((52, 3))
    t_pose_kps[0] = [0, 0, 0]  # Pelvis at origin

    # Define T-pose directions for SMPL-H
    # Y is Up, X is Side, Z is Forward
    for i in range(1, 52):
        p = parents[i]
        dist = np.linalg.norm(kps[0, i] - kps[0, p])
        name = names[i].lower()

        # Default direction: Up
        dir_vec = np.array([0.0, 1.0, 0.0])

        if any(x in name for x in ["hip", "knee", "ankle", "foot"]):
            dir_vec = np.array([0.0, -1.0, 0.0])  # Legs down
            if "hip" in name:
                # Add a small X offset to separate hips for side detection
                if name.startswith("l_"):
                    dir_vec[0] = 0.5
                else:
                    dir_vec[0] = -0.5
                dir_vec /= np.linalg.norm(dir_vec)
        elif (
            "collar" in name
            or "shoulder" in name
            or "elbow" in name
            or "wrist" in name
            or "index" in name
            or "middle" in name
            or "pinky" in name
            or "ring" in name
            or "thumb" in name
        ):
            if name.startswith("l_"):
                dir_vec = np.array([1.0, 0.0, 0.0])  # Left arm out (+X)
            else:
                dir_vec = np.array([-1.0, 0.0, 0.0])  # Right arm out (-X)

        t_pose_kps[i] = t_pose_kps[p] + dir_vec * dist

    for i, name in enumerate(names):
        bone = BoneData(name)
        p_idx = parents[i]
        if p_idx != -1 and p_idx != i:
            bone.parent_name = names[p_idx]
            bone.bone_parent_name = names[p_idx]

        # Rest Pose = Canonical T-Pose
        bone.rest_rotation = np.array([1, 0, 0, 0])  # Identity
        bone.head = t_pose_kps[i]
        bone.world_matrix = np.eye(4)
        bone.world_matrix[:3, :3] = np.eye(3)  # Identity world rotation
        bone.world_matrix[3, :3] = t_pose_kps[i]

        # Animation
        for f in range(T):
            # Backup convention: Transpose the world matrix before conversion
            bone.world_animation[f] = matrix_to_quaternion(world_rots[f, i].T)
            bone.world_location_animation[f] = global_kps[f, i]

        skel.add_bone(bone)
        skel.all_node_parents[name] = bone.parent_name
        skel.node_rest_rotations[name] = bone.rest_rotation
        skel.node_world_matrices[name] = bone.world_matrix
        skel.node_local_matrices[name] = np.eye(4) # Simplified local for NPZ

    return skel


# =============================================================================
# Retargeting Configuration & Presets
# =============================================================================


# Presets imported from .presets


# =============================================================================
# Heuristic & Fuzzy Matching System
# =============================================================================


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

    prefixes = [
        "bip01_",
        "bip001_",
        "joint_",
        "bone_",
        "def_",
        "rig_",
        "valvebiped_",
        "mixamorig_",
    ]
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix) :]

    # Remove underscores and dots for comparison
    name_lower = name_lower.replace("_", "").replace(".", "").replace("-", "")

    return name_lower


def detect_side(name: str) -> tuple[bool, bool]:
    """
    Detect if bone is left or right side with improved logic.
    Returns: (is_left, is_right)
    """
    name_lower = name.lower()

    # Explicit left markers
    is_left = any(
        x in name_lower
        for x in ["left", ".l", "_l", "l_", "lhand", "lfoot", "larm", "lleg"]
    ) or (
        name_lower.startswith("l")
        and len(name_lower) > 1
        and name_lower[1] in ["_", "."]
    )

    # Explicit right markers
    is_right = any(
        x in name_lower
        for x in ["right", ".r", "_r", "r_", "rhand", "rfoot", "rarm", "rleg"]
    ) or (
        name_lower.startswith("r")
        and len(name_lower) > 1
        and name_lower[1] in ["_", "."]
    )

    return is_left, is_right


def classify_bone(name: str) -> list[str]:
    """Classify bone into categories (root, spine, leg, arm, finger, etc.)"""
    name_norm = normalize_bone_name(name)
    categories = []

    for category, keywords in BONE_KEYWORDS.items():
        if any(kw in name_norm for kw in keywords):
            categories.append(category)

    return categories


def calculate_bone_similarity(
    source_name: str, target_name: str, use_aliases: bool = True
) -> float:
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
    if ("hand" in s_categories or "arm" in s_categories) and "finger" in t_categories:
        # A generic hand bone should never match a specific finger bone
        return 0.0
    if "finger" in s_categories and ("hand" in t_categories or "arm" in t_categories):
        return 0.0

    # Strategy 2: One contains the other (substring match)
    if s_norm in t_norm or t_norm in s_norm:
        overlap = min(len(s_norm), len(t_norm))
        total = max(len(s_norm), len(t_norm))
        score = 0.85 * (overlap / total)

        # Strict finger separation for substring matches
        if "finger" in s_categories or "finger" in t_categories:
            # If both are fingers, must share same type and segment
            s_type = next(
                (
                    c
                    for c in ["thumb", "index", "middle", "ring", "pinky"]
                    if c in s_categories
                ),
                None,
            )
            t_type = next(
                (
                    c
                    for c in ["thumb", "index", "middle", "ring", "pinky"]
                    if c in t_categories
                ),
                None,
            )
            if s_type != t_type:
                return 0.0

            # Segment check
            s_seg = (
                1
                if "1" in s_norm
                else (2 if "2" in s_norm else (3 if "3" in s_norm else 0))
            )
            t_seg = 0
            if "proximal" in t_norm:
                t_seg = 1
            elif "distal" in t_norm or "dist" in t_norm:
                t_seg = 3
            elif "medial" in t_norm or "inter" in t_norm:
                t_seg = 2
            elif "1" in t_norm:
                t_seg = 1
            elif "2" in t_norm:
                t_seg = 2
            elif "3" in t_norm:
                t_seg = 3

            if s_seg != 0 and t_seg != 0 and s_seg != t_seg:
                return 0.0

        # Apply category penalty if they don't share enough categories
        if s_categories and t_categories:
            intersect = set(s_categories) & set(t_categories)
            if not intersect:
                score *= 0.1  # Very heavy penalty for body part mismatch
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
                    score *= 0.3  # Heavy penalty for category mismatch
            return score

    # Strategy 5: Category matching (same type of bone)
    if s_categories and t_categories:
        # Check finger type equality FIRST
        s_finger_types = set(["thumb", "index", "middle", "ring", "pinky"]) & set(
            s_categories
        )
        t_finger_types = set(["thumb", "index", "middle", "ring", "pinky"]) & set(
            t_categories
        )
        if s_finger_types or t_finger_types:
            if s_finger_types != t_finger_types:
                return 0.0

        overlap = len(set(s_categories) & set(t_categories))
        if overlap > 0:
            score = 0.5 * (overlap / max(len(s_categories), len(t_categories)))

            # Segment specific adjustment for fingers
            if "finger" in s_categories:
                s_seg = (
                    1
                    if "1" in s_norm
                    else (2 if "2" in s_norm else (3 if "3" in s_norm else 0))
                )
                t_seg = 0
                if "proximal" in t_norm:
                    t_seg = 1
                elif "distal" in t_norm:
                    t_seg = 3
                elif "medial" in t_norm:
                    t_seg = 2
                elif "1" in t_norm:
                    t_seg = 1
                elif "2" in t_norm:
                    t_seg = 2
                elif "3" in t_norm:
                    t_seg = 3

                if s_seg != 0 and t_seg != 0:
                    if s_seg == t_seg:
                        score += 0.4
                    else:
                        return 0.0

            return score

    return 0.0  # No match


def find_best_bone_match(
    target_bone_name: str,
    source_skeleton: "Skeleton",
    already_mapped_sources: set[str],
    require_side_match: bool = True,
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




def get_fbx_rotation_order_str(node: fbx.FbxNode) -> str:
    order = node.RotationOrder.Get()
    mapping = {0: "xyz", 1: "xzy", 2: "yzx", 3: "yxz", 4: "zxy", 5: "zyx"}
    return mapping.get(order, "xyz")


def normalize_target_rig(node: fbx.FbxNode, parent_world_inv: np.ndarray = None, eval_time: fbx.FbxTime = None):
    """
    Replicate Blender's FBX normalization:
    1. Bake Pre-Rotations, Post-Rotations, and Pivots into the actual local transform.
    2. Force RotationOrder to XYZ.
    3. Ensure LclTranslation reflects the physical joint spacing.
    """
    if parent_world_inv is None:
        parent_world_inv = np.eye(4)

    if eval_time is None:
        eval_time = fbx.FbxTime(0)

    # 1. Capture current evaluated world transform at rest pose (usually T-Pose)
    world_mat = fbx_matrix_to_numpy(node.EvaluateGlobalTransform(eval_time))

    # 2. Reset hidden FBX offsets (Magic pivots/rotations) using high-level methods
    node.SetRotationActive(True)
    node.SetPivotState(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.FbxNode.EPivotState.ePivotActive)
    
    node.SetPreRotation(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.FbxVector4(0, 0, 0, 0))
    node.SetPostRotation(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.FbxVector4(0, 0, 0, 0))
    node.SetRotationPivot(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.FbxVector4(0, 0, 0, 0))
    node.SetScalingPivot(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.FbxVector4(0, 0, 0, 0))
    node.SetRotationOffset(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.FbxVector4(0, 0, 0, 0))
    node.SetScalingOffset(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.FbxVector4(0, 0, 0, 0))

    # 3. Force RotationOrder to XYZ
    node.SetRotationOrder(fbx.FbxNode.EPivotSet.eSourcePivot, fbx.EFbxRotationOrder.eEulerXYZ)

    # 4. Calculate New Clean Local Transformation
    # Result = inv(ParentWorld) * NodeWorld
    clean_local = parent_world_inv @ world_mat

    # Set cleaner Translation (extract from matrix row 3)
    pos = clean_local[3, :3]
    node.LclTranslation.Set(fbx.FbxDouble3(pos[0], pos[1], pos[2]))

    # Calculate cleaner Orientation (from matrix rotation part)
    # We use scipy to extract XYZ Euler safely from the clean 3x3
    rot_part = clean_local[:3, :3]
    # Remove scaling to get pure rotation
    scaling = np.linalg.norm(rot_part, axis=0)
    rot_normalized = rot_part / (scaling + 1e-12)
    
    r = R.from_matrix(rot_normalized)
    euler = r.as_euler("xyz", degrees=True)
    node.LclRotation.Set(fbx.FbxDouble3(euler[0], euler[1], euler[2]))

    # Recurse through children
    world_inv = np.linalg.inv(world_mat)
    for i in range(node.GetChildCount()):
        normalize_target_rig(node.GetChild(i), world_inv, eval_time)


# =============================================================================
# Technical FBX & SDK Utilities
# =============================================================================



def collect_skeleton_nodes(
    node: fbx.FbxNode,
    skeleton: Skeleton,
    immediate_parent: str = None,
    last_bone_parent: str = None,
    depth: int = 0,
    sampling_time: fbx.FbxTime = None,
    bind_pose_map: dict = None,
):
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
    skeleton.node_local_matrices[node_name] = node_local_mat
    skeleton.node_rest_rotations[node_name] = matrix_to_quaternion(node_world_mat)

    # 2. Bone Identification
    if attr:
        at = attr.GetAttributeType()
        # 3 = eSkeleton, 4 = eLimbNode, 2 = eNull
        if at in [3, 4]:
            is_bone = True
        elif at == 2 and (node.GetChildCount() > 0 or immediate_parent):
            is_bone = True

    name_lower = node_name.lower()
    keywords = [
        "hips",
        "hip",
        "spine",
        "neck",
        "head",
        "arm",
        "leg",
        "foot",
        "ankle",
        "knee",
        "shoulder",
        "elbow",
        "pelvis",
        "joint",
        "mixamo",
        "thigh",
        "upper",
        "forearm",
        "hand",
        "finger",
        "clavicle",
        "collar",
        "toe",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "upleg",
        "downleg",
        "wrist",
        "chest",
        "belly",
        "bone_",
        "character",
        "calf",
        "ball",
        "shin",
        "mid",
        "metacarpal",
        "subclavian",
        # G3/Daz Genesis specific keywords
        "abdomen",
        "shldr",
        "twist",
        "bend",
        "metatarsal",
        "necklower",
        "neckupper",
    ]
    if any(k in name_lower for k in keywords):
        is_bone = True

    # UE5 IK bones should be collected so they can be "virtually snapped"
    if "ik_" in name_lower or "_ik" in name_lower:
        is_bone = True

    if is_bone:
        existing = skeleton.get_bone_case_insensitive(node_name)
        if existing and (attr and attr.GetAttributeType() in [3, 4]):
            skeleton.bones.pop(existing.name.lower(), None)
        elif existing:
            is_bone = False

    if is_bone:
        bone = BoneData(node_name)
        bone.has_skeleton_attr = attr and attr.GetAttributeType() in [3, 4]
        bone.parent_name = immediate_parent
        bone.bone_parent_name = last_bone_parent
        bone.local_matrix = node_local_mat
        bone.world_matrix = node_world_mat
        bone.head = node_world_mat[3, :3]
        bone.rest_rotation = skeleton.node_rest_rotations[node_name]
        skeleton.add_bone(bone)
        last_bone_parent = node_name

    skeleton.all_node_parents[node_name] = immediate_parent
    imm_p = node_name
    for i in range(node.GetChildCount()):
        collect_skeleton_nodes(
            node.GetChild(i),
            skeleton,
            imm_p,
            last_bone_parent,
            depth + 1,
            sampling_time,
            bind_pose_map,
        )


def extract_animation(scene: FbxScene, skeleton: Skeleton):
    stack = scene.GetCurrentAnimationStack()
    if not stack:
        return
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
                bone.world_location_animation[f] = wmat[3, :3]  # NEW: world-space pos
        for i in range(node.GetChildCount()):
            sample(node.GetChild(i))

    sample(scene.GetRootNode())

# =============================================================================
# Animation Extraction & Application
# =============================================================================


def load_fbx(
    filepath: str, 
    sample_rest_frame: int = None, 
    use_bind_pose: bool = True,
    auto_normalize: bool = True
):
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
                    bind_pose_map[pnode.GetName()] = fbx_matrix_to_numpy(
                        pose.GetMatrix(j)
                    )
            if bind_pose_map:
                break  # Preference: use the first bind pose found

    skel = Skeleton(os.path.basename(filepath))
    collect_skeleton_nodes(
        scene.GetRootNode(),
        skel,
        sampling_time=fbx.FbxTime(0) if sample_rest_frame is not None else None,
        bind_pose_map=bind_pose_map if use_bind_pose else None,
    )

    # 3. Auto-Normalization Detection
    if auto_normalize is True:
        # Smart Auto-detect
        is_daz = False
        daz_keys = ["hip", "lthigh", "rthigh", "lshin", "rshin", "lcollar", "rcollar"]
        bone_names_lower = [b.lower() for b in skel.bones.keys()]
        match_count = sum(1 for k in daz_keys if any(k in b for b in bone_names_lower))
        
        if match_count >= 4:
            daz_candidates = ["hip", "lThigh", "rThigh", "lUpperArm", "rUpperArm"]
            found_pre_rot = False
            for c in daz_candidates:
                node = scene.FindNodeByName(c)
                if not node:
                    for i in range(scene.GetNodeCount()):
                        n = scene.GetNode(i)
                        if n.GetName().lower() == c.lower():
                            node = n
                            break
                if node:
                    pre = node.PreRotation.Get()
                    if abs(pre[0]) > 0.001 or abs(pre[1]) > 0.001 or abs(pre[2]) > 0.001:
                        found_pre_rot = True
                        break
            if found_pre_rot:
                is_daz = True
                print(f"[HY-Motion] Auto-Detected Daz Genesis rig. Applying Rig Normalization...")
        
        if not is_daz:
            auto_normalize = False # Don't normalize after all

    if auto_normalize: # This is True if it was forced or auto-detected successfully
        # Clear stacks and poses
        for i in range(scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)) - 1, -1, -1):
            s = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i)
            scene.DisconnectSrcObject(s)
            s.Destroy()
        
        for i in range(scene.GetPoseCount() - 1, -1, -1):
            p = scene.GetPose(i)
            scene.DisconnectSrcObject(p)
            p.Destroy()

        normalize_target_rig(scene.GetRootNode(), eval_time=fbx.FbxTime(0))
        
        print(f"[HY-Motion] Re-collecting clean skeleton data...")
        skel = Skeleton(os.path.basename(filepath))
        collect_skeleton_nodes(
            scene.GetRootNode(),
            skel,
            sampling_time=fbx.FbxTime(0) if sample_rest_frame is not None else None,
            bind_pose_map=None
        )

    unit = scene.GetGlobalSettings().GetSystemUnit()
    skel.unit_scale = unit.GetScaleFactor()

    # Refresh frame range if needed
    return manager, scene, skel



# =============================================================================
# Mapping & Skeleton Analysis
# =============================================================================


def load_bone_mapping(
    filepath: str, src_skel: Skeleton, tgt_skel: Skeleton
) -> dict[str, str]:
    is_ue5 = filepath and filepath.lower() == "ue5"
    if not is_ue5:
        # Auto-detect UE5 based on specific bone names
        if (
            tgt_skel.get_bone_case_insensitive("clavicle_l")
            and tgt_skel.get_bone_case_insensitive("upperarm_l")
            and tgt_skel.get_bone_case_insensitive("spine_01")
        ):
            is_ue5 = True
            print("[Retarget] Auto-detected UE5 skeleton structure.")

        # Auto-detect G9 based on specific bone names
        is_g9 = False
        if not is_ue5:
            has_upperarm = tgt_skel.get_bone_case_insensitive("l_upperarm") is not None
            has_shin = tgt_skel.get_bone_case_insensitive("l_shin") is not None
            has_hand = tgt_skel.get_bone_case_insensitive("l_hand") is not None

            if has_upperarm and has_shin and has_hand:
                is_g9 = True
                print("[Retarget] Auto-detected G9 skeleton structure.")

        # Auto-detect G3 based on specific bone names
        is_g3 = False
        if not is_ue5 and not is_g9:
            has_thigh_bend = (
                tgt_skel.get_bone_case_insensitive("lThighBend") is not None
            )
            has_shin_g3 = tgt_skel.get_bone_case_insensitive("lShin") is not None
            has_hand_g3 = tgt_skel.get_bone_case_insensitive("lHand") is not None

            if has_thigh_bend and has_shin_g3 and has_hand_g3:
                is_g3 = True
                print("[Retarget] Auto-detected G3 skeleton structure.")

    mapping = {} if (is_ue5 or is_g9 or is_g3) else BASE_BONE_MAPPING.copy()
    if is_ue5:
        # Pass the flag through the mapping for the animation retargeter
        mapping["__preset__"] = "ue5"
    elif is_g9:
        mapping["__preset__"] = "g9"
    elif is_g3:
        mapping["__preset__"] = "g3"

    # Check if target is UniRig (ArticulationXL) - look for any bone_X pattern
    is_unirig = False
    if not is_ue5 and not is_g9:
        is_unirig = tgt_skel.get_bone_case_insensitive("bone_0") is not None
        if not is_unirig:
            # Check for any bone_X naming pattern
            is_unirig = any("bone_" in name.lower() for name in tgt_skel.bones.keys())

    json_bones = {}
    if is_ue5:
        print(f"Detected UE5 mapping preset, using hardcoded UE5 mapping...")
        json_bones = UE5_MAPPING
    elif is_g9:
        print(f"Detected G9 mapping preset, using hardcoded G9 mapping...")
        json_bones = G9_MAPPING
    elif is_g3:
        print(f"Detected G3 mapping preset, using hardcoded G3 mapping...")
        json_bones = G3_MAPPING
    elif filepath and os.path.exists(filepath):
        print(f"Loading Mapping File: {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)
        json_bones = data.get("bones", {})
    elif is_unirig:
        # Try to find unirig.json in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        unirig_json_path = os.path.join(script_dir, "unirig.json")
        if os.path.exists(unirig_json_path):
            print(f"Detected UniRig target, loading mapping from: {unirig_json_path}")
            with open(unirig_json_path, "r") as f:
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

    # Pre-populate from initial mapping (hardcoded BASE_BONE_MAPPING keys)
    for s_name, t_val in mapping.items():
        if s_name == "__preset__":
            continue
        s_bone = src_skel.get_bone_case_insensitive(s_name)
        if not s_bone:
            continue

        t_names = [t_val] if isinstance(t_val, str) else t_val
        for tn in t_names:
            t_bone = tgt_skel.get_bone_case_insensitive(tn)
            if t_bone:
                mapped_targets.add(t_bone.name)
                mapped_sources.add(s_bone.name.lower())
                match_count += 1

    # Strategy:
    # 1. Name-based matching for stable semantic names.
    # 2. Geometric matching for everything else (especially UniRig bone_X).

    # Pre-calculate heights for normalization
    src_h = get_skeleton_height(src_skel)
    tgt_h = get_skeleton_height(tgt_skel)
    if src_h < 0.01:
        src_h = 1.0
    if tgt_h < 0.01:
        tgt_h = 1.0

    sorted_json = sorted(json_bones.items())

    # Phase 1: Semantic Name Matching (High Confidence)
    for gen_name, aliases in sorted_json:
        if not isinstance(aliases, list):
            aliases = [aliases]

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

        if not source_bone:
            continue

        # Find Target Bone (Semantic Match)
        # For UE5/Presets, we allow mapping ONE source bone to MULTIPLE target bones (chains)
        found_names = []
        for a in aliases:
            if is_unirig and a.startswith("bone_") and a != "bone_0":
                continue  # Skip unstable UniRig indices for name matching

            found = tgt_skel.get_bone_case_insensitive(a)
            if found:
                found_names.append(found.name)
                if not (is_ue5 or is_g9 or is_g3):
                    break  # Only UE5/G9/G3/Chains need multiple targets

        if found_names and source_bone.name.lower() not in mapped_sources:
            mapping[source_bone.name.lower()] = found_names
            match_count += len(found_names)
            for fn in found_names:
                mapped_targets.add(fn)
            mapped_sources.add(source_bone.name.lower())

    print(
        f"[Retarget] load_bone_mapping: Found {match_count} matches via semantic names."
    )

    if is_ue5 or is_g9 or is_g3:
        print(
            f"[Retarget] {('UE5' if is_ue5 else ('G9' if is_g9 else 'G3'))} preset active: Skipping structural matching and returning explicit mapping ({len(mapping)} bones)."
        )
        return mapping

    # Phase 2: Structural BFS Matching (Hierarchy-Aware)
    # We start from the root and traverse down
    # Helper to check if a bone is a finger
    def is_finger(name):
        n = name.lower()
        return any(
            k in n for k in ["thumb", "index", "middle", "ring", "pinky", "finger"]
        )

    # Helper to find root/pelvis bone in target skeleton
    def find_target_root(skel):
        """Find the root/pelvis bone using multiple fallback strategies."""

        # Strategy 1: Look for bone_0 (UniRig standard)
        root = skel.get_bone_case_insensitive("bone_0")
        if root:
            print(f"[Retarget] Found target root via 'bone_0': {root.name}")
            return root

        # Strategy 2: Name-based detection
        root_names = [
            "hip",
            "hips",
            "pelvis",
            "root",
            "cog",
            "center",
            "base",
            "spine_01",
        ]
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
            if not children:
                continue

            # Count descendants recursively
            def count_descendants(bone_name, visited=None):
                if visited is None:
                    visited = set()
                if bone_name in visited:
                    return 0
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
            if child_count >= 3:
                score += 100  # Bonus for typical root structure

            if score > best_score:
                best_score = score
                best_root = bone

        if best_root:
            print(
                f"[Retarget] Found target root via topology (score {best_score}): {best_root.name}"
            )
            return best_root

        # Strategy 4: Geometric detection - find bone closest to center with children on both sides
        center_y = 0
        for bone in skel.bones.values():
            center_y = max(center_y, bone.head[1])
        center_y /= 2  # Approximate vertical center

        for name, bone in skel.bones.items():
            children = skel.get_children(bone)
            if len(children) < 2:
                continue

            # Check if children spread both left and right
            has_left = any(c.head[0] < bone.head[0] - 0.05 for c in children)
            has_right = any(c.head[0] > bone.head[0] + 0.05 for c in children)

            if has_left and has_right:
                print(
                    f"[Retarget] Found target root via geometry (bilateral children): {bone.name}"
                )
                return bone

        print(f"[Retarget] WARNING: Could not find target root bone!")
        return None

    src_root = src_skel.get_bone_case_insensitive("Pelvis")
    tgt_root = find_target_root(tgt_skel)

    if src_root and tgt_root:
        print(
            f"[Retarget] Starting Structural BFS Matching from {src_root.name} -> {tgt_root.name}..."
        )

        # Queue of (src_bone, tgt_bone) pairs to explore children of
        queue = [(src_root, tgt_root)]

        while queue:
            s_parent, t_parent = queue.pop(0)

            s_children = src_skel.get_children(s_parent.name)
            t_children = tgt_skel.get_children(t_parent.name)

            if not s_children or not t_children:
                continue

            # Filter out fingers if requested
            s_children = [c for c in s_children if not is_finger(c.name)]
            t_children = [c for c in t_children if not is_finger(c.name)]

            # Side matching check
            def get_side(v, name):
                n = name.lower()
                if "left" in n or "l_" in n or ".l" in n:
                    return -1
                if "right" in n or "r_" in n or ".r" in n:
                    return 1
                if abs(v[0]) < 0.05 * (tgt_h if v is t_rel else src_h):
                    return 0  # Height-relative threshold
                return 1 if v[0] > 0 else -1

            # Match children of s_parent to children of t_parent
            for tc in t_children:
                if tc.name in mapped_targets:
                    continue

                best_sc = None
                best_score = 999999.0

                # 1. Try Name-based Hint (Fuzzy)
                # Check if any source child matches tc's name or its semantic meaning
                for sc in s_children:
                    if sc.name.lower() in mapped_sources:
                        continue

                    # Check if sc name matches tc name (fuzzy)
                    # Or if sc name matches the gen_name associated with tc in unirig.json
                    is_match = False
                    if sc.name.lower() == tc.name.lower():
                        is_match = True
                    else:
                        for gen_name, aliases in json_bones.items():
                            if (
                                sc.name.lower() == gen_name.lower()
                                or sc.name.lower()
                                in [
                                    a.lower()
                                    for a in (
                                        aliases
                                        if isinstance(aliases, list)
                                        else [aliases]
                                    )
                                ]
                            ):
                                if tc.name in (
                                    aliases if isinstance(aliases, list) else [aliases]
                                ):
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
                        if sc.name.lower() in mapped_sources:
                            continue

                        s_rel = sc.head - s_parent.head
                        s_side = get_side(s_rel, sc.name)
                        t_rel_norm = t_rel / tgt_h
                        s_rel_norm = s_rel / src_h

                        dist = np.linalg.norm(t_rel_norm - s_rel_norm)

                        # Relaxed side matching for limb chains
                        if t_side != s_side and dist > 0.2:
                            continue

                        if dist < best_score:
                            best_score = dist
                            best_sc = sc

                # 3. Single-child Fallback: If target has exactly 1 unmatched child and source has 1 unmatched child, force match
                if (
                    not best_sc
                    and len(t_children) == 1
                    and len(
                        [c for c in s_children if c.name.lower() not in mapped_sources]
                    )
                    == 1
                ):
                    remaining_s = [
                        c for c in s_children if c.name.lower() not in mapped_sources
                    ][0]
                    best_sc = remaining_s
                    best_score = 0.99  # Mark as forced match

                # Increased threshold from 0.5 to 1.0 to allow more matches in limb chains
                if best_sc and best_score < 1.0:
                    mapping[best_sc.name.lower()] = tc.name
                    mapped_targets.add(tc.name)
                    mapped_sources.add(best_sc.name.lower())
                    match_count += 1
                    queue.append((best_sc, tc))

        print(
            f"[Retarget] Structural BFS Matching complete. Total matches: {match_count}"
        )

    # Phase 3: Hand and Finger Mapping (Phase 2 of the plan)
    # Identify mapped wrists and explore their children
    wrists = [("L_Wrist", ["left", "l_"]), ("R_Wrist", ["right", "r_"])]
    for wrist_gen_name, side_keys in wrists:
        # Find the mapped target wrist
        target_wrist_name = None
        source_wrist_name = None

        # Look for the wrist in the current mapping
        # We need to find a match where both bones actually exist in the skeletons
        for s_name, t_val in mapping.items():
            s_lower = s_name.lower()
            if any(k in s_lower for k in ["wrist", "hand"]):
                if any(sk in s_lower for sk in side_keys):
                    # Handle if t_val is a list (UE5/G9 preset)
                    t_name = t_val[0] if isinstance(t_val, list) else t_val
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

        if not s_wrist or not t_wrist:
            continue

        s_fingers = [
            c for c in src_skel.get_children(s_wrist.name) if is_finger(c.name)
        ]
        t_fingers = [
            c for c in tgt_skel.get_children(t_wrist.name)
        ]  # All children of target wrist are likely fingers

        if not t_fingers:
            continue

        # Case A: Single-Finger Rig (e.g. SK_tira)
        if len(t_fingers) == 1:
            # Map source Middle finger to this single target finger
            # Find source middle finger
            src_middle = None
            for sf in s_fingers:
                if "middle" in sf.name.lower():
                    src_middle = sf
                    break
            if not src_middle and s_fingers:
                src_middle = s_fingers[len(s_fingers) // 2]  # Fallback to middle-ish

            if src_middle:
                mapping[src_middle.name.lower()] = t_fingers[0].name
                mapped_targets.add(t_fingers[0].name)
                mapped_sources.add(src_middle.name.lower())
                match_count += 1

                # Recursively map down the chain
                s_curr, t_curr = src_middle, t_fingers[0]
                while True:
                    s_next = [
                        c
                        for c in src_skel.get_children(s_curr.name)
                        if is_finger(c.name)
                    ]
                    t_next = tgt_skel.get_children(t_curr.name)
                    if not s_next or not t_next:
                        break

                    mapping[s_next[0].name.lower()] = t_next[0].name
                    mapped_targets.add(t_next[0].name)
                    mapped_sources.add(s_next[0].name.lower())
                    match_count += 1
                    s_curr, t_curr = s_next[0], t_next[0]

        # Case B: Multi-Finger Rig
        else:
            # Use geometric matching relative to wrist
            for tf in t_fingers:
                if tf.name in mapped_targets:
                    continue

                best_sf = None
                best_score = 999999.0
                t_rel = tf.head - t_wrist.head

                for sf in s_fingers:
                    if sf.name.lower() in mapped_sources:
                        continue
                    s_rel = sf.head - s_wrist.head
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
                        s_next = [
                            c
                            for c in src_skel.get_children(s_curr.name)
                            if is_finger(c.name)
                        ]
                        t_next = tgt_skel.get_children(t_curr.name)
                        if not s_next or not t_next:
                            break

                        mapping[s_next[0].name.lower()] = t_next[0].name
                        mapped_targets.add(t_next[0].name)
                        mapped_sources.add(s_next[0].name.lower())
                        match_count += 1
                        s_curr, t_curr = s_next[0], t_next[0]

    # Phase 4: Arm Fallback Detection
    # If arms were not mapped via structural BFS, try to find them using alternative strategies
    arm_bones = ["collar", "shoulder", "elbow", "wrist"]
    for side, side_keys in [("L", ["left", "l_"]), ("R", ["right", "r_"])]:
        # Check if this side's arm is already mapped
        side_arm_mapped = any(
            any(sk in s.lower() for sk in side_keys)
            and any(ab in s.lower() for ab in arm_bones)
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
                excluded_names = [
                    "armature",
                    "character",
                    "root",
                    "scene",
                    "skeleton",
                    "rig",
                ]
                if any(
                    ex == t_lower or t_lower.startswith(ex + "_")
                    for ex in excluded_names
                ):
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
                    if "pelvis" in mapped_s.lower():
                        if isinstance(mapped_t, list):
                            for t_name in mapped_t:
                                tgt_pelvis_bone = tgt_skel.get_bone_case_insensitive(
                                    t_name
                                )
                                if tgt_pelvis_bone:
                                    break
                        else:
                            tgt_pelvis_bone = tgt_skel.get_bone_case_insensitive(
                                mapped_t
                            )
                        if tgt_pelvis_bone:
                            break

                if src_pelvis and tgt_pelvis_bone:
                    src_rel = (src_bone.head - src_pelvis.head) / src_h
                    tgt_rel = (t_bone.head - tgt_pelvis_bone.head) / tgt_h

                    # Check side consistency
                    src_side = (
                        1 if src_rel[0] > 0.05 else (-1 if src_rel[0] < -0.05 else 0)
                    )
                    tgt_side = (
                        1 if tgt_rel[0] > 0.05 else (-1 if tgt_rel[0] < -0.05 else 0)
                    )

                    if src_side != tgt_side:
                        continue

                    dist = np.linalg.norm(src_rel - tgt_rel)
                    if dist < best_score:
                        best_score = dist
                        best_tgt = t_bone

            if best_tgt and best_score < 1.0:
                if (
                    src_bone.name.lower() not in mapped_sources
                    and best_tgt.name not in mapped_targets
                ):
                    mapping[src_bone.name.lower()] = best_tgt.name
                    mapped_targets.add(best_tgt.name)
                    mapped_sources.add(src_bone.name.lower())
                    match_count += 1

    print(f"[Retarget] Final load_bone_mapping: Found {match_count} total matches.")
    return mapping


# =============================================================================
# The Retargeting Engine
# =============================================================================

def apply_retargeted_animation(
    scene, skeleton, ret_rots, ret_locs, fstart, fend, source_time_mode=None
):
    if source_time_mode:
        scene.GetGlobalSettings().SetTimeMode(source_time_mode)
    else:
        # Default to 30 FPS (HyMotion standard) if no source time mode provided (e.g. NPZ)
        scene.GetGlobalSettings().SetTimeMode(fbx.FbxTime.EMode.eFrames30)

    tmode = scene.GetGlobalSettings().GetTimeMode()
    unit = scene.GetGlobalSettings().GetSystemUnit()

    # Clear old stacks
    for i in range(
        scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(FbxAnimStack.ClassId)) - 1,
        -1,
        -1,
    ):
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
            pre_q = R.from_euler("xyz", [pq_v[0], pq_v[1], pq_v[2]], degrees=True)
            post_q = R.from_euler("xyz", [poq_v[0], poq_v[1], poq_v[2]], degrees=True)

            ord_lower = get_fbx_rotation_order_str(node).lower()

            cx = node.LclRotation.GetCurve(layer, "X", True)
            cy = node.LclRotation.GetCurve(layer, "Y", True)
            cz = node.LclRotation.GetCurve(layer, "Z", True)
            cx.KeyModifyBegin()
            cy.KeyModifyBegin()
            cz.KeyModifyBegin()

            for f, q_local in ret_rots[name].items():
                t = FbxTime()
                t.SetFrame(f - fstart, tmode)  # Normalize to start at 0

                # Logic: R_eff = R_pre * R_local * inv(R_post)
                # Thus R_local = inv(R_pre) * R_eff * R_post
                ql = R.from_quat([q_local[1], q_local[2], q_local[3], q_local[0]])
                q_final = pre_q.inv() * ql * post_q

                # Convert back to Euler matching the node's custom rotation order
                e = q_final.as_euler(ord_lower, degrees=True)

                curve_map = {"x": cx, "y": cy, "z": cz}
                for i, axis in enumerate(ord_lower):
                    c = curve_map[axis]
                    idx = c.KeyAdd(t)[0]
                    c.KeySetValue(idx, float(e[i]))
                    c.KeySetInterpolation(
                        idx, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
                    )
            cx.KeyModifyEnd()
            cy.KeyModifyEnd()
            cz.KeyModifyEnd()

        if name in ret_locs:
            node.LclTranslation.ModifyFlag(
                fbx.FbxPropertyFlags.EFlags.eAnimatable, True
            )
            tx = node.LclTranslation.GetCurve(layer, "X", True)
            ty = node.LclTranslation.GetCurve(layer, "Y", True)
            tz = node.LclTranslation.GetCurve(layer, "Z", True)
            tx.KeyModifyBegin()
            ty.KeyModifyBegin()
            tz.KeyModifyBegin()
            # Correct translation to account for FBX's intrinsic world pose offset
            rest_world_pos = skeleton.node_world_matrices.get(name, np.eye(4))[3, :3]
            pnode = node.GetParent()
            pname = pnode.GetName() if pnode else None
            rest_parent_world_pos = (
                skeleton.node_world_matrices.get(pname, np.eye(4))[3, :3]
                if pname
                else np.zeros(3)
            )

            # This is the offset that FBX evaluates even when LclTranslation is zero
            intrinsic_offset = rest_world_pos - rest_parent_world_pos

            for f, loc in ret_locs[name].items():
                t = FbxTime()
                t.SetFrame(f - fstart, tmode)  # Normalize to start at 0
                for i, c in enumerate([tx, ty, tz]):
                    idx = c.KeyAdd(t)[0]
                    # Desired translation = target_relative_pos - intrinsic_fbx_offset
                    val = loc[i] - intrinsic_offset[i]
                    c.KeySetValue(idx, float(val))
                    c.KeySetInterpolation(
                        idx, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
                    )
            tx.KeyModifyEnd()
            ty.KeyModifyEnd()
            tz.KeyModifyEnd()

        for i in range(node.GetChildCount()):
            apply_node(node.GetChild(i))

    apply_node(scene.GetRootNode())


def retarget_animation(
    src_skel: Skeleton,
    tgt_skel: Skeleton,
    mapping: dict[str, str],
    force_scale: float = 0.0,
    yaw_offset: float = 0.0,
    neutral_fingers: bool = True,
    in_place: bool = False,
    in_place_x: bool = False,
    in_place_y: bool = False,
    in_place_z: bool = False,
    preserve_position: bool = False,
    auto_stride: bool = False,
    smart_arm_align: bool = False,
):
    print("Retargeting Animation...")

    is_ue5 = mapping.get("__preset__") == "ue5"
    is_g9 = mapping.get("__preset__") == "g9"
    is_g3 = mapping.get("__preset__") == "g3"
    # Remove internal flag from mapping
    mapping = {k: v for k, v in mapping.items() if k != "__preset__"}

    ret_rots = {}
    ret_locs = {}

    # 0. Smart Arm Alignment Pre-pass
    # If enabled, we "level" A-pose limbs to T-pose by modifying the rest rotation
    # of the target skeleton before the main retargeting process begins.
    if smart_arm_align:
        print("[Retarget] Applying Smart Arm Alignment pre-pass...")
        for tb_name, tb in tgt_skel.bones.items():
            # LEVELING REFINEMENT: Only level segments that determine limb angle (upper/lower).
            # Excluding terminal segments (hand, wrist, foot, ankle) allows them to keep their forward-facing orientation.
            is_leveling_segment = any(
                x in tb.name.lower()
                for x in [
                    "upperarm",
                    "lowerarm",
                    "thigh",
                    "calf",
                    "arm",
                    "leg",
                    "elbow",
                    "knee",
                ]
            )
            is_terminal_segment = any(
                x in tb.name.lower() for x in ["hand", "wrist", "foot", "ankle"]
            )

            if is_leveling_segment and not is_terminal_segment:
                children = tgt_skel.bone_children.get(tb.name, [])
                if children:
                    # Find a child bone to determine current rest direction
                    cb_name = children[0]
                    cb = tgt_skel.bones.get(cb_name.lower())
                    if cb:
                        curr_v = cb.head - tb.head
                        if np.linalg.norm(curr_v) > 1e-6:
                            curr_v /= np.linalg.norm(curr_v)
                            is_l = any(
                                x in tb.name.lower() for x in ["l_", "_l", "left"]
                            )
                            is_leg_part = any(
                                x in tb.name.lower()
                                for x in ["thigh", "calf", "leg", "knee"]
                            )

                            if is_leg_part:
                                target_v = np.array([0.0, -1.0, 0.0])  # Legs down
                            else:
                                target_v = (
                                    np.array([1.0, 0.0, 0.0])
                                    if is_l
                                    else np.array([-1.0, 0.0, 0.0])
                                )

                            align_mat = solve_rotation_between_vectors(curr_v, target_v)
                            # FIX: solve_rotation returns a column-major matrix, so convert directly via scipy
                            align_r = R.from_matrix(align_mat)
                            align_q_raw = align_r.as_quat()
                            align_q = np.array(
                                [
                                    align_q_raw[3],
                                    align_q_raw[0],
                                    align_q_raw[1],
                                    align_q_raw[2],
                                ]
                            )
                            # Apply world-space alignment to rest rotation
                            tb.rest_rotation = quaternion_multiply(
                                align_q, tb.rest_rotation
                            )
                            tgt_skel.node_rest_rotations[tb.name] = tb.rest_rotation
                            # print(f"  [Smart Align] Leveled: {tb.name}")

    # 0.5 Finger Neutralization Pre-pass (Source-Side)
    # If source has curled fingers in rest pose, we normalize them to "straight" (identity relative to parent)
    # This ensures that 0 animation rotation = straight fingers on target.
    if neutral_fingers:
        print("[Retarget] Applying Neutral Finger pre-pass (Source)...")
        for sb_name, sb in src_skel.bones.items():
            is_finger = any(
                f in sb.name.lower()
                for f in ["index", "middle", "ring", "pinky", "thumb", "toe"]
            )
            if is_finger:
                pname = sb.parent_name
                if pname:
                    p_bone = src_skel.get_bone_case_insensitive(pname)
                    if p_bone:
                        # Make rest rotation match parent's rest rotation (Neutralizes curl)
                        sb.rest_rotation = p_bone.rest_rotation
                        # print(f"  [Neutral] Standardized rest pose: {sb.name}")

    yaw_q_raw = R.from_euler("y", yaw_offset, degrees=True).as_quat()
    yaw_q = np.array([yaw_q_raw[3], yaw_q_raw[0], yaw_q_raw[1], yaw_q_raw[2]])

    # 1. Base Mapping
    active = []
    mapped_targets = set()
    mapped_sources = set()

    for s_key, t_val in mapping.items():
        s_bone = src_skel.get_bone_case_insensitive(s_key)
        if not s_bone:
            continue

        # Support both single target string and list of multiple targets (for chains)
        t_keys = [t_val] if isinstance(t_val, str) else t_val

        for t_key in t_keys:
            t_bone = tgt_skel.get_bone_case_insensitive(t_key)
            if t_bone:
                # Skip if target bone is already part of a mapping
                if t_bone.name.lower() in mapped_targets:
                    continue

                # Use the (potentially normalized) rest rotation directly
                off = quaternion_multiply(
                    quaternion_inverse(s_bone.rest_rotation), t_bone.rest_rotation
                )
                active.append((s_bone, t_bone, off))
                mapped_targets.add(t_bone.name.lower())

        mapped_sources.add(s_bone.name.lower())

    # 2. Smart fuzzy matching for unmapped bones using advanced algorithm
    # Sort target bones: Hips/Root first, then limbs, then fingers
    def sort_key(b):
        n = b.name.lower()
        if "hips" in n or "root" in n or "pelvis" in n:
            return 0
        if "spine" in n or "chest" in n:
            return 1
        if "neck" in n or "head" in n:
            return 2
        if "leg" in n or "arm" in n:
            return 3
        return 10  # Fingers and other bones last

    tgt_list = sorted(tgt_skel.bones.values(), key=sort_key)

    # Track matches with confidence scores for reporting
    fuzzy_matches = []

    # 2. Smart fuzzy matching for unmapped bones using advanced algorithm
    # DISABLED for presets to ensure ONLY the requested bones are mapped.
    if not (is_ue5 or is_g9 or is_g3):
        for t_bone in tgt_list:
            if t_bone.name.lower() in mapped_targets:
                continue

            # Use the advanced bone matching system
            best_source_name, confidence = find_best_bone_match(
                t_bone.name, src_skel, mapped_sources, require_side_match=True
            )

            # Accept match if confidence is above threshold
            if best_source_name and confidence >= 0.5:  # 50% confidence minimum
                s_bone = src_skel.bones[best_source_name]

                # Ensure this source bone hasn't been mapped yet
                if s_bone.name.lower() in mapped_sources:
                    continue

                # Calculate offset using normalized rest rotations
                off = quaternion_multiply(
                    quaternion_inverse(s_bone.rest_rotation), t_bone.rest_rotation
                )
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
    if force_scale > 1e-4:
        # Use manual scale (e.g. 100.0 to convert Meters to CM manually)
        scale = force_scale
        print(f"[Retarget] Using Manual Base Scale: {scale:.4f}")
    else:
        # Absolute Scaling: Automatic unit conversion (M -> CM usually)
        scale = src_skel.unit_scale / tgt_skel.unit_scale
        print(
            f"[Retarget] Using Automatic Base Scale: {scale:.4f} ({'Meters to CM' if scale > 1.0 else 'CM/M matching'})"
        )

    if auto_stride:
        # Proportional Retargeting: Scale movement by height ratio
        scale *= anatomical_scale
        print(
            f"[Retarget] Applying Auto-Stride: Final Scale {scale:.4f} (Anatomical Ratio {anatomical_scale:.4f})"
        )
    else:
        print(f"[Retarget] Auto-Stride OFF: Using physical displacement.")

    print(f"[INFO] Height Check - Source: {src_h_cm:.2f}cm, Target: {tgt_h_cm:.2f}cm")
    print(f"[INFO] Final Retargeting Scale: {scale:.4f}")

    tgt_world_anims = {}
    frames = (
        sorted(list(src_skel.bones.values())[0].world_animation.keys())
        if src_skel.bones
        else []
    )
    if not frames:
        frames = range(src_skel.frame_start, src_skel.frame_end + 1)

    # 1. World Rotations and Root Displacement Reference
    root_mapped = False
    # Detect Target Up Axis (Blender is Z-Up)
    t_up_axis = np.array([0, 1, 0])  # Default Y-Up
    t_head = tgt_skel.get_bone_case_insensitive("head")
    if t_head:
        if abs(t_head.head[2]) > abs(t_head.head[1]):
            t_up_axis = np.array([0, 0, 1])

    t_up_axis_idx = np.argmax(np.abs(t_up_axis))

    # Coordinate Transform: Map Source Up ([0,1,0]) to Target Up
    if t_up_axis[2] == 1:  # Z-Up (Blender)
        # Source Y -> Target Z, Source Z -> Target -Y, Source X -> Target X
        coord_q = R.from_euler("x", 90, degrees=True)
    else:  # Y-Up
        coord_q = R.identity()

    # Realignment Logic (Calculated in Target Space)
    realignment_tgt_q = R.identity()
    t_up = t_up_axis

    # NEW: Unified robust search for alignment anchors and root bone
    t_lhip, t_rhip, t_spine = None, None, None
    s_lhip, s_rhip, s_spine = None, None, None
    s_lfoot, s_rfoot, t_lfoot, t_rfoot = None, None, None, None
    root_bone_name = None
    s_root_bone = None

    # SMPL-H / HyMotion Reference Bone Names
    ROOT_S = ["pelvis"]
    LHIP_S = ["l_hip", "leftupleg"]
    RHIP_S = ["r_hip", "rightupleg"]
    SPINE_S = ["spine3", "spine2", "spine"]
    LFOOT_S = ["l_ankle", "leftfoot", "l_foot"]
    RFOOT_S = ["r_ankle", "rightfoot", "r_foot"]

    for s_bone, t_bone_val, _ in active:
        t_bone = t_bone_val[0] if isinstance(t_bone_val, list) else t_bone_val
        if not t_bone:
            continue

        sb_name = s_bone.name.lower()

        # 1. Root Detection (Based on mapping to 'Pelvis' or keywords)
        if any(x == sb_name for x in ROOT_S) or any(
            x in t_bone.name.lower() for x in ["hips", "pelvis", "root"]
        ):
            if not root_bone_name:
                root_bone_name = t_bone.name
                s_root_bone = s_bone
                # Note: Do not print here yet to keep log clean until verified

        # 2. Alignment Anchors
        if any(x == sb_name for x in LHIP_S):
            s_lhip, t_lhip = s_bone, t_bone
        elif any(x == sb_name for x in RHIP_S):
            s_rhip, t_rhip = s_bone, t_bone
        elif sb_name in SPINE_S:
            if (
                not s_spine
                or sb_name == "spine3"
                or (sb_name == "spine2" and s_spine.name.lower() == "spine")
            ):
                s_spine, t_spine = s_bone, t_bone

        # 3. Grounding Anchors (Feet/Ankles)
        if any(x == sb_name for x in LFOOT_S):
            s_lfoot, t_lfoot = s_bone, t_bone
        elif any(x == sb_name for x in RFOOT_S):
            s_rfoot, t_rfoot = s_bone, t_bone

    if s_lhip and s_rhip and t_lhip and t_rhip:
        # Construct Source Frame
        s_mid_hip = (s_lhip.world_matrix[3, :3] + s_rhip.world_matrix[3, :3]) * 0.5
        s_up = (
            (s_spine.world_matrix[3, :3] - s_mid_hip)
            if s_spine
            else np.array([0, 1, 0])
        )
        s_side = s_rhip.world_matrix[3, :3] - s_lhip.world_matrix[3, :3]
        s_fwd = np.cross(s_side, s_up)
        s_up = np.cross(s_fwd, s_side)

        s_fwd /= np.linalg.norm(s_fwd) + 1e-9
        s_up /= np.linalg.norm(s_up) + 1e-9
        s_side /= np.linalg.norm(s_side) + 1e-9
        s_mat = np.stack((s_side, s_up, s_fwd), axis=-1)

        # Construct Target Frame
        t_mid_hip = (t_lhip.world_matrix[3, :3] + t_rhip.world_matrix[3, :3]) * 0.5
        t_up_vec = (t_spine.world_matrix[3, :3] - t_mid_hip) if t_spine else t_up_axis
        t_side = t_rhip.world_matrix[3, :3] - t_lhip.world_matrix[3, :3]
        t_fwd = np.cross(t_side, t_up_vec)
        t_up = np.cross(t_fwd, t_side)

        t_fwd /= np.linalg.norm(t_fwd) + 1e-9
        t_up /= np.linalg.norm(t_up) + 1e-9
        t_side /= np.linalg.norm(t_side) + 1e-9
        t_mat = np.stack((t_side, t_up, t_fwd), axis=-1)
        real_mat = t_mat @ s_mat.T @ coord_q.as_matrix().T
        realignment_tgt_q = R.from_matrix(real_mat)
        print(
            f"[Retarget] Computed Alignment Matrix (Euler XYZ): {realignment_tgt_q.as_euler('xyz', degrees=True)}"
        )
    else:
        realignment_tgt_q = R.from_quat([0, 0, 0, 1])
        print(
            f"[Retarget] WARNING: Unable to compute alignment matrix - missing hip/thigh bones"
        )

    # Final Global Transformation
    yaw_q = R.from_euler("y" if t_up_axis[1] == 1 else "z", yaw_offset, degrees=True)
    global_transform_q = yaw_q * realignment_tgt_q * coord_q

    # TRAJECTORY DRIFT FIX: Use ONLY Yaw for trajectory translation.
    # Extracts the horizontal heading for movement to prevent 'climbing' or 'sinking'.
    fwd_transformed = global_transform_q.apply(
        [0, 0, 1] if t_up_axis[1] == 1 else [0, 1, 0]
    )
    if t_up_axis[1] == 1:
        angle = np.arctan2(fwd_transformed[0], fwd_transformed[2])
        global_pos_q = R.from_euler("y", angle)
    else:
        angle = np.arctan2(fwd_transformed[0], fwd_transformed[1])
        global_pos_q = R.from_euler("z", angle)

    gt_q_np = np.array(
        [
            global_transform_q.as_quat()[3],
            global_transform_q.as_quat()[0],
            global_transform_q.as_quat()[1],
            global_transform_q.as_quat()[2],
        ]
    )

    # 2. Retarget World Rotations
    tgt_world_anims = {}
    for s_bone, t_bone_val, _ in active:
        # Filter None bones out of the list (e.g. if neck_03 doesn't exist)
        if isinstance(t_bone_val, list):
            t_bone_list = [b for b in t_bone_val if b is not None]
        else:
            t_bone_list = [t_bone_val] if t_bone_val is not None else []

        if not t_bone_list:
            continue

        # [ROKOKO Rebuild] Establish cross-mapping for logical child discovery
        target_to_source = {}
        for s_tmp_name, t_tmp_val in mapping.items():
            t_tmp_list = t_tmp_val if isinstance(t_tmp_val, list) else [t_tmp_val]
            for tn in t_tmp_list:
                target_to_source[tn.lower()] = s_tmp_name

        num_targets = len(t_bone_list)
        s_rest_q = s_bone.rest_rotation

        for i, tb in enumerate(t_bone_list):
            if tb.name not in tgt_world_anims:
                tgt_world_anims[tb.name] = {}

            # Calculate rest pose offset in WORLD space
            s_rest_world_q = quaternion_multiply(gt_q_np, s_rest_q)
            t_rest_world_q = tb.rest_rotation
            off_q = quaternion_multiply(
                quaternion_inverse(s_rest_world_q), t_rest_world_q
            )

            # Identify if this is a limb that needs directional alignment
            # NOTE: 'hip' is excluded to preserve original source orientation/tilt
            # NOTE: 'twist' bones are excluded because they should carry the ROLL from s_rot_f_distributed,
            #       not the vector-aiming (Aim) component which is handled by their 'bend' parent.
            is_limb = (
                any(
                    x in s_bone.name.lower()
                    for x in [
                        "knee",
                        "shoulder",
                        "elbow",
                        "thigh",
                        "shin",
                        "upperarm",
                        "lowerarm",
                    ]
                )
                and "twist" not in tb.name.lower()
            )

            # [Logical Child Fix] Find the next MAJOR joint in the mapping
            # This skips tiny twist bones and looks at the actual bend joint (e.g. Shin for Thigh)
            s_logical_child = None

            def find_mapped_descendant(bone_name, skel, target_map):
                children = skel.bone_children.get(bone_name, [])
                for cn in children:
                    if cn.lower() in target_map:
                        return skel.get_bone_case_insensitive(cn)
                    # Recursive search
                    found = find_mapped_descendant(cn, skel, target_map)
                    if found:
                        return found
                return None

            s_logical_child = find_mapped_descendant(
                s_bone.name, src_skel, [k.lower() for k in mapping.keys()]
            )
            t_logical_child = find_mapped_descendant(
                tb.name, tgt_skel, [k.lower() for k in target_to_source.keys()]
            )

            # G3 Twist Detection for proper Swing/Twist decomposition
            is_twist_bone = "twist" in tb.name.lower()
            is_bend_bone = "bend" in tb.name.lower()

            for f in frames:
                s_rot_f = s_bone.world_animation.get(f, s_rest_q)

                # Chain distribution for G3 Bend+Twist bones
                if num_targets > 1:
                    rel_q = quaternion_multiply(quaternion_inverse(s_rest_q), s_rot_f)

                    if is_twist_bone or is_bend_bone:
                        # PROPER SWING-TWIST DECOMPOSITION
                        # We need the source's limb axis in Source-Local space.
                        s_rest_world_r = R.from_quat(
                            [s_rest_q[1], s_rest_q[2], s_rest_q[3], s_rest_q[0]]
                        )

                        s_v_rest_world = (
                            s_logical_child.head - s_bone.head
                            if s_logical_child
                            else np.array([1, 0, 0])
                        )
                        norm = np.linalg.norm(s_v_rest_world)
                        if norm > 1e-6:
                            s_v_rest_world = s_v_rest_world / norm
                        else:
                            s_v_rest_world = np.array([1, 0, 0])

                        # Transform world rest axis to bone-local rest axis
                        s_local_axis = s_rest_world_r.inv().apply(s_v_rest_world)

                        swing_q, twist_q = decompose_swing_twist(rel_q, s_local_axis)

                        if is_bend_bone:
                            # Bend bones get the SWING component (primary rotation direction)
                            s_rot_f_distributed = quaternion_multiply(s_rest_q, swing_q)
                        elif is_twist_bone:
                            # Twist bones get the TWIST component (roll around bone axis)
                            # This is critical for G3 stability - twist should NOT get full rotation
                            s_rot_f_distributed = quaternion_multiply(s_rest_q, twist_q)
                        else:
                            # Fallback: use index-based distribution
                            if i < num_targets - 1:
                                s_rot_f_distributed = quaternion_multiply(
                                    s_rest_q, swing_q
                                )
                            else:
                                s_rot_f_distributed = s_rot_f
                    else:
                        # Generic chain (e.g. spine): proportional SLERP
                        # Again, the last bone should carry the full rotation
                        if i == num_targets - 1:
                            s_rot_f_distributed = s_rot_f
                        else:
                            factor = (i + 1) / num_targets
                            r_ident = R.from_quat([0, 0, 0, 1])
                            r_rel = R.from_quat(
                                [rel_q[1], rel_q[2], rel_q[3], rel_q[0]]
                            )
                            slerp = Slerp([0, 1], R.concatenate([r_ident, r_rel]))
                            step_rel_r = slerp(factor)
                            step_rel_q_raw = step_rel_r.as_quat()
                            step_rel_q = np.array(
                                [
                                    step_rel_q_raw[3],
                                    step_rel_q_raw[0],
                                    step_rel_q_raw[1],
                                    step_rel_q_raw[2],
                                ]
                            )
                            s_rot_f_distributed = quaternion_multiply(
                                s_rest_q, step_rel_q
                            )
                else:
                    s_rot_f_distributed = s_rot_f

                # VECTOR-SPACE SOLVER (POSITION-BASED)
                # Uses actual animated world positions for bone direction instead of
                # rotating rest vectors — prevents drift from rest-pose convention mismatch
                if is_limb and s_logical_child and t_logical_child:
                    # 1. Source World Vector (Animated) - from actual keypoint positions
                    s_pos_f = s_bone.world_location_animation.get(f, s_bone.head)
                    sc_pos_f = s_logical_child.world_location_animation.get(
                        f, s_logical_child.head
                    )
                    s_v_anim_raw = sc_pos_f - s_pos_f
                    s_v_anim_len = np.linalg.norm(s_v_anim_raw)

                    if s_v_anim_len > 1e-6:
                        s_v_anim = s_v_anim_raw / s_v_anim_len
                        # Transform to target coordinate space
                        s_v_anim = global_transform_q.apply(s_v_anim)

                        # 2. Target World Vector (Rest) - Towards logical child
                        t_v_rest = t_logical_child.head - tb.head
                        t_v_rest_len = np.linalg.norm(t_v_rest)

                        if t_v_rest_len > 1e-6:
                            t_v_rest = t_v_rest / t_v_rest_len

                            # 3. Solve Directional Alignment (Aim)
                            aim_mat = solve_rotation_between_vectors(t_v_rest, s_v_anim)
                            aim_r = R.from_matrix(aim_mat)

                            # 4. Compute rest-pose rotation of target bone
                            t_rest_r = R.from_quat(
                                [
                                    t_rest_world_q[1],
                                    t_rest_world_q[2],
                                    t_rest_world_q[3],
                                    t_rest_world_q[0],
                                ]
                            )

                            # 5. Extract Source Twist (limb roll around bone axis)
                            s_rot_f_world = quaternion_multiply(
                                gt_q_np, s_rot_f_distributed
                            )
                            _, s_twist_q = decompose_swing_twist(
                                s_rot_f_world, s_v_anim
                            )
                            s_twist_r = R.from_quat(
                                [s_twist_q[1], s_twist_q[2], s_twist_q[3], s_twist_q[0]]
                            )

                            # Also extract source rest twist for delta computation
                            s_rest_world_r = R.from_quat(
                                [
                                    s_rest_world_q[1],
                                    s_rest_world_q[2],
                                    s_rest_world_q[3],
                                    s_rest_world_q[0],
                                ]
                            )
                            s_v_rest_dir = s_logical_child.head - s_bone.head
                            s_v_rest_len = np.linalg.norm(s_v_rest_dir)
                            if s_v_rest_len > 1e-6:
                                s_v_rest_dir = global_transform_q.apply(
                                    s_v_rest_dir / s_v_rest_len
                                )
                            else:
                                s_v_rest_dir = s_v_anim
                            s_rest_world_q2 = quaternion_multiply(gt_q_np, s_rest_q)
                            _, s_rest_twist_q = decompose_swing_twist(
                                s_rest_world_q2, s_v_rest_dir
                            )
                            s_rest_twist_r = R.from_quat(
                                [
                                    s_rest_twist_q[1],
                                    s_rest_twist_q[2],
                                    s_rest_twist_q[3],
                                    s_rest_twist_q[0],
                                ]
                            )

                            # Delta twist = animated twist relative to rest twist
                            delta_twist_r = s_twist_r * s_rest_twist_r.inv()

                            # Final: Target = Aim * Rest * DeltaTwist
                            t_rot_f_r = delta_twist_r * aim_r * t_rest_r
                            t_rot_f_raw = t_rot_f_r.as_quat()
                            t_rot_f = np.array(
                                [
                                    t_rot_f_raw[3],
                                    t_rot_f_raw[0],
                                    t_rot_f_raw[1],
                                    t_rot_f_raw[2],
                                ]
                            )
                        else:
                            s_rot_f_world = quaternion_multiply(
                                gt_q_np, s_rot_f_distributed
                            )
                            t_rot_f = quaternion_multiply(s_rot_f_world, off_q)
                    else:
                        s_rot_f_world = quaternion_multiply(
                            gt_q_np, s_rot_f_distributed
                        )
                        t_rot_f = quaternion_multiply(s_rot_f_world, off_q)
                else:
                    # Standard Rotation Mapping for Spine, Head, Hands
                    s_rot_f_world = quaternion_multiply(gt_q_np, s_rot_f_distributed)
                    t_rot_f = quaternion_multiply(s_rot_f_world, off_q)

                # Normalize quaternion to prevent drift accumulation
                t_rot_f = t_rot_f / (np.linalg.norm(t_rot_f) + 1e-12)
                tgt_world_anims[tb.name][f] = t_rot_f

    # NOTE: Head leveling removed - the head should follow the body naturally

    # Inherit parent rotation for uanimated head/neck bones
    # If the source head has identity rotation while the body is rotated,
    # the head animation was not properly baked. We fix this by inheriting
    # the parent's rotation.
    for s_bone, t_bone_val, _ in active:
        t_bone_list = t_bone_val if isinstance(t_bone_val, list) else [t_bone_val]
        # Check if any bone in this mapping is head or neck
        is_head_neck = any(
            any(x in tb.name.lower() for x in ["head", "neck"]) for tb in t_bone_list
        )

        if is_head_neck:
            # Check if the source head is uanimated (stays at rest for all frames)
            s_rest_q = s_bone.rest_rotation
            is_uanimated = True
            for f in frames[: min(10, len(frames))]:  # Check first 10 frames
                s_rot_f = s_bone.world_animation.get(f, s_rest_q)
                if np.linalg.norm(s_rot_f - s_rest_q) > 0.01:
                    is_uanimated = False
                    break

            if is_uanimated:
                print(
                    f"[FIX] {s_bone.name} detected as uanimated - inheriting parent rotation"
                )
                # Find parent bone
                pname = s_bone.parent_name
                if pname:
                    p_bone = src_skel.get_bone_case_insensitive(pname)
                    if p_bone:
                        for f in frames:
                            # Get parent's animated world rotation
                            p_rot_f = p_bone.world_animation.get(
                                f, p_bone.rest_rotation
                            )
                            # Transform to target space
                            p_rot_tgt = quaternion_multiply(gt_q_np, p_rot_f)

                            for tb in t_bone_list:
                                # Apply the same offset as we would for the head
                                off_q = quaternion_multiply(
                                    quaternion_inverse(
                                        quaternion_multiply(gt_q_np, s_rest_q)
                                    ),
                                    tb.rest_rotation,
                                )
                                t_rot_f = quaternion_multiply(p_rot_tgt, off_q)
                                tgt_world_anims[tb.name][f] = t_rot_f

    # 5. Displacement and Local Rotations
    # root_bone_name is already correctly detected from mapping/keywords above.
    print(f"[Retarget] Identified Target Root for Translation: {root_bone_name}")

    for s_bone, t_bone_val, _ in active:
        # Handle list or single bone
        t_bone_list = t_bone_val if isinstance(t_bone_val, list) else [t_bone_val]
        t_bone_main = t_bone_list[0]

        # Robust root name check
        is_root = t_bone_main.name == root_bone_name

        if is_root:
            ret_locs[t_bone_main.name] = {}

            # DELTA-BASED TRANSLATION
            # Instead of computing absolute positions (which causes floating due to
            # different body proportions), we:
            # 1. Start at the target's rest position
            # 2. Apply the source's per-frame DELTA (relative to frame 0) scaled appropriately
            # 3. Add a grounding correction to align feet with ground

            # Source reference position at frame 0
            s_rest_pos = s_bone.world_matrix[3, :3]
            s_pos_0 = s_bone.world_location_animation.get(frames[0], s_rest_pos)

            # Target rest position (where the hip should be at rest)
            t_rest_pos = t_bone_main.world_matrix[3, :3]

            # Grounding: Calculate how much to shift vertically so feet touch the ground
            cur_s_lfoot = s_lfoot or src_skel.get_bone_case_insensitive("L_Ankle")
            cur_s_rfoot = s_rfoot or src_skel.get_bone_case_insensitive("R_Ankle")

            # Target floor: minimum foot Y in rest pose
            cur_t_lfoot = t_lfoot or tgt_skel.get_bone_case_insensitive("leftfoot")
            cur_t_rfoot = t_rfoot or tgt_skel.get_bone_case_insensitive("rightfoot")
            t_floor = 0.0
            if cur_t_lfoot or cur_t_rfoot:
                foot_vals = []
                if cur_t_lfoot:
                    foot_vals.append(cur_t_lfoot.head[t_up_axis_idx])
                if cur_t_rfoot:
                    foot_vals.append(cur_t_rfoot.head[t_up_axis_idx])
                t_floor = min(foot_vals) if foot_vals else 0.0

            # Source floor at frame 0 (in target units)
            s_floor_f0_tgt = 0.0
            if cur_s_lfoot or cur_s_rfoot:
                s_foot_at_f0 = []
                for sf in [cur_s_lfoot, cur_s_rfoot]:
                    if sf:
                        p = sf.world_location_animation.get(frames[0], sf.head)
                        p_tgt = global_pos_q.apply(p * scale)
                        s_foot_at_f0.append(p_tgt[t_up_axis_idx])
                s_floor_f0_tgt = min(s_foot_at_f0) if s_foot_at_f0 else 0.0

            # Source hip at frame 0 in target space (for reference/delta computation)
            s_hip_f0_tgt = global_pos_q.apply(s_pos_0 * scale)

            # CORRECT INITIAL POSITION:
            # Use the TARGET'S rest hip position as anchor.
            # The source's vertical movement (delta from frame 0) is applied ON TOP
            # of the target's rest position, so the character maintains its native height.
            initial_pos = t_rest_pos.copy()

            # Horizontal centering: center based on feet centroid at frame 0
            foot_pos_0 = []
            for sf in [cur_s_lfoot, cur_s_rfoot]:
                if sf:
                    p = sf.world_location_animation.get(frames[0], sf.head)
                    foot_pos_0.append(global_pos_q.apply(p * scale))

            if foot_pos_0:
                s_center_0 = np.mean(foot_pos_0, axis=0)
            else:
                s_center_0 = s_hip_f0_tgt

            # Keep horizontal at target rest position
            for i in range(3):
                if i != t_up_axis_idx:
                    initial_pos[i] = t_rest_pos[i] + (s_hip_f0_tgt[i] - s_center_0[i])

            print(f"[Retarget] Target rest hip pos: {t_rest_pos.round(2)}")
            print(f"[Retarget] Initial hip pos: {initial_pos.round(2)}")

            if preserve_position:
                print(f"[Retarget] Preserving absolute world position.")

            for f in frames:
                s_pos_f = s_bone.world_location_animation.get(f, s_rest_pos)

                # Delta from frame 0 (in source space, scaled and rotated to target)
                s_delta = s_pos_f - s_pos_0
                t_delta = global_pos_q.apply(s_delta * scale)

                # Target position = initial grounded position + delta
                t_pos_f = initial_pos + t_delta

                if preserve_position:
                    t_pos_f = global_pos_q.apply(s_pos_f * scale)

                # Lock movement based on granular toggles or the legacy in_place flag
                lock_x = in_place_x or in_place
                lock_z = in_place_z or in_place
                lock_y = in_place_y

                if t_up_axis[1] == 1:  # Y-Up
                    if lock_x:
                        t_pos_f[0] = t_rest_pos[0]
                    if lock_z:
                        t_pos_f[2] = t_rest_pos[2]
                    if lock_y:
                        t_pos_f[1] = t_rest_pos[1]
                else:  # Z-Up
                    if lock_z:
                        t_pos_f[2] = t_rest_pos[2]
                    if lock_x:
                        t_pos_f[0] = t_rest_pos[0]
                    if lock_y:
                        t_pos_f[1] = t_rest_pos[1]

                # Convert to parent-local space
                pname = t_bone_main.parent_name
                p_world_mat = tgt_skel.node_world_matrices.get(pname, np.eye(4))
                p_rot_q = tgt_world_anims.get(pname, {}).get(f)

                if p_rot_q is not None:
                    p_rot_r = R.from_quat(
                        [p_rot_q[1], p_rot_q[2], p_rot_q[3], p_rot_q[0]]
                    )
                else:
                    p_rest_q = tgt_skel.node_rest_rotations.get(
                        pname, np.array([1, 0, 0, 0])
                    )
                    p_rot_r = R.from_quat(
                        [p_rest_q[1], p_rest_q[2], p_rest_q[3], p_rest_q[0]]
                    )

                local_pos_f = p_rot_r.inv().apply(t_pos_f - p_world_mat[3, :3])

                ret_locs[t_bone_main.name][f] = local_pos_f


    bone_parent_map = {}
    for s_bone_active, t_bone_val, _ in active:
        t_bone_list = t_bone_val if isinstance(t_bone_val, list) else [t_bone_val]
        for tb in t_bone_list:
            bone_parent_map[tb.name] = tb.parent_name

    # Also need parent info for unmapped bones in the chain
    for bname, bdata in tgt_skel.bones.items():
        if bdata.name not in bone_parent_map:
            bone_parent_map[bdata.name] = bdata.parent_name

    # Cache for computed FBX world rotations
    fbx_world_cache = {}  # (bone_name, frame) -> quaternion

    def get_fbx_world(bone_name, frame):
        """Compute what the FBX renderer would calculate as this bone's world rotation."""
        cache_key = (bone_name, frame)
        if cache_key in fbx_world_cache:
            return fbx_world_cache[cache_key]

        # If this bone is mapped and has a retargeted world rotation,
        # its FBX world IS that world rotation (that's our goal)
        if bone_name in tgt_world_anims and frame in tgt_world_anims[bone_name]:
            result = tgt_world_anims[bone_name][frame]
            fbx_world_cache[cache_key] = result
            return result

        # Unmapped bone: its FBX world = parent_fbx_world * rest_local
        parent = tgt_skel.all_node_parents.get(bone_name)
        if parent is None or parent == bone_name:
            # Root node - return identity or rest rotation
            result = tgt_skel.node_rest_rotations.get(bone_name, np.array([1, 0, 0, 0]))
            fbx_world_cache[cache_key] = result
            return result

        parent_world = get_fbx_world(parent, frame)

        # This bone's rest local = inv(parent_rest_world) * this_rest_world
        parent_rest = tgt_skel.node_rest_rotations.get(parent, np.array([1, 0, 0, 0]))
        bone_rest = tgt_skel.node_rest_rotations.get(bone_name, np.array([1, 0, 0, 0]))
        rest_local = quaternion_multiply(quaternion_inverse(parent_rest), bone_rest)

        result = quaternion_multiply(parent_world, rest_local)
        result = result / (np.linalg.norm(result) + 1e-12)
        fbx_world_cache[cache_key] = result
        return result

    # Cache for computed FBX world positions
    fbx_world_pos_cache = {}

    def get_fbx_world_pos(bone_name, frame):
        """Compute what the FBX renderer would calculate as this bone's world position."""
        cache_key = (bone_name, frame)
        if cache_key in fbx_world_pos_cache:
            return fbx_world_pos_cache[cache_key]

        # 1. Is this bone animated in translation (The Root)?
        if bone_name in ret_locs and frame in ret_locs[bone_name]:
            l_pos = ret_locs[bone_name][frame]
            
            pname = tgt_skel.all_node_parents.get(bone_name)
            if pname and pname != bone_name:
                p_pos = get_fbx_world_pos(pname, frame)
                q = get_fbx_world(pname, frame)
                if q is not None:
                    p_rot = R.from_quat([q[1], q[2], q[3], q[0]])
                else:
                    p_rot = R.identity()
                result = p_pos + p_rot.apply(l_pos)
            else:
                # Absolute root animation
                result = l_pos
            
            fbx_world_pos_cache[cache_key] = result
            return result

        # 2. Normal bone or Meta-node: WorldPos = ParentWorldPos + ParentWorldRot * LocalRestTranslation
        pname = tgt_skel.all_node_parents.get(bone_name)
        if pname is None or pname == bone_name:
            # This is an absolute scene root node or a node with no tracked parent
            # Use its rest world position as the absolute anchor
            if bone_name in tgt_skel.node_world_matrices:
                result = tgt_skel.node_world_matrices[bone_name][3, :3]
            else:
                result = np.zeros(3)
            fbx_world_pos_cache[cache_key] = result
            return result

        p_pos = get_fbx_world_pos(pname, frame)
        p_rot_q = get_fbx_world(pname, frame)
        p_rot = R.from_quat([p_rot_q[1], p_rot_q[2], p_rot_q[3], p_rot_q[0]])
        
        # We need the local rest translation relative to the ACTUAL FBX parent
        # tgt_skel.node_local_matrices contains this for all nodes
        if bone_name in tgt_skel.node_local_matrices:
            l_transl = tgt_skel.node_local_matrices[bone_name][3, :3]
        else:
            # Fallback if node not found
            result = p_pos 
            fbx_world_pos_cache[cache_key] = result
            return result
            
        result = p_pos + p_rot.apply(l_transl)
        fbx_world_pos_cache[cache_key] = result
        return result

    for s_bone, t_bone_val, _ in active:
        t_bone_list = t_bone_val if isinstance(t_bone_val, list) else [t_bone_val]

        for i, tb in enumerate(t_bone_list):
            ret_rots[tb.name] = {}
            pname = tgt_skel.all_node_parents.get(tb.name)

            for f in frames:
                # Get what the FBX parent's world rotation will be at this frame
                if pname:
                    parent_world_f = get_fbx_world(pname, f)
                else:
                    parent_world_f = np.array([1, 0, 0, 0])

                # World to local: local = inv(parent_world) * desired_world
                desired_world = tgt_world_anims[tb.name][f]
                l_rot = quaternion_multiply(
                    quaternion_inverse(parent_world_f), desired_world
                )
                # Normalize to prevent floating-point drift
                l_rot = l_rot / (np.linalg.norm(l_rot) + 1e-12)
                ret_rots[tb.name][f] = l_rot

    # --- NEW: UE5 IK Bone Virtual Snapping ---
    # We use an ordered dictionary to ensure parents (like gun/root) are snapped first
    ik_map = [
        ("ik_hand_gun", "hand_r"),
        ("ik_hand_l", "hand_l"),
        ("ik_hand_r", "hand_r"),
        ("ik_foot_l", "foot_l"),
        ("ik_foot_r", "foot_r"),
    ]
    
    for ik_name, bio_name in ik_map:
        t_ik = tgt_skel.get_bone_case_insensitive(ik_name)
        t_bio = tgt_skel.get_bone_case_insensitive(bio_name)
        
        if t_ik and t_bio:
            print(f"[Retarget] Snapping IK bone {t_ik.name} to {t_bio.name}")
            ret_rots[t_ik.name] = {}
            ret_locs[t_ik.name] = {}
            
            pname = t_ik.parent_name
            for f in frames:
                # 1. Rotation Snapping
                bio_world_q = tgt_world_anims.get(t_bio.name, {}).get(f)
                if bio_world_q is None:
                    bio_world_q = tgt_skel.node_rest_rotations.get(t_bio.name, np.array([1, 0, 0, 0]))
                
                if pname:
                    parent_world_f = get_fbx_world(pname, f)
                else:
                    parent_world_f = np.array([1, 0, 0, 0])
                
                l_rot = quaternion_multiply(quaternion_inverse(parent_world_f), bio_world_q)
                ret_rots[t_ik.name][f] = l_rot / (np.linalg.norm(l_rot) + 1e-12)

                # 2. Translation Snapping
                bio_world_pos = get_fbx_world_pos(t_bio.name, f)
                if pname:
                    p_pos = get_fbx_world_pos(pname, f)
                    p_rot_q = get_fbx_world(pname, f)
                    p_rot = R.from_quat([p_rot_q[1], p_rot_q[2], p_rot_q[3], p_rot_q[0]])
                    
                    # local = inv(p_rot) * (world - p_pos)
                    l_pos = p_rot.inv().apply(bio_world_pos - p_pos)
                    ret_locs[t_ik.name][f] = l_pos
                else:
                    ret_locs[t_ik.name][f] = bio_world_pos

    return ret_rots, ret_locs, active

# =============================================================================
# Scene & Saving Utilities
# =============================================================================


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
                    if texture and hasattr(texture, "GetFileName"):
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
                            relative_path = os.path.join(
                                f"{fbx_base}_textures", filename
                            )
                            texture.SetFileName(relative_path)
                            texture.SetRelativeFileName(relative_path)

    if copied_count > 0:
        print(
            f"[Retarget] Copied {copied_count} texture file(s) to {os.path.basename(texture_dir)}"
        )

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
        ios.SetBoolProp(EXP_FBX_MATERIAL, True)  # Include materials
        ios.SetBoolProp(EXP_FBX_TEXTURE, True)  # Include textures
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
        raise RuntimeError(
            f"Failed to initialize FBX exporter: {exporter.GetStatus().GetErrorString()}"
        )



    # Export the scene
    if not exporter.Export(scene):
        raise RuntimeError(
            f"Failed to export FBX: {exporter.GetStatus().GetErrorString()}"
        )

    exporter.Destroy()
    print(f"[Retarget] Saved FBX to: {path}")

# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", required=True)
    parser.add_argument("--target", "-t", required=True)
    parser.add_argument(
        "--mapping",
        "-m",
        default="",
        help="Optional bone mapping file (uses hardcoded mappings if not provided)",
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--yaw", "-y", type=float, default=0.0)
    parser.add_argument("--scale", "-sc", type=float, default=0.0)
    parser.add_argument(
        "--no-neutral",
        dest="neutral",
        action="store_false",
        help="Disable neutral finger rest-pose",
    )
    parser.add_argument(
        "--in-place", action="store_true", help="Lock horizontal movement"
    )
    parser.add_argument(
        "--auto-stride",
        action="store_true",
        help="Scale movement to target proportions (Relative retargeting)",
    )
    parser.add_argument(
        "--smart-arm-align",
        action="store_true",
        help="Automatically correct A-pose limbs to T-pose (Fixes shriveled arms)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=None,
        help="Force bake FBX pre-rotations and pivots (Blender-style normalization).",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Force disable rig normalization.",
    )
    parser.set_defaults(neutral=True)
    args = parser.parse_args()

    if args.source.lower().endswith(".npz"):
        print(f"Loading NPZ Source: {args.source}")
        src_man, src_scene = None, None
        src_skel = load_npz(args.source)
    else:
        # Sampling characters often use frame 0 as bind, but real FBX files have a Bind Pose
        # reachable by setting current animation stack to None.
        src_man, src_scene, src_skel = load_fbx(
            args.source, sample_rest_frame=None, use_bind_pose=True
        )
        src_h = get_skeleton_height(src_skel)
        # FALLBACK: If Bind Pose is collapsed (height ~0), try frame 0
        if src_h < 0.1:
            src_man, src_scene, _ = load_fbx(
                args.source, sample_rest_frame=0, use_bind_pose=True
            )
            # Refresh skeleton with sampled data
            src_skel = Skeleton(os.path.basename(args.source))
            collect_skeleton_nodes(
                src_scene.GetRootNode(), src_skel, sampling_time=FbxTime()
            )

        extract_animation(src_scene, src_skel)

    # For target skeleton, disable bind pose as some FBX files (esp. Daz G9)
    # have bind poses with incorrect/identity rotations for bones
    tgt_man, tgt_scene, tgt_skel = load_fbx(args.target, use_bind_pose=False, auto_normalize=args.normalize)


    mapping = load_bone_mapping(args.mapping, src_skel, tgt_skel)

    rots, locs, active = retarget_animation(
        src_skel,
        tgt_skel,
        mapping,
        force_scale=args.scale,
        yaw_offset=args.yaw,
        neutral_fingers=args.neutral,
        in_place=args.in_place,
        auto_stride=args.auto_stride,
        smart_arm_align=args.smart_arm_align,
    )

    print(f"\n[Retarget] Bone Mapping Results:\n  Total: {len(active)} bones mapped")
    if len(active) < 15:
        print(
            f"  [WARNING] Very few bones mapped ({len(active)}). Retargeting might be poor for high-fidelity animations."
        )

    src_time_mode = (
        src_scene.GetGlobalSettings().GetTimeMode()
        if src_scene
        else tgt_scene.GetGlobalSettings().GetTimeMode()
    )
    apply_retargeted_animation(
        tgt_scene,
        tgt_skel,
        rots,
        locs,
        src_skel.frame_start,
        src_skel.frame_end,
        src_time_mode,
    )

    save_fbx(tgt_man, tgt_scene, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
