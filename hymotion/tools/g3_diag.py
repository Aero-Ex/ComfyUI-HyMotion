"""G3 Retargeting Diagnostic - Dumps rotation data to find distortion cause."""
import sys, os, numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(__file__))
from retarget_fbx import (
    load_npz, load_fbx, load_bone_mapping, get_skeleton_height,
    matrix_to_quaternion, quaternion_multiply, quaternion_inverse,
    extract_animation
)

SRC = "/home/aero/comfy/ComfyUI/output/hymotion_npz/motiondata_20260128_162755994_fd930663_000.npz"
TGT = "/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx"

print("=== LOADING SOURCE (NPZ) ===")
src_skel = load_npz(SRC)

print(f"\n=== SOURCE SKELETON INFO ===")
print(f"Bones: {len(src_skel.bones)}")
print(f"Frames: {src_skel.frame_start}-{src_skel.frame_end}")
print(f"Unit Scale: {src_skel.unit_scale}")

# Check source rotations for sanity
key_bones = ['Pelvis', 'L_Hip', 'L_Knee', 'Spine1', 'L_Shoulder', 'L_Elbow']
print(f"\n=== SOURCE REST ROTATIONS ===")
for bn in key_bones:
    b = src_skel.get_bone_case_insensitive(bn)
    if b:
        print(f"  {bn}: rest_q={b.rest_rotation}, head={b.head}")

print(f"\n=== SOURCE WORLD ROTATIONS (frames 0, 15, 30) ===")
for bn in key_bones:
    b = src_skel.get_bone_case_insensitive(bn)
    if b:
        for f in [0, 15, 30]:
            wq = b.world_animation.get(f)
            wpos = b.world_location_animation.get(f)
            if wq is not None:
                r = R.from_quat([wq[1], wq[2], wq[3], wq[0]])
                euler = r.as_euler('xyz', degrees=True)
                print(f"  {bn} f{f}: euler={euler.round(1)}, pos={wpos.round(3) if wpos is not None else 'N/A'}")

# Check for rotation magnitude explosion (drift indicator)
print(f"\n=== SOURCE ROTATION MAGNITUDE CHECK ===")
for bn in key_bones:
    b = src_skel.get_bone_case_insensitive(bn)
    if b:
        norms = []
        for f in range(src_skel.frame_start, min(src_skel.frame_end+1, 50)):
            wq = b.world_animation.get(f)
            if wq is not None:
                norms.append(np.linalg.norm(wq))
        if norms:
            print(f"  {bn}: quat_norm min={min(norms):.6f} max={max(norms):.6f} mean={np.mean(norms):.6f}")

print(f"\n=== LOADING TARGET (FBX) ===")
tgt_man, tgt_scene, tgt_skel = load_fbx(TGT, use_bind_pose=False)
print(f"Bones: {len(tgt_skel.bones)}")
print(f"Unit Scale: {tgt_skel.unit_scale}")
print(f"Height: {get_skeleton_height(tgt_skel):.4f}")

g3_key_bones = ['hip', 'lThighBend', 'lThighTwist', 'lShin', 'lFoot',
                'abdomenLower', 'abdomenUpper', 'chestLower',
                'lShldrBend', 'lShldrTwist', 'lForearmBend', 'lForearmTwist', 'lHand']

print(f"\n=== TARGET REST ROTATIONS (G3 Key Bones) ===")
for bn in g3_key_bones:
    b = tgt_skel.get_bone_case_insensitive(bn)
    if b:
        r = R.from_quat([b.rest_rotation[1], b.rest_rotation[2], b.rest_rotation[3], b.rest_rotation[0]])
        euler = r.as_euler('xyz', degrees=True)
        print(f"  {bn}: euler={euler.round(1)}, head={b.head.round(3)}, parent={b.parent_name}")

print(f"\n=== TARGET BONE HIERARCHY (G3 Leg Chain) ===")
for bn in ['hip', 'lThighBend', 'lThighTwist', 'lShin', 'lFoot', 'lToe']:
    b = tgt_skel.get_bone_case_insensitive(bn)
    if b:
        children = tgt_skel.bone_children.get(b.name, [])
        print(f"  {bn} -> children: {children}")

print(f"\n=== TARGET WORLD MATRICES (G3 Key Bones) ===")
for bn in g3_key_bones:
    b = tgt_skel.get_bone_case_insensitive(bn)
    if b:
        wm = b.world_matrix
        print(f"  {bn}: pos={wm[3,:3].round(3)}")
        # Check if world matrix rotation part is identity-ish (T-pose check)
        rot_part = wm[:3,:3]
        # Check if rows are unit length
        row_norms = [np.linalg.norm(rot_part[i]) for i in range(3)]
        print(f"    row_norms={[round(n,4) for n in row_norms]}")

# Now check the mapping
print(f"\n=== BONE MAPPING ===")
mapping = load_bone_mapping('', src_skel, tgt_skel)
preset = mapping.get('__preset__', 'unknown')
print(f"Preset: {preset}")
clean_mapping = {k:v for k,v in mapping.items() if k != '__preset__'}
for sk, tv in sorted(clean_mapping.items()):
    sb = src_skel.get_bone_case_insensitive(sk)
    if sb:
        print(f"  {sk} -> {tv}")

# Critical test: compute offset quaternion for key pairs and check sanity
print(f"\n=== OFFSET QUATERNIONS (Source Rest vs Target Rest) ===")
test_pairs = [
    ('Pelvis', 'hip'),
    ('L_Hip', 'lThighBend'),
    ('L_Knee', 'lShin'),
    ('Spine1', 'abdomenLower'),
    ('L_Shoulder', 'lShldrBend'),
    ('L_Elbow', 'lForearmBend'),
]
for s_name, t_name in test_pairs:
    sb = src_skel.get_bone_case_insensitive(s_name)
    tb = tgt_skel.get_bone_case_insensitive(t_name)
    if sb and tb:
        off = quaternion_multiply(quaternion_inverse(sb.rest_rotation), tb.rest_rotation)
        r = R.from_quat([off[1], off[2], off[3], off[0]])
        euler = r.as_euler('xyz', degrees=True)
        print(f"  {s_name}->{t_name}: offset_euler={euler.round(1)}")

# Test: What does the vector solver produce vs standard for frame 0 and 30
print(f"\n=== VECTOR SOLVER COMPARISON (L_Hip->lThighBend) ===")
sb = src_skel.get_bone_case_insensitive('L_Hip')
sc = src_skel.get_bone_case_insensitive('L_Knee')
tb = tgt_skel.get_bone_case_insensitive('lThighBend')
tc = tgt_skel.get_bone_case_insensitive('lShin')
if sb and sc and tb and tc:
    for f in [0, 15, 30, 45]:
        s_pos = sb.world_location_animation.get(f, sb.head)
        sc_pos = sc.world_location_animation.get(f, sc.head)
        s_v = sc_pos - s_pos
        s_v_len = np.linalg.norm(s_v)
        if s_v_len > 1e-6:
            s_v_norm = s_v / s_v_len
        else:
            s_v_norm = np.array([0,0,0])
        
        t_v = tc.head - tb.head
        t_v_len = np.linalg.norm(t_v)
        if t_v_len > 1e-6:
            t_v_norm = t_v / t_v_len
        else:
            t_v_norm = np.array([0,0,0])
        
        dot = np.dot(s_v_norm, t_v_norm)
        print(f"  f{f}: src_dir={s_v_norm.round(3)}, tgt_rest_dir={t_v_norm.round(3)}, dot={dot:.4f}")

# Check: Are source world positions changing frame to frame?
print(f"\n=== SOURCE POSITION TRAVEL (should change per frame) ===")
for bn in ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle']:
    b = src_skel.get_bone_case_insensitive(bn)
    if b:
        p0 = b.world_location_animation.get(0, b.head)
        p30 = b.world_location_animation.get(30, b.head)
        diff = np.linalg.norm(p30 - p0)
        print(f"  {bn}: f0={p0.round(3)}, f30={p30.round(3)}, travel={diff:.4f}")

print("\n=== DIAGNOSTIC COMPLETE ===")
