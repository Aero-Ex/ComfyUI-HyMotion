"""Rotation Drift Diagnostic - Traces retargeted rotations frame by frame."""
import sys, os, numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(__file__))
from retarget_fbx import (
    load_npz, load_fbx, load_bone_mapping, retarget_animation,
    get_skeleton_height, quaternion_multiply, quaternion_inverse
)

SRC = "/home/aero/comfy/ComfyUI/output/hymotion_npz/motiondata_20260128_162755994_fd930663_000.npz"
TGT = "/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx"

src_skel = load_npz(SRC)
tgt_man, tgt_scene, tgt_skel = load_fbx(TGT, use_bind_pose=False)
mapping = load_bone_mapping('', src_skel, tgt_skel)

rots, locs, active = retarget_animation(
    src_skel, tgt_skel, mapping,
    smart_arm_align=True, auto_stride=True
)

# Check key bones for rotation drift
check_bones = ['hip', 'lThighBend', 'lThighTwist', 'lShin', 'lFoot',
               'abdomenLower', 'abdomenUpper', 'chestLower', 'chestUpper',
               'lShldrBend', 'lShldrTwist', 'lForearmBend', 'lHand',
               'neckLower', 'head']

print("\n" + "="*80)
print("ROTATION DRIFT ANALYSIS")
print("="*80)

for bone_name in check_bones:
    if bone_name not in rots:
        print(f"\n{bone_name}: NOT IN RETARGETED OUTPUT")
        continue
    
    bone_rots = rots[bone_name]
    frames = sorted(bone_rots.keys())
    if not frames:
        continue
    
    # Check quaternion norms and euler angle ranges
    norms = []
    eulers = []
    for f in frames:
        q = bone_rots[f]
        norms.append(np.linalg.norm(q))
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        eulers.append(r.as_euler('xyz', degrees=True))
    
    eulers = np.array(eulers)
    norms = np.array(norms)
    
    # Detect drift: large euler angle changes between consecutive frames
    max_delta = 0
    max_delta_frame = 0
    for i in range(1, len(frames)):
        delta = np.max(np.abs(eulers[i] - eulers[i-1]))
        if delta > max_delta:
            max_delta = delta
            max_delta_frame = frames[i]
    
    # Detect if rotation magnitude is growing over time
    euler_mag = np.linalg.norm(eulers, axis=1)
    mag_trend = euler_mag[-1] - euler_mag[0] if len(euler_mag) > 1 else 0
    
    print(f"\n{bone_name}:")
    print(f"  Frames: {frames[0]}-{frames[-1]} ({len(frames)} frames)")
    print(f"  Quat norm: min={norms.min():.6f} max={norms.max():.6f}")
    print(f"  Euler X range: [{eulers[:,0].min():.1f}, {eulers[:,0].max():.1f}]")
    print(f"  Euler Y range: [{eulers[:,1].min():.1f}, {eulers[:,1].max():.1f}]")
    print(f"  Euler Z range: [{eulers[:,2].min():.1f}, {eulers[:,2].max():.1f}]")
    print(f"  Max frame-to-frame delta: {max_delta:.1f}deg at frame {max_delta_frame}")
    print(f"  Magnitude trend (last-first): {mag_trend:.1f}deg")
    
    # Print first few frames and around drift point
    print(f"  Frame-by-frame (first 5 + around max delta):")
    show_frames = list(range(min(5, len(frames))))
    # Add frames around max delta
    max_idx = frames.index(max_delta_frame) if max_delta_frame in frames else 0
    for offset in [-2, -1, 0, 1, 2]:
        idx = max_idx + offset
        if 0 <= idx < len(frames) and idx not in show_frames:
            show_frames.append(idx)
    show_frames.sort()
    
    for idx in show_frames:
        f = frames[idx]
        q = bone_rots[f]
        e = eulers[idx]
        delta_str = ""
        if idx > 0:
            d = np.max(np.abs(eulers[idx] - eulers[idx-1]))
            delta_str = f" (Δ={d:.1f}°)"
        print(f"    f{f}: euler=[{e[0]:7.1f}, {e[1]:7.1f}, {e[2]:7.1f}]{delta_str}")

# Also check: source world rotations for same bones
print("\n" + "="*80)
print("SOURCE WORLD ROTATION CHECK (for comparison)")
print("="*80)

src_check = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Shoulder', 'L_Elbow', 'Spine1']
for bn in src_check:
    b = src_skel.get_bone_case_insensitive(bn)
    if b:
        print(f"\n{bn}:")
        for f in [0, 5, 10, 15, 20, 25, 30]:
            wq = b.world_animation.get(f)
            if wq is not None:
                r = R.from_quat([wq[1], wq[2], wq[3], wq[0]])
                e = r.as_euler('xyz', degrees=True)
                wpos = b.world_location_animation.get(f, b.head)
                print(f"    f{f}: euler=[{e[0]:7.1f}, {e[1]:7.1f}, {e[2]:7.1f}], pos={wpos.round(3)}")

print("\n=== DONE ===")
