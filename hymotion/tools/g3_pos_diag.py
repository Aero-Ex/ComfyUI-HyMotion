"""Position diagnostic - traces hip Y values to find floating cause."""
import sys, os, numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(__file__))
from retarget_fbx import (
    load_npz, load_fbx, load_bone_mapping, retarget_animation, get_skeleton_height
)

SRC = "/home/aero/comfy/ComfyUI/output/hymotion_npz/motiondata_20260128_162755994_fd930663_000.npz"
TGT = "/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx"

src_skel = load_npz(SRC)
tgt_man, tgt_scene, tgt_skel = load_fbx(TGT, use_bind_pose=False)
mapping = load_bone_mapping('', src_skel, tgt_skel)

rots, locs, active = retarget_animation(
    src_skel, tgt_skel, mapping, smart_arm_align=True, auto_stride=True
)

# Check source hip position (Y = height in source)
s_pelvis = src_skel.get_bone_case_insensitive('Pelvis')
s_lfoot = src_skel.get_bone_case_insensitive('L_Ankle')
s_rfoot = src_skel.get_bone_case_insensitive('R_Ankle')

print("\n" + "="*80)
print("SOURCE POSITIONS (in meters, Y-up)")
print("="*80)
for f in range(0, min(180, src_skel.frame_end+1), 5):
    p = s_pelvis.world_location_animation.get(f, s_pelvis.head)
    lf = s_lfoot.world_location_animation.get(f, s_lfoot.head) if s_lfoot else None
    rf = s_rfoot.world_location_animation.get(f, s_rfoot.head) if s_rfoot else None
    min_foot = min(lf[1] if lf is not None else 999, rf[1] if rf is not None else 999)
    print(f"  f{f:3d}: pelvis_y={p[1]:.3f}m, min_foot_y={min_foot:.3f}m, pelvis_above_feet={p[1]-min_foot:.3f}m")

# Check the retargeted translation output
print("\n" + "="*80)
print("RETARGETED ROOT TRANSLATION (in target units, should be cm)")
print("="*80)
# Find which bone got translations
for bone_name, frame_locs in locs.items():
    print(f"\nBone: {bone_name}")
    frames = sorted(frame_locs.keys())
    for f in frames[::5]:  # Every 5 frames
        loc = frame_locs[f]
        print(f"  f{f:3d}: x={loc[0]:8.2f}, y={loc[1]:8.2f}, z={loc[2]:8.2f}")
    
    # Check vertical range
    y_vals = [frame_locs[f][1] for f in frames]
    print(f"\n  Y range: [{min(y_vals):.2f}, {max(y_vals):.2f}]")
    print(f"  Y delta (max-min): {max(y_vals)-min(y_vals):.2f}")
    
    # Source pelvis height range for comparison
    src_y = [s_pelvis.world_location_animation.get(f, s_pelvis.head)[1] for f in frames]
    print(f"\n  Source pelvis Y range: [{min(src_y):.3f}m, {max(src_y):.3f}m]")
    print(f"  Source Y delta: {(max(src_y)-min(src_y))*100:.2f}cm")

# Check the target rest pose
print("\n" + "="*80)
print("TARGET SKELETON REST POSITIONS")
print("="*80)
for bn in ['Genesis3Female', 'hip', 'pelvis', 'lThighBend', 'lShin', 'lFoot']:
    b = tgt_skel.get_bone_case_insensitive(bn)
    if b:
        print(f"  {b.name}: head={b.head.round(2)}, parent={b.parent_name}")
        wm = b.world_matrix
        print(f"    world_pos={wm[3,:3].round(2)}")

print("\n=== DONE ===")
