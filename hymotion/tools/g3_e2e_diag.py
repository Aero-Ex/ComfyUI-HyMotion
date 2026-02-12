"""End-to-end diagnostic: Check PreRotation, rotation order, and verify FBX output."""
import sys, os, numpy as np
from scipy.spatial.transform import Rotation as R
import fbx
from fbx import FbxTime

sys.path.insert(0, os.path.dirname(__file__))
from retarget_fbx import (
    load_npz, load_fbx, load_bone_mapping, retarget_animation,
    apply_retargeted_animation, save_fbx, get_fbx_rotation_order_str
)

SRC = "/home/aero/comfy/ComfyUI/output/hymotion_npz/motiondata_20260128_162755994_fd930663_000.npz"
TGT = "/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx"
OUT = "/tmp/g3_diag_test.fbx"

# 1. Check PreRotation and PostRotation of key G3 bones
print("="*80)
print("PART 1: PreRotation / PostRotation / RotationOrder of G3 bones")
print("="*80)

tgt_man, tgt_scene, tgt_skel = load_fbx(TGT, use_bind_pose=False)

def check_node_props(node, depth=0):
    name = node.GetName()
    key_bones = ['hip', 'pelvis', 'abdomenLower', 'abdomenUpper', 'chestLower', 'chestUpper',
                 'lThighBend', 'lThighTwist', 'lShin', 'lFoot',
                 'lShldrBend', 'lShldrTwist', 'lForearmBend', 'lForearmTwist', 'lHand',
                 'lCollar', 'neckLower', 'neckUpper', 'head', 'Genesis3Female']
    if name in key_bones:
        pre = node.PreRotation.Get()
        post = node.PostRotation.Get()
        rot = node.LclRotation.Get()
        rot_order = get_fbx_rotation_order_str(node)
        print(f"  {name}:")
        print(f"    PreRotation:  [{pre[0]:.2f}, {pre[1]:.2f}, {pre[2]:.2f}]")
        print(f"    PostRotation: [{post[0]:.2f}, {post[1]:.2f}, {post[2]:.2f}]")
        print(f"    LclRotation:  [{rot[0]:.2f}, {rot[1]:.2f}, {rot[2]:.2f}]")
        print(f"    RotationOrder: {rot_order}")
    for i in range(node.GetChildCount()):
        check_node_props(node.GetChild(i), depth+1)

check_node_props(tgt_scene.GetRootNode())

# 2. Run retarget and check output
print("\n" + "="*80)
print("PART 2: Retarget and verify output")
print("="*80)

src_skel = load_npz(SRC)
tgt_man2, tgt_scene2, tgt_skel2 = load_fbx(TGT, use_bind_pose=False)
mapping = load_bone_mapping('', src_skel, tgt_skel2)

rots, locs, active = retarget_animation(
    src_skel, tgt_skel2, mapping, smart_arm_align=True, auto_stride=True
)

# Check what LOCAL rotations we're writing
print("\nLocal rotations being written (key bones, frames 0,10,20,30):")
for bn in ['hip', 'lThighBend', 'lThighTwist', 'lShin', 'lForearmBend', 'abdomenLower', 'abdomenUpper']:
    if bn in rots:
        print(f"\n  {bn}:")
        for f in [0, 10, 20, 30]:
            if f in rots[bn]:
                q = rots[bn][f]
                r = R.from_quat([q[1], q[2], q[3], q[0]])
                e = r.as_euler('xyz', degrees=True)
                print(f"    f{f}: euler=[{e[0]:7.1f}, {e[1]:7.1f}, {e[2]:7.1f}], norm={np.linalg.norm(q):.6f}")

# Check translations
print("\nTranslations being written (hip):")
for bn in locs:
    print(f"\n  {bn}:")
    for f in [0, 10, 20, 30, 50, 100, 150]:
        if f in locs[bn]:
            loc = locs[bn][f]
            print(f"    f{f}: [{loc[0]:8.2f}, {loc[1]:8.2f}, {loc[2]:8.2f}]")

# 3. Apply and export, then re-read
print("\n" + "="*80)
print("PART 3: Export and re-read to verify FBX output")
print("="*80)

src_time_mode = tgt_scene2.GetGlobalSettings().GetTimeMode()
apply_retargeted_animation(tgt_scene2, tgt_skel2, rots, locs, src_skel.frame_start, src_skel.frame_end, src_time_mode)
save_fbx(tgt_man2, tgt_scene2, OUT)

# Re-read the output
print("\nRe-reading exported FBX...")
from fbx import FbxManager, FbxImporter, FbxScene, FbxIOSettings
mgr = FbxManager.Create()
ios = FbxIOSettings.Create(mgr, "IOSRoot")
mgr.SetIOSettings(ios)
imp = FbxImporter.Create(mgr, "")
imp.Initialize(OUT, -1, ios)
scene = FbxScene.Create(mgr, "")
imp.Import(scene)
imp.Destroy()

tmode = scene.GetGlobalSettings().GetTimeMode()

def check_exported_node(node, key_bones):
    name = node.GetName()
    if name in key_bones:
        print(f"\n  {name}:")
        # Check LclRotation at various frames
        for f in [0, 10, 20, 30]:
            t = FbxTime()
            t.SetFrame(f, tmode)
            # Evaluate the node's local transform at this time
            local_rot = node.EvaluateLocalRotation(t)
            world_rot = node.EvaluateGlobalTransform(t)
            
            lr = [local_rot[0], local_rot[1], local_rot[2]]
            wr = world_rot.GetR()
            wt = world_rot.GetT()
            
            print(f"    f{f}: local=[{lr[0]:7.1f}, {lr[1]:7.1f}, {lr[2]:7.1f}], world_r=[{wr[0]:7.1f}, {wr[1]:7.1f}, {wr[2]:7.1f}], world_t=[{wt[0]:7.1f}, {wt[1]:7.1f}, {wt[2]:7.1f}]")
    
    for i in range(node.GetChildCount()):
        check_exported_node(node.GetChild(i), key_bones)

check_bones = ['hip', 'pelvis', 'lThighBend', 'lThighTwist', 'lShin', 'lFoot',
               'abdomenLower', 'abdomenUpper', 'lShldrBend', 'lForearmBend', 'head']
check_exported_node(scene.GetRootNode(), check_bones)

print("\n=== DONE ===")
