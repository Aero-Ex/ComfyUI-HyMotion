"""Quick check of bone_children and parent hierarchy for G3."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from retarget_fbx import load_fbx

TGT = "/home/aero/Documents/G3F-Tpose-WithoutLocks.fbx"
tgt_man, tgt_scene, tgt_skel = load_fbx(TGT, use_bind_pose=False)

print("=== ALL bone_children entries ===")
for parent, children in sorted(tgt_skel.bone_children.items()):
    print(f"  {parent}: {children}")

print(f"\n=== Key bone details ===")
for bn in ['hip', 'pelvis', 'abdomenLower', 'abdomenUpper', 'chestLower',
           'lThighBend', 'lThighTwist', 'lShin', 'lFoot', 'lToe',
           'lShldrBend', 'lShldrTwist', 'lForearmBend', 'lForearmTwist', 'lHand',
           'lCollar', 'Genesis3Female']:
    b = tgt_skel.get_bone_case_insensitive(bn)
    if b:
        print(f"  {b.name}: parent={b.parent_name}, bone_parent={b.bone_parent_name}")
    else:
        print(f"  {bn}: NOT FOUND")
