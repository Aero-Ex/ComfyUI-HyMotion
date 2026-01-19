"""
Detailed diagnostic to show which bones are mapped vs missing.
Usage: python diagnose_mapping_detail.py <target_fbx>
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retargetfbxnpzfull import load_fbx, load_npz, load_bone_mapping, Skeleton

SOURCE_FILE = r"D:\@home\aero\comfy\ComfyUI\output\hymotion_fbx\motion_20260118_091426163_89203df0_002.fbx"

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_mapping_detail.py <target_fbx>")
        sys.exit(1)
    
    target_file = sys.argv[1]
    
    print(f"Loading source: {os.path.basename(SOURCE_FILE)}")
    src_man, src_scene, src_skel = load_fbx(SOURCE_FILE, sample_rest_frame=0)
    print(f"Source bones: {len(src_skel.bones)}")
    
    print(f"\nLoading target: {os.path.basename(target_file)}")
    tgt_man, tgt_scene, tgt_skel = load_fbx(target_file)
    print(f"Target bones: {len(tgt_skel.bones)}")
    
    print("\n" + "="*60)
    print("RUNNING MAPPING...")
    print("="*60)
    
    mapping = load_bone_mapping("", src_skel, tgt_skel)
    
    print("\n" + "="*60)
    print("MAPPING RESULTS")
    print("="*60)
    
    # Categorize source bones
    categories = {
        'Torso': ['pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head'],
        'Left Arm': ['l_collar', 'l_shoulder', 'l_elbow', 'l_wrist'],
        'Right Arm': ['r_collar', 'r_shoulder', 'r_elbow', 'r_wrist'],
        'Left Leg': ['l_hip', 'l_knee', 'l_ankle', 'l_foot'],
        'Right Leg': ['r_hip', 'r_knee', 'r_ankle', 'r_foot'],
        'Left Fingers': ['l_thumb1', 'l_thumb2', 'l_thumb3', 'l_index1', 'l_index2', 'l_index3', 
                         'l_middle1', 'l_middle2', 'l_middle3', 'l_ring1', 'l_ring2', 'l_ring3',
                         'l_pinky1', 'l_pinky2', 'l_pinky3'],
        'Right Fingers': ['r_thumb1', 'r_thumb2', 'r_thumb3', 'r_index1', 'r_index2', 'r_index3', 
                          'r_middle1', 'r_middle2', 'r_middle3', 'r_ring1', 'r_ring2', 'r_ring3',
                          'r_pinky1', 'r_pinky2', 'r_pinky3'],
    }
    
    for cat_name, bones in categories.items():
        print(f"\n{cat_name}:")
        for bone in bones:
            if bone in mapping:
                print(f"  ✅ {bone} -> {mapping[bone]}")
            else:
                print(f"  ❌ {bone} (NOT MAPPED)")
    
    # Show target bones that have no mapping
    mapped_targets = set(mapping.values())
    unmapped_targets = [b for b in tgt_skel.bones.keys() if b not in mapped_targets]
    
    print(f"\n{'='*60}")
    print(f"UNMAPPED TARGET BONES ({len(unmapped_targets)}):")
    print("="*60)
    for b in sorted(unmapped_targets):
        bone = tgt_skel.bones[b]
        children = tgt_skel.get_children(b)
        child_names = [c.name for c in children] if children else []
        print(f"  {b} (parent: {bone.bone_parent_name}, children: {child_names})")

if __name__ == "__main__":
    main()
