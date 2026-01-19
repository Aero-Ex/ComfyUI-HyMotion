import sys
import os
import numpy as np

# Import the retargeting module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retargetfbxnpzfull import load_fbx, load_npz, load_bone_mapping, get_skeleton_height

# Target FBX files to test
TARGET_FILES = [
    r"D:\rigged_1768725759_articulationxl.fbx",
    r"D:\rigged_1768725846_articulationxl.fbx",
    r"D:\rigged_1768725894_articulationxl.fbx",
    r"D:\rigged_1768725910_articulationxl.fbx",
    r"D:\rigged_1768725926_articulationxl.fbx",
    r"D:\rigged_1768725943_articulationxl.fbx",
    r"D:\SK_teddy.fbx",
]

# Sources
NPZ_SOURCE = r"D:\@home\aero\comfy\ComfyUI\output\hymotion_npz\motion_20260119_085655633_f1bd3a14_000.npz"
FBX_SOURCE = r"D:\@home\aero\comfy\ComfyUI\output\hymotion_fbx\motion_20260119_085654402_b5efb627_000.fbx"

def test_parity():
    print("=" * 100)
    print("RETARGETING PARITY BATCH TEST: NPZ vs FBX")
    print("=" * 100)

    # Load sources once
    print(f"Loading NPZ Source: {os.path.basename(NPZ_SOURCE)}")
    npz_skel = load_npz(NPZ_SOURCE)
    npz_h = get_skeleton_height(npz_skel)
    
    print(f"Loading FBX Source: {os.path.basename(FBX_SOURCE)}")
    fbx_man, fbx_scene, fbx_skel = load_fbx(FBX_SOURCE, sample_rest_frame=0)
    fbx_h = get_skeleton_height(fbx_skel)

    print(f"\nSource Heights - NPZ: {npz_h:.4f}, FBX: {fbx_h:.4f}")
    print("-" * 100)

    summary = []

    for tgt_path in TARGET_FILES:
        tgt_name = os.path.basename(tgt_path)
        print(f"\nTarget: {tgt_name}")
        
        if not os.path.exists(tgt_path):
            print(f"  ERROR: File not found")
            summary.append((tgt_name, "MISSING", "MISSING", "FAIL"))
            continue

        # Load target
        try:
            tgt_man, tgt_scene, tgt_skel = load_fbx(tgt_path)
            tgt_h = get_skeleton_height(tgt_skel)
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            summary.append((tgt_name, "ERROR", "ERROR", "FAIL"))
            continue

        # Test NPZ mapping
        npz_mapping = load_bone_mapping("", npz_skel, tgt_skel)
        npz_scale = tgt_h / npz_h if npz_h > 0 else 0
        
        # Test FBX mapping
        fbx_mapping = load_bone_mapping("", fbx_skel, tgt_skel)
        fbx_scale = tgt_h / fbx_h if fbx_h > 0 else 0

        # Check parity
        npz_set = set(npz_mapping.items())
        fbx_set = set(fbx_mapping.items())
        parity = "YES" if npz_set == fbx_set else "NO"
        
        print(f"  NPZ Mapping: {len(npz_mapping)} bones, Scale: {npz_scale:.4f}")
        print(f"  FBX Mapping: {len(fbx_mapping)} bones, Scale: {fbx_scale:.4f}")
        print(f"  Parity: {parity}")

        if parity == "NO":
            only_npz = npz_set - fbx_set
            only_fbx = fbx_set - npz_set
            if only_npz: print(f"    Only in NPZ: {sorted(list(only_npz))}")
            if only_fbx: print(f"    Only in FBX: {sorted(list(only_fbx))}")
        
        summary.append((tgt_name, len(npz_mapping), len(fbx_mapping), parity))
        print("-" * 100)

    # Final Summary Table
    print("\n" + "=" * 100)
    print(f"{'Target FBX':<40} | {'NPZ Bones':<10} | {'FBX Bones':<10} | {'Parity'}")
    print("-" * 100)
    for name, npz_c, fbx_c, p in summary:
        print(f"{name:<40} | {str(npz_c):<10} | {str(fbx_c):<10} | {p}")
    print("=" * 100)

if __name__ == "__main__":
    test_parity()
